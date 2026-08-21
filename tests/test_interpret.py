"""REAL Triton kernels executed on CPU via the interpreter (TRITON_INTERPRET=1).

The interpreter runs the exact kernel IR instruction-by-instruction with numpy
— this validates kernel logic (indexing, masking, reduction order, scatter)
without a GPU. Tensor-core numerics and eviction hints are the only deltas.

Run:  TRITON_INTERPRET=1 pytest -m interpret -q
(the env var must be set BEFORE triton is first imported, hence a dedicated
marker excluded from the default CPU suite)
"""

import os

import pytest
import torch

pytestmark = pytest.mark.interpret

if os.getenv("TRITON_INTERPRET", "0") != "1":
    pytest.skip("requires TRITON_INTERPRET=1 (set before triton import)",
                allow_module_level=True)

from tests.conftest import make_moe_inputs
from treemoe.kernels.op1_tree_moe import tree_moe_forward
from treemoe.kernels.op4_commit import fused_verify_commit
from treemoe.ref.tree_moe_ref import tree_moe_forward_ref
from treemoe.ref.verify_ref import tree_verify_ref
from tests.test_kernel_semantics import _random_tree

N, E, H, I = 64, 8, 64, 128  # H=64 exercises the shrunken-tile path (bk=64)


@pytest.mark.parametrize("budget", [2, 4, 8])
def test_full_op1_pipeline_interpreted(rng, budget):
    """fused Kernel A + gemm1 + deterministic gemm2 + combine vs reference."""
    x, w1, w2, w3, router, accept = make_moe_inputs(N, E, H, I, rng)
    out = tree_moe_forward(x, w1, w2, w3, router, accept, budget, deterministic=True)
    ref = tree_moe_forward_ref(x, w1, w2, w3, router, accept, budget)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_op1_atomic_path_interpreted(rng):
    x, w1, w2, w3, router, accept = make_moe_inputs(N, E, H, I, rng)
    out = tree_moe_forward(x, w1, w2, w3, router, accept, 8, deterministic=False)
    ref = tree_moe_forward_ref(x, w1, w2, w3, router, accept, 8)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_op1_packed_weight_path_interpreted(rng):
    """16-bit weights take the PACK_W=1 branch (u32-packed loads + bit-unpack);
    validates the little-endian unpack numerics end-to-end vs the reference.
    fp16, not bf16: the interpreter's torch-bf16 -> numpy bridge is broken
    (numpy has no bf16), garbling packed AND unpacked paths equally; fp16
    executes the identical packed code path and is bit-exact vs unpacked."""
    x, w1, w2, w3, router, accept = make_moe_inputs(N, E, H, I, rng,
                                                    dtype=torch.float16)
    out = tree_moe_forward(x, w1, w2, w3, router, accept, 4, deterministic=True)
    ref = tree_moe_forward_ref(x, w1, w2, w3, router, accept, 4)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_op1_degrade_branch_interpreted(rng):
    x, w1, w2, w3, router, _ = make_moe_inputs(N, E, H, I, rng)
    accept = torch.zeros(N)  # all nodes below tau -> top-1 gates
    out = tree_moe_forward(x, w1, w2, w3, router, accept, 8)
    ref = tree_moe_forward_ref(x, w1, w2, w3, router, accept, 8)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("seed", range(10))
def test_op4_greedy_kernel_interpreted(seed):
    """the actual _tree_verify_greedy_kernel (+ argmax kernel) vs reference."""
    g = torch.Generator().manual_seed(seed)
    n, vocab = 16, 64
    tokens, parent, children = _random_tree(g, n, vocab)
    logits = torch.randn(n, vocab, generator=g)
    if seed % 3 == 0:
        for node in range(n):
            if children[node]:
                logits[node, tokens[children[node][0]]] += 100.0

    res = fused_verify_commit(logits, torch.softmax(logits, -1), tokens, parent,
                              children, kv=None, temperature=0.0)
    ref = tree_verify_ref(logits, torch.softmax(logits, -1), tokens, parent,
                          children, temperature=0.0)
    assert int(res.num_accepted) == int(ref.num_accepted)
    m = int(ref.num_accepted)
    assert torch.equal(res.accepted_slots[:m], ref.accepted_slots[:m])
    assert int(res.bonus_token) == int(ref.bonus_token)


def test_op4_all_negative_logits_interpreted():
    """regression: argmax kernel -inf init on an all-negative logits row."""
    tokens = torch.tensor([0, 5, 2])
    children = [[1], [2], []]
    logits = torch.full((3, 8), -10.0)
    logits[0, 5] = -1.5
    logits[1, 2] = -2.0
    logits[2, 7] = -3.0
    res = fused_verify_commit(logits, torch.softmax(logits, -1), tokens,
                              torch.tensor([-1, 0, 1]), children, kv=None,
                              temperature=0.0)
    assert int(res.num_accepted) == 2 and int(res.bonus_token) == 7


def test_op4_kv_commit_kernel_interpreted(tiny_config):
    """_kv_commit_kernel vs PagedKVCache.commit_tree on identical caches."""
    from treemoe.model.kv_cache import PagedKVCache

    g = torch.Generator().manual_seed(5)
    kv_a = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=torch.float32)
    kv_b = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=torch.float32)
    for kv in (kv_a, kv_b):
        kv.seq_len = 10
        kv._ensure_capacity(10)
    scratch = torch.randn(
        tiny_config.num_layers, 1, kv_a.block_size,
        tiny_config.num_kv_heads, tiny_config.head_dim, generator=g,
    )
    kv_a.k[:, kv_a.tree_block] = scratch.squeeze(1)
    kv_a.v[:, kv_a.tree_block] = scratch.squeeze(1) + 1
    kv_b.k[:, kv_b.tree_block] = scratch.squeeze(1)
    kv_b.v[:, kv_b.tree_block] = scratch.squeeze(1) + 1

    n, vocab = 8, 32
    tokens, parent, children = _random_tree(g, n, vocab)
    logits = torch.randn(n, vocab, generator=g)
    for node in range(n):  # force a non-empty accept chain
        if children[node]:
            logits[node, tokens[children[node][0]]] += 100.0

    res = fused_verify_commit(logits, torch.softmax(logits, -1), tokens, parent,
                              children, kv=kv_a, temperature=0.0, max_depth=6)
    kv_b.commit_tree(res.accepted_slots)
    assert torch.equal(kv_a.k, kv_b.k) and torch.equal(kv_a.v, kv_b.v)
    assert kv_a.seq_len == kv_b.seq_len
