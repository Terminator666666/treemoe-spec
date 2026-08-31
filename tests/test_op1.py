"""Task 2.1: op1/op3 reference behaviour + Triton parity (12 cases, plan §Task 2.1)."""

import pytest
import torch

from tests.conftest import make_moe_inputs
from treemoe.ref.tree_moe_ref import (
    budget_route_ref,
    route_and_bucket_ref,
    tree_moe_forward_ref,
)

N, E, H, I = 64, 8, 64, 128


# ---------------- budget routing (op3) ----------------

@pytest.mark.parametrize("budget", [2, 4, 8])
def test_budget_route_respects_budget(rng, budget):
    gates = torch.softmax(torch.randn(N, E, generator=rng), dim=-1)
    accept = torch.rand(N, generator=rng)
    ids, g = budget_route_ref(gates, accept, budget)
    assert ids.unique().numel() <= budget
    assert torch.allclose(g.sum(-1), torch.ones(N), atol=1e-5)


def test_budget_route_b8_is_identity_topk(rng):
    """B=8 (lossless mode) must reproduce plain top-2 routing."""
    gates = torch.softmax(torch.randn(N, E, generator=rng), dim=-1)
    accept = torch.ones(N)  # no degradation branch
    ids, g = budget_route_ref(gates, accept, 8)
    ref_g, ref_ids = gates.topk(2, dim=-1)
    assert torch.equal(ids.sort(-1).values, ref_ids.sort(-1).values)
    assert torch.allclose(g, ref_g / ref_g.sum(-1, keepdim=True), atol=1e-5)


def test_low_prob_nodes_degrade_to_top1(rng):
    gates = torch.softmax(torch.randn(N, E, generator=rng), dim=-1)
    accept = torch.zeros(N)  # everything below tau
    _, g = budget_route_ref(gates, accept, 8)
    assert torch.equal(g[:, 1], torch.zeros(N))
    assert torch.equal(g[:, 0], torch.ones(N))


# ---------------- bucketing (kernel A semantics) ----------------

def test_bucket_offsets_and_stability(rng):
    x, w1, w2, w3, router, accept = make_moe_inputs(N, E, H, I, rng)
    ids, _, slots, offsets = route_and_bucket_ref(x, router, accept, 8)
    assert offsets[-1] == 2 * N
    flat = ids.reshape(-1)
    for e in range(E):
        seg = slots[offsets[e] : offsets[e + 1]]
        assert (flat[seg] == e).all()
        tokens = seg // 2
        assert (tokens.diff() >= 0).all()  # DFS order preserved inside expert


def test_bucket_all_tokens_one_expert(rng):
    """Extreme case: router forces every token to experts {0,1}.

    Sign-proof construction: bias-free logit = w.x flips with x's sign, so
    instead pin x[:,0] > 0 and give experts 0/1 the only nonzero weights.
    """
    x, w1, w2, w3, router, accept = make_moe_inputs(N, E, H, I, rng)
    x[:, 0] = x[:, 0].abs() + 1.0
    router.zero_()
    router[0, 0] = 2.0  # logit0 = 2*x0 > logit1 = x0 > 0 = others
    router[1, 0] = 1.0
    _, _, _, offsets = route_and_bucket_ref(x, router, accept, 8)
    assert offsets[1] - offsets[0] == N and offsets[2] - offsets[1] == N
    assert offsets[2] == offsets[-1]


# ---------------- full forward (ref self-consistency) ----------------

def test_ref_forward_matches_naive_at_b8(rng):
    """B=8 reference must equal the plain HF-style per-expert loop."""
    import torch.nn.functional as F

    x, w1, w2, w3, router, _ = make_moe_inputs(N, E, H, I, rng)
    accept = torch.ones(N)
    out = tree_moe_forward_ref(x, w1, w2, w3, router, accept, 8)

    logits = F.linear(x, router)
    gates = torch.softmax(logits, dim=-1)
    topg, topi = gates.topk(2, dim=-1)
    topg = topg / topg.sum(-1, keepdim=True)
    naive = torch.zeros_like(x)
    for e in range(E):
        for k in range(2):
            sel = topi[:, k] == e
            if sel.any():
                xe = x[sel]
                h = F.silu(xe @ w1[e].t()) * (xe @ w3[e].t())
                naive[sel] += (h @ w2[e].t()) * topg[sel, k : k + 1]
    assert torch.allclose(out, naive, rtol=1e-4, atol=1e-5)


def test_empty_expert_contributes_nothing(rng):
    x, w1, w2, w3, router, accept = make_moe_inputs(N, E, H, I, rng)
    x[:, 0] = x[:, 0].abs() + 1.0
    router[7].zero_()
    router[7, 0] = -100.0  # logit7 <= -100: expert 7 never routed (sign-proof)
    w1[7] = float("nan")  # would poison output if touched
    w3[7] = float("nan")
    out = tree_moe_forward_ref(x, w1, w2, w3, router, torch.ones(N), 7)
    assert not out.isnan().any()


# ---------------- Triton parity (GPU only) ----------------

@pytest.mark.gpu
@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("budget", [4, 8])
def test_triton_matches_ref(n, budget):
    """Kernel error vs an fp32 golden must stay within 2x the bf16 reference's
    own error (FlashAttention-style criterion).

    Fixed absolute tolerances are wrong at real shapes: with H=4096/I=14336
    bf16 accumulation, kernel and cuBLAS reference legitimately differ by
    ~1 ulp of the output magnitude (0.125 at |y|~16-32), as measured on a
    4090 (diag_op1.py: all 4 fused/packed combos gave identical max|d|).
    """
    from treemoe.kernels.op1_tree_moe import tree_moe_forward

    g = torch.Generator().manual_seed(7)
    x, w1, w2, w3, router, accept = make_moe_inputs(n, 8, 4096, 14336, g, dtype=torch.bfloat16)
    x, w1, w2, w3 = (t.cuda() for t in (x, w1, w2, w3))
    router, accept = router.cuda(), accept.cuda()
    out = tree_moe_forward(x, w1, w2, w3, router, accept, budget)
    ref = tree_moe_forward_ref(x, w1, w2, w3, router, accept, budget)
    golden = tree_moe_forward_ref(
        x.float(), w1.float(), w2.float(), w3.float(), router.float(), accept, budget
    )
    err_out = (out.float() - golden).abs().max().item()
    err_ref = (ref.float() - golden).abs().max().item()
    assert err_out <= 2 * err_ref + 1e-4, (
        f"kernel err {err_out:.4f} > 2x reference err {err_ref:.4f} + 1e-4"
    )


@pytest.mark.gpu
def test_triton_deterministic_bitwise():
    """deterministic=True must be bitwise reproducible across runs (fp32
    addition is non-associative; the atomic fast path is not)."""
    from treemoe.kernels.op1_tree_moe import tree_moe_forward

    g = torch.Generator().manual_seed(11)
    x, w1, w2, w3, router, accept = make_moe_inputs(64, 8, 4096, 14336, g, dtype=torch.bfloat16)
    x, w1, w2, w3 = (t.cuda() for t in (x, w1, w2, w3))
    router, accept = router.cuda(), accept.cuda()
    outs = [tree_moe_forward(x, w1, w2, w3, router, accept, 8, deterministic=True).clone()
            for _ in range(5)]
    for o in outs[1:]:
        assert torch.equal(o, outs[0])  # bitwise, not allclose


@pytest.mark.gpu
@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("budget", [2, 4, 8])
def test_fused_route_bucket_triton_matches_torch(n, budget):
    """Fused single-CTA Kernel A vs the torch composition, on device.
    n=128 covers the extended range (O((2N)^2)=256^2 rank, ~0.5KB/thread
    spill accepted to kill the torch-fallback launch gap)."""
    from treemoe.kernels.op1_tree_moe import (
        BM, _route_bucket_fused_kernel, route_and_bucket,
    )

    g = torch.Generator().manual_seed(3)
    e, h = 8, 4096
    x = torch.randn(n, h, generator=g, dtype=torch.bfloat16).cuda()
    router = (torch.randn(e, h, generator=g, dtype=torch.bfloat16) * 0.1).cuda()
    accept = torch.rand(n, generator=g).cuda()

    ids_t, gates_t, padded_t, blk_t, s2r_t, max_blocks = route_and_bucket(
        x, router, accept, budget
    )
    topk = torch.zeros(2 * n, dtype=torch.long, device="cuda")
    gates = torch.zeros(2 * n, dtype=torch.float32, device="cuda")
    padded = torch.full((max_blocks * BM,), -2, dtype=torch.long, device="cuda")
    blk = torch.full((max_blocks,), -2, dtype=torch.long, device="cuda")
    s2r = torch.zeros(2 * n, dtype=torch.long, device="cuda")
    _route_bucket_fused_kernel[(1,)](
        x, router, accept, topk, gates, padded, blk, s2r,
        budget, 0.05, N=n, E=e, EP=16, H=h, BK=128,
        MAX_BPE=(2 * n + BM - 1) // BM, BLOCK_M=BM, MAX_BLOCKS=max_blocks,
    )
    assert torch.equal(topk, ids_t.reshape(-1))
    torch.testing.assert_close(gates, gates_t.reshape(-1).float(), rtol=1e-3, atol=1e-3)
    assert torch.equal(padded, padded_t)
    assert torch.equal(blk, blk_t)
    assert torch.equal(s2r, s2r_t)
