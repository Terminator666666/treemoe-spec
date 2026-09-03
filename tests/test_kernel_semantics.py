"""CPU-executable cross-validation of kernel-side semantics (no GPU needed).

Two layers of defence added after the theoretical audit:
  1. kernels.route_and_bucket (the graph-safe torch composition actually fed to
     the Triton GEMMs) must agree with route_and_bucket_ref segment-for-segment,
     and slot_to_row must be the exact inverse of the padded layout.
  2. A line-by-line Python port of _tree_verify_greedy_kernel's control flow
     must reproduce tree_verify_ref's greedy accept path on random trees —
     catches algorithmic bugs in the serial DFS before a GPU is available.
"""

import pytest
import torch

from tests.conftest import make_moe_inputs
from treemoe.kernels.op1_tree_moe import BM, route_and_bucket
from treemoe.ref.tree_moe_ref import route_and_bucket_ref
from treemoe.ref.verify_ref import tree_verify_ref

N, E, H, I = 64, 8, 64, 128


# ---------------- op1 kernel A: padded layout vs reference ----------------

@pytest.mark.parametrize("budget", [2, 4, 8])
def test_padded_layout_matches_ref_segments(rng, budget):
    x, _, _, _, router, accept = make_moe_inputs(N, E, H, I, rng)
    ids_r, gates_r, sorted_slots, offsets = route_and_bucket_ref(x, router, accept, budget)
    ids_k, gates_k, padded, blk_experts, slot_to_row, max_blocks = route_and_bucket(
        x, router, accept, budget
    )
    assert torch.equal(ids_r, ids_k) and torch.allclose(gates_r, gates_k)

    max_bpe = (2 * N + BM - 1) // BM
    for e in range(E):
        seg_ref = sorted_slots[offsets[e] : offsets[e + 1]]
        region = padded[e * max_bpe * BM : (e + 1) * max_bpe * BM]
        seg_pad = region[region >= 0]
        assert torch.equal(seg_pad, seg_ref)  # same slots, same DFS order
        # -1 padding must be a strict suffix of the region (dense prefix)
        n_real = seg_pad.numel()
        assert (region[:n_real] >= 0).all() and (region[n_real:] == -1).all()


def test_slot_to_row_is_exact_inverse(rng):
    x, _, _, _, router, accept = make_moe_inputs(N, E, H, I, rng)
    _, _, padded, _, slot_to_row, _ = route_and_bucket(x, router, accept, 8)
    # every real slot appears exactly once, and slot_to_row points back at it
    assert torch.equal(padded[slot_to_row], torch.arange(2 * N))


def test_block_expert_ids_consistent(rng):
    x, _, _, _, router, accept = make_moe_inputs(N, E, H, I, rng)
    _, _, padded, blk_experts, _, max_blocks = route_and_bucket(x, router, accept, 4)
    max_bpe = max_blocks // E
    for b in range(max_blocks):
        rows = padded[b * BM : (b + 1) * BM]
        if blk_experts[b] < 0:
            assert (rows == -1).all()  # masked block: kernel exits, rows unused
        else:
            assert blk_experts[b] == b // max_bpe
            assert (rows >= 0).any()   # used block must carry >=1 real slot


# ---------------- op4 kernel: greedy DFS simulation vs reference ----------------

def _simulate_greedy_kernel(argmax, tree_tokens, children, max_depth):
    """Line-by-line port of _tree_verify_greedy_kernel (op4_commit.py)."""
    counts = [len(c) for c in children]
    starts = [0] * len(children)
    for i in range(1, len(children)):
        starts[i] = starts[i - 1] + counts[i - 1]
    flat = [c for kids in children for c in kids] or [0]

    accepted_slots = [-1] * len(tree_tokens)
    node, count = 0, 0
    for _d in range(max_depth):
        target_top = int(argmax[node])
        accepted = -1
        for j in range(counts[node]):
            c = flat[starts[node] + j]
            if int(tree_tokens[c]) == target_top and accepted < 0:
                accepted = c
        go = accepted >= 0
        accepted_slots[count] = accepted  # idempotent -1 store on reject
        count += 1 if go else 0
        node = accepted if go else node
    return accepted_slots, int(argmax[node]), count, node


def _random_tree(g, n=16, vocab=64):
    """Random tree in DFS order (parent < child), random branching."""
    parent = [-1]
    for i in range(1, n):
        parent.append(int(torch.randint(0, i, (1,), generator=g)))
    children = [[] for _ in range(n)]
    for i in range(1, n):
        children[parent[i]].append(i)
    tokens = torch.randint(0, vocab, (n,), generator=g)
    # siblings must carry distinct tokens (EAGLE-2 top-k invariant)
    for kids in children:
        seen = set()
        for c in kids:
            while int(tokens[c]) in seen:
                tokens[c] = (tokens[c] + 1) % vocab
            seen.add(int(tokens[c]))
    return tokens, torch.tensor(parent), children


@pytest.mark.parametrize("seed", range(20))
def test_greedy_kernel_sim_matches_ref(seed):
    g = torch.Generator().manual_seed(seed)
    n, vocab, max_depth = 16, 64, 6
    tokens, parent, children = _random_tree(g, n, vocab)
    logits = torch.randn(n, vocab, generator=g)
    if seed % 3 == 0:  # exercise the full-accept path too
        for node in range(n):
            if children[node]:
                logits[node, tokens[children[node][0]]] += 100.0

    ref = tree_verify_ref(logits, torch.softmax(logits, -1), tokens, parent,
                          children, temperature=0.0)
    argmax = logits.argmax(-1)
    sim_slots, sim_bonus, sim_count, _ = _simulate_greedy_kernel(
        argmax, tokens, children, max_depth
    )
    assert sim_count == int(ref.num_accepted)
    assert sim_slots[:sim_count] == ref.accepted_slots[:sim_count].tolist()
    assert sim_bonus == int(ref.bonus_token)


def test_greedy_all_negative_logits():
    """Regression for the -1.0 argmax-init bug: all-negative logits row."""
    n, vocab = 3, 8
    tokens = torch.tensor([0, 5, 2])
    children = [[1], [2], []]
    logits = torch.full((n, vocab), -10.0)
    logits[0, 5] = -1.5  # argmax=5 even though every value < 0
    logits[1, 2] = -2.0
    logits[2, 7] = -3.0
    ref = tree_verify_ref(logits, torch.softmax(logits, -1), tokens,
                          torch.tensor([-1, 0, 1]), children, temperature=0.0)
    assert int(ref.num_accepted) == 2 and int(ref.bonus_token) == 7
    sim_slots, sim_bonus, sim_count, _ = _simulate_greedy_kernel(
        logits.argmax(-1), tokens, children, 6
    )
    assert sim_count == 2 and sim_bonus == 7


# ---------------- fused Kernel A: register-level algorithm simulation ----------------

def _simulate_fused_route_bucket(x, router, accept, budget, tau=0.05, ep=16):
    """Line-by-line torch port of _budget_bucket_fused_kernel (op1_tree_moe.py)."""
    n = x.shape[0]
    e = router.shape[0]
    max_bpe = (2 * n + BM - 1) // BM
    max_blocks = e * max_bpe

    logits = torch.full((n, ep), float("-inf"))
    logits[:, :e] = (x @ router.t()).float()
    gates = torch.softmax(logits, dim=-1)                       # pad lanes -> 0

    scores = (accept[:, None] * gates).sum(0)
    scores[e:] = float("-inf")
    keep = torch.zeros(ep, dtype=torch.bool)
    for i in range(e):                                           # first-occurrence top-B
        cand = torch.where(keep, torch.tensor(float("-inf")), scores)
        am = int(cand.argmax())
        if i < budget:
            keep[am] = True

    mg = torch.where(keep[None, :], gates, torch.zeros(()))
    i1 = mg.argmax(1)
    g1 = mg.max(1).values
    mg2 = mg.clone()
    mg2[torch.arange(n), i1] = 0.0
    i2 = mg2.argmax(1)
    g2 = mg2.max(1).values
    s = g1 + g2
    tg1, tg2 = g1 / s, g2 / s
    degrade = accept < tau
    tg1 = torch.where(degrade, torch.ones(()), tg1)
    tg2 = torch.where(degrade, torch.zeros(()), tg2)

    slots = torch.arange(2 * n)
    fe = torch.stack([i1, i2], dim=1).reshape(-1)                # tl.interleave
    eq = fe[:, None] == fe[None, :]
    lower = slots[None, :] < slots[:, None]
    rank = (eq & lower).sum(1)
    dest = fe * (max_bpe * BM) + rank

    pos = torch.arange(max_blocks * BM)
    hit = pos[:, None] == dest[None, :]
    padded = (hit * (slots[None, :] + 1)).sum(1) - 1             # -1 where no slot

    counts = (fe[None, :] == torch.arange(ep)[:, None]).sum(1)
    blocks_needed = (counts + BM - 1) // BM
    blk = torch.arange(max_blocks)
    need = blocks_needed[blk // max_bpe]
    used = (blk % max_bpe) < need
    blk_ids = torch.where(used, blk // max_bpe, torch.full((), -1, dtype=torch.long))
    topk_flat = torch.stack([i1, i2], dim=1).reshape(-1)
    gates_flat = torch.stack([tg1, tg2], dim=1).reshape(-1)
    return topk_flat, gates_flat, padded, blk_ids, dest


@pytest.mark.parametrize("budget", [2, 4, 8])
def test_fused_kernel_a_sim_matches_torch_composition(rng, budget):
    x, _, _, _, router, accept = make_moe_inputs(N, E, H, I, rng)
    topk_f, gates_f, padded_f, blk_f, dest_f = _simulate_fused_route_bucket(
        x, router, accept, budget
    )
    ids_t, gates_t, padded_t, blk_t, s2r_t, _ = route_and_bucket(x, router, accept, budget)
    assert torch.equal(topk_f, ids_t.reshape(-1))
    assert torch.allclose(gates_f, gates_t.reshape(-1).float(), atol=1e-6)
    assert torch.equal(padded_f, padded_t)
    assert torch.equal(blk_f, blk_t)
    assert torch.equal(dest_f, s2r_t)


def test_fused_kernel_a_sim_degrade_branch(rng):
    x, _, _, _, router, _ = make_moe_inputs(N, E, H, I, rng)
    accept = torch.zeros(N)  # all below tau -> top-1 gates [1, 0]
    _, gates_f, _, _, _ = _simulate_fused_route_bucket(x, router, accept, 8)
    assert torch.equal(gates_f.reshape(N, 2)[:, 0], torch.ones(N))
    assert torch.equal(gates_f.reshape(N, 2)[:, 1], torch.zeros(N))


# ---------------- op4 online softmax: blockwise rescaling simulation ----------------

@pytest.mark.parametrize("vb", [4, 16, 64])
def test_online_softmax_sim_matches_torch(vb):
    """Port of the Milakov-Gimelshein online pass in _postprocess_softmax_kernel:
    running max with sum rescaling must equal torch.softmax on any block size."""
    g = torch.Generator().manual_seed(3)
    v = 100  # deliberately not a multiple of vb (exercises the mask path)
    logits = torch.randn(v, generator=g) * 10
    vmax, vsum = -1e38, 0.0
    for v0 in range(0, v, vb):
        blk = logits[v0 : v0 + vb]
        nmax = max(vmax, float(blk.max()))
        vsum = vsum * torch.exp(torch.tensor(vmax - nmax)).item() + float(
            torch.exp(blk - nmax).sum()
        )
        vmax = nmax
    probs = torch.exp(logits - vmax) / vsum
    torch.testing.assert_close(probs, torch.softmax(logits, -1), rtol=1e-5, atol=1e-7)
