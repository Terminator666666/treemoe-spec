"""Op2 (draft-guided expert prefetch) unit tests — spec §3.2 / plan Task 4.1-4.2.

Coverage gap found during the 4090 bring-up: op2 had zero tests. CPU tests
cover RouterPredictor semantics; gpu-marked tests cover the H2D ring buffer
(bitwise copy correctness, event ordering, eviction) and the L2-warm kernel
(must not mutate weights).
"""

import pytest
import torch

from treemoe.kernels.op2_prefetch import HostExpertPool, RouterPredictor, l2_warm

N, L, E, H = 6, 4, 8, 32


def _predictor(seed: int = 0) -> RouterPredictor:
    torch.manual_seed(seed)
    return RouterPredictor(hidden=H, num_layers=L, num_experts=E)


# ---------------- RouterPredictor (CPU) ----------------

def test_predictor_output_shape():
    p = _predictor()
    out = p(torch.randn(N, H))
    assert out.shape == (N, L, E)


def test_bitmap_respects_budget_exactly():
    p = _predictor()
    for budget in (1, 2, 4, 8):
        bm = p.predict_bitmap(torch.randn(N, H), budget)
        assert bm.shape == (L, E) and bm.dtype == torch.bool
        assert (bm.sum(-1) == budget).all()


def test_bitmap_keeps_highest_demand_experts():
    p = _predictor()
    feats = torch.randn(N, H)
    bm = p.predict_bitmap(feats, 3)
    demand = torch.softmax(p(feats), dim=-1).sum(0)      # [L, E], same math
    expect = torch.zeros_like(demand, dtype=torch.bool)
    expect.scatter_(1, demand.topk(3, dim=-1).indices, True)
    assert torch.equal(bm, expect)


def test_recall_full_budget_is_one_and_monotone():
    p = _predictor()
    feats = torch.randn(N, H)
    true_topk = torch.randint(0, E, (N, L, 2))
    r = [p.recall_at(feats, true_topk, k) for k in (1, 2, 4, E)]
    assert r[-1] == 1.0                       # predicting everything hits all
    assert all(a <= b + 1e-9 for a, b in zip(r, r[1:]))  # recall grows with k


# ---------------- HostExpertPool ring buffer (GPU) ----------------

EI, EH = 16, 8   # tiny expert shape [I, H]


def _host_weights(g: torch.Generator):
    def mk(*shape):
        t = torch.randn(*shape, generator=g, dtype=torch.bfloat16)
        try:
            return t.pin_memory()
        except RuntimeError:
            return t
    return mk(E, EI, EH), mk(E, EH, EI), mk(E, EI, EH)   # w1, w2, w3


@pytest.mark.gpu
def test_pool_prefetch_then_lookup_bitwise():
    g = torch.Generator().manual_seed(0)
    w1, w2, w3 = _host_weights(g)
    pool = HostExpertPool(num_slots=4, expert_shape=(EI, EH))
    pool.prefetch(3, 5, w1, w2, w3)
    pool.prefetch(3, 1, w1, w2, w3)
    got = pool.lookup(3, 5)
    assert got is not None
    torch.cuda.synchronize()
    assert torch.equal(got["w1"].cpu(), w1[5])
    assert torch.equal(got["w2"].cpu(), w2[5])
    assert torch.equal(got["w3"].cpu(), w3[5])
    assert pool.lookup(3, 1) is not None
    assert pool.lookup(0, 0) is None          # never prefetched


@pytest.mark.gpu
def test_pool_duplicate_prefetch_consumes_no_slot():
    g = torch.Generator().manual_seed(1)
    w1, w2, w3 = _host_weights(g)
    pool = HostExpertPool(num_slots=2, expert_shape=(EI, EH))
    pool.prefetch(0, 0, w1, w2, w3)
    for _ in range(5):
        pool.prefetch(0, 0, w1, w2, w3)       # dedup: cursor must not advance
    pool.prefetch(0, 1, w1, w2, w3)
    assert pool.lookup(0, 0) is not None and pool.lookup(0, 1) is not None


@pytest.mark.gpu
def test_pool_ring_eviction_order():
    g = torch.Generator().manual_seed(2)
    w1, w2, w3 = _host_weights(g)
    pool = HostExpertPool(num_slots=2, expert_shape=(EI, EH))
    for e in (0, 1, 2):                       # 3 fetches into 2 slots
        pool.prefetch(0, e, w1, w2, w3)
    assert pool.lookup(0, 0) is None          # oldest evicted
    torch.cuda.synchronize()
    assert torch.equal(pool.lookup(0, 1)["w1"].cpu(), w1[1])
    assert torch.equal(pool.lookup(0, 2)["w1"].cpu(), w1[2])


# ---------------- l2_warm (GPU) ----------------

@pytest.mark.gpu
def test_l2_warm_does_not_mutate_weights():
    g = torch.Generator().manual_seed(3)
    w = torch.randn(E, EI * EH, generator=g, dtype=torch.bfloat16).cuda()
    before = w.clone()
    stream = torch.cuda.Stream()
    l2_warm(w, torch.tensor([0, 3, 7]), stream)
    torch.cuda.synchronize()
    assert torch.equal(w, before)
