"""Op2 (draft-guided expert prefetch) unit tests — spec §3.2 / plan Task 4.1-4.2.

Coverage gap found during the 4090 bring-up: op2 had zero tests. CPU tests
cover RouterPredictor semantics; gpu-marked tests cover the H2D ring buffer
(bitwise copy correctness, event ordering, eviction) and the L2-warm kernel
(must not mutate weights).
"""

import pytest
import torch

from treemoe.kernels.op2_prefetch import (
    HostExpertPool, LayerPrefetcher, RouterPredictor, l2_warm,
)

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


# ---------------- LayerPrefetcher (engine integration, CPU) ----------------

def _offload_all(w):
    import dataclasses
    return dataclasses.replace(
        w, layers=[dataclasses.replace(lw, experts_on_gpu=False) for lw in w.layers])


def _forward_pair(cfg, w, rng, prefetcher):
    """(resident logits, prefetched-offload logits) on the same weights."""
    from treemoe.model.kv_cache import PagedKVCache
    from treemoe.model.mixtral import MixtralForward, naive_moe

    ids = torch.randint(0, cfg.vocab_size, (5,), generator=rng)
    pos = torch.arange(5)
    kv1 = PagedKVCache(cfg, num_blocks=8, device="cpu", dtype=cfg.dtype)
    resident = MixtralForward(w, kv1, moe_fn=naive_moe).forward(ids, pos)
    kv2 = PagedKVCache(cfg, num_blocks=8, device="cpu", dtype=cfg.dtype)
    off = MixtralForward(_offload_all(w), kv2, moe_fn=naive_moe,
                         prefetcher=prefetcher).forward(ids, pos)
    return resident, off


def test_layer_prefetcher_forward_matches_resident(tiny_config, rng):
    from test_parity import random_weights

    w = random_weights(tiny_config, rng)
    pf = LayerPrefetcher(_offload_all(w).layers, depth=2)
    resident, off = _forward_pair(tiny_config, w, rng, pf)
    assert torch.equal(resident, off)          # bitwise: staging is a pure copy


def test_layer_prefetcher_cycles_buffers_and_partial_offload(tiny_config, rng):
    """More offloaded layers than buffers (depth=2, 4 offloaded) + one resident
    layer in the middle: buffer reuse ordering and subset mapping."""
    import dataclasses

    from test_parity import random_weights
    from treemoe.model.kv_cache import PagedKVCache
    from treemoe.model.mixtral import MixtralForward, naive_moe

    cfg = dataclasses.replace(tiny_config, num_layers=5)
    w = random_weights(cfg, rng)
    ids = torch.randint(0, cfg.vocab_size, (5,), generator=rng)
    pos = torch.arange(5)
    kv1 = PagedKVCache(cfg, num_blocks=8, device="cpu", dtype=cfg.dtype)
    resident = MixtralForward(w, kv1, moe_fn=naive_moe).forward(ids, pos)

    w_off = dataclasses.replace(w, layers=[
        dataclasses.replace(lw, experts_on_gpu=(i == 2)) for i, lw in enumerate(w.layers)])
    pf = LayerPrefetcher(w_off.layers, depth=2)
    assert pf.offload_ids == [0, 1, 3, 4]
    kv2 = PagedKVCache(cfg, num_blocks=8, device="cpu", dtype=cfg.dtype)
    off = MixtralForward(w_off, kv2, moe_fn=naive_moe, prefetcher=pf).forward(ids, pos)
    assert torch.equal(resident, off)
    # two forward passes reuse the same buffers correctly
    kv3 = PagedKVCache(cfg, num_blocks=8, device="cpu", dtype=cfg.dtype)
    off2 = MixtralForward(w_off, kv3, moe_fn=naive_moe, prefetcher=pf).forward(ids, pos)
    assert torch.equal(resident, off2)


def test_layer_prefetcher_bitmap_copies_only_predicted_rows(rng):
    """Bitmap mode copies exactly the predicted rows; unpredicted rows keep the
    previous buffer occupant's data (stale by contract, spec §3.2 / op3)."""
    from treemoe.model.weights import LayerWeights

    E_, I_, H_ = 4, 6, 8

    def lw():
        return LayerWeights(
            input_layernorm=torch.ones(H_), post_attn_layernorm=torch.ones(H_),
            attn={}, router=torch.zeros(E_, H_),
            w1=torch.randn(E_, I_, H_, generator=rng),
            w2=torch.randn(E_, H_, I_, generator=rng),
            w3=torch.randn(E_, I_, H_, generator=rng),
            experts_on_gpu=False)

    layers = [lw(), lw()]
    pf = LayerPrefetcher(layers, depth=1)

    pf.begin()                                  # bitmap=None: full copies
    assert torch.equal(pf.acquire(0)["w1"], layers[0].w1)
    pf.release(0)
    assert torch.equal(pf.acquire(1)["w2"], layers[1].w2)
    pf.release(1)

    bm = torch.ones(2, E_, dtype=torch.bool)
    bm[1] = False
    bm[1, 0] = True                             # layer 1: only expert 0 predicted
    pf.set_bitmap(bm)
    pf.begin()
    pf.acquire(0)                               # full copy: buffer holds layer 0
    pf.release(0)
    got = pf.acquire(1)                         # depth=1: same buffer, row-copy
    assert torch.equal(got["w1"][0], layers[1].w1[0])   # predicted row is fresh
    assert torch.equal(got["w1"][1], layers[0].w1[1])   # unpredicted row: stale
    assert pf.staged_rows_total == 13           # pass 1: 8 full; pass 2: 4 + 1


def test_layer_prefetcher_jit_stages_only_exact_routed_rows(rng):
    from treemoe.model.weights import LayerWeights

    num_experts, intermediate, hidden = 4, 6, 8

    def layer():
        return LayerWeights(
            input_layernorm=torch.ones(hidden), post_attn_layernorm=torch.ones(hidden),
            attn={}, router=torch.zeros(num_experts, hidden),
            w1=torch.randn(num_experts, intermediate, hidden, generator=rng),
            w2=torch.randn(num_experts, hidden, intermediate, generator=rng),
            w3=torch.randn(num_experts, intermediate, hidden, generator=rng),
            experts_on_gpu=False,
        )

    layers = [layer(), layer()]
    pf = LayerPrefetcher(layers, depth=1, auto_bitmap=True, jit_staging=True)
    pf.set_bitmap(torch.ones(2, num_experts, dtype=torch.bool))
    pf.begin(is_verification=True)

    assert pf._staged_rows[0] == set()
    assert pf.staged_rows_total == 0
    buf = pf.acquire(0)
    assert pf.prepare_experts(0, {1, 3}) == (2, 0)
    assert pf._staged_rows[0] == {1, 3}
    assert torch.equal(buf["w1"][1], layers[0].w1[1])
    assert torch.equal(buf["w2"][3], layers[0].w2[3])
    pf.release(0)

    buf = pf.acquire(1)
    assert pf._staged_rows[1] == set()
    assert pf.prepare_experts(1, {2}) == (1, 0)
    assert torch.equal(buf["w3"][2], layers[1].w3[2])
    assert pf.staged_rows_total == 3
    assert pf.jit_rows_total == 3
    assert pf.jit_verify_rows_total == 3
    assert pf.repair_rows_total == 0
    assert pf.repair_misses == 0


def test_layer_budget_plan_overrides_temporal_bitmap_and_counts_repair(rng):
    from treemoe.model.weights import LayerWeights

    num_experts, intermediate, hidden = 4, 6, 8

    def layer():
        return LayerWeights(
            input_layernorm=torch.ones(hidden), post_attn_layernorm=torch.ones(hidden),
            attn={}, router=torch.zeros(num_experts, hidden),
            w1=torch.randn(num_experts, intermediate, hidden, generator=rng),
            w2=torch.randn(num_experts, hidden, intermediate, generator=rng),
            w3=torch.randn(num_experts, intermediate, hidden, generator=rng),
            experts_on_gpu=False,
        )

    pf = LayerPrefetcher([layer(), layer()], depth=2, auto_bitmap=True)
    plan = torch.tensor([
        [True, True, False, False],
        [True, True, True, False],
    ])
    pf.set_budget_plan(plan)
    pf.begin()

    assert pf._bitmap is not None and torch.equal(pf._bitmap, plan)
    assert pf.staged_rows_total == 5
    assert pf.staged_bytes_total == 5 * pf.expert_row_bytes
    assert pf.repair(0, {0, 2, 3}) == 2
    assert pf.repair_rows_total == 2
    assert pf.repair_bytes_total == 2 * pf.expert_row_bytes

    pf.set_budget_plan(None)                    # initial/prefill plan means full
    pf.begin()
    assert pf._bitmap is None
    assert all(rows is None for rows in pf._staged_rows.values())


def _repairing_moe_fn(pf):
    """naive_moe wrapper implementing the exact-offload contract: derive the
    routed expert set (same math as naive_moe's own top-2) and repair the
    staged buffer BEFORE the expert weights are read."""
    import torch.nn.functional as F

    from treemoe.model.mixtral import naive_moe

    def fn(x, lw, layer_idx):
        gates = torch.softmax(F.linear(x.float(), lw.router.float()), dim=-1)
        ids = set(gates.topk(2, dim=-1).indices.flatten().tolist())
        pf.prepare_experts(layer_idx, ids)
        return naive_moe(x, lw, layer_idx)

    return fn


def test_layer_prefetcher_repair_makes_bitmap_lossless(tiny_config, rng):
    """Exact-offload contract (cf. DualDeadline 2026): a deliberately terrible
    bitmap (expert 0 only) + repair() must still be bitwise identical to the
    resident forward — mispredictions become on-demand copies, not errors."""
    from test_parity import random_weights
    from treemoe.model.kv_cache import PagedKVCache
    from treemoe.model.mixtral import MixtralForward, naive_moe

    w = random_weights(tiny_config, rng)
    ids = torch.randint(0, tiny_config.vocab_size, (5,), generator=rng)
    pos = torch.arange(5)

    kv1 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
    resident = MixtralForward(w, kv1, moe_fn=naive_moe).forward(ids, pos)

    w_off = _offload_all(w)
    pf = LayerPrefetcher(w_off.layers, depth=2)
    bm = torch.zeros(tiny_config.num_layers, tiny_config.num_experts, dtype=torch.bool)
    bm[:, 0] = True                             # predict only expert 0 everywhere
    pf.set_bitmap(bm)
    kv2 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
    off = MixtralForward(w_off, kv2, moe_fn=_repairing_moe_fn(pf), prefetcher=pf).forward(ids, pos)

    assert pf.repair_misses > 0                 # the bad bitmap really missed
    assert torch.equal(resident, off)           # ...and repair kept it lossless


def test_layer_prefetcher_jit_forward_matches_resident(tiny_config, rng):
    from test_parity import random_weights
    from treemoe.model.kv_cache import PagedKVCache
    from treemoe.model.mixtral import MixtralForward, naive_moe

    w = random_weights(tiny_config, rng)
    ids = torch.randint(0, tiny_config.vocab_size, (5,), generator=rng)
    pos = torch.arange(5)

    kv1 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
    resident = MixtralForward(w, kv1, moe_fn=naive_moe).forward(ids, pos)

    w_off = _offload_all(w)
    pf = LayerPrefetcher(w_off.layers, depth=2, auto_bitmap=True, jit_staging=True)
    kv2 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
    off = MixtralForward(
        w_off, kv2, moe_fn=_repairing_moe_fn(pf), prefetcher=pf,
    ).forward(ids, pos)

    assert torch.equal(resident, off)
    assert pf.jit_rows_total > 0
    assert pf.repair_rows_total == 0
    assert pf.repair_misses == 0


def test_layer_prefetcher_auto_bitmap_temporal(tiny_config, rng):
    """Zero-training temporal predictor: pass 2's bitmap = pass 1's observed
    expert sets. Both passes must stay bitwise exact, and pass 2 must actually
    restrict copies (2 tokens x top-2 => <=4 of 8 experts staged)."""
    from test_parity import random_weights
    from treemoe.model.kv_cache import PagedKVCache
    from treemoe.model.mixtral import MixtralForward, naive_moe

    w = random_weights(tiny_config, rng)
    w_off = _offload_all(w)
    pf = LayerPrefetcher(w_off.layers, depth=2, auto_bitmap=True)
    moe_fn = _repairing_moe_fn(pf)

    for step in range(2):
        ids = torch.randint(0, tiny_config.vocab_size, (2,), generator=rng)
        pos = torch.arange(2)
        kv1 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
        resident = MixtralForward(w, kv1, moe_fn=naive_moe).forward(ids, pos)
        kv2 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
        off = MixtralForward(w_off, kv2, moe_fn=moe_fn, prefetcher=pf).forward(ids, pos)
        assert torch.equal(resident, off), f"pass {step} diverged"
        if step == 0:
            assert all(s is None for s in pf._staged_rows.values())  # no history yet
        else:
            # temporal bitmap = pass-1 usage: <=4 experts/layer staged up-front
            assert pf._bitmap is not None and (pf._bitmap.sum(-1) <= 4).all()
            assert all(s is not None for s in pf._staged_rows.values())


def test_layer_prefetcher_router_hint_stages_top_budget(rng):
    """router_hint stages exactly the top-budget experts per layer by the
    layer's own router demand over the draft features (first pass: no
    temporal history). With history, staging is capped at
    max(budget, |observed|) rows: observed experts first, remaining slots
    by demand. Disabled hint falls back to full copies."""
    from treemoe.model.weights import LayerWeights

    E_, I_, H_ = 4, 6, 8

    def lw():
        return LayerWeights(
            input_layernorm=torch.ones(H_), post_attn_layernorm=torch.ones(H_),
            attn={}, router=torch.randn(E_, H_, generator=rng),
            w1=torch.randn(E_, I_, H_, generator=rng),
            w2=torch.randn(E_, H_, I_, generator=rng),
            w3=torch.randn(E_, I_, H_, generator=rng),
            experts_on_gpu=False)

    layers = [lw(), lw()]
    feats = torch.randn(3, H_, generator=rng)

    pf = LayerPrefetcher(layers, depth=2, auto_bitmap=True)
    pf.router_hint(feats, budget=2)
    pf.begin()
    demand_order = {}
    for li in (0, 1):
        logits = feats.float() @ layers[li].router.float().t()
        demand = torch.softmax(logits, -1).sum(0)
        demand_order[li] = demand.argsort(descending=True).tolist()
        assert pf._staged_rows[li] == set(demand_order[li][:2])

    # second pass with observed usage: cap = max(budget=2, |used|), observed
    # experts always staged, remaining slots by demand rank
    pf.repair(0, {3})            # 1 observed  -> cap 2: {3} + top-1 demand
    pf.repair(1, {0, 1, 2})      # 3 observed  -> cap 3: exactly the used set
    pf.begin()
    want0 = {3} | {next(e for e in demand_order[0] if e != 3)}
    assert pf._staged_rows[0] == want0
    assert pf._staged_rows[1] == {0, 1, 2}

    pf2 = LayerPrefetcher(layers, depth=2, auto_bitmap=True)
    pf2.use_router_hint = False
    pf2.router_hint(feats, budget=2)
    pf2.begin()
    assert all(s is None for s in pf2._staged_rows.values())


def test_spec_engine_offload_router_hint_lossless(tiny_config, rng):
    """Engine wiring red line: spec decode over an offloaded target with the
    draft-guided router hint (deliberately restrictive top-3 of 8) + repair
    must equal AR greedy on resident weights, token for token."""
    from test_parity import random_weights
    from test_spec_lossless import TinyDraft, ar_greedy
    from treemoe.engine.loop import SpecDecodeEngine
    from treemoe.model.kv_cache import PagedKVCache
    from treemoe.model.mixtral import MixtralForward, naive_moe

    w = random_weights(tiny_config, rng)
    prompt = torch.randint(0, tiny_config.vocab_size, (5,), generator=rng)

    kv1 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
    ar = ar_greedy(MixtralForward(w, kv1, moe_fn=naive_moe), prompt.clone(), 24)

    w_off = _offload_all(w)
    pf = LayerPrefetcher(w_off.layers, depth=2, auto_bitmap=True)
    kv2 = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
    target = MixtralForward(w_off, kv2, moe_fn=_repairing_moe_fn(pf), prefetcher=pf)
    eng = SpecDecodeEngine(target, TinyDraft(tiny_config.vocab_size),
                           tree_size=8, max_depth=3, expert_budget=3)
    spec = eng.generate(prompt.clone(), max_new_tokens=24, eos_token_id=-1)

    assert spec == ar
    assert pf._hint is not None                     # the hint path really ran
    assert pf._bitmap is not None                   # ...and produced a bitmap


@pytest.mark.gpu
def test_layer_prefetcher_gpu_forward_matches_resident(tiny_config, rng):
    """Real side-stream + event ordering: pinned-host offload forward must be
    bitwise identical to the resident forward on GPU."""
    import dataclasses

    from test_parity import random_weights
    from treemoe.model.kv_cache import PagedKVCache
    from treemoe.model.mixtral import MixtralForward, naive_moe

    def mv(t):
        return t.cuda()

    w0 = random_weights(tiny_config, rng)
    layers_gpu = [dataclasses.replace(
        lw, input_layernorm=mv(lw.input_layernorm),
        post_attn_layernorm=mv(lw.post_attn_layernorm),
        attn={k: mv(v) for k, v in lw.attn.items()}, router=mv(lw.router),
        w1=mv(lw.w1), w2=mv(lw.w2), w3=mv(lw.w3)) for lw in w0.layers]
    w = dataclasses.replace(w0, embed_tokens=mv(w0.embed_tokens),
                            final_norm=mv(w0.final_norm), lm_head=mv(w0.lm_head),
                            layers=layers_gpu)
    ids = torch.randint(0, tiny_config.vocab_size, (5,), generator=rng).cuda()
    pos = torch.arange(5, device="cuda")

    kv1 = PagedKVCache(tiny_config, num_blocks=8, device="cuda", dtype=tiny_config.dtype)
    resident = MixtralForward(w, kv1, moe_fn=naive_moe).forward(ids, pos)

    def pin(t):
        try:
            return t.pin_memory()
        except RuntimeError:
            return t

    layers_off = [dataclasses.replace(
        lw_gpu, w1=pin(lw_cpu.w1), w2=pin(lw_cpu.w2), w3=pin(lw_cpu.w3),
        experts_on_gpu=False) for lw_gpu, lw_cpu in zip(layers_gpu, w0.layers)]
    w_off = dataclasses.replace(w, layers=layers_off)
    pf = LayerPrefetcher(w_off.layers, depth=2)
    kv2 = PagedKVCache(tiny_config, num_blocks=8, device="cuda", dtype=tiny_config.dtype)
    off = MixtralForward(w_off, kv2, moe_fn=naive_moe, prefetcher=pf).forward(ids, pos)
    torch.cuda.synchronize()
    assert torch.equal(resident, off)


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
