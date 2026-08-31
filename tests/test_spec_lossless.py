"""Task 1.3 gate: speculative decoding output == AR greedy (lossless, B=8);
plus measurement-pipeline sanity (Task 0.2/0.3) on synthetic traces."""

import pytest
import torch

from tests.test_parity import random_weights
from treemoe.engine.loop import SpecDecodeEngine
from treemoe.model.kv_cache import PagedKVCache
from treemoe.model.mixtral import MixtralForward, naive_moe


class TinyDraft:
    """Draft model stub reusing the *target* itself in feature space is overkill
    for tiny tests; instead propose tokens from a fixed table (worst-case draft:
    correctness must hold regardless of draft quality)."""

    def __init__(self, vocab: int):
        self.vocab = vocab

    def reset(self):
        pass

    def step(self, tokens, features, positions):
        t = tokens.shape[0]
        logits = torch.zeros(t, self.vocab)
        for i in range(t):
            for k in range(4):
                logits[i, (int(tokens[i]) * 7 + k + 1) % self.vocab] = 4.0 - k
        return features, logits


@pytest.fixture()
def engine_pair(tiny_config, rng):
    w = random_weights(tiny_config, rng)

    def fresh(moe_fn=naive_moe):
        kv = PagedKVCache(tiny_config, num_blocks=8, device="cpu", dtype=tiny_config.dtype)
        return MixtralForward(w, kv, moe_fn=moe_fn)

    return fresh, tiny_config


def ar_greedy(model, prompt, n):
    pos = torch.arange(prompt.shape[0])
    logits = model.forward(prompt, pos)
    out = [int(logits[-1].argmax())]
    while len(out) < n:
        cur = torch.tensor([out[-1]])
        p = torch.tensor([model.kv.seq_len])
        logits = model.forward(cur, p)
        out.append(int(logits[-1].argmax()))
    return out


def test_spec_decode_lossless_vs_ar(engine_pair, rng):
    fresh, cfg = engine_pair
    prompt = torch.randint(0, cfg.vocab_size, (5,), generator=rng)

    ar = ar_greedy(fresh(), prompt.clone(), 12)

    target = fresh()
    eng = SpecDecodeEngine(target, TinyDraft(cfg.vocab_size),
                           tree_size=8, max_depth=3, expert_budget=8)
    spec = eng.generate(prompt.clone(), max_new_tokens=12)
    assert spec == ar  # lossless red line (plan Task 1.3)


def test_spec_decode_lossless_vs_ar_long(engine_pair):
    # 40 tokens x multiple prompts: catches KV position-bookkeeping bugs that a
    # 12-token single-seed run misses (e.g. the root's KV never being committed
    # only shows up once an uncommitted step's context matters — regression for
    # the [root]+accepted commit fix). eos disabled so lengths always compare.
    fresh, cfg = engine_pair
    for seed in (1, 7, 42):
        g = torch.Generator().manual_seed(seed)
        prompt = torch.randint(0, cfg.vocab_size, (5,), generator=g)
        ar = ar_greedy(fresh(), prompt.clone(), 40)
        eng = SpecDecodeEngine(fresh(), TinyDraft(cfg.vocab_size),
                               tree_size=8, max_depth=3, expert_budget=8)
        spec = eng.generate(prompt.clone(), max_new_tokens=40, eos_token_id=-1)
        assert spec == ar, f"seed {seed}"


def random_eagle_weights(cfg, g):
    from treemoe.model.eagle import EagleWeights

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=cfg.dtype) * 0.05

    return EagleWeights(
        fc=r(cfg.hidden_dim, 2 * cfg.hidden_dim),
        attn={
            "q_proj": r(cfg.num_heads * cfg.head_dim, cfg.hidden_dim),
            "k_proj": r(cfg.num_kv_heads * cfg.head_dim, cfg.hidden_dim),
            "v_proj": r(cfg.num_kv_heads * cfg.head_dim, cfg.hidden_dim),
            "o_proj": r(cfg.hidden_dim, cfg.num_heads * cfg.head_dim),
        },
        input_layernorm=torch.ones(cfg.hidden_dim, dtype=cfg.dtype),
        post_attn_layernorm=torch.ones(cfg.hidden_dim, dtype=cfg.dtype),
        mlp_gate=r(cfg.intermediate_dim, cfg.hidden_dim),
        mlp_up=r(cfg.intermediate_dim, cfg.hidden_dim),
        mlp_down=r(cfg.hidden_dim, cfg.intermediate_dim),
    )


def test_spec_lossless_with_eagle_draft(engine_pair, rng):
    # real EagleDraftModel end-to-end on CPU: exercises the committed-KV /
    # begin_tree / tree_mask paths (prompt conditioning, ancestor-only
    # attention, rejected-branch pruning). Red line: still bitwise == AR.
    from treemoe.model.eagle import EagleDraftModel

    fresh, cfg = engine_pair
    w = random_weights(cfg, torch.Generator().manual_seed(5))
    ew = random_eagle_weights(cfg, torch.Generator().manual_seed(6))
    for seed in (3, 11):
        g = torch.Generator().manual_seed(seed)
        prompt = torch.randint(0, cfg.vocab_size, (6,), generator=g)
        ar = ar_greedy(fresh(), prompt.clone(), 30)
        target = fresh()
        draft = EagleDraftModel(ew, cfg, w.embed_tokens, w.lm_head)
        eng = SpecDecodeEngine(target, draft, tree_size=8, max_depth=3,
                               expert_budget=8)
        spec = eng.generate(prompt.clone(), max_new_tokens=30, eos_token_id=-1)
        assert spec == ar, f"seed {seed}"
        assert draft._ck is not None
        # committed KV holds exactly the accepted sequence (prompt[1:] + outputs
        # minus the trailing bonus/root not yet committed) — rejected branches
        # never leak in: committed length == last root_pos - 1 + 1
        assert draft._ck.shape[0] == target.kv.seq_len - 1


def test_eagle_tree_mask_chain_equals_causal(tiny_config):
    # a linear chain stepped one node per level with explicit ancestor masks
    # must reproduce the same features/logits as one causal batch: the tree
    # topology mask is exactly "ancestors + self".
    from treemoe.model.eagle import EagleDraftModel

    cfg = tiny_config
    g = torch.Generator().manual_seed(9)
    w = random_weights(cfg, g)
    ew = random_eagle_weights(cfg, g)

    toks = torch.randint(0, cfg.vocab_size, (3,), generator=g)
    feats = torch.randn(3, cfg.hidden_dim, generator=g, dtype=cfg.dtype)
    poss = torch.arange(4, 7)

    a = EagleDraftModel(ew, cfg, w.embed_tokens, w.lm_head)
    a.begin_tree()
    fa, la = a.step(toks, feats, poss)  # tree_mask=None -> batch-causal chain

    b = EagleDraftModel(ew, cfg, w.embed_tokens, w.lm_head)
    b.begin_tree()
    for i in range(3):
        m = torch.ones(1, i + 1, dtype=torch.bool)  # ancestors 0..i-1 + self
        fb, lb = b.step(toks[i:i + 1], feats[i:i + 1], poss[i:i + 1], tree_mask=m)
        assert torch.allclose(fa[i], fb[0], atol=1e-5)
        assert torch.allclose(la[i], lb[0], atol=1e-4)


def test_accept_length_stat_updates(engine_pair, rng):
    fresh, cfg = engine_pair
    prompt = torch.randint(0, cfg.vocab_size, (5,), generator=rng)
    eng = SpecDecodeEngine(fresh(), TinyDraft(cfg.vocab_size),
                           tree_size=8, max_depth=3)
    eng.generate(prompt, max_new_tokens=10)
    assert eng.stats.steps >= 1
    assert eng.stats.mean_accept_len >= 1.0  # bonus token guarantees >=1/step


def test_analyze_tree_group_simulation():
    from measurements.analyze import simulate_tree_node_groups

    groups = simulate_tree_node_groups(seq_len=100, n_nodes=16, seed=0)
    assert groups
    for nodes, parents in groups:
        assert len(nodes) <= 16
        assert all(0 <= p < 100 for p in nodes)
        assert parents[0] is None
