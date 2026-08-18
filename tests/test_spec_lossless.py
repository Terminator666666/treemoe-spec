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
