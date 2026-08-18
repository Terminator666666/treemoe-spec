"""Tree construction invariants (engine/tree.py) — pure CPU logic tests."""

import torch

from treemoe.engine.tree import build_eagle2_tree


def fake_draft_step(vocab: int = 64, hidden: int = 8):
    """Deterministic fake draft model: logits depend on token id."""

    def step(tokens, features, positions):
        t = tokens.shape[0]
        logits = torch.zeros(t, vocab)
        for i in range(t):
            base = int(tokens[i]) % vocab
            for k in range(4):
                logits[i, (base + k + 1) % vocab] = 4.0 - k  # distinct top-4
        return features, logits

    return step


def build(tree_size=16, max_depth=4):
    return build_eagle2_tree(
        fake_draft_step(), torch.tensor(3), torch.zeros(8), root_pos=10,
        tree_size=tree_size, max_depth=max_depth, branch_k=4, device="cpu",
    )


def test_root_and_padding():
    t = build()
    assert int(t.parent[0]) == -1
    assert t.num_valid <= t.size
    assert (t.tokens[t.num_valid:] == -1).all()


def test_parents_precede_children_dfs():
    t = build()
    for i in range(1, t.num_valid):
        assert 0 <= int(t.parent[i]) < i  # DFS numbering: parent before child


def test_attn_mask_is_ancestor_closure():
    t = build()
    for i in range(t.num_valid):
        assert bool(t.attn_mask[i, i])
        j = int(t.parent[i])
        while j >= 0:
            assert bool(t.attn_mask[i, j])
            j = int(t.parent[j])
    # no visibility into non-ancestors (sibling check)
    for i in range(1, t.num_valid):
        for j in range(1, t.num_valid):
            if not t.attn_mask[i, j]:
                continue
            # j visible to i => j is ancestor-or-self of i
            k = i
            ok = False
            while k >= 0:
                if k == j:
                    ok = True
                    break
                k = int(t.parent[k])
            assert ok


def test_accept_prob_monotone_along_paths():
    t = build()
    for i in range(1, t.num_valid):
        p = int(t.parent[i])
        assert float(t.accept_prob[i]) <= float(t.accept_prob[p]) + 1e-6


def test_children_adjacency_consistent():
    t = build()
    for parent_idx, kids in enumerate(t.children):
        for c in kids:
            assert int(t.parent[c]) == parent_idx
