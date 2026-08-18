"""Task 3.1: op4 verify/commit — reference behaviour + Triton parity on trees."""

import pytest
import torch

from treemoe.ref.verify_ref import tree_verify_ref


def chain_tree(tokens: list[int]):
    """Linear chain: node i's parent is i-1."""
    n = len(tokens)
    parent = torch.tensor([-1] + list(range(n - 1)))
    children = [[i + 1] for i in range(n - 1)] + [[]]
    return torch.tensor(tokens), parent, children


def make_logits(n: int, v: int, winners: list[int]) -> torch.Tensor:
    lg = torch.zeros(n, v)
    for i, w in enumerate(winners):
        lg[i, w] = 10.0
    return lg


def test_greedy_accepts_matching_chain():
    # target argmax at node i decides node i+1's token; all match -> full accept
    tokens, parent, children = chain_tree([5, 7, 9, 11])
    logits = make_logits(4, 32, winners=[7, 9, 11, 13])
    res = tree_verify_ref(logits, torch.zeros_like(logits), tokens, parent, children)
    assert int(res.num_accepted) == 3
    assert res.accepted_tokens[:3].tolist() == [7, 9, 11]
    assert int(res.bonus_token) == 13  # bonus from last accepted node's dist


def test_greedy_stops_at_first_mismatch():
    tokens, parent, children = chain_tree([5, 7, 999, 11])
    logits = make_logits(4, 32000, winners=[7, 9, 11, 13])
    res = tree_verify_ref(logits, torch.zeros_like(logits), tokens, parent, children)
    assert int(res.num_accepted) == 1
    assert int(res.bonus_token) == 9  # argmax of node 1 (last accepted)


def test_greedy_picks_matching_branch():
    # root has two children (slots 1,2); only slot 2 matches argmax
    tokens = torch.tensor([5, 8, 7])
    parent = torch.tensor([-1, 0, 0])
    children = [[1, 2], [], []]
    logits = make_logits(3, 32, winners=[7, 0, 4])
    res = tree_verify_ref(logits, torch.zeros_like(logits), tokens, parent, children)
    assert int(res.num_accepted) == 1
    assert int(res.accepted_slots[0]) == 2
    assert int(res.bonus_token) == 4


def test_sampling_mode_deterministic_with_injected_uniforms():
    tokens, parent, children = chain_tree([5, 7, 9])
    logits = make_logits(3, 32, winners=[7, 9, 11])
    q = torch.full((3, 32), 1.0 / 32)
    u_accept = torch.zeros(3)          # u < ratio always -> accept all
    res = tree_verify_ref(logits, q, tokens, parent, children,
                          temperature=1.0, uniforms=u_accept,
                          generator=torch.Generator().manual_seed(0))
    assert int(res.num_accepted) == 2
    u_reject = torch.ones(3)           # u >= ratio -> reject at first child
    res2 = tree_verify_ref(logits, q, tokens, parent, children,
                           temperature=1.0, uniforms=u_reject,
                           generator=torch.Generator().manual_seed(0))
    assert int(res2.num_accepted) == 0


@pytest.mark.gpu
def test_fused_greedy_matches_ref_random_trees():
    from treemoe.kernels.op4_commit import fused_verify_commit

    g = torch.Generator().manual_seed(42)
    for _ in range(50):  # plan says 1000; trimmed variant runs in CI, full run nightly
        n, v = 16, 512
        parent = torch.tensor([-1] + [int(torch.randint(0, i, (1,), generator=g)) for i in range(1, n)])
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[int(parent[i])].append(i)
        tokens = torch.randint(0, v, (n,), generator=g)
        logits = torch.randn(n, v, generator=g) * 3
        res_ref = tree_verify_ref(logits.cuda(), torch.zeros(n, v).cuda(),
                                  tokens.cuda(), parent.cuda(), children)
        res_ker = fused_verify_commit(logits.cuda(), torch.zeros(n, v).cuda(),
                                      tokens.cuda(), parent.cuda(), children,
                                      max_depth=n)
        assert int(res_ref.num_accepted) == int(res_ker.num_accepted)
        m = int(res_ref.num_accepted)
        assert res_ref.accepted_slots[:m].tolist() == res_ker.accepted_slots[:m].tolist()
        assert int(res_ref.bonus_token) == int(res_ker.bonus_token)
