"""EAGLE-2 dynamic draft tree, padded to a static shape (spec §2).

EAGLE-2 tree construction: expand top-K children per frontier node, score each
node by its *global* acceptance probability (product of branch probs along the
path), keep the best `tree_size` nodes overall, then re-serialize in DFS order
(DFS order maximizes in-expert token locality for op1, spec §3.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DraftTree:
    tokens: torch.Tensor         # [N] int64, -1 for padding slots
    parent: torch.Tensor         # [N] int64, -1 for root/padding
    accept_prob: torch.Tensor    # [N] fp32 global acceptance probability
    attn_mask: torch.Tensor      # [N, N] bool: j visible to i iff j is ancestor-or-self
    children: list[list[int]]    # adjacency in final DFS numbering
    num_valid: int

    @property
    def size(self) -> int:
        return int(self.tokens.shape[0])


def build_eagle2_tree(
    draft_step_fn,               # (tokens[T], features[T,H], positions[T]) -> (feat', logits)
    root_token: torch.Tensor,    # [] committed last token
    root_feature: torch.Tensor,  # [H]
    root_pos: int,
    tree_size: int = 64,
    max_depth: int = 6,
    branch_k: int = 4,
    device: str = "cuda",
) -> DraftTree:
    """Grow the candidate tree breadth-first, then prune to top tree_size nodes."""
    # candidate pool entries: (token, parent_idx_in_pool, depth, logprob_path, feature)
    pool_tokens = [int(root_token)]
    pool_parent = [-1]
    pool_depth = [0]
    pool_logp = [0.0]
    pool_feat = [root_feature]

    frontier = [0]
    for depth in range(1, max_depth + 1):
        if not frontier:
            break
        toks = torch.tensor([pool_tokens[i] for i in frontier], device=device)
        feats = torch.stack([pool_feat[i] for i in frontier])
        poss = torch.tensor([root_pos + pool_depth[i] for i in frontier], device=device)
        new_feats, logits = draft_step_fn(toks, feats, poss)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        topv, topi = logprobs.topk(branch_k, dim=-1)
        next_frontier = []
        for fi, pool_i in enumerate(frontier):
            for k in range(branch_k):
                pool_tokens.append(int(topi[fi, k]))
                pool_parent.append(pool_i)
                pool_depth.append(depth)
                pool_logp.append(pool_logp[pool_i] + float(topv[fi, k]))
                pool_feat.append(new_feats[fi])
                next_frontier.append(len(pool_tokens) - 1)
        # EAGLE-2 dynamic expansion: only the globally most promising nodes expand
        next_frontier.sort(key=lambda i: -pool_logp[i])
        frontier = next_frontier[: max(2, branch_k * 2)]

    # global top-(tree_size-1) selection by acceptance proxy exp(logp); root always kept
    order = sorted(range(1, len(pool_tokens)), key=lambda i: -pool_logp[i])
    keep = [0] + [i for i in order if _ancestors_kept(i, order[: tree_size - 1], pool_parent)][: tree_size - 1]
    keep = _close_under_ancestors(keep, pool_parent)[:tree_size]

    # DFS re-serialization
    kids_of: dict[int, list[int]] = {i: [] for i in keep}
    for i in keep:
        p = pool_parent[i]
        if p in kids_of and i != 0:
            kids_of[p].append(i)
    dfs: list[int] = []

    def _walk(i: int) -> None:
        dfs.append(i)
        for c in kids_of[i]:
            _walk(c)

    _walk(0)
    remap = {old: new for new, old in enumerate(dfs)}

    n = tree_size
    tokens = torch.full((n,), -1, dtype=torch.long, device=device)
    parent = torch.full((n,), -1, dtype=torch.long, device=device)
    accept = torch.zeros(n, device=device)
    for old in dfs:
        new = remap[old]
        tokens[new] = pool_tokens[old]
        parent[new] = remap[pool_parent[old]] if pool_parent[old] in remap else -1
        accept[new] = float(torch.tensor(pool_logp[old]).exp())

    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    for i in range(len(dfs)):
        j = i
        while j >= 0:
            mask[i, j] = True
            j = int(parent[j])

    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, len(dfs)):
        children[int(parent[i])].append(i)

    return DraftTree(
        tokens=tokens, parent=parent, accept_prob=accept,
        attn_mask=mask, children=children, num_valid=len(dfs),
    )


def _ancestors_kept(i: int, kept: list[int], parent: list[int]) -> bool:
    kept_set = set(kept) | {0}
    p = parent[i]
    while p != -1:
        if p not in kept_set:
            return False
        p = parent[p]
    return True


def _close_under_ancestors(keep: list[int], parent: list[int]) -> list[int]:
    s = set(keep)
    for i in list(keep):
        p = parent[i]
        while p != -1 and p not in s:
            s.add(p)
            p = parent[p]
    # preserve original priority order, ancestors first via stable sort by depth
    return sorted(s, key=lambda i: (_depth(i, parent), keep.index(i) if i in keep else 0))


def _depth(i: int, parent: list[int]) -> int:
    d = 0
    while parent[i] != -1:
        i = parent[i]
        d += 1
    return d
