"""EAGLE-2 dynamic draft tree, padded to a static shape (spec §2).

EAGLE-2 tree construction: expand top-K children per frontier node, score each
node by its *global* acceptance probability (product of branch probs along the
path), keep the best `tree_size` nodes overall, then re-serialize in DFS order
(DFS order maximizes in-expert token locality for op1, spec §3.1).
"""

from __future__ import annotations

import inspect
from contextlib import nullcontext
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
    features: torch.Tensor = None  # [num_valid, H] draft features, DFS order
                                   # (op2 router hint input, spec §3.2)

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
    performance_tracer=None,
) -> DraftTree:
    """Grow the candidate tree breadth-first, then prune to top tree_size nodes."""
    # candidate pool entries: (token, parent_idx_in_pool, depth, logprob_path, feature)
    pool_tokens = [int(root_token)]
    pool_parent = [-1]
    pool_depth = [0]
    pool_logp = [0.0]
    pool_feat = [root_feature]

    frontier = [0]
    # topology-aware drafting: if the draft model accepts a tree_mask, pass
    # per-level ancestor visibility so a node never attends siblings or other
    # branches (matches official EAGLE-2; wrong context degrades draft quality
    # and therefore mean accepted length). Duck-typed drafts without the param
    # (tests) keep the legacy batch-causal behaviour.
    try:
        _wants_mask = "tree_mask" in inspect.signature(draft_step_fn).parameters
    except (TypeError, ValueError):
        _wants_mask = False
    kv_index: dict[int, int] = {}  # pool id -> row in the draft's tree KV
    kv_count = 0
    for depth in range(1, max_depth + 1):
        if not frontier:
            break
        level_record = (
            performance_tracer.begin_draft_level(depth, len(frontier), kv_count)
            if performance_tracer is not None else None
        )
        toks = torch.tensor([pool_tokens[i] for i in frontier], device=device)
        feats = torch.stack([pool_feat[i] for i in frontier])
        poss = torch.tensor([root_pos + pool_depth[i] for i in frontier], device=device)
        phase = performance_tracer.phase(level_record, "draft_forward") \
            if performance_tracer is not None else nullcontext()
        with phase:
            if _wants_mask:
                t = len(frontier)
                m = torch.zeros(t, kv_count + t, dtype=torch.bool)
                for r, pool_i in enumerate(frontier):
                    m[r, kv_count + r] = True  # self
                    p = pool_parent[pool_i]
                    while p != -1:
                        m[r, kv_index[p]] = True
                        p = pool_parent[p]
                new_feats, logits = draft_step_fn(toks, feats, poss,
                                                  tree_mask=m.to(device))
                for r, pool_i in enumerate(frontier):
                    kv_index[pool_i] = kv_count + r
                kv_count += t
            else:
                new_feats, logits = draft_step_fn(toks, feats, poss)
        phase = performance_tracer.phase(level_record, "candidate_select") \
            if performance_tracer is not None else nullcontext()
        with phase:
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            topv, topi = logprobs.topk(branch_k, dim=-1)
            # ONE packed D2H per level. token ids (<32000) are exact in fp32.
            packed = torch.cat([topi.float(), topv], dim=1).tolist()
        if level_record is not None:
            level_record.update({
                "frontier_pool_nodes": frontier.copy(),
                "input_tokens": [pool_tokens[i] for i in frontier],
                "positions": [root_pos + pool_depth[i] for i in frontier],
                "candidate_token_ids": [
                    [int(value) for value in row[:branch_k]] for row in packed
                ],
                "candidate_logprob": [
                    [float(value) for value in row[branch_k:]] for row in packed
                ],
            })
        next_frontier = []
        for fi, pool_i in enumerate(frontier):
            row = packed[fi]
            for k in range(branch_k):
                pool_tokens.append(int(row[k]))
                pool_parent.append(pool_i)
                pool_depth.append(depth)
                pool_logp.append(pool_logp[pool_i] + row[branch_k + k])
                pool_feat.append(new_feats[fi])
                next_frontier.append(len(pool_tokens) - 1)
        # EAGLE-2 dynamic expansion: only the globally most promising nodes expand
        next_frontier.sort(key=lambda i: -pool_logp[i])
        frontier = next_frontier[: max(2, branch_k * 2)]
        if level_record is not None:
            level_record["generated_candidates"] = len(next_frontier)
            level_record["next_frontier_nodes"] = len(frontier)
            level_record["selected_frontier_pool_nodes"] = frontier.copy()

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

    # serialize entirely host-side, then upload once: the old per-element
    # GPU-tensor writes and int(parent[j]) ancestor walks cost ~2N tiny H2D
    # launches + O(N*depth) D2H syncs per step
    n = tree_size
    tok_new = [-1] * n
    par_new = [-1] * n
    logp_new = [float("-inf")] * n
    for old in dfs:
        new = remap[old]
        tok_new[new] = pool_tokens[old]
        par_new[new] = remap[pool_parent[old]] if pool_parent[old] in remap else -1
        logp_new[new] = pool_logp[old]

    # ancestor-closure mask, one O(N) pass: DFS order guarantees parent < child,
    # so row(i) = row(parent) | self. The old per-row while-parent walk cost
    # O(N*depth) Python iterations (~0.2ms at N=64, growing with tree size).
    mask_cpu = torch.zeros(n, n, dtype=torch.bool)
    for i in range(len(dfs)):
        p = par_new[i]
        if p >= 0:
            mask_cpu[i] = mask_cpu[p]
        mask_cpu[i, i] = True

    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, len(dfs)):
        children[par_new[i]].append(i)

    tokens = torch.tensor(tok_new, dtype=torch.long, device=device)
    parent = torch.tensor(par_new, dtype=torch.long, device=device)
    # fp32 exp matches the old torch.tensor(logp).exp() numerics bitwise;
    # padding slots get exp(-inf) = 0.0 like the old torch.zeros prefill
    accept = torch.tensor(logp_new, dtype=torch.float32).exp().to(device)
    mask = mask_cpu.to(device)

    return DraftTree(
        tokens=tokens, parent=parent, accept_prob=accept,
        attn_mask=mask, children=children, num_valid=len(dfs),
        features=torch.stack([pool_feat[i] for i in dfs]),
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
