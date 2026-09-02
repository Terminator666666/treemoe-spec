"""Pure-PyTorch reference for op1 (tree-aware expert-stationary MoE) with
embedded op3 (budget-constrained verification routing). Spec §3.1 + §3.3.

This file is the numerical anchor: Triton kernels in treemoe/kernels must match
it to rtol=1e-3 (BF16). Everything is written for clarity, not speed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def budget_route_ref(
    gates: torch.Tensor,            # [N, E] fp32 softmax gate probs
    node_accept_prob: torch.Tensor, # [N] fp32 EAGLE-2 global acceptance prob
    expert_budget: int,             # B in [2, E]
    top1_threshold: float = 0.05,   # tau: nodes below get top-1 routing (spec §3.3 step 4)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (topk_ids [N,2], topk_gates [N,2]); evicted slots renormalized.

    Steps (spec §3.3):
      1. s_e = sum_n p_accept(n) * g_{n,e}   (acceptance-weighted expert demand)
      2. keep K = TopB(s)
      3. reroute evicted top-2 choices to the best expert within K, renormalize
      4. low-probability nodes degrade to top-1 (second slot weight = 0)
    """
    n, e = gates.shape
    scores = (node_accept_prob.unsqueeze(1) * gates).sum(0)          # [E]
    keep = torch.topk(scores, k=expert_budget).indices                # [B]
    keep_mask = torch.zeros(e, dtype=torch.bool, device=gates.device)
    keep_mask[keep] = True

    masked_gates = gates.masked_fill(~keep_mask, 0.0)
    topk_gates, topk_ids = masked_gates.topk(2, dim=-1)               # within K
    # HF Mixtral semantics: p_k / (p_1 + p_2), NOT softmax (these are already
    # probabilities; softmax would re-exponentiate and break B=8 losslessness)
    topk_gates = topk_gates / topk_gates.sum(-1, keepdim=True)

    degrade = node_accept_prob < top1_threshold                       # [N]
    topk_gates = torch.where(
        degrade.unsqueeze(1),
        torch.stack([torch.ones(n, device=gates.device), torch.zeros(n, device=gates.device)], dim=1),
        topk_gates,
    )
    return topk_ids, topk_gates


def route_and_bucket_ref(
    x: torch.Tensor,               # [N, H] bf16, DFS order
    router_weight: torch.Tensor,   # [E, H]
    node_accept_prob: torch.Tensor,
    expert_budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kernel A reference. Returns (topk_ids, topk_gates, sorted_slots, expert_offsets).

    sorted_slots: [2N] flat (token, k-slot) indices sorted by (expert, dfs order),
    encoded as token*2+k. expert_offsets: [E+1] prefix sums. All GPU tensors,
    nothing read back to CPU (CUDA Graph red line).
    """
    logits = F.linear(x, router_weight)
    gates = torch.softmax(logits.float(), dim=-1)
    topk_ids, topk_gates = budget_route_ref(gates, node_accept_prob, expert_budget)

    n = x.shape[0]
    e = router_weight.shape[0]
    flat_expert = topk_ids.reshape(-1)                                # [2N], slot i = token i//2, k i%2
    # stable sort by expert keeps DFS order inside each expert segment
    order = torch.argsort(flat_expert, stable=True)
    sorted_slots = order                                              # token = order//2, k = order%2
    counts = torch.bincount(flat_expert, minlength=e)
    expert_offsets = torch.zeros(e + 1, dtype=torch.long, device=x.device)
    expert_offsets[1:] = counts.cumsum(0)
    return topk_ids, topk_gates, sorted_slots, expert_offsets


def tree_moe_forward_ref(
    x: torch.Tensor,               # [N, H] bf16
    w1: torch.Tensor,              # [E, I, H] bf16
    w2: torch.Tensor,              # [E, H, I] bf16
    w3: torch.Tensor,              # [E, I, H] bf16
    router_weight: torch.Tensor,   # [E, H]
    node_accept_prob: torch.Tensor,
    expert_budget: int,
) -> torch.Tensor:
    """Full op1 reference: route+bucket then expert-major FFN accumulation."""
    topk_ids, topk_gates, sorted_slots, expert_offsets = route_and_bucket_ref(
        x, router_weight, node_accept_prob, expert_budget
    )
    out = torch.zeros_like(x)
    e = w1.shape[0]
    for ei in range(e):
        s, t = int(expert_offsets[ei]), int(expert_offsets[ei + 1])
        if s == t:
            continue  # empty expert: kernel exits immediately (masked semantics)
        slots = sorted_slots[s:t]
        tokens = slots // 2
        ks = slots % 2
        xe = x[tokens]
        h = F.silu(xe @ w1[ei].t()) * (xe @ w3[ei].t())
        contrib = (h @ w2[ei].t()) * topk_gates[tokens, ks].unsqueeze(1).to(x.dtype)
        out.index_add_(0, tokens, contrib)
    return out
