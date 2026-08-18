"""Op2: draft-guided router pre-execution & expert prefetch (spec §3.2).

Components:
  RouterPredictor  - 1M-param cross-layer head: EAGLE draft features (~= target
                     penultimate hidden) -> per-layer expert logits, one fused
                     GEMM f[64,4096] @ Wp[4096, 32*8].
  l2_warm          - resident-weights config: touch predicted experts' weight
                     ranges so they land in L2 before verification reads them.
  HostExpertPool   - offload config (the real battleground): cold experts live
                     in host pinned memory; predicted experts are copied H2D on
                     a side stream into a ring buffer, >=4 layers ahead
                     (352MB @ PCIe Gen5 ~ 5.5 ms per expert-layer, spec §3.2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class RouterPredictor(torch.nn.Module):
    def __init__(self, hidden: int = 4096, num_layers: int = 32, num_experts: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.proj = torch.nn.Linear(hidden, num_layers * num_experts, bias=False)

    def forward(self, draft_features: torch.Tensor) -> torch.Tensor:
        """[N, H] -> per-layer expert logits [N, L, E]."""
        return self.proj(draft_features.float()).view(
            -1, self.num_layers, self.num_experts
        )

    @torch.inference_mode()
    def predict_bitmap(self, draft_features: torch.Tensor, budget: int) -> torch.Tensor:
        """Tree-wide OR + top-budget truncation -> bool bitmap [L, E] (spec §3.2)."""
        logits = self.forward(draft_features)                  # [N, L, E]
        demand = torch.softmax(logits, dim=-1).sum(0)          # [L, E]
        keep = demand.topk(budget, dim=-1).indices             # [L, B]
        bitmap = torch.zeros_like(demand, dtype=torch.bool)
        bitmap.scatter_(1, keep, True)
        return bitmap

    def recall_at(self, draft_features: torch.Tensor, true_topk: torch.Tensor, k: int) -> float:
        """Offline eval (plan Task 4.1 gate: recall@4 >= 0.70). true_topk: [N, L, 2]."""
        pred = self.forward(draft_features).topk(k, dim=-1).indices     # [N, L, k]
        hit = (true_topk.unsqueeze(-1) == pred.unsqueeze(-2)).any(-1)   # [N, L, 2]
        return float(hit.float().mean())


if HAS_TRITON:

    @triton.jit
    def _l2_warm_kernel(w_ptr, n_bytes, STRIDE: tl.constexpr):
        """Read 1 element per 128B cache line across a weight region."""
        pid = tl.program_id(0)
        offs = pid * 1024 + tl.arange(0, 1024)
        ptrs = offs.to(tl.int64) * STRIDE
        x = tl.load(w_ptr + ptrs, mask=ptrs < n_bytes // 2, other=0)
        # fold into a dummy store to defeat DCE (single scalar, negligible traffic)
        tl.store(w_ptr + 0, tl.load(w_ptr + 0) + x.to(tl.bfloat16).sum() * 0)


def l2_warm(weight: torch.Tensor, expert_ids: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """Touch predicted experts' w1/w2/w3 rows on the prefetch stream."""
    if not (HAS_TRITON and weight.is_cuda):
        return
    numel_per_expert = weight[0].numel()
    with torch.cuda.stream(stream):
        for e in expert_ids.tolist():
            base = weight.view(weight.shape[0], -1)[e]
            grid = (max(1, numel_per_expert // (1024 * 64)),)
            _l2_warm_kernel[grid](base, numel_per_expert * 2, STRIDE=64)


class HostExpertPool:
    """Ring-buffered H2D expert prefetcher for the offload configuration."""

    def __init__(self, num_slots: int, expert_shape: tuple[int, ...],
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.slots = [
            {
                "w1": torch.empty(expert_shape, device=device, dtype=dtype),
                "w3": torch.empty(expert_shape, device=device, dtype=dtype),
                "w2": torch.empty((expert_shape[1], expert_shape[0]), device=device, dtype=dtype),
            }
            for _ in range(num_slots)
        ]
        self.slot_key: list[tuple[int, int] | None] = [None] * num_slots
        self.events = [torch.cuda.Event() for _ in range(num_slots)]
        self.cursor = 0
        self.stream = torch.cuda.Stream()

    def lookup(self, layer: int, expert: int) -> dict[str, torch.Tensor] | None:
        try:
            i = self.slot_key.index((layer, expert))
        except ValueError:
            return None
        self.events[i].wait(torch.cuda.current_stream())  # copy-done ordering
        return self.slots[i]

    def prefetch(self, layer: int, expert: int, host_w1, host_w2, host_w3) -> None:
        """Enqueue async H2D copy on the side stream (call >=4 layers ahead)."""
        if (layer, expert) in self.slot_key:
            return
        i = self.cursor
        self.cursor = (self.cursor + 1) % len(self.slots)
        self.slot_key[i] = (layer, expert)
        with torch.cuda.stream(self.stream):
            self.slots[i]["w1"].copy_(host_w1[expert], non_blocking=True)
            self.slots[i]["w3"].copy_(host_w3[expert], non_blocking=True)
            self.slots[i]["w2"].copy_(host_w2[expert], non_blocking=True)
            self.events[i].record(self.stream)
