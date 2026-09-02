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
  LayerPrefetcher  - engine integration of the offload path: depth-buffered
                     stacked [E,I,H] staging that op1 consumes directly (no
                     D2D re-gather), copied ahead of compute on a side stream.
                     4090 measured (bench_op2): pinned H2D 23.8 GB/s, compute
                     slowdown under active prefetch +5.8%, lead ~4.1*B layers.
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


class LayerPrefetcher:
    """Ahead-of-time H2D staging of offloaded layers (engine side of spec §3.2).

    Each of the `depth` buffers is a stacked w1/w2/w3 triple shaped like one
    layer's experts, so op1's expert-stationary kernel consumes it directly —
    the H2D copy lands in place, no D2D re-gather. Copies run on a side stream
    ahead of the compute stream; two event rings order overwrite-after-use
    (free) and use-after-copy (ready).

    Bitmap mode (`set_bitmap`, [L, E] bool from RouterPredictor.predict_bitmap):
    only predicted experts' rows are copied. On its own this is lossy
    (unpredicted rows keep stale data); consumers that call `repair()` with
    the routed expert set between routing and the expert GEMMs make it EXACT:
    predicted hits overlap with compute for free, mispredictions cost an
    on-demand copy on the compute stream (cf. DualDeadline 2026's exact
    offloaded-MoE prefetching). `bitmap=None` copies every row and is bitwise
    equivalent to MixtralForward's synchronous staging path.

    auto_bitmap=True is a zero-training predictor (cf. MoE-SpeQ 2025): the
    bitmap for each pass is the expert set repair() observed in the previous
    pass — consecutive spec-decode steps extend the same sequence, so routing
    has strong temporal locality. First pass falls back to full copies.
    Requires repair()-calling consumers (it both records usage and restores
    exactness on misses).

    On CPU tensors (tiny-config tests) copies degrade to synchronous — the
    buffer-cycling logic is identical and testable without a GPU.
    """

    def __init__(self, layers, depth: int = 2, auto_bitmap: bool = False):
        self.layers = layers
        self.offload_ids = [i for i, lw in enumerate(layers) if not lw.experts_on_gpu]
        self.depth = max(1, min(depth, max(1, len(self.offload_ids))))
        self.auto_bitmap = auto_bitmap
        self.use_router_hint = True         # gate for the draft-guided hint
        self._hint: torch.Tensor | None = None      # [L, E] fp32 router demand
        self._hint_budget = 0
        self._routers: torch.Tensor | None = None  # [L, E, H] fp32 cache
        self._bufs: list[dict[str, torch.Tensor]] | None = None
        self._buf_of: dict[int, int] = {}   # layer idx -> buffer slot (this pass)
        self._queue: list[int] = []         # offloaded layers not yet scheduled
        self._bitmap: torch.Tensor | None = None
        self._staged_rows: dict[int, set[int] | None] = {}  # None = all rows
        self._used_prev: dict[int, set[int]] = {}  # repair() observations, last pass
        self._used_cur: dict[int, set[int]] = {}
        self.repair_misses = 0              # cumulative mispredicted experts
        self._cuda = False
        self._stream = None
        self._ready: list[torch.cuda.Event] = []
        self._free: list[torch.cuda.Event] = []

    def set_bitmap(self, bitmap: torch.Tensor | None) -> None:
        """[L, E] bool, rows to copy per layer; None = all (lossless). Moved to
        CPU here (one sync), so per-layer scheduling stays sync-free."""
        self._bitmap = None if bitmap is None else bitmap.detach().to("cpu", torch.bool)

    def router_hint(self, features: torch.Tensor, budget: int,
                    accept_prob: torch.Tensor | None = None) -> None:
        """Draft-guided router pre-execution (spec §3.2 headline path).

        Runs every layer's own router over the draft tree's EAGLE features
        (which approximate the target's hidden trajectory) and stores the
        aggregated softmax demand. Same idea as MoE-SpeQ'25's draft-predicted
        prefetch, but training-free: the predictor IS the target's router.
        begin() merges the demand ranking with the temporal set under a
        per-layer row cap; prediction only affects staging, repair() keeps
        every pass exact. Gated behind auto_bitmap so the --no-auto-bitmap
        full-copy baseline stays prediction-free."""
        if not (self.auto_bitmap and self.use_router_hint and self.offload_ids):
            return
        if self._routers is None:
            self._routers = torch.stack([lw.router for lw in self.layers]).float()
        logits = torch.einsum("nh,leh->nle", features.float(), self._routers)
        probs = torch.softmax(logits, dim=-1)                        # [N, L, E]
        if accept_prob is not None:
            # match op3's demand score s_e = sum_n p_accept(n) * g_{n,e} so the
            # predictor ranks experts the way verification will actually route
            probs = probs * accept_prob.float().view(-1, 1, 1)
        self._hint = probs.sum(0).cpu()                              # [L, E]
        self._hint_budget = budget

    def _ensure_buffers(self, lw) -> None:
        dev = lw.router.device
        self._cuda = dev.type == "cuda"
        self._bufs = [
            {k: torch.empty_like(getattr(lw, k), device=dev) for k in ("w1", "w2", "w3")}
            for _ in range(self.depth)
        ]
        if self._cuda:
            self._stream = torch.cuda.Stream()
            self._ready = [torch.cuda.Event() for _ in range(self.depth)]
            self._free = [torch.cuda.Event() for _ in range(self.depth)]

    def begin(self) -> None:
        """Start a forward pass: schedule the first `depth` offloaded layers."""
        if not self.offload_ids:
            return
        if self._bufs is None:
            self._ensure_buffers(self.layers[self.offload_ids[0]])
        if self._used_cur:
            self._used_prev = self._used_cur
        self._used_cur = {}
        if self.auto_bitmap:
            num_experts = self.layers[self.offload_ids[0]].w1.shape[0]
            if self._hint is not None:
                # capped merge of the two zero-training predictors. In the
                # transfer-bound regime the cost is staged BYTES, not misses:
                # a plain union stages unbounded extra rows and loses more on
                # PCIe than the avoided misses cost (measured on 4090: 2.2x
                # slower despite hit_rate 0.907 vs 0.846). So each layer
                # stages at most max(budget, |observed|) rows -- op3 caps the
                # routed set at `budget`, so this covers the worst case.
                # Observed experts first (repeat routing is the common case),
                # remaining slots to the highest-demand hint experts.
                order = self._hint.argsort(dim=-1, descending=True)
                bm = torch.zeros(len(self.layers), num_experts, dtype=torch.bool)
                for li in self.offload_ids:
                    used = self._used_prev.get(li, set())
                    k = min(num_experts, max(self._hint_budget, len(used)))
                    row = set(used)
                    for e in order[li].tolist():
                        if len(row) >= k:
                            break
                        row.add(e)
                    bm[li, sorted(row)] = True
                self.set_bitmap(bm)
            else:
                self.set_bitmap(self.temporal_bitmap(num_experts))
        self._staged_rows.clear()
        self._buf_of.clear()
        self._queue = list(self.offload_ids)
        for _ in range(min(self.depth, len(self._queue))):
            self._schedule_next()

    def _schedule_next(self) -> None:
        layer_idx = self._queue.pop(0)
        slot = len(self._buf_of) % self.depth
        self._buf_of[layer_idx] = slot
        lw, buf = self.layers[layer_idx], self._bufs[slot]
        rows = (None if self._bitmap is None
                else self._bitmap[layer_idx].nonzero().flatten().tolist())
        self._staged_rows[layer_idx] = None if rows is None else set(rows)

        def copy_rows():
            for k in ("w1", "w2", "w3"):
                src, dst = getattr(lw, k), buf[k]
                if rows is None:
                    dst.copy_(src, non_blocking=True)
                else:
                    for e in rows:
                        dst[e].copy_(src[e], non_blocking=True)

        if self._cuda:
            with torch.cuda.stream(self._stream):
                self._free[slot].wait(self._stream)   # overwrite-after-use
                copy_rows()
                self._ready[slot].record(self._stream)
        else:
            copy_rows()

    def acquire(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Return the staged w1/w2/w3 for this layer, ordered after its copy."""
        slot = self._buf_of[layer_idx]
        if self._cuda:
            self._ready[slot].wait(torch.cuda.current_stream())  # use-after-copy
        return self._bufs[slot]

    def repair(self, layer_idx: int, expert_ids) -> int:
        """Exact-offload contract: make the staged buffer correct for the
        experts routing actually selected (e.g. op1 Routing.expert_ids()).
        Call AFTER routing, BEFORE the expert GEMMs — the on-demand copies
        run on the current stream, so subsequent kernels are ordered after
        them. Also records usage for auto_bitmap. Returns #misses copied."""
        ids = {int(i) for i in expert_ids}
        self._used_cur[layer_idx] = ids
        staged = self._staged_rows.get(layer_idx)
        if staged is None:      # full copy this pass: nothing can be stale
            return 0
        missing = sorted(ids - staged)
        if missing:
            slot = self._buf_of[layer_idx]
            if self._cuda:
                self._ready[slot].wait(torch.cuda.current_stream())
            lw, buf = self.layers[layer_idx], self._bufs[slot]
            for k in ("w1", "w2", "w3"):
                src, dst = getattr(lw, k), buf[k]
                for e in missing:
                    dst[e].copy_(src[e], non_blocking=True)
            staged.update(missing)
            self.repair_misses += len(missing)
        return len(missing)

    def temporal_bitmap(self, num_experts: int) -> torch.Tensor | None:
        """Zero-training expert predictor: per-layer expert sets observed by
        repair() in the previous pass. Layers without an observation get an
        all-ones row (full copy). None until any history exists."""
        if not self._used_prev:
            return None
        bitmap = torch.ones(len(self.layers), num_experts, dtype=torch.bool)
        for li, used in self._used_prev.items():
            row = torch.zeros(num_experts, dtype=torch.bool)
            row[sorted(used)] = True
            bitmap[li] = row
        return bitmap

    def release(self, layer_idx: int) -> None:
        """Mark the layer consumed (compute enqueued) and refill the pipeline."""
        slot = self._buf_of[layer_idx]
        if self._cuda:
            self._free[slot].record(torch.cuda.current_stream())
        if self._queue:
            self._schedule_next()
