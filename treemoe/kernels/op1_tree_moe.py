"""Op1: tree-aware expert-stationary MoE kernel (embeds op3 budget routing).

v1 = two-kernel fallback declared in spec §3.1 risk table, structured as:
  Kernel A  route_and_bucket : fp32 router GEMM (cuBLAS) + budget routing +
            stable (expert, DFS) bucketing — all GPU tensor ops, zero CPU
            readback, CUDA-Graph safe. v2 fuses this into one Triton CTA.
  Kernel B1 grouped w1/w3 + SiLU-mul  (expert-stationary tiles, Triton)
  Kernel B2 grouped w2 + gate-scaled scatter-add, split-k over I (Triton)

Static shapes everywhere: launch grids sized for the worst case (all 2N slots
to one expert); padding blocks carry expert_id = -1 and exit immediately
(DeepGEMM masked-layout semantics).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU-only dev box: reference path still importable
    HAS_TRITON = False

from treemoe.ref.tree_moe_ref import budget_route_ref, tree_moe_forward_ref

BM = 16    # slot rows per block (M is tiny: <=16 tokens/expert typically)
BN = 128   # intermediate/out columns per block
BK = 128   # reduction tile
SPLIT_K = 8


# --------------------------------------------------------------------------
# Kernel A — route_and_bucket (graph-safe torch composition, v2: fuse in Triton)
# --------------------------------------------------------------------------

def route_and_bucket(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    node_accept_prob: torch.Tensor,
    expert_budget: int,
    block_m: int = BM,
):
    """Returns (topk_ids, topk_gates, padded_slots, block_expert_ids, num_blocks_max).

    padded_slots: [E * ceil(2N/BM) * BM] slot ids (token*2+k), -1 padded so every
    BM-row block belongs to exactly one expert. block_expert_ids: [max_blocks]
    expert id per block, -1 for unused blocks. Shapes depend only on (N, E).
    """
    n, _ = x.shape
    e = router_weight.shape[0]
    logits = F.linear(x.float(), router_weight.float())
    gates = torch.softmax(logits, dim=-1)
    topk_ids, topk_gates = budget_route_ref(gates, node_accept_prob, expert_budget)

    flat_expert = topk_ids.reshape(-1)                       # [2N]
    order = torch.argsort(flat_expert, stable=True)          # DFS-stable in-expert
    counts = torch.bincount(flat_expert, minlength=e)        # [E]
    blocks_per_expert = (counts + block_m - 1) // block_m    # [E]
    max_blocks_per_expert = (2 * n + block_m - 1) // block_m
    max_blocks = e * max_blocks_per_expert

    device = x.device
    padded_slots = torch.full((max_blocks * block_m,), -1, dtype=torch.long, device=device)
    block_expert_ids = torch.full((max_blocks,), -1, dtype=torch.long, device=device)

    # scatter each expert's slot segment into its private padded region
    seg_starts = torch.zeros(e, dtype=torch.long, device=device)
    seg_starts[1:] = counts.cumsum(0)[:-1]
    pos_in_expert = torch.arange(2 * n, device=device) - seg_starts[flat_expert[order]]
    dest = flat_expert[order] * (max_blocks_per_expert * block_m) + pos_in_expert
    padded_slots[dest] = order

    blk_idx = torch.arange(max_blocks, device=device)
    blk_expert = blk_idx // max_blocks_per_expert
    blk_local = blk_idx % max_blocks_per_expert
    used = blk_local < blocks_per_expert[blk_expert]
    block_expert_ids[used] = blk_expert[used]

    # inverse permutation: slot id (token*2+k) -> padded row, for the
    # deterministic combine kernel (fixed-order reduction, no atomics)
    slot_to_row = torch.empty(2 * n, dtype=torch.long, device=device)
    slot_to_row[order] = dest
    return topk_ids, topk_gates, padded_slots, block_expert_ids, slot_to_row, max_blocks


# --------------------------------------------------------------------------
# Kernel B1 — grouped w1/w3 GEMM + SiLU⊙ (expert-stationary)
# --------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _route_bucket_fused_kernel(
        x_ptr, router_ptr, accept_ptr,
        topk_ids_ptr, gates_flat_ptr, padded_slots_ptr,
        block_expert_ids_ptr, slot_to_row_ptr,
        expert_budget, tau,
        N: tl.constexpr, E: tl.constexpr, EP: tl.constexpr,   # EP = 16 (pow2 pad)
        H: tl.constexpr, BK: tl.constexpr,
        MAX_BPE: tl.constexpr, BLOCK_M: tl.constexpr, MAX_BLOCKS: tl.constexpr,
    ):
        """Kernel A v2: the whole route+bucket pipeline in ONE program.

        Production precedent: vLLM fuses topk_softmax and moe_align_block_size
        into single CUDA kernels (csrc/moe/) instead of chains of torch ops —
        at decode batch sizes launch overhead rivals the math. This replaces
        ~8 kernel launches per layer with 1.

        Serial-friendly sizes: N<=128 nodes, E=8 experts; the O((2N)^2) stable
        rank is a 128x128 vectorized comparison, trivial for one CTA.
        """
        offs_n = tl.arange(0, N)                  # tree nodes (N = pow2 tree size)
        offs_e = tl.arange(0, EP)
        e_valid = offs_e < E

        # ---- 1. router GEMM, fp32 accumulate (HPC-Ops finding) ----
        acc = tl.zeros((N, EP), dtype=tl.float32)
        for k0 in range(0, H, BK):
            ks = k0 + tl.arange(0, BK)
            x_t = tl.load(x_ptr + offs_n[:, None] * H + ks[None, :]).to(tl.float32)
            w_t = tl.load(router_ptr + offs_e[:, None] * H + ks[None, :],
                          mask=e_valid[:, None], other=0.0).to(tl.float32)
            acc += tl.dot(x_t, tl.trans(w_t))
        logits = tl.where(e_valid[None, :], acc, -float("inf"))

        # ---- 2. row softmax ----
        rmax = tl.max(logits, axis=1)
        expl = tl.exp(logits - rmax[:, None])
        gates = expl / tl.sum(expl, axis=1)[:, None]          # [N, EP]

        # ---- 3. op3 budget routing: acceptance-weighted expert demand ----
        accept = tl.load(accept_ptr + offs_n).to(tl.float32)  # [N]
        scores = tl.sum(accept[:, None] * gates, axis=0)      # [EP]
        scores = tl.where(e_valid, scores, -float("inf"))
        keep = tl.zeros((EP,), dtype=tl.int1)
        for i in tl.static_range(E):                          # top-B, first-occurrence ties
            cand = tl.where(keep, -float("inf"), scores)
            am = tl.argmax(cand, axis=0)
            keep = keep | ((offs_e == am) & (i < expert_budget))

        # ---- 4. in-budget top-2 + p/(p1+p2) renorm + tau degradation ----
        mg = tl.where(keep[None, :], gates, 0.0)
        i1 = tl.argmax(mg, axis=1)                            # [N]
        g1 = tl.max(mg, axis=1)
        mg2 = tl.where(offs_e[None, :] == i1[:, None], 0.0, mg)
        i2 = tl.argmax(mg2, axis=1)
        g2 = tl.max(mg2, axis=1)
        s = g1 + g2
        tg1 = g1 / s
        tg2 = g2 / s
        degrade = accept < tau
        tg1 = tl.where(degrade, 1.0, tg1)
        tg2 = tl.where(degrade, 0.0, tg2)

        # slot layout: slot 2t = (t, k=0), slot 2t+1 = (t, k=1)
        tl.store(topk_ids_ptr + offs_n * 2, i1.to(tl.int64))
        tl.store(topk_ids_ptr + offs_n * 2 + 1, i2.to(tl.int64))
        tl.store(gates_flat_ptr + offs_n * 2, tg1)
        tl.store(gates_flat_ptr + offs_n * 2 + 1, tg2)

        # ---- 5. stable (expert, DFS) bucketing via O((2N)^2) rank ----
        # fe stays in registers (tl.interleave): same-CTA global store->load
        # reread is an L1-coherence hazard production kernels avoid
        slots = tl.arange(0, 2 * N)
        fe = tl.interleave(i1, i2).to(tl.int32)               # [2N] expert per slot
        eq = fe[:, None] == fe[None, :]
        lower = slots[None, :] < slots[:, None]
        rank = tl.sum((eq & lower).to(tl.int32), axis=1)      # stable in-expert rank
        dest = fe.to(tl.int64) * (MAX_BPE * BLOCK_M) + rank.to(tl.int64)
        tl.store(slot_to_row_ptr + slots, dest)

        # padded_slots as an inverse scatter computed in registers: one plain
        # store pass, no -1 prefill + barrier (cross-lane store races)
        pos = tl.arange(0, MAX_BLOCKS * BLOCK_M).to(tl.int64)
        hit = pos[:, None] == dest[None, :]                   # [R, 2N]
        val = tl.sum(tl.where(hit, slots[None, :] + 1, 0), axis=1) - 1  # -1 if no slot
        tl.store(padded_slots_ptr + pos, val.to(tl.int64))

        # ---- 6. per-block expert ids (blocks past ceil(count/BM) masked -1) ----
        counts = tl.sum((fe[None, :] == offs_e[:, None]).to(tl.int32), axis=1)  # [EP]
        blocks_needed = (counts + BLOCK_M - 1) // BLOCK_M
        blk = tl.arange(0, MAX_BLOCKS)
        blk_e = blk // MAX_BPE
        need = tl.sum(tl.where(blk_e[:, None] == offs_e[None, :],
                               blocks_needed[None, :], 0), axis=1)
        used = (blk % MAX_BPE) < need
        tl.store(block_expert_ids_ptr + blk,
                 tl.where(used, blk_e, -1).to(tl.int64))

    @triton.jit
    def _moe_gemm1_kernel(
        x_ptr, w1_ptr, w3_ptr, h_ptr,
        padded_slots_ptr, block_expert_ids_ptr,
        H: tl.constexpr, I: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)   # slot block (expert-major -> expert-stationary)
        pid_n = tl.program_id(1)   # intermediate-dim block
        expert = tl.load(block_expert_ids_ptr + pid_m)
        if expert < 0:
            return  # masked padding block (DeepGEMM-style, shape stays static)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        slots = tl.load(padded_slots_ptr + offs_m)               # [BM]
        m_mask = slots >= 0
        tokens = tl.where(m_mask, slots // 2, 0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        w_base = expert.to(tl.int64) * I * H
        for k0 in range(0, H, BLOCK_K):
            xk = k0 + offs_k
            x_tile = tl.load(
                x_ptr + tokens[:, None] * H + xk[None, :],
                mask=m_mask[:, None], other=0.0,
            )
            w_off = w_base + offs_n[:, None] * H + xk[None, :]
            w1_t = tl.load(w1_ptr + w_off)                        # [BN, BK]
            w3_t = tl.load(w3_ptr + w_off)
            acc1 += tl.dot(x_tile, tl.trans(w1_t))
            acc3 += tl.dot(x_tile, tl.trans(w3_t))

        h = acc1 * tl.sigmoid(acc1) * acc3                        # SiLU(a)⊙b, fp32
        # store to workspace in padded-row layout: row = global slot-block row
        tl.store(
            h_ptr + offs_m[:, None].to(tl.int64) * I + offs_n[None, :],
            h.to(tl.bfloat16),
            mask=m_mask[:, None],
        )

    @triton.jit
    def _moe_gemm2_kernel(
        h_ptr, w2_ptr, out_ptr, gates_ptr,
        padded_slots_ptr, block_expert_ids_ptr,
        H: tl.constexpr, I: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLIT: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)   # output(H)-dim block
        pid_s = tl.program_id(2)   # split-k segment over I (occupancy for tiny M)
        expert = tl.load(block_expert_ids_ptr + pid_m)
        if expert < 0:
            return

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        slots = tl.load(padded_slots_ptr + offs_m)
        m_mask = slots >= 0
        tokens = tl.where(m_mask, slots // 2, 0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        seg = I // SPLIT
        w_base = expert.to(tl.int64) * H * I
        for k0 in range(pid_s * seg, (pid_s + 1) * seg, BLOCK_K):
            hk = k0 + offs_k
            h_tile = tl.load(
                h_ptr + offs_m[:, None].to(tl.int64) * I + hk[None, :],
                mask=m_mask[:, None], other=0.0,
            )
            w2_t = tl.load(w2_ptr + w_base + offs_n[:, None] * I + hk[None, :])
            acc += tl.dot(h_tile.to(tl.bfloat16), tl.trans(w2_t))

        gate = tl.load(gates_ptr + slots, mask=m_mask, other=0.0)  # [BM] fp32
        acc = acc * gate[:, None]
        # scatter-accumulate into fp32 out buffer (atomic: token may appear in 2 slots)
        tl.atomic_add(
            out_ptr + tokens[:, None].to(tl.int64) * H + offs_n[None, :],
            acc, mask=m_mask[:, None],
        )

    @triton.jit
    def _moe_gemm2_det_kernel(
        h_ptr, w2_ptr, partial_ptr, gates_ptr,
        padded_slots_ptr, block_expert_ids_ptr,
        R: tl.constexpr,           # padded rows = max_blocks * BLOCK_M
        H: tl.constexpr, I: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLIT: tl.constexpr,
    ):
        """Deterministic variant: split-k partials go to a private workspace row
        (plain store, no atomics); _combine_kernel reduces them in fixed order.
        fp32 addition is non-associative, so atomic accumulation order flips
        bits run-to-run and can flip argmax at near-ties — unacceptable for the
        lossless spec==AR red line."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_s = tl.program_id(2)
        expert = tl.load(block_expert_ids_ptr + pid_m)
        if expert < 0:
            return

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        slots = tl.load(padded_slots_ptr + offs_m)
        m_mask = slots >= 0
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        seg = I // SPLIT
        w_base = expert.to(tl.int64) * H * I
        for k0 in range(pid_s * seg, (pid_s + 1) * seg, BLOCK_K):
            hk = k0 + offs_k
            h_tile = tl.load(
                h_ptr + offs_m[:, None].to(tl.int64) * I + hk[None, :],
                mask=m_mask[:, None], other=0.0,
            )
            w2_t = tl.load(w2_ptr + w_base + offs_n[:, None] * I + hk[None, :])
            acc += tl.dot(h_tile.to(tl.bfloat16), tl.trans(w2_t))

        gate = tl.load(gates_ptr + slots, mask=m_mask, other=0.0)
        acc = acc * gate[:, None]
        dst = (pid_s.to(tl.int64) * R + offs_m[:, None]) * H + offs_n[None, :]
        tl.store(partial_ptr + dst, acc, mask=m_mask[:, None])

    @triton.jit
    def _combine_kernel(
        partial_ptr, slot_to_row_ptr, out_ptr,
        R: tl.constexpr, H: tl.constexpr,
        BLOCK_N: tl.constexpr, SPLIT: tl.constexpr,
    ):
        """out[t] = sum_s partial[s, row(2t)] + sum_s partial[s, row(2t+1)],
        fixed iteration order -> bitwise deterministic across runs."""
        t = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        r0 = tl.load(slot_to_row_ptr + 2 * t)
        r1 = tl.load(slot_to_row_ptr + 2 * t + 1)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for s in range(0, SPLIT):
            acc += tl.load(partial_ptr + (s * R + r0) * H + offs_n)  # r0 int64 promotes
        for s in range(0, SPLIT):
            acc += tl.load(partial_ptr + (s * R + r1) * H + offs_n)
        tl.store(out_ptr + t.to(tl.int64) * H + offs_n, acc)


# --------------------------------------------------------------------------
# Host wrapper
# --------------------------------------------------------------------------

class _Workspace:
    """Static buffers reused across steps (CUDA Graph friendly)."""

    def __init__(self, n: int, e: int, hidden: int, inter: int, device):
        max_blocks = e * ((2 * n + BM - 1) // BM)
        self.max_blocks = max_blocks
        self.rows = max_blocks * BM
        self.h = torch.zeros(self.rows, inter, dtype=torch.bfloat16, device=device)
        self.out_f32 = torch.zeros(n, hidden, dtype=torch.float32, device=device)
        self.partial = None  # lazily allocated for deterministic mode
        # fused Kernel A outputs (static, rewritten every step)
        self.topk_flat = torch.zeros(2 * n, dtype=torch.long, device=device)
        self.gates_flat = torch.zeros(2 * n, dtype=torch.float32, device=device)
        self.padded_slots = torch.full((self.rows,), -1, dtype=torch.long, device=device)
        self.block_expert_ids = torch.full((max_blocks,), -1, dtype=torch.long, device=device)
        self.slot_to_row = torch.zeros(2 * n, dtype=torch.long, device=device)

    def get_partial(self, hidden: int, device):
        if self.partial is None:
            # [SPLIT_K, rows, H] fp32; only rows of real slots are touched, so
            # HBM traffic ~ SPLIT_K * 2N rows, not the full allocation
            self.partial = torch.empty(
                SPLIT_K, self.rows, hidden, dtype=torch.float32, device=device
            )
        return self.partial


_ws_cache: dict[tuple, _Workspace] = {}


def tree_moe_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    router_weight: torch.Tensor,
    node_accept_prob: torch.Tensor,
    expert_budget: int,
    out: torch.Tensor | None = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """Spec §3.1 entry point. Falls back to the reference on CPU / no Triton.

    deterministic=True (default): split-k partials + fixed-order combine,
    bitwise reproducible (required by the lossless red-line test); costs one
    extra fp32 partial round-trip (~SPLIT_K*2N*H*8B per layer).
    deterministic=False: atomic_add fast path for benchmarking.
    """
    if not HAS_TRITON or not x.is_cuda:
        return tree_moe_forward_ref(
            x, w1, w2, w3, router_weight, node_accept_prob, expert_budget
        )

    n, hidden = x.shape
    e, inter, _ = w1.shape
    key = (n, e, hidden, inter, x.device.index)
    ws = _ws_cache.get(key)
    if ws is None:
        ws = _ws_cache[key] = _Workspace(n, e, hidden, inter, x.device)

    max_bpe = (2 * n + BM - 1) // BM
    # fused Kernel A: single-CTA route+bucket (1 launch vs ~8 torch ops);
    # register footprint of the O((2N)^2) rank limits it to N<=64, E<=16
    use_fused_a = (n & (n - 1)) == 0 and n <= 64 and e <= 16 and hidden % BK == 0
    if use_fused_a:
        _route_bucket_fused_kernel[(1,)](
            x, router_weight, node_accept_prob,
            ws.topk_flat, ws.gates_flat, ws.padded_slots,
            ws.block_expert_ids, ws.slot_to_row,
            expert_budget, 0.05,
            N=n, E=e, EP=16, H=hidden, BK=BK,
            MAX_BPE=max_bpe, BLOCK_M=BM, MAX_BLOCKS=ws.max_blocks,
            num_warps=4,
        )
        gates_flat = ws.gates_flat
        padded_slots, block_expert_ids = ws.padded_slots, ws.block_expert_ids
        slot_to_row, max_blocks = ws.slot_to_row, ws.max_blocks
    else:
        _topk_ids, topk_gates, padded_slots, block_expert_ids, slot_to_row, max_blocks = route_and_bucket(
            x, router_weight, node_accept_prob, expert_budget
        )
        gates_flat = topk_gates.reshape(-1).float().contiguous()   # index by slot id

    # num_warps=4/num_stages=4: vLLM fused_moe production default for M<=32
    # decode — "smallest batches are memory-latency bound, a deeper pipeline
    # hides the weight loads" (vllm fused_moe.py get_default_config)
    grid1 = (max_blocks, inter // BN)
    _moe_gemm1_kernel[grid1](
        x, w1, w3, ws.h, padded_slots, block_expert_ids,
        H=hidden, I=inter, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_warps=4, num_stages=4,
    )
    grid2 = (max_blocks, hidden // BN, SPLIT_K)
    if deterministic:
        partial = ws.get_partial(hidden, x.device)
        _moe_gemm2_det_kernel[grid2](
            ws.h, w2, partial, gates_flat, padded_slots, block_expert_ids,
            R=ws.rows, H=hidden, I=inter,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT=SPLIT_K,
            num_warps=4, num_stages=4,
        )
        _combine_kernel[(n, hidden // BN)](
            partial, slot_to_row, ws.out_f32,
            R=ws.rows, H=hidden, BLOCK_N=BN, SPLIT=SPLIT_K,
        )
    else:
        ws.out_f32.zero_()
        _moe_gemm2_kernel[grid2](
            ws.h, w2, ws.out_f32, gates_flat, padded_slots, block_expert_ids,
            H=hidden, I=inter, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT=SPLIT_K,
            num_warps=4, num_stages=4,
        )
    result = ws.out_f32.to(x.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result
