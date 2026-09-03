"""Op1: tree-aware expert-stationary MoE kernel (embeds op3 budget routing).

v1 = two-kernel fallback declared in spec §3.1 risk table, structured as:
    Router      F.linear(BF16) + FP32 softmax, exactly matching HF numerics.
    Kernel A    acceptance-weighted budget routing + stable (expert, DFS)
                            bucketing in one Triton CTA, with zero CPU readback.
  Kernel B1 grouped w1/w3 + SiLU-mul  (expert-stationary tiles, Triton)
  Kernel B2 grouped w2 + gate-scaled scatter-add, split-k over I (Triton)

Static shapes everywhere: launch grids sized for the worst case (all 2N slots
to one expert); padding blocks carry expert_id = -1 and exit immediately
(DeepGEMM masked-layout semantics).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU-only dev box: reference path still importable
    HAS_TRITON = False

# Triton interpreter (TRITON_INTERPRET=1) executes kernels instruction-by-
# instruction on CPU via numpy — lets us run the REAL kernels on GPU-less
# boxes. Must be set before import; eviction hints become no-ops there.
_INTERPRET = os.getenv("TRITON_INTERPRET", "0") == "1"

from treemoe.ref.tree_moe_ref import budget_route_ref, tree_moe_forward_ref

BM = 16    # slot rows per block (M is tiny: <=16 tokens/expert typically)
# Per-GEMM tiles, chosen by GPU-less static analysis (benchmarks/static_analysis.py:
# AOT compile for sm_90a + ptxas -v register counts -> theoretical occupancy).
# Both GEMMs are memory-bound streams; occupancy = latency hiding = bandwidth.
BN1 = 64   # gemm1 N over I=14336
BK1 = 64   # gemm1 K over H (u32-packed loads: regs 102->80, occ 25%->38%)
BN2 = 64   # gemm2 N over H
BK2 = 128  # gemm2 K over I (within split-K segment). 4090 sweep
           # (benchmarks/sweep_op1.py, 108 configs): BK2 dominates -- 128 is
           # 1.30x faster than the ptxas-chosen 32 despite lower occupancy
           # (116 regs). A streaming kernel needs bytes-in-flight, not
           # occupancy: BK=32 kept only ~8KB/CTA in flight vs gemm1's ~48KB.
           # On-silicon: gemm2 484 -> ~935 GB/s, whole op1 at 89% of peak.
# SPLIT_K=2 (sweep): larger K segments per CTA beat extra parallelism, and it
# halves the det-path fp32 partial round-trip as a bonus
SPLIT_K = 2
# launch configs as module constants so benchmarks/sweep_op1.py can patch
# them for on-silicon tuning
GEMM1_WARPS, GEMM1_STAGES = 8, 3   # ptxas sweep: 80 regs, zero spill, 38% occ
GEMM2_WARPS, GEMM2_STAGES = 8, 3   # 4090 sweep top-1 (ties within noise)


# --------------------------------------------------------------------------
# Kernel A — exact-HF router followed by fused budget/demand/bucket
# --------------------------------------------------------------------------

def route_and_bucket(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    node_accept_prob: torch.Tensor,
    expert_budget: int,
    block_m: int = BM,
    top1_threshold: float = 0.05,
    return_demand: bool = False,
    router_gates: torch.Tensor | None = None,
    routing_objective: str = "mass",
):
    """Returns (topk_ids, topk_gates, padded_slots, block_expert_ids, num_blocks_max).

    padded_slots: [E * ceil(2N/BM) * BM] slot ids (token*2+k), -1 padded so every
    BM-row block belongs to exactly one expert. block_expert_ids: [max_blocks]
    expert id per block, -1 for unused blocks. Shapes depend only on (N, E).
    """
    n, _ = x.shape
    e = router_weight.shape[0]
    if router_gates is None:
        logits = F.linear(x, router_weight)
        gates = torch.softmax(logits.float(), dim=-1)
    else:
        gates = router_gates
    topk_ids, topk_gates = budget_route_ref(
        gates, node_accept_prob, expert_budget,
        top1_threshold=top1_threshold,
        routing_objective=routing_objective,
    )

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
    result = (topk_ids, topk_gates, padded_slots, block_expert_ids,
              slot_to_row, max_blocks)
    if return_demand:
        demand = (node_accept_prob.float().unsqueeze(1) * gates).sum(0)
        return (*result, demand)
    return result


# --------------------------------------------------------------------------
# Kernel B1 — grouped w1/w3 GEMM + SiLU⊙ (expert-stationary)
# --------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _budget_bucket_fused_kernel(
        gates_ptr, accept_ptr,
        topk_ids_ptr, gates_flat_ptr, padded_slots_ptr,
        block_expert_ids_ptr, slot_to_row_ptr, demand_ptr,
        expert_budget, tau,
        N: tl.constexpr, E: tl.constexpr, EP: tl.constexpr,   # EP = 16 (pow2 pad)
        MAX_BPE: tl.constexpr, BLOCK_M: tl.constexpr, MAX_BLOCKS: tl.constexpr,
        CRITICAL_PATH: tl.constexpr,
    ):
        """Fuse acceptance-weighted budgeting and bucketing in one program.

        Production precedent: vLLM fuses topk_softmax and moe_align_block_size
        into single CUDA kernels (csrc/moe/) instead of chains of torch ops —
        at decode batch sizes launch overhead rivals the math. Router gates are
        computed by PyTorch first so BF16 GEMM and FP32 softmax exactly follow
        the HF path across Triton/cuBLAS versions.

        Production sizes: N<=64 nodes, E=8 experts; the O((2N)^2) stable rank
        stays spill-free in one CTA. N=128 uses the exact torch fallback.
        """
        offs_n = tl.arange(0, N)                  # tree nodes (N = pow2 tree size)
        offs_e = tl.arange(0, EP)
        e_valid = offs_e < E

        # ---- 1. exact-HF router probabilities, padded to EP lanes ----
        gates = tl.load(
            gates_ptr + offs_n[:, None] * E + offs_e[None, :],
            mask=e_valid[None, :], other=0.0,
        ).to(tl.float32)

        # ---- 2. op3 budget routing: acceptance-weighted expert demand ----
        accept = tl.load(accept_ptr + offs_n).to(tl.float32)  # [N]
        scores = tl.sum(accept[:, None] * gates, axis=0)      # [EP]
        scores = tl.where(e_valid, scores, -float("inf"))
        tl.store(demand_ptr + offs_e, scores, mask=e_valid)
        keep = tl.zeros((EP,), dtype=tl.int1)
        if CRITICAL_PATH:
            full_i1 = tl.argmax(gates, axis=1)
            full_second = tl.where(
                offs_e[None, :] == full_i1[:, None], -float("inf"), gates
            )
            full_i2 = tl.argmax(full_second, axis=1)
            needed = ((offs_e[None, :] == full_i1[:, None])
                             | ((offs_e[None, :] == full_i2[:, None])
                                 & (accept[:, None] >= tau)))
            criticality = tl.max(
                tl.where(needed, accept[:, None], -float("inf")), axis=0,
            )
            for rank in tl.static_range(E):
                available = ~keep & e_valid
                best_criticality = tl.max(
                    tl.where(available, criticality, -float("inf")), axis=0,
                )
                same_risk = criticality == best_criticality
                candidate = tl.where(
                    available & same_risk, scores, -float("inf")
                )
                best_expert = tl.argmax(candidate, axis=0)
                keep = keep | ((offs_e == best_expert) & (rank < expert_budget))
        else:
            for rank in tl.static_range(E):
                candidate = tl.where(keep, -float("inf"), scores)
                best_expert = tl.argmax(candidate, axis=0)
                keep = keep | ((offs_e == best_expert) & (rank < expert_budget))

        # ---- 3. in-budget top-2 + p/(p1+p2) renorm + tau degradation ----
        masked_gates = tl.where(keep[None, :], gates, 0.0)
        i1 = tl.argmax(masked_gates, axis=1)
        g1 = tl.max(masked_gates, axis=1)
        second_gates = tl.where(
            offs_e[None, :] == i1[:, None], 0.0, masked_gates
        )
        i2 = tl.argmax(second_gates, axis=1)
        g2 = tl.max(second_gates, axis=1)
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

        # ---- 4. stable (expert, DFS) bucketing via O((2N)^2) rank ----
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
        # store pass, no -1 prefill + barrier (cross-lane store races).
        # BLOCKED over the padded-row space: the monolithic [R, 2N] hit matrix
        # (131072 lanes at R=1024) was the top ptxas spill source (7.6KB/thread
        # at 4 warps); RB=2N rows per iteration keeps the live set at [2N, 2N]
        RB: tl.constexpr = 2 * N
        for r0 in range(0, MAX_BLOCKS * BLOCK_M, RB):
            pos = (r0 + tl.arange(0, RB)).to(tl.int64)
            hit = pos[:, None] == dest[None, :]               # [RB, 2N]
            val = tl.sum(tl.where(hit, slots[None, :] + 1, 0), axis=1) - 1
            tl.store(padded_slots_ptr + pos, val.to(tl.int64))

        # ---- 5. per-block expert ids (blocks past ceil(count/BM) masked -1) ----
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
        PACK_W: tl.constexpr,   # 1 iff weights are 16-bit (u64-packed loads)
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
        offs_k4 = tl.arange(0, BLOCK_K // 4)   # one u64 lane = 4 bf16

        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # SASS audit (nvdisasm): Triton 3.7 lowers mma B-operand loads in the
        # dot layout -> 16-bit scalar LDG.E.U16, regardless of hints/tile/
        # orientation, and its cp.async width == the element width (it never
        # vectorizes across lanes here). Reinterpreting the 16-bit weight
        # stream as u64 quads forces 8-byte lanes: LDG.E.U16 x32 -> one
        # cp.async 0x8 per 4 elements (sm_89 AOT: u16 32x2B, u32 24x4B,
        # u64 12x8B per tile) -- the streamed-weight issue path shrinks 2-8x
        # and with it the IMAD/IADD3 address-gen work that dominates the
        # static instruction mix. Unpack is shifts+ands in-register -- free
        # next to a DRAM-bound stream. Trade-off: Triton drops the
        # evict_first hint on this path (acceptable: a 23.8GB/step stream
        # never fits 50MB L2; evict_last on reused x/h is kept).
        w_base = expert.to(tl.int64) * I * H
        if PACK_W:
            w1_u64 = w1_ptr.to(tl.pointer_type(tl.uint64), bitcast=True)
            w3_u64 = w3_ptr.to(tl.pointer_type(tl.uint64), bitcast=True)
        for k0 in range(0, H, BLOCK_K):
            xk = k0 + offs_k
            xk = tl.max_contiguous(tl.multiple_of(xk, BLOCK_K), BLOCK_K)
            # x rows are re-read by every n-block CTA of this expert -> pin in
            # L2 (PTX ld.global.L2::evict_last)
            x_tile = tl.load(
                x_ptr + tokens[:, None] * H + xk[None, :],
                mask=m_mask[:, None], other=0.0,
                eviction_policy="evict_last",
            )
            if PACK_W:
                k64 = k0 // 4 + offs_k4
                k64 = tl.max_contiguous(tl.multiple_of(k64, BLOCK_K // 4), BLOCK_K // 4)
                w_off64 = w_base // 4 + offs_n[:, None] * (H // 4) + k64[None, :]
                w1_64 = tl.load(w1_u64 + w_off64, eviction_policy="evict_first")
                w3_64 = tl.load(w3_u64 + w_off64, eviction_policy="evict_first")
                # little-endian: u64 = [e0 e1 e2 e3] -> u32 pairs -> u16 lanes
                w1_32 = tl.interleave((w1_64 & 0xFFFFFFFF).to(tl.uint32),
                                      (w1_64 >> 32).to(tl.uint32))
                w3_32 = tl.interleave((w3_64 & 0xFFFFFFFF).to(tl.uint32),
                                      (w3_64 >> 32).to(tl.uint32))
                w1_t = tl.interleave(
                    (w1_32 & 0xFFFF).to(tl.uint16).to(w1_ptr.dtype.element_ty, bitcast=True),
                    (w1_32 >> 16).to(tl.uint16).to(w1_ptr.dtype.element_ty, bitcast=True),
                )
                w3_t = tl.interleave(
                    (w3_32 & 0xFFFF).to(tl.uint16).to(w3_ptr.dtype.element_ty, bitcast=True),
                    (w3_32 >> 16).to(tl.uint16).to(w3_ptr.dtype.element_ty, bitcast=True),
                )
            else:  # 32-bit weights (interpreter parity tests): loads already wide
                w_off = w_base + offs_n[:, None] * H + xk[None, :]
                w1_t = tl.load(w1_ptr + w_off, eviction_policy="evict_first")
                w3_t = tl.load(w3_ptr + w_off, eviction_policy="evict_first")
            acc1 += tl.dot(x_tile, tl.trans(w1_t))
            acc3 += tl.dot(x_tile, tl.trans(w3_t))

        h = acc1 * tl.sigmoid(acc1) * acc3                        # SiLU(a)⊙b, fp32
        # store to workspace in padded-row layout: row = global slot-block row;
        # gemm2 reads h back immediately -> keep resident (st.global evict_last)
        tl.store(
            h_ptr + offs_m[:, None].to(tl.int64) * I + offs_n[None, :],
            h.to(h_ptr.dtype.element_ty),
            mask=m_mask[:, None],
            eviction_policy="evict_last",
        )

    @triton.jit
    def _moe_gemm2_kernel(
        h_ptr, w2_ptr, out_ptr, gates_ptr,
        padded_slots_ptr, block_expert_ids_ptr,
        H: tl.constexpr, I: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLIT: tl.constexpr, PACK_W: tl.constexpr,
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
        offs_k4 = tl.arange(0, BLOCK_K // 4)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        seg = I // SPLIT
        # u64-packed weight stream: see _moe_gemm1_kernel SASS-audit note
        w_base = expert.to(tl.int64) * H * I
        if PACK_W:
            w2_u64 = w2_ptr.to(tl.pointer_type(tl.uint64), bitcast=True)
        for k0 in range(pid_s * seg, (pid_s + 1) * seg, BLOCK_K):
            hk = k0 + offs_k
            hk = tl.max_contiguous(tl.multiple_of(hk, BLOCK_K), BLOCK_K)
            h_tile = tl.load(
                h_ptr + offs_m[:, None].to(tl.int64) * I + hk[None, :],
                mask=m_mask[:, None], other=0.0,
                eviction_policy="evict_last",   # h reused by all pid_n CTAs
            )
            if PACK_W:
                k64 = k0 // 4 + offs_k4
                k64 = tl.max_contiguous(tl.multiple_of(k64, BLOCK_K // 4), BLOCK_K // 4)
                w2_64 = tl.load(w2_u64 + w_base // 4 + offs_n[:, None] * (I // 4) + k64[None, :],
                                eviction_policy="evict_first")
                w2_32 = tl.interleave((w2_64 & 0xFFFFFFFF).to(tl.uint32),
                                      (w2_64 >> 32).to(tl.uint32))
                w2_t = tl.interleave(
                    (w2_32 & 0xFFFF).to(tl.uint16).to(w2_ptr.dtype.element_ty, bitcast=True),
                    (w2_32 >> 16).to(tl.uint16).to(w2_ptr.dtype.element_ty, bitcast=True),
                )
            else:
                w2_t = tl.load(w2_ptr + w_base + offs_n[:, None] * I + hk[None, :],
                               eviction_policy="evict_first")  # streamed once
            acc += tl.dot(h_tile, tl.trans(w2_t))

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
        SPLIT: tl.constexpr, PACK_W: tl.constexpr,
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
        offs_k4 = tl.arange(0, BLOCK_K // 4)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        seg = I // SPLIT
        # u64-packed weight stream: see _moe_gemm1_kernel SASS-audit note
        w_base = expert.to(tl.int64) * H * I
        if PACK_W:
            w2_u64 = w2_ptr.to(tl.pointer_type(tl.uint64), bitcast=True)
        for k0 in range(pid_s * seg, (pid_s + 1) * seg, BLOCK_K):
            hk = k0 + offs_k
            hk = tl.max_contiguous(tl.multiple_of(hk, BLOCK_K), BLOCK_K)
            h_tile = tl.load(
                h_ptr + offs_m[:, None].to(tl.int64) * I + hk[None, :],
                mask=m_mask[:, None], other=0.0,
                eviction_policy="evict_last",
            )
            if PACK_W:
                k64 = k0 // 4 + offs_k4
                k64 = tl.max_contiguous(tl.multiple_of(k64, BLOCK_K // 4), BLOCK_K // 4)
                w2_64 = tl.load(w2_u64 + w_base // 4 + offs_n[:, None] * (I // 4) + k64[None, :],
                                eviction_policy="evict_first")
                w2_32 = tl.interleave((w2_64 & 0xFFFFFFFF).to(tl.uint32),
                                      (w2_64 >> 32).to(tl.uint32))
                w2_t = tl.interleave(
                    (w2_32 & 0xFFFF).to(tl.uint16).to(w2_ptr.dtype.element_ty, bitcast=True),
                    (w2_32 >> 16).to(tl.uint16).to(w2_ptr.dtype.element_ty, bitcast=True),
                )
            else:
                w2_t = tl.load(w2_ptr + w_base + offs_n[:, None] * I + hk[None, :],
                               eviction_policy="evict_first")
            acc += tl.dot(h_tile, tl.trans(w2_t))

        gate = tl.load(gates_ptr + slots, mask=m_mask, other=0.0)
        acc = acc * gate[:, None]
        dst = (pid_s.to(tl.int64) * R + offs_m[:, None]) * H + offs_n[None, :]
        # combine kernel reads partials right back -> keep resident in L2
        tl.store(partial_ptr + dst, acc, mask=m_mask[:, None],
                 eviction_policy="evict_last")

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

    def __init__(self, n: int, e: int, hidden: int, inter: int, device, dtype):
        max_blocks = e * ((2 * n + BM - 1) // BM)
        self.max_blocks = max_blocks
        self.num_experts = e
        self.rows = max_blocks * BM
        self.h = torch.zeros(self.rows, inter, dtype=dtype, device=device)
        self.out_f32 = torch.zeros(n, hidden, dtype=torch.float32, device=device)
        self.partial = None  # lazily allocated for deterministic mode
        # fused Kernel A outputs (static, rewritten every step)
        self.topk_flat = torch.zeros(2 * n, dtype=torch.long, device=device)
        self.gates_flat = torch.zeros(2 * n, dtype=torch.float32, device=device)
        self.padded_slots = torch.full((self.rows,), -1, dtype=torch.long, device=device)
        self.block_expert_ids = torch.full((max_blocks,), -1, dtype=torch.long, device=device)
        self.slot_to_row = torch.zeros(2 * n, dtype=torch.long, device=device)
        self.demand = torch.zeros(e, dtype=torch.float32, device=device)

    def get_partial(self, hidden: int, device):
        if self.partial is None:
            # [SPLIT_K, rows, H] fp32; only rows of real slots are touched, so
            # HBM traffic ~ SPLIT_K * 2N rows, not the full allocation
            self.partial = torch.empty(
                SPLIT_K, self.rows, hidden, dtype=torch.float32, device=device
            )
        return self.partial


_ws_cache: dict[tuple, _Workspace] = {}


class Routing:
    """Phase-1 result of tree_moe_forward: routing + bucketing, no expert
    weights touched. Lets an offload engine learn WHICH experts this layer
    needs before the GEMMs read the weights, so a lossy prefetch can be
    repaired into an exact one (cf. DualDeadline 2026 / MoE-SpeQ 2025)."""

    __slots__ = ("ws", "gates_flat", "padded_slots", "block_expert_ids",
                 "slot_to_row", "max_blocks", "demand")

    def __init__(self, ws, gates_flat, padded_slots, block_expert_ids,
                 slot_to_row, max_blocks, demand):
        self.ws = ws
        self.gates_flat = gates_flat
        self.padded_slots = padded_slots
        self.block_expert_ids = block_expert_ids
        self.slot_to_row = slot_to_row
        self.max_blocks = max_blocks
        self.demand = demand

    def expert_ids(self) -> list[int]:
        """Experts actually routed this layer. ONE small D2H sync
        (max_blocks int64s) -- the price of the exact-offload contract."""
        ids = self.block_expert_ids[: self.max_blocks].tolist()
        return sorted({int(i) for i in ids if i >= 0})

    def exclude_experts(self, expert_ids) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        """Hybrid CPU-expert dispatch (Fiddler, arXiv:2402.07033): drop these
        experts from the GPU pass and return [(e, token_idx, gates)] host
        tensors so the caller can run their FFN on CPU from the pinned host
        copy -- below the ~8-token break-even (bench_cpu_expert) that beats
        streaming a 352MB expert over PCIe.

        GPU-side surgery, O(cold slots): zero their gates (atomic path),
        mark their blocks unused so both GEMMs skip them, and zero their
        partial rows so the fixed-order combine adds exact zeros. The GPU
        output then equals a pass that never routed these experts."""
        ws = self.ws
        seg_rows = self.padded_slots.shape[0] // ws.num_experts
        picked: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        all_slots = []
        for e in expert_ids:
            seg = self.padded_slots[e * seg_rows:(e + 1) * seg_rows]
            slots = seg[seg >= 0]
            if slots.numel() == 0:
                continue
            picked.append((int(e), (slots // 2).cpu(),
                           self.gates_flat[slots].float().cpu()))
            all_slots.append(slots)
        if not picked:
            return picked
        slots = torch.cat(all_slots)
        self.gates_flat[slots] = 0.0
        partial = ws.get_partial(ws.out_f32.shape[1], self.padded_slots.device)
        partial[:, self.slot_to_row[slots]] = 0.0
        drop = torch.isin(
            self.block_expert_ids,
            torch.tensor([e for e, _, _ in picked],
                         device=self.block_expert_ids.device))
        self.block_expert_ids[drop] = -1
        return picked


def route_experts(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    node_accept_prob: torch.Tensor,
    expert_budget: int,
    inter: int,
    top1_threshold: float = 0.05,
    routing_objective: str = "mass",
) -> Routing:
    """Phase 1 of tree_moe_forward: op3 budget routing + bucketing (fused
    Kernel A or the torch fallback). Needs only the resident router weights;
    `inter` is the FFN intermediate size (host-known, avoids touching w1).
    Pass the result to tree_moe_forward(routing=...) to run the GEMMs."""
    if not 2 <= expert_budget <= router_weight.shape[0]:
        raise ValueError(
            f"expert_budget must be in [2, {router_weight.shape[0]}]"
        )
    if routing_objective not in {"mass", "critical_path"}:
        raise ValueError("routing_objective must be 'mass' or 'critical_path'")
    n, hidden = x.shape
    e = router_weight.shape[0]
    key = (n, e, hidden, inter, x.device.index if x.is_cuda else -1, x.dtype)
    ws = _ws_cache.get(key)
    if ws is None:
        ws = _ws_cache[key] = _Workspace(n, e, hidden, inter, x.device, x.dtype)

    max_bpe = (2 * n + BM - 1) // BM
    # Exact-HF router math followed by a single-CTA budget+bucket kernel.
    # N<=64 compiles with zero spill. N=128's O((2N)^2) rank matrix spills
    # ~0.5KB/thread and has shown cross-warp argmax miscompilation on sm_89;
    # keep it on the exact torch fallback rather than expanding the kernel's
    # correctness domain beyond the production tree sizes.
    use_fused_a = ((n & (n - 1)) == 0 and 16 <= n <= 64 and e <= 16
                   and os.getenv("TREEMOE_FUSED_A") != "0")  # debug: force torch routing
    logits = F.linear(x, router_weight)
    gates = torch.softmax(logits.float(), dim=-1).contiguous()
    if use_fused_a:
        _budget_bucket_fused_kernel[(1,)](
            gates, node_accept_prob,
            ws.topk_flat, ws.gates_flat, ws.padded_slots,
            ws.block_expert_ids, ws.slot_to_row, ws.demand,
            expert_budget, top1_threshold,
            N=n, E=e, EP=16,
            MAX_BPE=max_bpe, BLOCK_M=BM, MAX_BLOCKS=ws.max_blocks,
            CRITICAL_PATH=routing_objective == "critical_path",
            # 32 warps: ptxas shows the O((2N)^2) rank matrix spills 7.6KB/thread
            # at 4 warps; spreading it over 1024 threads -> regs 255->64,
            # spill -> ~1KB (static_analysis.py sweep). Single-CTA latency win.
            num_warps=32,
        )
        return Routing(ws, ws.gates_flat, ws.padded_slots,
                   ws.block_expert_ids, ws.slot_to_row, ws.max_blocks,
                   ws.demand)
    (_topk_ids, topk_gates, padded_slots, block_expert_ids,
     slot_to_row, max_blocks, demand) = route_and_bucket(
        x, router_weight, node_accept_prob, expert_budget,
        top1_threshold=top1_threshold,
        return_demand=True,
        router_gates=gates,
        routing_objective=routing_objective,
    )
    gates_flat = topk_gates.reshape(-1).float().contiguous()   # index by slot id
    return Routing(ws, gates_flat, padded_slots, block_expert_ids,
                   slot_to_row, max_blocks, demand)


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
    routing: Routing | None = None,
    routing_objective: str = "mass",
) -> torch.Tensor:
    """Spec §3.1 entry point. Falls back to the reference on CPU / no Triton.

    deterministic=True (default): split-k partials + fixed-order combine,
    bitwise reproducible (required by the lossless red-line test); costs one
    extra fp32 partial round-trip (~SPLIT_K*2N*H*8B per layer).
    deterministic=False: atomic_add fast path for benchmarking.

    Under TRITON_INTERPRET=1 the Triton kernels execute on CPU (numpy
    interpreter) — the real kernel code paths, minus tensor cores.

    routing: pre-computed phase-1 handle from route_experts() (two-phase
    offload path); None runs routing inline (identical launches).
    """
    if not HAS_TRITON or not (x.is_cuda or _INTERPRET):
        return tree_moe_forward_ref(
            x, w1, w2, w3, router_weight, node_accept_prob, expert_budget,
            routing_objective=routing_objective,
        )

    n, hidden = x.shape
    e, inter, _ = w1.shape
    if routing is None:
        routing = route_experts(x, router_weight, node_accept_prob,
                                expert_budget, inter,
                                routing_objective=routing_objective)
    ws = routing.ws
    gates_flat = routing.gates_flat
    padded_slots, block_expert_ids = routing.padded_slots, routing.block_expert_ids
    slot_to_row, max_blocks = routing.slot_to_row, routing.max_blocks

    # per-shape tile params: tiny interpreter configs (H=64, I=128) must not
    # read past the reduction dim; tl.dot still needs K>=16
    bk1 = BK1 if hidden % BK1 == 0 else hidden
    bn1 = BN1 if inter % BN1 == 0 else inter        # gemm1 N-dim = I
    bn2 = BN2 if hidden % BN2 == 0 else hidden      # gemm2/combine N-dim = H
    seg = inter // SPLIT_K
    bk2 = min(BK2, seg) if seg % min(BK2, seg) == 0 else seg
    # u64-packed weight loads need 16-bit elements and strides/tiles % 4
    pack_w = int(w1.element_size() == 2 and hidden % 4 == 0 and inter % 4 == 0
                 and bk1 % 4 == 0 and bk2 % 4 == 0)
    if os.getenv("TREEMOE_PACK_W") == "0":  # debug: force plain weight loads
        pack_w = 0

    # num_warps=4/num_stages=4: vLLM fused_moe production default for M<=32
    # decode — "smallest batches are memory-latency bound, a deeper pipeline
    # hides the weight loads" (vllm fused_moe.py get_default_config)
    grid1 = (max_blocks, inter // bn1)
    _moe_gemm1_kernel[grid1](
        x, w1, w3, ws.h, padded_slots, block_expert_ids,
        H=hidden, I=inter, BLOCK_M=BM, BLOCK_N=bn1, BLOCK_K=bk1, PACK_W=pack_w,
        num_warps=GEMM1_WARPS, num_stages=GEMM1_STAGES,
    )
    grid2 = (max_blocks, hidden // bn2, SPLIT_K)
    if deterministic:
        partial = ws.get_partial(hidden, x.device)
        _moe_gemm2_det_kernel[grid2](
            ws.h, w2, partial, gates_flat, padded_slots, block_expert_ids,
            R=ws.rows, H=hidden, I=inter,
            BLOCK_M=BM, BLOCK_N=bn2, BLOCK_K=bk2, SPLIT=SPLIT_K, PACK_W=pack_w,
            num_warps=GEMM2_WARPS, num_stages=GEMM2_STAGES,
        )
        _combine_kernel[(n, hidden // bn2)](
            partial, slot_to_row, ws.out_f32,
            R=ws.rows, H=hidden, BLOCK_N=bn2, SPLIT=SPLIT_K,
        )
    else:
        ws.out_f32.zero_()
        _moe_gemm2_kernel[grid2](
            ws.h, w2, ws.out_f32, gates_flat, padded_slots, block_expert_ids,
            H=hidden, I=inter, BLOCK_M=BM, BLOCK_N=bn2, BLOCK_K=bk2, SPLIT=SPLIT_K,
            PACK_W=pack_w,
            num_warps=GEMM2_WARPS, num_stages=GEMM2_STAGES,
        )
    result = ws.out_f32.to(x.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result
