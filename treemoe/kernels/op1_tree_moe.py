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
    return topk_ids, topk_gates, padded_slots, block_expert_ids, max_blocks


# --------------------------------------------------------------------------
# Kernel B1 — grouped w1/w3 GEMM + SiLU⊙ (expert-stationary)
# --------------------------------------------------------------------------

if HAS_TRITON:

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


# --------------------------------------------------------------------------
# Host wrapper
# --------------------------------------------------------------------------

class _Workspace:
    """Static buffers reused across steps (CUDA Graph friendly)."""

    def __init__(self, n: int, e: int, hidden: int, inter: int, device):
        max_blocks = e * ((2 * n + BM - 1) // BM)
        self.h = torch.zeros(max_blocks * BM, inter, dtype=torch.bfloat16, device=device)
        self.out_f32 = torch.zeros(n, hidden, dtype=torch.float32, device=device)


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
) -> torch.Tensor:
    """Spec §3.1 entry point. Falls back to the reference on CPU / no Triton."""
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

    _topk_ids, topk_gates, padded_slots, block_expert_ids, max_blocks = route_and_bucket(
        x, router_weight, node_accept_prob, expert_budget
    )
    gates_flat = topk_gates.reshape(-1).float().contiguous()   # index by slot id

    ws.out_f32.zero_()
    grid1 = (max_blocks, inter // BN)
    _moe_gemm1_kernel[grid1](
        x, w1, w3, ws.h, padded_slots, block_expert_ids,
        H=hidden, I=inter, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    grid2 = (max_blocks, hidden // BN, SPLIT_K)
    _moe_gemm2_kernel[grid2](
        ws.h, w2, ws.out_f32, gates_flat, padded_slots, block_expert_ids,
        H=hidden, I=inter, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT=SPLIT_K,
    )
    result = ws.out_f32.to(x.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result
