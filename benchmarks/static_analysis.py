"""Static (GPU-less) bottleneck analysis of the op1/op4 Triton kernels.

Compiles every kernel ahead-of-time for sm_90 (H100/H200), then runs
`ptxas -v` on the generated PTX to extract the resource footprint that
determines occupancy on real hardware:

  * registers / thread  (spills = the #1 silent perf killer)
  * static shared memory
  * theoretical occupancy on GH100 (65536 regs/SM, 228KB smem, 64 warps)
  * wave count of the real Mixtral-shape grid on 132 SMs

This runs on a CPU-only box: Triton's compiler pipeline down to cubin is
pure host code; only *executing* the cubin needs a GPU.  Register counts
and spill decisions come from the very same ptxas the GPU box will use,
so the numbers transfer 1:1.

Usage:  python benchmarks/static_analysis.py
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# point Triton at the pip-installed ptxas (CPU-only triton wheel may not bundle one)
_PTXAS_CANDIDATES = glob.glob(
    "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvcc/bin/ptxas"
) + glob.glob(
    os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/cuda_nvcc/bin/ptxas")
)
if _PTXAS_CANDIDATES:
    os.environ.setdefault("TRITON_PTXAS_PATH", _PTXAS_CANDIDATES[0])

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402

from treemoe.kernels import op1_tree_moe as op1  # noqa: E402
from treemoe.kernels import op4_commit as op4  # noqa: E402

# ---------------------------------------------------------------- GH100 facts
SM_COUNT = 132          # H100 SXM / H200 (same GH100 die)
REGS_PER_SM = 65536
SMEM_PER_SM = 228 * 1024
MAX_WARPS_PER_SM = 64
MAX_CTAS_PER_SM = 32
REG_ALLOC_UNIT = 256    # regs allocated per warp in units of 256

# Mixtral-8x7B + tree N=64 shapes
N, E, H, I = 64, 8, 4096, 14336
BM, SPLIT_K = op1.BM, op1.SPLIT_K
BN1, BK1, BN2, BK2 = op1.BN1, op1.BK1, op1.BN2, op1.BK2
MAX_BLOCKS = E * ((2 * N + BM - 1) // BM)   # 64
ROWS = MAX_BLOCKS * BM                       # 1024


@dataclass
class Spec:
    name: str
    fn: object
    signature: dict[str, str]
    constexprs: dict[str, int]
    num_warps: int
    num_stages: int
    grid: tuple[int, ...]
    note: str = ""
    # filled in after compile
    regs: int = 0
    smem: int = 0
    spill_st: int = 0
    spill_ld: int = 0
    occupancy: float = 0.0
    limiter: str = ""
    waves: float = 0.0
    extra: dict = field(default_factory=dict)


def _mixtral_specs() -> list[Spec]:
    i64, f32, bf16 = "*i64", "*fp32", "*bf16"
    return [
        Spec(
            "op1 Kernel A (fused budget+bucket)", op1._budget_bucket_fused_kernel,
            {"gates_ptr": f32, "accept_ptr": f32,
             "topk_ids_ptr": i64, "gates_flat_ptr": f32, "padded_slots_ptr": i64,
             "block_expert_ids_ptr": i64, "slot_to_row_ptr": i64, "demand_ptr": f32,
             "expert_budget": "i32", "tau": "fp32",
             "N": "constexpr", "E": "constexpr", "EP": "constexpr",
             "MAX_BPE": "constexpr",
             "BLOCK_M": "constexpr", "MAX_BLOCKS": "constexpr",
             "CRITICAL_PATH": "constexpr"},
            {"N": N, "E": E, "EP": 16,
             "MAX_BPE": (2 * N + BM - 1) // BM, "BLOCK_M": BM,
             "MAX_BLOCKS": MAX_BLOCKS, "CRITICAL_PATH": False},
            num_warps=32, num_stages=1, grid=(1,),
            note="single CTA by design; latency-critical, not occupancy-critical",
        ),
        Spec(
            "op1 Kernel A (critical-path)", op1._budget_bucket_fused_kernel,
            {"gates_ptr": f32, "accept_ptr": f32,
             "topk_ids_ptr": i64, "gates_flat_ptr": f32, "padded_slots_ptr": i64,
             "block_expert_ids_ptr": i64, "slot_to_row_ptr": i64, "demand_ptr": f32,
             "expert_budget": "i32", "tau": "fp32",
             "N": "constexpr", "E": "constexpr", "EP": "constexpr",
             "MAX_BPE": "constexpr", "BLOCK_M": "constexpr",
             "MAX_BLOCKS": "constexpr", "CRITICAL_PATH": "constexpr"},
            {"N": N, "E": E, "EP": 16,
             "MAX_BPE": (2 * N + BM - 1) // BM, "BLOCK_M": BM,
             "MAX_BLOCKS": MAX_BLOCKS, "CRITICAL_PATH": True},
            num_warps=32, num_stages=1, grid=(1,),
            note="lexicographic acceptance-risk coverage; separate constexpr variant",
        ),
        Spec(
            "op1 GEMM1 (w1/w3 + SiLU)", op1._moe_gemm1_kernel,
            {"x_ptr": bf16, "w1_ptr": bf16, "w3_ptr": bf16, "h_ptr": bf16,
             "padded_slots_ptr": i64, "block_expert_ids_ptr": i64,
             "H": "constexpr", "I": "constexpr", "BLOCK_M": "constexpr",
             "BLOCK_N": "constexpr", "BLOCK_K": "constexpr", "PACK_W": "constexpr"},
            {"H": H, "I": I, "BLOCK_M": BM, "BLOCK_N": BN1, "BLOCK_K": BK1,
             "PACK_W": 1},
            num_warps=op1.GEMM1_WARPS, num_stages=op1.GEMM1_STAGES,
            grid=(MAX_BLOCKS, I // BN1),
            note="dominant kernel: streams w1+w3 (2/3 of expert bytes)",
        ),
        Spec(
            "op1 GEMM2 det (split-K partials)", op1._moe_gemm2_det_kernel,
            {"h_ptr": bf16, "w2_ptr": bf16, "partial_ptr": f32, "gates_ptr": f32,
             "padded_slots_ptr": i64, "block_expert_ids_ptr": i64,
             "R": "constexpr", "H": "constexpr", "I": "constexpr",
             "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",
             "SPLIT": "constexpr", "PACK_W": "constexpr"},
            {"R": ROWS, "H": H, "I": I, "BLOCK_M": BM, "BLOCK_N": BN2,
             "BLOCK_K": min(BK2, I // SPLIT_K), "SPLIT": SPLIT_K, "PACK_W": 1},
            num_warps=op1.GEMM2_WARPS, num_stages=op1.GEMM2_STAGES,
            grid=(MAX_BLOCKS, H // BN2, SPLIT_K),
            note="streams w2 (1/3 of expert bytes)",
        ),
        Spec(
            "op1 GEMM2 atomic (fast path)", op1._moe_gemm2_kernel,
            {"h_ptr": bf16, "w2_ptr": bf16, "out_ptr": f32, "gates_ptr": f32,
             "padded_slots_ptr": i64, "block_expert_ids_ptr": i64,
             "H": "constexpr", "I": "constexpr", "BLOCK_M": "constexpr",
             "BLOCK_N": "constexpr", "BLOCK_K": "constexpr", "SPLIT": "constexpr",
             "PACK_W": "constexpr"},
            {"H": H, "I": I, "BLOCK_M": BM, "BLOCK_N": BN2,
             "BLOCK_K": min(BK2, I // SPLIT_K), "SPLIT": SPLIT_K, "PACK_W": 1},
            num_warps=op1.GEMM2_WARPS, num_stages=op1.GEMM2_STAGES,
            grid=(MAX_BLOCKS, H // BN2, SPLIT_K),
        ),
        Spec(
            "op1 combine (fixed-order reduce)", op1._combine_kernel,
            {"partial_ptr": f32, "slot_to_row_ptr": i64, "out_ptr": f32,
             "R": "constexpr", "H": "constexpr", "BLOCK_N": "constexpr",
             "SPLIT": "constexpr"},
            {"R": ROWS, "H": H, "BLOCK_N": BN2, "SPLIT": SPLIT_K},
            num_warps=4, num_stages=1, grid=(N, H // BN2),
        ),
        Spec(
            "op4 tree-verify greedy", op4._tree_verify_greedy_kernel,
            {"argmax_ptr": i64, "tree_tokens_ptr": i64, "child_start_ptr": i64,
             "child_list_ptr": i64, "child_count_ptr": i64,
             "accepted_slots_ptr": i64, "bonus_ptr": i64,
             "num_accepted_ptr": i64, "next_root_ptr": i64,
             "MAX_DEPTH": "constexpr"},
            {"MAX_DEPTH": 6},
            num_warps=1, num_stages=1, grid=(1,),
            note="single CTA serial DFS; latency-critical, not occupancy-critical",
        ),
        Spec(
            "op4 argmax (V=32000)", op4._argmax_kernel,
            {"x_ptr": f32, "out_ptr": i64, "V": "constexpr", "VB": "constexpr"},
            {"V": 32000, "VB": 1024},
            num_warps=4, num_stages=1, grid=(N,),
        ),
        Spec(
            "op4 online softmax", op4._postprocess_softmax_kernel,
            {"logits_ptr": f32, "probs_ptr": f32, "prev_tokens_ptr": i64,
             "V": "constexpr", "VB": "constexpr",
             "temperature": "fp32", "rep_penalty": "fp32", "num_prev": "i32"},
            {"V": 32000, "VB": 1024},
            num_warps=4, num_stages=1, grid=(N,),
        ),
        Spec(
            "op4 KV commit", op4._kv_commit_kernel,
            {"k_ptr": bf16, "v_ptr": bf16, "accepted_slots_ptr": i64,
             "dest_block_ptr": i64, "dest_off_ptr": i64, "num_accepted_ptr": i64,
             "tree_block": "constexpr", "BLOCK_SIZE": "constexpr",
             "KVH_HD": "constexpr", "stride_layer": "constexpr",
             "stride_block": "constexpr", "stride_slot": "constexpr",
             "PACK": "constexpr"},
            {"tree_block": 0, "BLOCK_SIZE": 64, "KVH_HD": 8 * 128,
             "stride_layer": 256 * 64 * 1024, "stride_block": 64 * 1024,
             "stride_slot": 1024, "PACK": 1},
            num_warps=4, num_stages=1, grid=(32, 6),
        ),
    ]


_PTXAS_RE = re.compile(
    r"Used (\d+) registers(?:.*?(\d+) bytes smem)?", re.S)
_SPILL_RE = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")


def compile_and_measure(spec: Spec, build_dir: str) -> None:
    src = triton.compiler.ASTSource(
        fn=spec.fn, signature=spec.signature, constexprs=spec.constexprs)
    k = triton.compile(
        src, target=GPUTarget("cuda", 90, 32),
        options={"num_warps": spec.num_warps, "num_stages": spec.num_stages})
    ptx = k.asm["ptx"]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", spec.name).strip("_")
    fname = os.path.join(build_dir, safe + ".ptx")
    with open(fname, "w") as f:
        f.write(ptx)
    # triton emits .target sm_90a (Hopper wgmma etc.) -> match ptxas arch
    arch = "sm_90a" if ".target sm_90a" in ptx else "sm_90"
    out = subprocess.run(
        [os.environ.get("TRITON_PTXAS_PATH", "ptxas"), f"-arch={arch}", "-v",
         "-o", "/dev/null", fname],
        capture_output=True, text=True).stderr
    m = _PTXAS_RE.search(out)
    if m:
        spec.regs = int(m.group(1))
        spec.smem = getattr(k.metadata, "shared", 0) or int(m.group(2) or 0)
    else:
        spec.smem = getattr(k.metadata, "shared", 0)
    sp = _SPILL_RE.search(out)
    if sp:
        spec.spill_st, spec.spill_ld = int(sp.group(1)), int(sp.group(2))

    # ---- theoretical occupancy on GH100 ----
    warps = spec.num_warps
    threads = warps * 32
    regs_per_warp = -(-spec.regs * 32 // REG_ALLOC_UNIT) * REG_ALLOC_UNIT
    lim_regs = (REGS_PER_SM // (regs_per_warp * warps)) if spec.regs else MAX_CTAS_PER_SM
    lim_smem = (SMEM_PER_SM // spec.smem) if spec.smem else MAX_CTAS_PER_SM
    lim_warps = MAX_WARPS_PER_SM // warps
    ctas = min(lim_regs, lim_smem, lim_warps, MAX_CTAS_PER_SM)
    spec.occupancy = ctas * warps / MAX_WARPS_PER_SM
    spec.limiter = min(
        (("regs", lim_regs), ("smem", lim_smem), ("warp-slots", lim_warps),
         ("cta-slots", MAX_CTAS_PER_SM)), key=lambda t: t[1])[0]
    total_ctas = 1
    for g in spec.grid:
        total_ctas *= g
    spec.waves = total_ctas / (ctas * SM_COUNT)
    spec.extra = {"ctas_per_sm": ctas, "total_ctas": total_ctas,
                  "threads": threads}


def main() -> None:
    build_dir = os.path.join(os.path.dirname(__file__), "..", "build", "ptx")
    os.makedirs(build_dir, exist_ok=True)
    specs = _mixtral_specs()
    print(f"target sm_90 (H100/H200, {SM_COUNT} SMs)  "
          f"shapes: N={N} E={E} H={H} I={I} "
          f"g1 {BM}x{BN1}x{BK1} g2 {BM}x{BN2}x{BK2} splitK={SPLIT_K}\n")
    hdr = (f"{'kernel':38s} {'regs':>4s} {'spill':>7s} {'smem':>7s} "
           f"{'occ':>5s} {'lim':>10s} {'grid CTAs':>9s} {'waves':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for s in specs:
        try:
            compile_and_measure(s, build_dir)
        except Exception as exc:  # keep going, report at the end
            print(f"{s.name:38s} COMPILE FAILED: {type(exc).__name__}: {exc}")
            continue
        spill = f"{s.spill_st}/{s.spill_ld}" if (s.spill_st or s.spill_ld) else "0"
        print(f"{s.name:38s} {s.regs:4d} {spill:>7s} {s.smem:7d} "
              f"{s.occupancy:5.0%} {s.limiter:>10s} "
              f"{s.extra['total_ctas']:9d} {s.waves:6.2f}")
        if s.note:
            print(f"{'':38s} └─ {s.note}")
    print("\nreading the table:")
    print("  spill > 0        -> register pressure forcing local-memory traffic (fix first)")
    print("  occ  < 25%       -> too few warps to hide HBM latency on a memory-bound kernel")
    print("  waves < 1        -> grid can't fill the GPU; kernel is launch/latency bound")
    print("  PTX dumped to build/ptx/ for manual inspection (ld.global.L2::evict_* hints etc.)")


if __name__ == "__main__":
    main()
