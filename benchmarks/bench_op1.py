"""Task 2.4: op1 microbenchmark vs BF16 baselines (spec §3.1 acceptance gate).

Gate: at N=64, op1 latency <= 0.8x vLLM fused_moe OR HBM bytes read -30%
(measure bytes with: ncu --metrics dram__bytes_read.sum <this script>).

Baselines are optional imports; missing ones are reported and skipped.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

E, H, I = 8, 4096, 14336

# HBM peak by device name (GB/s) for bandwidth-utilization reporting;
# override with --peak-gbs for cards not listed
_PEAK_GBS = {"4090": 1008, "4080": 717, "A100": 2039, "A10": 600,
             "L40S": 864, "H100": 3350, "H200": 4800, "5070": 672}


def detect_peak_gbs() -> float | None:
    name = torch.cuda.get_device_name(0)
    for key, gbs in _PEAK_GBS.items():
        if key in name:
            return float(gbs)
    return None


def timed(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def bench_ours(x, w1, w2, w3, router, accept, budget, deterministic=True):
    from treemoe.kernels.op1_tree_moe import tree_moe_forward

    return timed(lambda: tree_moe_forward(x, w1, w2, w3, router, accept, budget,
                                          deterministic=deterministic))


def profile_ours(x, w1, w2, w3, router, accept, budget, deterministic=False):
    """Per-kernel CUDA time breakdown (torch.profiler; no ncu perms needed)."""
    from torch.profiler import ProfilerActivity, profile

    from treemoe.kernels.op1_tree_moe import tree_moe_forward

    for _ in range(5):  # warmup + compile
        tree_moe_forward(x, w1, w2, w3, router, accept, budget, deterministic=deterministic)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            tree_moe_forward(x, w1, w2, w3, router, accept, budget, deterministic=deterministic)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))


def resolve_vllm_fused_moe() -> tuple[str, Callable] | None:
    """Resolve the legacy fused_moe or current fused_experts API."""
    try:
        package = importlib.import_module("vllm.model_executor.layers.fused_moe")
    except ImportError:
        return None

    candidate = getattr(package, "fused_moe", None)
    if callable(candidate):
        return "fused_moe", candidate
    candidate = getattr(package, "fused_experts", None)
    if callable(candidate):
        return "fused_experts", candidate

    try:
        module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.fused_moe"
        )
    except ImportError:
        return None
    candidate = getattr(module, "fused_moe", None)
    if callable(candidate):
        return "fused_moe", candidate
    candidate = getattr(module, "fused_experts", None)
    return ("fused_experts", candidate) if callable(candidate) else None


def bench_vllm(x, w1, w2, w3, router):
    resolved = resolve_vllm_fused_moe()
    if resolved is None:
        return None
    api, fused_moe = resolved
    w13 = torch.cat([w1, w3], dim=1).contiguous()  # vLLM packs gate+up

    def run():
        # Match op1's input/output boundary: both timings start from hidden
        # states and include router-logit computation plus exact top-2 MoE.
        gating = torch.nn.functional.linear(x, router).float()
        if api == "fused_moe":
            return fused_moe(x, w13, w2, gating, topk=2, renormalize=True)
        topk_logits, topk_ids = gating.topk(2, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)
        return fused_moe(x, w13, w2, topk_weights, topk_ids)

    return timed(run)


def bench_megablocks_note() -> str:
    return "megablocks: run via its dMoE layer harness, see benchmarks/README"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--block-m", type=int, choices=(16, 32), default=16,
                    help="expert token tile; run separate processes to compare BM16/BM32")
    ap.add_argument("--peak-gbs", type=float, default=None,
                    help="HBM peak GB/s for utilization %% (auto-detected for known cards)")
    ap.add_argument("--profile", action="store_true",
                    help="print a per-kernel CUDA time breakdown per tree size (atomic path)")
    args = ap.parse_args()
    os.environ["TREEMOE_BLOCK_M"] = str(args.block_m)
    assert torch.cuda.is_available(), "kernel benchmark requires a GPU"
    peak = args.peak_gbs or detect_peak_gbs()
    print(f"device: {torch.cuda.get_device_name(0)}"
          + (f"  (peak {peak:.0f} GB/s)" if peak else "  (peak unknown: pass --peak-gbs)"))
    try:
        import vllm
        print(f"vllm: {vllm.__version__}")
    except ImportError:
        print("vllm: not installed (baseline column will be skipped)")
    print("scope: hidden states -> router logits -> exact top-2 MoE; BF16 weights/activations")
    from treemoe.kernels.op1_tree_moe import GEMM1_WARPS, GEMM2_WARPS
    print(f"tile: BM={args.block_m}, gemm1_warps={GEMM1_WARPS}, "
          f"gemm2_warps={GEMM2_WARPS}")

    g = torch.Generator(device="cuda").manual_seed(0)
    w1 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w3 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    router = torch.randn(E, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1

    print(
        f"{'N':>5} {'blocks':>7} {'experts':>7} {'grid':>4} {'cap':>4} "
        f"{'det(us)':>10} {'atomic(us)':>11} {'uniqGB/s':>9} {'uniq%':>6} "
        f"{'loadGB/s':>9} "
        f"{'vllm(us)':>10} {'ratio':>7}"
    )
    for n in args.tree_sizes:
        x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=g)
        # All ones disable low-probability top-1 degradation. With B=E=8,
        # ours and vLLM therefore execute the same lossless top-2 semantics.
        accept = torch.ones(n, device="cuda")
        # deterministic=True is the red-line path (fp32 partial round-trip +
        # combine launch); deterministic=False (atomic) is the production
        # perf path -- GB/s, util and the vLLM gate are measured on it
        t_det = bench_ours(x, w1, w2, w3, router, accept, args.budget, deterministic=True)
        t_ours = bench_ours(x, w1, w2, w3, router, accept, args.budget, deterministic=False)
        t_vllm = bench_vllm(x, w1, w2, w3, router)
        from treemoe.kernels.op1_tree_moe import route_experts
        routing = route_experts(x, router, accept, args.budget, I)
        active_blocks = int((
            routing.block_expert_ids[:routing.launch_blocks] >= 0
        ).sum())
        active_experts = len(routing.expert_ids())
        # uniqGB/s estimates compulsory weight traffic if each active expert
        # is fetched once. loadGB/s counts every block's logical load; repeated
        # blocks of one expert can hit L2, so this value may exceed HBM peak.
        expert_bytes = (w1.nbytes + w2.nbytes + w3.nbytes) / E
        unique_gbs = active_experts * expert_bytes / (t_ours * 1e-6) / 1e9
        logical_gbs = active_blocks * expert_bytes / (t_ours * 1e-6) / 1e9
        unique_util = f"{unique_gbs / peak:5.0%}" if peak else "   n/a"
        ratio = f"{t_ours / t_vllm:.2f}" if t_vllm else "n/a"
        result = (
            f"{n:>5} {active_blocks:>7} {active_experts:>7} "
            f"{routing.launch_blocks:>4} {routing.max_blocks:>4} "
            f"{t_det:>10.1f} {t_ours:>11.1f} "
            f"{unique_gbs:>9.0f} {unique_util:>6} {logical_gbs:>9.0f} "
            f"{t_vllm or float('nan'):>10.1f} {ratio:>7}"
        )
        print(result)
        if args.profile:
            profile_ours(x, w1, w2, w3, router, accept, args.budget)
        if n == 64 and t_vllm:
            gate = t_ours <= 0.8 * t_vllm
            print(f"  gate(N=64, <=0.8x vLLM): {'PASS' if gate else 'FAIL (check dram bytes)'}")
        print("traffic: uniqGB/s reads each active expert once; loadGB/s counts per-block "
                    "loads and includes cache reuse. Use Nsight Compute for measured DRAM.")
    print(bench_megablocks_note())


if __name__ == "__main__":
    main()
