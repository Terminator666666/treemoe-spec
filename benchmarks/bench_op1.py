"""Task 2.4: op1 microbenchmark vs BF16 baselines (spec §3.1 acceptance gate).

Gate: at N=64, op1 latency <= 0.8x vLLM fused_moe OR HBM bytes read -30%
(measure bytes with: ncu --metrics dram__bytes_read.sum <this script>).

Baselines are optional imports; missing ones are reported and skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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


def bench_vllm(x, w1, w2, w3, router):
    try:
        from vllm.model_executor.layers.fused_moe import fused_moe
    except ImportError:
        return None
    w13 = torch.cat([w1, w3], dim=1).contiguous()  # vLLM packs gate+up

    def run():
        # Match op1's input/output boundary: both timings start from hidden
        # states and include router-logit computation plus exact top-2 MoE.
        gating = x.float() @ router.t().float()
        return fused_moe(x, w13, w2, gating, topk=2, renormalize=True)

    return timed(run)


def bench_megablocks_note() -> str:
    return "megablocks: run via its dMoE layer harness, see benchmarks/README"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--peak-gbs", type=float, default=None,
                    help="HBM peak GB/s for utilization %% (auto-detected for known cards)")
    ap.add_argument("--profile", action="store_true",
                    help="print a per-kernel CUDA time breakdown per tree size (atomic path)")
    args = ap.parse_args()
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

    g = torch.Generator(device="cuda").manual_seed(0)
    w1 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w3 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    router = torch.randn(E, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1

    print(
        f"{'N':>5} {'blocks':>7} {'grid':>4} {'cap':>4} {'det(us)':>10} "
        f"{'atomic(us)':>11} {'GB/s':>8} {'util':>6} "
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
        # Every BM-row block is a separate CTA group and streams its expert's
        # weights once. Skewed experts spanning multiple blocks therefore
        # reread weights; count blocks, not unique experts.
        expert_bytes = (w1.nbytes + w2.nbytes + w3.nbytes) / E
        byts = active_blocks * expert_bytes
        gbs = byts / (t_ours * 1e-6) / 1e9
        util = f"{gbs / peak:5.0%}" if peak else "   n/a"
        ratio = f"{t_ours / t_vllm:.2f}" if t_vllm else "n/a"
        result = (
            f"{n:>5} {active_blocks:>7} {routing.launch_blocks:>4} "
            f"{routing.max_blocks:>4} {t_det:>10.1f} {t_ours:>11.1f} "
            f"{gbs:>8.0f} {util:>6} "
            f"{t_vllm or float('nan'):>10.1f} {ratio:>7}"
        )
        print(result)
        if args.profile:
            profile_ours(x, w1, w2, w3, router, accept, args.budget)
        if n == 64 and t_vllm:
            gate = t_ours <= 0.8 * t_vllm
            print(f"  gate(N=64, <=0.8x vLLM): {'PASS' if gate else 'FAIL (check dram bytes)'}")
    print(bench_megablocks_note())


if __name__ == "__main__":
    main()
