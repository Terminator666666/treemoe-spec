"""On-silicon tile-config sweep for op1 (finally have a GPU).

4090 torch.profiler breakdown (N=64, atomic path):
  gemm1 2.036ms -> 923 GB/s = 92% of peak   (BN1=64 BK1=64 nw8 ns3)
  gemm2 1.942ms -> 484 GB/s = 48% of peak   (BN2=32 BK2=32 SPLIT_K=4 nw4 ns4)
gemm2 starves: BK2=32 means 112 K-iterations of 2KB w2 tiles, ~8KB
in-flight per CTA -- not enough to hide DRAM latency (gemm1 keeps ~48KB
in flight). The static (ptxas-only) choice optimized occupancy, but
bytes-in-flight is what a streaming kernel needs. Sweep on hardware:

  python benchmarks/sweep_op1.py            # gemm2 sweep (the sick one)
  python benchmarks/sweep_op1.py --gemm 1   # gemm1 sweep
Each config is correctness-checked against the bf16 reference first.
Timing is the median of event-timed reps; a config only "beats" the default
if it clears AsmEvo's variance-aware commit margin (arXiv:2608.20711 Eq.2):
  speedup >= 1 + max(eps, k*cv)
which stops clock jitter on an unlocked 4090 from promoting noise.
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

E, H, I = 8, 4096, 14336
EPS, K_CV = 0.002, 0.85   # AsmEvo defaults


def make_inputs(n: int):
    g = torch.Generator(device="cuda").manual_seed(0)
    w1 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w3 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    router = torch.randn(E, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=g)
    accept = torch.rand(n, device="cuda", generator=g)
    return x, w1, w2, w3, router, accept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemm", type=int, choices=[1, 2], default=2)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    assert torch.cuda.is_available()

    from treemoe.kernels import op1_tree_moe as op1
    from treemoe.ref.tree_moe_ref import tree_moe_forward_ref

    inputs = make_inputs(args.n)
    x, w1, w2, w3, router, accept = inputs
    ref = tree_moe_forward_ref(x, w1, w2, w3, router, accept, args.budget).float()
    err_tol = 0.5  # loose sanity gate; winners get re-validated by pytest

    if args.gemm == 2:
        combos = [
            ("BN2", "BK2", "SPLIT_K", "GEMM2_WARPS", "GEMM2_STAGES"),
            itertools.product((32, 64), (32, 64, 128), (2, 4, 8), (4, 8), (3, 4, 5)),
        ]
    else:
        combos = [
            ("BN1", "BK1", "GEMM1_WARPS", "GEMM1_STAGES"),
            itertools.product((32, 64, 128), (32, 64, 128), (4, 8), (2, 3, 4)),
        ]
    names, grid = combos
    defaults = {k: getattr(op1, k) for k in names}

    def run(cfg: dict) -> tuple[float, float]:
        """-> (median_us, cv) via CUDA events (immune to host jitter)."""
        for k, v in cfg.items():
            setattr(op1, k, v)
        op1._ws_cache.clear()
        fwd = lambda: op1.tree_moe_forward(  # noqa: E731
            x, w1, w2, w3, router, accept, args.budget, deterministic=False)
        for _ in range(5):
            out = fwd()
        err = (out.float() - ref).abs().max().item()
        if err > err_tol:
            raise ValueError(f"wrong result, max|d|={err:.3f}")
        torch.cuda.synchronize()
        times = []
        for _ in range(args.iters):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            fwd()
            t1.record()
            t1.synchronize()
            times.append(t0.elapsed_time(t1) * 1e3)  # us
        med = statistics.median(times)
        cv = statistics.pstdev(times) / med if med else 0.0
        return med, cv

    print(f"device: {torch.cuda.get_device_name(0)}  sweeping gemm{args.gemm} "
          f"(N={args.n}, budget={args.budget}, atomic path)")
    print("  ".join(f"{k:>12}" for k in names) + f" {'us':>10} {'cv':>6}")
    results = []
    for vals in grid:
        cfg = dict(zip(names, vals))
        if args.gemm == 2 and (I // cfg["SPLIT_K"]) % cfg["BK2"] != 0:
            continue  # split segment must be divisible by the K tile
        try:
            t, cv = run(cfg)
        except Exception as e:
            print("  ".join(f"{v:>12}" for v in vals) + f"     FAIL  {str(e)[:60]}")
            continue
        results.append((t, cv, cfg))
        print("  ".join(f"{v:>12}" for v in vals) + f" {t:>10.1f} {cv:>6.1%}")

    for k, v in defaults.items():  # restore
        setattr(op1, k, v)
    op1._ws_cache.clear()

    results.sort(key=lambda r: r[0])
    print("\ntop 5:")
    for t, cv, cfg in results[:5]:
        print(f"  {t:>8.1f}us (cv {cv:.1%})  {cfg}")
    base = dict(defaults)
    hit = next(((t, cv) for t, cv, c in results
                if all(c[k] == base[k] for k in c)), None)
    if hit and results:
        tb, cvb = hit
        best_t, best_cv, best_cfg = results[0]
        margin = 1 + max(EPS, K_CV * max(cvb, best_cv))
        speedup = tb / best_t
        verdict = ("COMMIT (clears variance-aware margin)"
                   if speedup >= margin else
                   "keep default (inside noise margin -- AsmEvo Eq.2)")
        print(f"\ncurrent default: {tb:.1f}us -> best: {best_t:.1f}us "
              f"({speedup:.3f}x, margin {margin:.3f}x) => {verdict}")


if __name__ == "__main__":
    main()
