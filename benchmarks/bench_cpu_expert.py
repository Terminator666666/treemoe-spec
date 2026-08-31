"""Fiddler-style break-even: compute a mispredicted expert ON THE HOST CPU
instead of streaming its weights over PCIe (Fiddler, arXiv:2402.07033).

A prefetch miss in op2's exact-offload path costs one expert transfer
(3 x [I,H] bf16 = 336MB at Mixtral shapes ~ 14ms at the 4090's measured
23.8GB/s pinned H2D). But the expert FFN for the FEW tokens routed to that
expert is a tiny GEMM the host CPU can do while the GPU works on other
experts: ship x down (t x 4096 x 2B = 8KB/token), SwiGLU on CPU, ship y back.
This script measures the CPU side and prints the break-even token count.

Runs anywhere (no GPU needed); the break-even is host-CPU dependent, so rerun
on the actual serving host (AutoDL box) before quoting numbers in the thesis.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

H, I = 4096, 14336  # Mixtral-8x7B expert shapes


def cpu_expert_ms(t: int, dtype: torch.dtype, iters: int) -> float:
    g = torch.Generator().manual_seed(0)
    w1 = torch.randn(I, H, generator=g).to(dtype) * 0.02
    w3 = torch.randn(I, H, generator=g).to(dtype) * 0.02
    w2 = torch.randn(H, I, generator=g).to(dtype) * 0.02
    x = torch.randn(t, H, generator=g).to(dtype)
    for _ in range(3):  # warmup (oneDNN JIT / cache)
        y = (F.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = (F.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()
        times.append((time.perf_counter() - t0) * 1e3)
    del y
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--pcie-gbs", type=float, default=23.8,
                    help="measured pinned H2D bandwidth (4090 bench_op2: 23.8)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    expert_bytes = 3 * I * H * (2 if dtype == torch.bfloat16 else 4)
    transfer_ms = expert_bytes / (args.pcie_gbs * 1e9) * 1e3
    gflop_per_tok = 3 * 2 * I * H / 1e9

    print(f"threads={torch.get_num_threads()} dtype={args.dtype} "
          f"expert={expert_bytes / 1e6:.0f}MB transfer@{args.pcie_gbs}GB/s="
          f"{transfer_ms:.2f}ms")
    print(f"{'tokens':>7} {'cpu_ms':>8} {'GFLOP/s':>8} {'xfer_ms':>8}  winner")
    break_even = None
    t1_ms = None
    for t in args.tokens:
        ms = cpu_expert_ms(t, dtype, args.iters)
        if t == 1:
            t1_ms = ms
        gflops = gflop_per_tok * t / (ms / 1e3)
        win = "cpu" if ms < transfer_ms else "transfer"
        if win == "transfer" and break_even is None:
            break_even = t
        print(f"{t:>7} {ms:>8.2f} {gflops:>8.1f} {transfer_ms:>8.2f}  {win}")
    if t1_ms is not None:
        # at t=1 the CPU is DRAM-bound (reads the whole expert once), so the
        # break-even is really host-DRAM-BW vs PCIe-BW; this line tells you
        # which side of that a given host is on
        print(f"implied host DRAM streaming BW at t=1: "
              f"{expert_bytes / (t1_ms / 1e3) / 1e9:.1f} GB/s "
              f"(PCIe reference: {args.pcie_gbs} GB/s)")
    if break_even is None:
        print("break-even: beyond tested range (CPU wins everywhere tested)")
    else:
        print(f"break-even: CPU compute wins below ~{break_even} tokens/expert")


if __name__ == "__main__":
    main()
