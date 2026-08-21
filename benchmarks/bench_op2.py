"""Op2 microbenchmark: H2D bandwidth + prefetch/compute overlap (no model needed).

Answers the two numbers the offload design (spec §3.2) depends on:
  1. real PCIe H2D bandwidth, pinned vs pageable (352MB per expert-layer
     => how many layers ahead the prefetcher must run);
  2. overlap quality: how much does op1 compute slow down while the ring
     buffer streams experts on its side stream, and what copy bandwidth
     survives the contention.

  python benchmarks/bench_op2.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

E, H, I = 8, 4096, 14336
EXPERT_MB = 3 * I * H * 2 / 2**20  # w1+w2+w3 bf16 per expert


def h2d_gbs(src: torch.Tensor, iters: int = 10) -> float:
    dst = torch.empty_like(src, device="cuda")
    dst.copy_(src, non_blocking=True)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return src.nbytes * iters / (time.perf_counter() - t0) / 1e9


def main() -> None:
    assert torch.cuda.is_available()
    from treemoe.kernels import op1_tree_moe as op1
    from treemoe.kernels.op2_prefetch import HostExpertPool

    print(f"device: {torch.cuda.get_device_name(0)}  expert-layer = {EXPERT_MB:.0f}MB x {E} experts")

    # ---- 1. raw H2D bandwidth, one expert (w1+w3 [I,H] + w2 [H,I]) ----
    one = torch.randn(3 * I * H, dtype=torch.bfloat16)
    pageable = h2d_gbs(one)
    try:
        pinned = h2d_gbs(one.pin_memory())
    except RuntimeError:
        pinned = float("nan")
    print(f"\nH2D bandwidth: pinned {pinned:.1f} GB/s | pageable {pageable:.1f} GB/s")
    per_expert_ms = EXPERT_MB / 1024 / pinned * 1e3
    print(f"=> {per_expert_ms:.1f} ms per expert (pinned); budget B experts/layer "
          f"= {per_expert_ms:.1f}*B ms of copy per layer")

    # ---- 2. overlap: op1 compute + ring-buffer prefetch on the side stream ----
    g = torch.Generator(device="cuda").manual_seed(0)
    w1 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w3 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    router = torch.randn(E, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
    x = torch.randn(64, H, device="cuda", dtype=torch.bfloat16, generator=g)
    accept = torch.rand(64, device="cuda", generator=g)
    fwd = lambda: op1.tree_moe_forward(  # noqa: E731
        x, w1, w2, w3, router, accept, 8, deterministic=False)

    hg = torch.Generator().manual_seed(1)
    try:
        hw1 = torch.randn(E, I, H, dtype=torch.bfloat16, generator=hg).pin_memory()
        hw2 = torch.randn(E, H, I, dtype=torch.bfloat16, generator=hg).pin_memory()
        hw3 = torch.randn(E, I, H, dtype=torch.bfloat16, generator=hg).pin_memory()
    except RuntimeError:
        print("\n(pin_memory unavailable: skipping overlap section)")
        return
    pool = HostExpertPool(num_slots=4, expert_shape=(I, H))

    # Per-stream timing with CUDA events: a device-wide synchronize would wait
    # for the side-stream copies too and report copy-bound wall time instead of
    # the actual compute slowdown.
    ITERS = 30

    def timed_compute(prefetch_backlog: int = 0):
        for _ in range(5):
            fwd()
        torch.cuda.synchronize()
        if prefetch_backlog:  # issue after the sync so copies overlap the timed region
            for k in range(prefetch_backlog):
                pool.prefetch(k // E % 32, k % E, hw1, hw2, hw3)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(ITERS):
            fwd()
        e1.record()
        e1.synchronize()  # waits for the main stream only
        return e0.elapsed_time(e1) / ITERS * 1e3  # us

    t_alone = timed_compute()
    # enough unique-key prefetches (dedup never skips) to outlast the compute window
    n_pref = int(ITERS * t_alone / 1e3 / per_expert_ms) + 4
    t_busy = timed_compute(prefetch_backlog=n_pref)
    torch.cuda.synchronize()  # drain the side stream before reporting

    print(f"\nop1 alone:                {t_alone:8.1f} us")
    print(f"op1 under active prefetch: {t_busy:8.1f} us  "
          f"(compute slowdown {t_busy / t_alone - 1:+.1%})")
    print(f"({n_pref} experts = {n_pref * EXPERT_MB / 1024:.1f} GB streamed on the side stream "
          f"during the timed region)")
    print(f"=> to hide B experts/layer behind a ~{t_busy/1e3:.1f}ms MoE layer, "
          f"prefetch must run ~{per_expert_ms / (t_busy / 1e3):.1f}*B layers ahead")


if __name__ == "__main__":
    main()
