"""Task 2.4: op1 microbenchmark vs BF16 baselines (spec §3.1 acceptance gate).

Gate: at N=64, op1 latency <= 0.8x vLLM fused_moe OR HBM bytes read -30%
(measure bytes with: ncu --metrics dram__bytes_read.sum <this script>).

Baselines are optional imports; missing ones are reported and skipped.
"""

from __future__ import annotations

import argparse
import time

import torch

E, H, I = 8, 4096, 14336


def timed(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def bench_ours(x, w1, w2, w3, router, accept, budget):
    from treemoe.kernels.op1_tree_moe import tree_moe_forward

    return timed(lambda: tree_moe_forward(x, w1, w2, w3, router, accept, budget))


def bench_vllm(x, w1, w2, w3, router):
    try:
        from vllm.model_executor.layers.fused_moe import fused_moe
    except ImportError:
        return None
    w13 = torch.cat([w1, w3], dim=1).contiguous()  # vLLM packs gate+up
    gating = x.float() @ router.t().float()
    return timed(lambda: fused_moe(x, w13, w2, gating, topk=2, renormalize=True))


def bench_megablocks_note() -> str:
    return "megablocks: run via its dMoE layer harness, see benchmarks/README"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--budget", type=int, default=8)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "kernel benchmark requires a GPU"

    g = torch.Generator(device="cuda").manual_seed(0)
    w1 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w3 = torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    router = torch.randn(E, H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1

    print(f"{'N':>5} {'ours(us)':>10} {'vllm(us)':>10} {'ratio':>7}")
    for n in args.tree_sizes:
        x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=g)
        accept = torch.rand(n, device="cuda", generator=g)
        t_ours = bench_ours(x, w1, w2, w3, router, accept, args.budget)
        t_vllm = bench_vllm(x, w1, w2, w3, router)
        ratio = f"{t_ours / t_vllm:.2f}" if t_vllm else "n/a"
        print(f"{n:>5} {t_ours:>10.1f} {t_vllm or float('nan'):>10.1f} {ratio:>7}")
        if n == 64 and t_vllm:
            gate = t_ours <= 0.8 * t_vllm
            print(f"  gate(N=64, <=0.8x vLLM): {'PASS' if gate else 'FAIL (check dram bytes)'}")
    print(bench_megablocks_note())


if __name__ == "__main__":
    main()
