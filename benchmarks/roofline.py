"""Analytical roofline projection of TreeMoE-Spec speedup (no GPU needed).

Both the AR baseline and tree verification at decode batch sizes are firmly
HBM-bandwidth bound on Hopper (arithmetic intensity of the verify GEMMs is
~2N/2B = 64 bf16-flops/byte at N=64, far under GH100's ~206 flops/byte machine
balance), so step time ≈ bytes moved / bandwidth.  This script does exact
byte accounting from the real kernel/data layout and combines it with MAT
(mean accepted tokens per step) to project end-to-end speedup:

    speedup = MAT * bytes_AR / bytes_tree        (both memory-bound)

MAT is the one quantity that cannot be produced on a CPU-only box (it needs
the real Mixtral + EAGLE-2 weights), so it is swept over the plausible range;
EAGLE-2 reports 3.5-4.3 for Mixtral-8x7B at temperature 0.  Phase-0
measurements (measurements/collect_routing.py) pin down the (budget B -> MAT)
curve on the experiment machine; re-run this script with those numbers.

Usage:  python benchmarks/roofline.py [--ctx 512] [--gpu h200|h100]
"""

from __future__ import annotations

import argparse

GB = 1 << 30

# ---------------------------------------------------------------- Mixtral-8x7B
L = 32          # layers
H = 4096        # hidden
I = 14336       # expert intermediate
E = 8           # experts
TOP_K = 2
V = 32000       # vocab
KVH_HD = 8 * 128            # kv_heads * head_dim
BYTES_W = 2                 # experts stay native BF16 (project mandate)

# tree / engine shape
N = 64          # tree nodes
D = 6           # tree depth (draft steps)
SPLIT_K = 4     # deterministic split-K partial factor (op1)

# per-layer weight bytes
EXPERT_B = 3 * I * H * BYTES_W                 # w1+w2+w3 of ONE expert  (352MB)
ATTN_B = (2 * H * H + 2 * H * KVH_HD) * BYTES_W   # q,o + k,v (GQA)      (84MB)
LM_HEAD_B = V * H * BYTES_W                    # read once per step      (262MB)

# EAGLE-2 draft: 1 decoder layer (dense FFN, same dims) + reuses target
# embeddings/lm_head; D sequential expansion passes re-read it every step.
DRAFT_LAYER_B = ATTN_B + 3 * I * H * BYTES_W // 1   # ~436MB
DRAFT_STEP_B = D * (DRAFT_LAYER_B + LM_HEAD_B)

GPUS = {  # (HBM bandwidth B/s, name)
    "h200": (4.8e12, "H200 141GB (4.8 TB/s)"),
    "h100": (3.35e12, "H100 80GB SXM (3.35 TB/s)"),
}


def kv_bytes(ctx: int, n_query_groups: int = 1) -> int:
    """KV cache read per step: FlashAttention streams K,V once per query block;
    N=64 tree queries fit ONE block, so tree and AR read the same bytes."""
    return 2 * L * KVH_HD * BYTES_W * ctx * n_query_groups


def ar_step_bytes(ctx: int) -> int:
    moe = TOP_K * EXPERT_B * L                # top-2 experts per layer
    return moe + ATTN_B * L + LM_HEAD_B + kv_bytes(ctx)


def tree_step_bytes(budget: int, ctx: int) -> dict[str, float]:
    """One draft+verify+commit step with expert budget B per layer."""
    moe_w = budget * EXPERT_B * L
    # op1 activation overhead (per layer, x32): h workspace write+read (bf16)
    # + deterministic split-K fp32 partial write+read (~9% tax, spec §3.1)
    h_ws = 2 * (2 * N) * I * BYTES_W
    partial = 2 * SPLIT_K * (2 * N) * H * 4
    acts = (h_ws + partial) * L
    lm = LM_HEAD_B                            # weight streamed once for all 64 nodes
    return {
        "moe_w": moe_w, "attn_w": ATTN_B * L, "lm_head": lm,
        "acts": acts, "draft": DRAFT_STEP_B, "kv": kv_bytes(ctx),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=512, help="context length")
    ap.add_argument("--gpu", choices=GPUS, default="h200")
    args = ap.parse_args()
    bw, gpu_name = GPUS[args.gpu]

    ar_b = ar_step_bytes(args.ctx)
    ar_tpot = ar_b / bw * 1e3
    print(f"model Mixtral-8x7B BF16 | tree N={N} D={D} | ctx={args.ctx} | {gpu_name}")
    print(f"AR baseline: {ar_b / GB:.2f} GB/step -> TPOT >= {ar_tpot:.2f} ms "
          f"(memory-bound floor)\n")

    print("tree step byte breakdown (GB):")
    hdr = f"{'B':>2s} {'moe_w':>7s} {'attn_w':>7s} {'lm_head':>7s} {'acts':>6s} {'draft':>6s} {'kv':>6s} {'total':>7s} {'vs AR':>6s}"
    print(hdr)
    budgets = (2, 3, 4, 6, 8)
    totals = {}
    for b in budgets:
        d = tree_step_bytes(b, args.ctx)
        tot = sum(d.values())
        totals[b] = tot
        print(f"{b:2d} {d['moe_w']/GB:7.2f} {d['attn_w']/GB:7.2f} {d['lm_head']/GB:7.2f} "
              f"{d['acts']/GB:6.2f} {d['draft']/GB:6.2f} {d['kv']/GB:6.2f} "
              f"{tot/GB:7.2f} {tot/ar_b:5.2f}x")

    print("\nprojected end-to-end speedup = MAT * bytes_AR / bytes_tree")
    print("(MAT = mean accepted tokens/step incl. bonus token; EAGLE-2 paper: "
          "3.5-4.3 on Mixtral @ T=0.\n Lower budget B degrades routing fidelity "
          "-> MAT drops; the B<->MAT curve comes from Phase-0 measurements.)")
    mats = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
    print(f"\n{'B \\ MAT':>8s} " + " ".join(f"{m:>6.1f}" for m in mats))
    for b in budgets:
        ratio = ar_b / totals[b]
        cells = " ".join(f"{m * ratio:>5.2f}x" for m in mats)
        print(f"{b:8d} {cells}")

    print(f"\nTPOT floor at MAT=3.5 ({gpu_name}):")
    for b in budgets:
        tpot = totals[b] / bw / 3.5 * 1e3
        print(f"  B={b}: {tpot:5.2f} ms/token  (AR floor {ar_tpot:.2f} ms)")

    print("\ncaveats (this is a projection, not a measurement):")
    print("  * assumes perfect bandwidth utilisation on both sides; real kernels")
    print("    reach 70-90% — ratios are more robust than absolute TPOT")
    print("  * ignores kernel launch latency (mitigated by CUDA Graphs + fused")
    print("    Kernel A) and the op4 verify/commit serial tail (<40us/step)")
    print("  * MAT column values are hypotheses until Phase-0 runs on the GPU box")


if __name__ == "__main__":
    main()
