"""End-to-end TPOT / accept-length benchmark with budget & tree-size sweeps
(paper main figure: tau-TPOT Pareto over B, plan Task 2.5 / Phase 5)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def run_config(engine_factory, budget: int, tree_size: int, prompts: list[torch.Tensor],
               max_new_tokens: int = 128) -> dict:
    eng, pf = engine_factory(budget=budget, tree_size=tree_size)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_tokens = 0
    for p in prompts:
        out = eng.generate(p, max_new_tokens=max_new_tokens)
        total_tokens += len(out)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    r = {
        "budget": budget,
        "tree_size": tree_size,
        "tpot_ms": wall / total_tokens * 1e3,
        "accept_len": eng.stats.mean_accept_len,
        "hit_rate": float("nan"),
    }
    if pf is not None and pf.routed_total:
        r["hit_rate"] = 1.0 - pf.repair_misses / pf.routed_total
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="checkpoints/mixtral-8x7b-instruct")
    ap.add_argument("--eagle-path", default="checkpoints/eagle-mixtral/model.safetensors")
    ap.add_argument("--budgets", type=int, nargs="+", default=[3, 4, 5, 6, 8])
    ap.add_argument("--tree-sizes", type=int, nargs="+", default=[64])
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--layout", choices=["resident", "offload"], default="resident",
                    help="offload: all expert weights pinned in host RAM, streamed "
                         "by op2 LayerPrefetcher with exact bitmap repair")
    ap.add_argument("--prefetch-depth", type=int, default=2)
    ap.add_argument("--no-auto-bitmap", action="store_true",
                    help="offload only: disable the temporal predictor "
                         "(every pass copies all experts; isolates repair overhead)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from treemoe.engine.loop import SpecDecodeEngine
    from treemoe.kernels.op1_tree_moe import route_experts, tree_moe_forward
    from treemoe.kernels.op2_prefetch import LayerPrefetcher
    from treemoe.model.config import MixtralConfig
    from treemoe.model.eagle import EagleDraftModel, load_eagle_weights
    from treemoe.model.kv_cache import PagedKVCache
    from treemoe.model.mixtral import MixtralForward
    from treemoe.model.weights import load_mixtral_weights

    cfg = MixtralConfig()
    if args.layout == "offload":
        weights = load_mixtral_weights(
            args.model_dir, cfg, layout="offload",
            offload_layers=set(range(cfg.num_layers)),
        )
    else:
        weights = load_mixtral_weights(args.model_dir, cfg)
    eagle_w = load_eagle_weights(args.eagle_path)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    prompts = [
        tok(f"Question {i}: explain topic {i} in detail.", return_tensors="pt")
        .input_ids[0].cuda()
        for i in range(args.num_prompts)
    ]

    def factory(budget: int, tree_size: int):
        kv = PagedKVCache(cfg, num_blocks=256)
        pf = None
        if args.layout == "offload":
            pf = LayerPrefetcher(weights.layers, depth=args.prefetch_depth,
                                 auto_bitmap=not args.no_auto_bitmap)
            pf.routed_total = 0  # bench counter alongside pf.repair_misses

        def moe_fn(x, lw, layer_idx, _b=budget):
            accept = moe_fn.current_accept_prob  # set by engine step; fallback ones
            routing = route_experts(x, lw.router, accept[: x.shape[0]], _b,
                                    inter=lw.w1.shape[1])
            if pf is not None:
                # exact-offload contract: one small D2H, then on-demand copies
                # for mispredicted experts BEFORE the GEMMs read lw.w1/w2/w3
                # (lw.* aliases the prefetcher's staged ring buffer here)
                ids = routing.expert_ids()
                pf.routed_total += len(ids)
                pf.repair(layer_idx, ids)
            return tree_moe_forward(x, lw.w1, lw.w2, lw.w3, lw.router,
                                    accept[: x.shape[0]], _b, routing=routing)

        moe_fn.current_accept_prob = torch.ones(tree_size, device="cuda")
        target = MixtralForward(weights, kv, moe_fn=moe_fn, prefetcher=pf)
        draft = EagleDraftModel(eagle_w, cfg, weights.embed_tokens, weights.lm_head)
        return SpecDecodeEngine(target, draft, tree_size=tree_size, expert_budget=budget), pf

    print(f"{'B':>3} {'N':>5} {'TPOT(ms)':>10} {'accept_len':>11} {'hit_rate':>9}")
    for n in args.tree_sizes:
        for b in args.budgets:
            r = run_config(factory, b, n, prompts)
            print(f"{r['budget']:>3} {r['tree_size']:>5} {r['tpot_ms']:>10.2f} "
                  f"{r['accept_len']:>11.2f} {r['hit_rate']:>9.3f}")


if __name__ == "__main__":
    main()
