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
    for i, p in enumerate(prompts):
        out = eng.generate(p, max_new_tokens=max_new_tokens)
        total_tokens += len(out)
        el = time.perf_counter() - t0
        print(f"  B={budget} N={tree_size} prompt {i + 1}/{len(prompts)} "
              f"({el:.0f}s elapsed, {el / total_tokens * 1e3:.0f}ms/tok)",
              file=sys.stderr, flush=True)
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
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--layout", choices=["resident", "offload"], default="resident",
                    help="offload: all expert weights pinned in host RAM, streamed "
                         "by op2 LayerPrefetcher with exact bitmap repair")
    ap.add_argument("--prefetch-depth", type=int, default=2)
    ap.add_argument("--no-auto-bitmap", action="store_true",
                    help="offload only: disable the temporal predictor "
                         "(every pass copies all experts; isolates repair overhead)")
    ap.add_argument("--random-weights", action="store_true",
                    help="no checkpoint needed: random weights at real Mixtral "
                         "shapes. TPOT/hit_rate/streaming numbers are valid "
                         "(memory traffic ignores values); accept_len is NOT.")
    args = ap.parse_args()

    from treemoe.engine.loop import SpecDecodeEngine
    from treemoe.kernels.op1_tree_moe import route_experts, tree_moe_forward
    from treemoe.kernels.op2_prefetch import LayerPrefetcher
    from treemoe.model.config import MixtralConfig
    from treemoe.model.eagle import EagleDraftModel, EagleWeights, load_eagle_weights
    from treemoe.model.kv_cache import PagedKVCache
    from treemoe.model.mixtral import MixtralForward
    from treemoe.model.weights import load_mixtral_weights, random_mixtral_weights

    cfg = MixtralConfig()
    offload = set(range(cfg.num_layers)) if args.layout == "offload" else None
    if args.random_weights:
        weights = random_mixtral_weights(cfg, layout=args.layout,
                                         offload_layers=offload)
        g = torch.Generator().manual_seed(1)
        hd = cfg.hidden_dim

        def rw(*s):
            return (torch.randn(*s, generator=g) * 0.02).to(cfg.dtype).cuda()

        eagle_w = EagleWeights(
            fc=rw(hd, 2 * hd),
            attn={"q_proj": rw(cfg.num_heads * cfg.head_dim, hd),
                  "k_proj": rw(cfg.num_kv_heads * cfg.head_dim, hd),
                  "v_proj": rw(cfg.num_kv_heads * cfg.head_dim, hd),
                  "o_proj": rw(hd, cfg.num_heads * cfg.head_dim)},
            input_layernorm=torch.ones(hd, dtype=cfg.dtype, device="cuda"),
            post_attn_layernorm=torch.ones(hd, dtype=cfg.dtype, device="cuda"),
            mlp_gate=rw(cfg.intermediate_dim, hd),
            mlp_up=rw(cfg.intermediate_dim, hd),
            mlp_down=rw(hd, cfg.intermediate_dim),
        )
        pg = torch.Generator().manual_seed(2)
        prompts = [torch.randint(0, cfg.vocab_size, (32,), generator=pg).cuda()
                   for _ in range(args.num_prompts)]
    else:
        from transformers import AutoTokenizer

        if args.layout == "offload":
            weights = load_mixtral_weights(
                args.model_dir, cfg, layout="offload", offload_layers=offload,
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

    print(f"{'B':>3} {'N':>5} {'TPOT(ms)':>10} {'accept_len':>11} {'hit_rate':>9}",
          flush=True)
    for n in args.tree_sizes:
        for b in args.budgets:
            r = run_config(factory, b, n, prompts,
                           max_new_tokens=args.max_new_tokens)
            print(f"{r['budget']:>3} {r['tree_size']:>5} {r['tpot_ms']:>10.2f} "
                  f"{r['accept_len']:>11.2f} {r['hit_rate']:>9.3f}", flush=True)


if __name__ == "__main__":
    main()
