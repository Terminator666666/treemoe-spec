"""End-to-end TPOT / accept-length benchmark with budget & tree-size sweeps
(paper main figure: tau-TPOT Pareto over B, plan Task 2.5 / Phase 5)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# AutoDL images ship OMP_NUM_THREADS=0 (invalid): libgomp warns and the CPU
# cold-expert arm gets a nondeterministic thread count. Sanitize before torch.
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() \
        or int(os.environ["OMP_NUM_THREADS"]) <= 0:
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F


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
        "cold": getattr(pf, "cold_total", 0) if pf is not None else 0,
    }
    if pf is not None and pf.routed_total:
        r["hit_rate"] = 1.0 - pf.repair_misses / pf.routed_total
    # the engine/prefetcher/kv sit in closure reference cycles (moe_fn attrs):
    # without an explicit collect the previous config's staging ring survives
    # into the next one and B-sweeps OOM on 24GB cards
    del eng, pf
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None,
                    help="Mixtral checkpoint dir (default: auto-detect "
                         "/root/autodl-tmp/Mixtral-8x7B-Instruct-v0.1, then "
                         "checkpoints/mixtral-8x7b-instruct)")
    ap.add_argument("--eagle-path", default=None,
                    help="EAGLE draft weights, .safetensors or .bin "
                         "(default: auto-detect)")
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
    ap.add_argument("--cpu-expert-threshold", type=int, default=0,
                    help="hybrid dispatch (Fiddler): missing experts routed "
                         "fewer than N tokens are computed on host CPU from the "
                         "pinned copy instead of repair-copied over PCIe (0 = "
                         "off; ~8 = bench_cpu_expert break-even on the 4090). "
                         "Exact arithmetic but different rounding vs all-GPU.")
    ap.add_argument("--no-router-hint", action="store_true",
                    help="offload ablation: keep the temporal bitmap but disable "
                         "the draft-guided router hint")
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
    from treemoe.model.weights import (default_eagle_path, default_model_dir,
                                       load_mixtral_weights,
                                       random_mixtral_weights)

    if args.model_dir is None:
        args.model_dir = default_model_dir()
    if args.eagle_path is None:
        args.eagle_path = default_eagle_path()

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

        print(f"loading mixtral from {args.model_dir} (layout={args.layout}, "
              f"~93GB stream{', host pinning' if args.layout == 'offload' else ''}"
              "; expect tens of minutes)", flush=True)
        if args.layout == "offload":
            weights = load_mixtral_weights(
                args.model_dir, cfg, layout="offload", offload_layers=offload,
            )
        else:
            weights = load_mixtral_weights(args.model_dir, cfg)
        print(f"loading eagle from {args.eagle_path}", flush=True)
        eagle_w = load_eagle_weights(args.eagle_path)
        tok = AutoTokenizer.from_pretrained(args.model_dir)
        print("weights ready, starting benchmark", flush=True)
        # EAGLE was trained on ShareGPT conversations and the official tau is
        # measured on chat-templated MT-bench prompts; raw untemplated text
        # systematically deflates acceptance. MT-bench-style instructions:
        instructions = [
            "Compose an engaging travel blog post about a recent trip to Hawaii.",
            "Explain the difference between TCP and UDP to a beginner.",
            "Write a short story about a robot learning to paint.",
            "What are the main causes of the French Revolution?",
            "Describe the process of photosynthesis step by step.",
            "Draft an email to a professor asking for a recommendation letter.",
            "Explain how a hash table works and when to use one.",
            "Summarize the plot of Romeo and Juliet in three paragraphs.",
            "Give practical tips for improving sleep quality.",
            "Explain quantum entanglement in simple terms.",
            "Write a product description for a smart water bottle.",
            "Compare renewable and fossil fuel energy sources.",
            "Describe how vaccines train the immune system.",
            "Outline a beginner workout plan for building strength.",
            "Explain the significance of the Turing test.",
            "Write a recipe for a vegetarian pasta dinner.",
            "Describe the water cycle and its main stages.",
            "Explain why the sky is blue using physics.",
            "Draft a cover letter for a software engineering internship.",
            "Discuss the pros and cons of remote work.",
        ]
        prompts = []
        for i in range(args.num_prompts):
            text = tok.apply_chat_template(
                [{"role": "user", "content": instructions[i % len(instructions)]}],
                tokenize=False, add_generation_prompt=True,
            )
            # template already contains <s>; add_special_tokens would double it
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
            prompts.append(ids[0].cuda())

    def factory(budget: int, tree_size: int):
        kv = PagedKVCache(cfg, num_blocks=256)
        pf = None
        if args.layout == "offload":
            pf = LayerPrefetcher(weights.layers, depth=args.prefetch_depth,
                                 auto_bitmap=not args.no_auto_bitmap)
            pf.use_router_hint = not args.no_router_hint
            pf.routed_total = 0  # bench counter alongside pf.repair_misses
            pf.cold_total = 0    # experts computed on host CPU (hybrid arm)
        host_layers = weights.layers  # pinned host copy for the CPU cold path

        def moe_fn(x, lw, layer_idx, _b=budget):
            accept = moe_fn.current_accept_prob  # set by engine step; fallback ones
            routing = route_experts(x, lw.router, accept[: x.shape[0]], _b,
                                    inter=lw.w1.shape[1])
            cold_x = []
            if pf is not None:
                # exact-offload contract: one small D2H, then on-demand copies
                # for mispredicted experts BEFORE the GEMMs read lw.w1/w2/w3
                # (lw.* aliases the prefetcher's staged ring buffer here)
                ids = routing.expert_ids()
                pf.routed_total += len(ids)
                staged = pf._staged_rows.get(layer_idx)
                if args.cpu_expert_threshold > 0 and staged is not None:
                    counts = (routing.padded_slots.view(
                        routing.ws.num_experts, -1) >= 0).sum(-1).tolist()
                    cold = [e for e in ids if e not in staged
                            and counts[e] < args.cpu_expert_threshold]
                    if cold:
                        # cold misses -> host CPU FFN (Fiddler-style) instead of
                        # streaming 352MB/expert. Gather inputs now so the host
                        # compute overlaps the GPU GEMMs below; kept out of
                        # repair() so the temporal bitmap won't stage them either.
                        pf.cold_total += len(cold)
                        ids = [e for e in ids if e not in cold]
                        for e, toks, gates in routing.exclude_experts(cold):
                            dev_toks = toks.to(x.device)
                            cold_x.append((e, dev_toks, gates, x[dev_toks].cpu()))
                pf.repair(layer_idx, ids)
            out = tree_moe_forward(x, lw.w1, lw.w2, lw.w3, lw.router,
                                   accept[: x.shape[0]], _b, routing=routing)
            for e, dev_toks, gates, xc in cold_x:
                hw = host_layers[layer_idx]
                y = (F.silu(xc @ hw.w1[e].t()) * (xc @ hw.w3[e].t())) @ hw.w2[e].t()
                out.index_add_(0, dev_toks, (gates.unsqueeze(1) * y.float())
                               .to(out.dtype).to(x.device))
            return out

        moe_fn.current_accept_prob = torch.ones(tree_size, device="cuda")
        target = MixtralForward(weights, kv, moe_fn=moe_fn, prefetcher=pf)
        # real EAGLE checkpoint facts (official config.json): rms_norm_eps=1e-6,
        # no rope_theta field -> Llama-default 10000 (target uses 1e-5 / 1e6)
        draft_kw = {} if args.random_weights else dict(rms_eps=1e-6, rope_theta=1e4)
        draft = EagleDraftModel(eagle_w, cfg, weights.embed_tokens, weights.lm_head,
                                **draft_kw)
        return SpecDecodeEngine(target, draft, tree_size=tree_size, expert_budget=budget), pf

    print(f"{'B':>3} {'N':>5} {'TPOT(ms)':>10} {'accept_len':>11} {'hit_rate':>9} {'cold':>6}",
          flush=True)
    for n in args.tree_sizes:
        for b in args.budgets:
            r = run_config(factory, b, n, prompts,
                           max_new_tokens=args.max_new_tokens)
            print(f"{r['budget']:>3} {r['tree_size']:>5} {r['tpot_ms']:>10.2f} "
                  f"{r['accept_len']:>11.2f} {r['hit_rate']:>9.3f} {r['cold']:>6}", flush=True)


if __name__ == "__main__":
    main()
