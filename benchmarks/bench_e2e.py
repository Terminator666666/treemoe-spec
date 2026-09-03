"""End-to-end TPOT / accept-length benchmark with budget & tree-size sweeps
(paper main figure: tau-TPOT Pareto over B, plan Task 2.5 / Phase 5)."""

from __future__ import annotations

import argparse
import json
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


@torch.inference_mode()
def ar_generate(target, prompt: torch.Tensor, max_new_tokens: int,
                eos_token_id: int = 2) -> list[int]:
    target.kv.reset()
    positions = torch.arange(prompt.shape[0], device=prompt.device)
    logits = target.forward(prompt, positions)
    last = logits[-1].argmax().view(1)
    output = [int(last)]
    while len(output) < max_new_tokens and output[-1] != eos_token_id:
        position = torch.tensor([target.kv.seq_len], device=prompt.device)
        logits = target.forward(last, position)
        last = logits[-1].argmax().view(1)
        output.append(int(last))
    return output


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
    target_steps = eng.stats.steps
    staged_rows = getattr(pf, "staged_rows_total", 0) if pf is not None else 0
    repair_rows = getattr(pf, "repair_rows_total", 0) if pf is not None else 0
    budget_histogram = (
        eng.layer_budget_allocator.budget_histogram.tolist()
        if eng.layer_budget_allocator is not None else None
    )
    planned_rows = (
        sum(budget * count for budget, count in enumerate(budget_histogram))
        if budget_histogram is not None else 0
    )
    row_gib = getattr(pf, "expert_row_bytes", 0) / 2**30 if pf is not None else 0.0
    r = {
        "budget": budget,
        "tree_size": tree_size,
        "tpot_ms": wall / total_tokens * 1e3,
        "accept_len": eng.stats.mean_accept_len,
        "target_steps": target_steps,
        "hit_rate": float("nan"),
        "cold": getattr(pf, "cold_total", 0) if pf is not None else 0,
        "staged_rows": staged_rows,
        "repair_rows": repair_rows,
        "staged_gib": staged_rows * row_gib,
        "repair_gib": repair_rows * row_gib,
        "planned_rows_per_step": planned_rows / max(target_steps, 1),
        "repair_rows_per_step": repair_rows / max(target_steps, 1),
        "h2d_gib_per_token": (staged_rows + repair_rows) * row_gib / total_tokens,
        "layer_budgets": (
            eng.layer_budget_allocator.plan.budgets.tolist()
            if eng.layer_budget_allocator is not None else None
        ),
        "budget_histogram": budget_histogram,
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


def run_lossless_check(engine_factory, tree_size: int,
                       prompts: list[torch.Tensor], max_new_tokens: int,
                       output_path: Path) -> None:
    ar_engine, ar_prefetcher = engine_factory(budget=8, tree_size=tree_size)
    ar_outputs = []
    for index, prompt in enumerate(prompts):
        ar_outputs.append(ar_generate(ar_engine.target, prompt, max_new_tokens))
        print(f"  lossless AR prompt {index + 1}/{len(prompts)} ready",
              file=sys.stderr, flush=True)
    del ar_engine, ar_prefetcher
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    spec_engine, spec_prefetcher = engine_factory(budget=8, tree_size=tree_size)
    rows = []
    all_match = True
    for index, (prompt, ar_output) in enumerate(zip(prompts, ar_outputs, strict=True)):
        steps_before = spec_engine.stats.steps
        tokens_before = spec_engine.stats.tokens
        spec_output = spec_engine.generate(prompt, max_new_tokens=max_new_tokens)
        steps = spec_engine.stats.steps - steps_before
        accepted_tokens = spec_engine.stats.tokens - tokens_before
        first_mismatch = next(
            (offset for offset, pair in enumerate(zip(ar_output, spec_output, strict=False))
             if pair[0] != pair[1]),
            None,
        )
        if first_mismatch is None and len(ar_output) != len(spec_output):
            first_mismatch = min(len(ar_output), len(spec_output))
        matched = first_mismatch is None
        all_match &= matched
        accept_len = accepted_tokens / max(steps, 1)
        rows.append({
            "prompt_index": index,
            "input_ids": prompt.cpu().tolist(),
            "ar_tokens": ar_output,
            "spec_tokens": spec_output,
            "match": matched,
            "first_mismatch": first_mismatch,
            "target_steps": steps,
            "mean_accept_len": accept_len,
        })
        verdict = "PASS" if matched else f"FAIL@{first_mismatch}"
        print(f"  lossless prompt {index + 1}/{len(prompts)}: {verdict} "
              f"tokens={len(spec_output)} accept_len={accept_len:.2f}", flush=True)

    artifact = {
        "mode": "lossless",
        "expert_budget": 8,
        "top1_threshold": 0.0,
        "tree_size": tree_size,
        "max_new_tokens": max_new_tokens,
        "all_match": all_match,
        "prompts": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"lossless artifact: {output_path}", flush=True)
    del spec_engine, spec_prefetcher
    gc.collect()
    torch.cuda.empty_cache()
    if not all_match:
        raise RuntimeError("B=8 speculative output diverged from AR; see artifact")


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
    budget_policy = ap.add_mutually_exclusive_group()
    budget_policy.add_argument("--adaptive-layer-budget", action="store_true",
                               help="reallocate per-layer budgets under a fixed "
                                    "global expert-row budget")
    budget_policy.add_argument("--uniform-layer-budget", action="store_true",
                               help="strict control: fixed B per layer with the "
                                    "same demand bitmap and repair machinery")
    ap.add_argument("--layer-budget-min", type=int, default=2,
                    help="minimum experts retained per layer in adaptive mode")
    ap.add_argument("--layer-budget-max", type=int, default=None,
                    help="maximum experts retained per layer in adaptive mode "
                         "(default: all experts)")
    ap.add_argument("--budget-ema-decay", type=float, default=0.8,
                    help="EMA decay for adaptive layer demand")
    ap.add_argument("--random-weights", action="store_true",
                    help="no checkpoint needed: random weights at real Mixtral "
                         "shapes. TPOT/hit_rate/streaming numbers are valid "
                         "(memory traffic ignores values); accept_len is NOT.")
    ap.add_argument("--top1-threshold", type=float, default=0.0,
                    help="op3: tree nodes with global acceptance prob below "
                        "this degrade to top-1 routing (spec §3.3 step 4). "
                        "Disabled by default because it is approximate. In "
                        "the offload regime this saves GEMM flops only "
                        "(staged bytes unchanged) but perturbs verification "
                        "logits for most deep nodes.")
    ap.add_argument("--ar-baseline", action="store_true",
                    help="run plain greedy AR through the same offload plumbing "
                         "instead of the sweep (speedup denominator). Uses "
                         "budget=8: AR must be exact, no expert truncation.")
    ap.add_argument("--check-lossless", action="store_true",
                    help="compare B=8 speculative tokens against AR on every "
                         "prompt and save a detailed JSON artifact")
    ap.add_argument("--output-json", type=Path,
                    default=Path("artifacts/e2e_lossless.json"))
    args = ap.parse_args()
    if args.check_lossless and args.top1_threshold != 0:
        ap.error("--check-lossless requires --top1-threshold 0")

    from treemoe.engine.loop import SpecDecodeEngine
    from treemoe.engine.layer_budget import LayerBudgetAllocator
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
        # real MT-bench questions (first turns), same eval set as the official
        # EAGLE numbers — vendored from SafeAILab/EAGLE eagle/data/mt_bench
        mt_path = Path(__file__).resolve().parent / "data" / "mt_bench.jsonl"
        with open(mt_path) as f:
            instructions = [json.loads(line)["turns"][0] for line in f]
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
            # engine sets this per forward (prefill=ones, tree step=accept_prob);
            # fallback for any other caller: real tokens, acceptance prob 1
            accept = moe_fn.current_accept_prob
            if accept.shape[0] != x.shape[0]:
                accept = torch.ones(x.shape[0], device=x.device)
            layer_budgets = getattr(moe_fn, "current_layer_budgets", None)
            layer_budget = (
                int(layer_budgets[layer_idx]) if layer_budgets is not None else _b
            )
            routing = route_experts(x, lw.router, accept, layer_budget,
                                    inter=lw.w1.shape[1],
                                    top1_threshold=args.top1_threshold)
            observer = getattr(moe_fn, "demand_observer", None)
            if observer is not None:
                observer(layer_idx, routing.demand)
            cold_x = []
            if pf is not None:
                # exact-offload contract: one small D2H, then on-demand copies
                # for mispredicted experts BEFORE the GEMMs read lw.w1/w2/w3
                # (lw.* aliases the prefetcher's staged ring buffer here)
                ids = routing.expert_ids()
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
                pf.routed_total += len(ids)
                pf.repair(layer_idx, ids)
            out = tree_moe_forward(x, lw.w1, lw.w2, lw.w3, lw.router,
                                   accept, layer_budget, routing=routing)
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
        allocator = None
        if args.adaptive_layer_budget or args.uniform_layer_budget:
            allocator = LayerBudgetAllocator(
                cfg.num_layers, cfg.num_experts, average_budget=budget,
                min_budget=args.layer_budget_min,
                max_budget=args.layer_budget_max,
                ema_decay=args.budget_ema_decay,
                adaptive=args.adaptive_layer_budget,
            )
        return SpecDecodeEngine(
            target, draft, tree_size=tree_size, expert_budget=budget,
            layer_budget_allocator=allocator,
        ), pf

        print(f"{'B':>3} {'N':>5} {'TPOT(ms)':>10} {'accept_len':>11} {'steps':>6} "
            f"{'expert_hit':>10} {'budget_hist':>18} {'planR/s':>8} "
            f"{'repairR/s':>9} {'H2DGiB/tok':>10} {'cold':>6}",
          flush=True)
    if args.check_lossless:
        run_lossless_check(
            factory, args.tree_sizes[0], prompts, args.max_new_tokens,
            args.output_json,
        )
        return
    if args.ar_baseline:
        # same kv/prefetcher/moe_fn plumbing as the spec arms, no draft/tree:
        # prefill, then one token per forward. budget=8 => exact AR outputs.
        eng, pf = factory(budget=8, tree_size=16)
        target = eng.target
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total = 0
        for i, p in enumerate(prompts):
            output = ar_generate(target, p, args.max_new_tokens)
            total += len(output)
            el = time.perf_counter() - t0
            print(f"  AR prompt {i + 1}/{len(prompts)} ({el:.0f}s elapsed, "
                  f"{el / total * 1e3:.0f}ms/tok)", file=sys.stderr, flush=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        hit = float("nan")
        if pf is not None and pf.routed_total:
            hit = 1.0 - pf.repair_misses / pf.routed_total
        print(f"{'AR':>3} {'-':>5} {wall / total * 1e3:>10.2f} {'-':>11} "
              f"{hit:>9.3f} {'-':>6}", flush=True)
        return

    for n in args.tree_sizes:
        for b in args.budgets:
            r = run_config(factory, b, n, prompts,
                           max_new_tokens=args.max_new_tokens)
            histogram = r["budget_histogram"]
            if histogram is None:
                plan_hist = "-"
            else:
                plan_hist = "/".join(
                    f"{budget}x{count}" for budget, count in enumerate(histogram)
                    if count
                )
            print(f"{r['budget']:>3} {r['tree_size']:>5} {r['tpot_ms']:>10.2f} "
                f"{r['accept_len']:>11.2f} {r['target_steps']:>6} "
                f"{r['hit_rate']:>10.3f} {plan_hist:>18} "
                f"{r['planned_rows_per_step']:>8.1f} "
                f"{r['repair_rows_per_step']:>9.1f} "
                f"{r['h2d_gib_per_token']:>10.2f} {r['cold']:>6}", flush=True)


if __name__ == "__main__":
    main()
