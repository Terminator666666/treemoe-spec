"""End-to-end TPOT / accept-length benchmark with budget & tree-size sweeps
(paper main figure: tau-TPOT Pareto over B, plan Task 2.5 / Phase 5)."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
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


def _profiler_error_artifact(
    output_dir: Path, label: str, artifact: str, error: Exception,
) -> str:
    path = output_dir / f"{label}.{artifact}-error.txt"
    message = f"{type(error).__name__}: {error}\n"
    try:
        path.write_text(message)
    except OSError:
        print(f"profiler {artifact} export failed: {message.strip()}",
              file=sys.stderr, flush=True)
    return str(path)


def _export_profiler_artifacts(
    profiler, output_dir: Path, label: str, profile_memory: bool,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    chrome_path = output_dir / f"{label}.chrome.json"
    try:
        profiler.export_chrome_trace(str(chrome_path))
        artifacts["chrome_trace"] = str(chrome_path)
    except Exception as error:
        artifacts["chrome_trace"] = _profiler_error_artifact(
            output_dir, label, "chrome", error,
        )

    table_path = output_dir / f"{label}.operators.txt"
    try:
        table_path.write_text(profiler.key_averages(
            group_by_input_shape=True,
        ).table(sort_by="self_cuda_time_total", row_limit=500))
        artifacts["operator_table"] = str(table_path)
    except Exception as error:
        artifacts["operator_table"] = _profiler_error_artifact(
            output_dir, label, "operators", error,
        )

    if profile_memory:
        memory_path = output_dir / f"{label}.memory.html"
        try:
            profiler.export_memory_timeline(str(memory_path), device="cuda:0")
            artifacts["memory_timeline"] = str(memory_path)
        except Exception as error:
            artifacts["memory_timeline"] = _profiler_error_artifact(
                output_dir, label, "memory", error,
            )
    else:
        artifacts["memory_timeline"] = "disabled; see execution JSON snapshots"
    return artifacts


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
               max_new_tokens: int = 128,
               profiler_dir: Path | None = None,
               profiler_label: str = "run",
               profiler_memory: bool = False,
               profiler_with_stack: bool = False) -> dict:
    eng, pf = engine_factory(budget=budget, tree_size=tree_size)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_tokens = 0
    profiler_context = nullcontext(None)
    if profiler_dir is not None:
        profiler_context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=profiler_memory,
            with_stack=profiler_with_stack,
        )
    with profiler_context as profiler:
        for i, p in enumerate(prompts):
            out = eng.generate(p, max_new_tokens=max_new_tokens)
            total_tokens += len(out)
            el = time.perf_counter() - t0
            print(f"  B={budget} N={tree_size} prompt {i + 1}/{len(prompts)} "
                  f"({el:.0f}s elapsed, {el / total_tokens * 1e3:.0f}ms/tok)",
                  file=sys.stderr, flush=True)
            torch.cuda.synchronize()
            inference_wall = time.perf_counter() - t0
    profiler_artifacts = None
    if profiler is not None:
        profiler_artifacts = _export_profiler_artifacts(
            profiler, profiler_dir, profiler_label, profiler_memory,
        )
    target_steps = eng.stats.steps
    staged_rows = getattr(pf, "staged_rows_total", 0) if pf is not None else 0
    repair_rows = getattr(pf, "repair_rows_total", 0) if pf is not None else 0
    jit_verify_rows = (
        getattr(pf, "jit_verify_rows_total", 0) if pf is not None else 0
    )
    budget_histogram = (
        eng.layer_budget_allocator.budget_histogram.tolist()
        if eng.layer_budget_allocator is not None else None
    )
    allocator = eng.layer_budget_allocator
    planned_rows = (
        sum(budget * count for budget, count in enumerate(budget_histogram))
        if budget_histogram is not None
        else budget * eng.target.cfg.num_layers * target_steps
    )
    row_gib = getattr(pf, "expert_row_bytes", 0) / 2**30 if pf is not None else 0.0
    r = {
        "budget": budget,
        "tree_size": tree_size,
        "generated_tokens": total_tokens,
        "tpot_ms": inference_wall / total_tokens * 1e3,
        "accept_len": eng.stats.mean_accept_len,
        "target_steps": target_steps,
        "hit_rate": float("nan"),
        "cold": getattr(pf, "cold_total", 0) if pf is not None else 0,
        "staging_mode": (
            "jit_exact" if pf is not None and pf.jit_staging
            else "predictive" if pf is not None else "resident"
        ),
        "staged_rows": staged_rows,
        "jit_rows": getattr(pf, "jit_rows_total", 0) if pf is not None else 0,
        "jit_verify_rows": jit_verify_rows,
        "repair_rows": repair_rows,
        "staged_gib": staged_rows * row_gib,
        "repair_gib": repair_rows * row_gib,
        "planned_rows_per_step": planned_rows / max(target_steps, 1),
        "jit_rows_per_step": jit_verify_rows / max(target_steps, 1),
        "stage_rows_per_step": (
            jit_verify_rows if pf is not None and pf.jit_staging else planned_rows
        ) / max(target_steps, 1),
        "repair_rows_per_step": repair_rows / max(target_steps, 1),
        "h2d_gib_per_token": (staged_rows + repair_rows) * row_gib / total_tokens,
        "layer_budgets": (
            eng.layer_budget_allocator.plan.budgets.tolist()
            if eng.layer_budget_allocator is not None else None
        ),
        "budget_histogram": budget_histogram,
        "budget_trace": (
            allocator.budget_trace if allocator is not None else None
        ),
        "demand_trace": (
            allocator.demand_trace if allocator is not None else None
        ),
        "expert_trace": getattr(eng.target.moe_fn, "expert_trace", None),
        "execution_trace": (
            eng.performance_tracer.to_dict()
            if eng.performance_tracer is not None else None
        ),
        "profiler_artifacts": profiler_artifacts,
    }
    if pf is not None and pf.routed_total and not pf.jit_staging:
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
    ap.add_argument("--warmup-new-tokens", type=int, default=0,
                    help="run one unmeasured prompt first to compile kernels "
                         "and initialize CUDA libraries")
    ap.add_argument("--layout", choices=["resident", "offload"], default="resident",
                    help="offload: all expert weights pinned in host RAM, streamed "
                         "by op2 LayerPrefetcher with exact bitmap repair")
    ap.add_argument("--prefetch-depth", type=int, default=2)
    ap.add_argument("--predictive-prefetch", action="store_true",
                    help="offload ablation: restore previous-pass/router-hint "
                         "prefetch plus synchronous repair; default stages "
                         "the exact routed experts just in time")
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
    ap.add_argument("--layer-budget-objective", choices=["log_mass", "mass"],
                    default="log_mass",
                    help="global allocation utility; log_mass protects "
                         "multiplicative fidelity, mass is the additive ablation")
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
    ap.add_argument("--routing-objective", choices=["mass", "critical_path"],
                    default="mass",
                    help="select the layer expert set by aggregate mass or "
                         "protect experts needed by high-acceptance nodes")
    ap.add_argument("--atomic-moe", action="store_true",
                    help="performance ablation: use nondeterministic atomic "
                         "GEMM2 accumulation instead of fixed-order combine")
    ap.add_argument("--compare-routing-objectives", action="store_true",
                    help="run mass then critical_path after one weight load")
    ap.add_argument("--ar-baseline", action="store_true",
                    help="run plain greedy AR through the same offload plumbing "
                         "instead of the sweep (speedup denominator). Uses "
                         "budget=8: AR must be exact, no expert truncation.")
    ap.add_argument("--check-lossless", action="store_true",
                    help="compare B=8 speculative tokens against AR on every "
                         "prompt and save a detailed JSON artifact")
    ap.add_argument("--output-json", type=Path,
                    default=Path("artifacts/e2e_lossless.json"))
    ap.add_argument("--budget-trace-json", type=Path, default=None,
                    help="write per-verification per-layer budgets for "
                         "adaptive allocation diagnosis")
    ap.add_argument("--execution-trace-json", type=Path, default=None,
                    help="write opt-in full-stage CUDA/host timings, routing, "
                         "prefetch, tree topology, and accepted paths")
    ap.add_argument("--execution-trace-baseline-first", action="store_true",
                    help="run the same configurations without tracing first and "
                         "store their TPOT as the performance baseline")
    ap.add_argument("--execution-trace-detail",
                    choices=["full", "progressive"], default="full",
                    help="progressive stores only tree, accepted path, natural "
                         "top-2 routing, row bytes, and timings")
    ap.add_argument("--torch-profiler-dir", type=Path, default=None,
                    help="write full CPU/CUDA kernel Chrome trace, operator "
                         "table, and optional GPU memory timeline")
    ap.add_argument("--torch-profiler-memory", action="store_true",
                    help="capture per-allocation memory events and timeline; "
                         "disabled by default due Kineto compatibility bugs")
    ap.add_argument("--torch-profiler-with-stack", action="store_true",
                    help="capture Python stacks; substantially increases trace "
                         "size and may trigger old Kineto decoding bugs")
    args = ap.parse_args()
    if args.check_lossless and args.top1_threshold != 0:
        ap.error("--check-lossless requires --top1-threshold 0")
    if args.check_lossless and args.atomic_moe:
        ap.error("--check-lossless requires deterministic MoE reduction")
    if args.execution_trace_baseline_first \
            and args.execution_trace_json is None:
        ap.error(
            "--execution-trace-baseline-first requires --execution-trace-json"
        )

    from treemoe.engine.loop import SpecDecodeEngine
    from treemoe.engine.layer_budget import LayerBudgetAllocator
    from treemoe.engine.perf_trace import ExecutionTracer
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

    def factory(budget: int, tree_size: int, routing_objective: str | None = None,
                enable_trace: bool | None = None):
        objective = routing_objective or args.routing_objective
        if enable_trace is None:
            enable_trace = args.execution_trace_json is not None
        tracer = (
            ExecutionTracer(detail=args.execution_trace_detail)
            if enable_trace else None
        )
        kv = PagedKVCache(cfg, num_blocks=256)
        pf = None
        if args.layout == "offload":
            pf = LayerPrefetcher(weights.layers, depth=args.prefetch_depth,
                                 auto_bitmap=not args.no_auto_bitmap,
                                 jit_staging=not args.predictive_prefetch)
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
            layer_record = tracer.current_layer if tracer is not None else None
            phase = tracer.phase(layer_record, "route") \
                if tracer else nullcontext()
            with phase:
                routing = route_experts(
                    x, lw.router, accept, layer_budget,
                    inter=lw.w1.shape[1],
                    top1_threshold=args.top1_threshold,
                    routing_objective=objective,
                )
            observer = getattr(moe_fn, "demand_observer", None)
            if observer is not None:
                observer(layer_idx, routing.demand)
            cold_x = []
            ids = None
            if pf is not None or tracer is not None:
                phase = tracer.phase(layer_record, "expert_ids_d2h", cuda=False) \
                    if tracer else nullcontext()
                with phase:
                    ids = routing.expert_ids()
            routed_ids = (ids or []).copy()
            staged_snapshot = None
            missing_before_repair = []
            staged_nonrouted = None
            jit_stage_rows = 0
            if pf is not None:
                # One small D2H exposes the actual routed set. Default JIT mode
                # copies exactly that set; the predictive ablation repairs any
                # misses. In both modes lw.* aliases the staging ring buffer.
                if args.budget_trace_json is not None \
                        and layer_budget < cfg.num_experts:
                    if layer_idx == 0:
                        moe_fn.expert_trace.append(
                            [None for _ in range(cfg.num_layers)]
                        )
                    moe_fn.expert_trace[-1][layer_idx] = ids.copy()
                staged = pf._staged_rows.get(layer_idx)
                staged_snapshot = None if staged is None else set(staged)
                if args.cpu_expert_threshold > 0 and staged is not None:
                    counts = torch.bincount(
                        routing.topk_ids.reshape(-1),
                        minlength=routing.ws.num_experts,
                    ).tolist()
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
                if staged_snapshot is not None and not pf.jit_staging:
                    missing_before_repair = sorted(set(ids) - staged_snapshot)
                    staged_nonrouted = sorted(staged_snapshot - set(ids))
                pf.routed_total += len(ids)
                phase_name = "jit_stage" if pf.jit_staging else "repair"
                phase = tracer.phase(layer_record, phase_name) \
                    if tracer else nullcontext()
                with phase:
                    jit_stage_rows, repair_rows = pf.prepare_experts(layer_idx, ids)
            else:
                repair_rows = 0
            if tracer is not None:
                phase = tracer.phase(layer_record, "routing_snapshot")
                if tracer.detail == "progressive":
                    with phase:
                        original_ids = routing.router_gates.topk(
                            2, dim=-1,
                        ).indices.detach().cpu().tolist()
                    layer_record.update({
                        "budget": layer_budget,
                        "tree_nodes": x.shape[0],
                        "staging_mode": (
                            "jit_exact" if pf is not None and pf.jit_staging
                            else "predictive" if pf is not None else "resident"
                        ),
                        "jit_stage_rows": jit_stage_rows,
                        "repair_rows": repair_rows,
                        "nodes": [
                            {"node": node, "original_top2_experts": experts}
                            for node, experts in enumerate(original_ids)
                        ],
                        "expert_row_bytes": (
                            pf.expert_row_bytes if pf is not None else 0
                        ),
                    })
                else:
                    with phase:
                        slot_counts = torch.bincount(
                            routing.topk_ids.reshape(-1),
                            minlength=routing.ws.num_experts,
                        ).tolist()
                        demand = routing.demand.detach().float().cpu().tolist()
                        original_prob, original_ids = routing.router_gates.topk(
                            2, dim=-1,
                        )
                        routing_detail = torch.cat([
                            routing.router_gates,
                            original_ids.float(), original_prob,
                            routing.topk_ids.float(), routing.gates_flat.view(-1, 2),
                        ], dim=-1).detach().float().cpu().tolist()
                    staged = (
                        pf._staged_rows.get(layer_idx) if pf is not None else None
                    )
                    layer_record.update({
                    "budget": layer_budget,
                    "tree_nodes": x.shape[0],
                    "routed_experts": routed_ids,
                    "gpu_experts": ids or [],
                    "cold_experts": [item[0] for item in cold_x],
                    "staged_experts_before_repair": (
                        None if staged_snapshot is None else sorted(staged_snapshot)
                    ),
                    "staged_experts_after_repair": (
                        None if staged is None else sorted(staged)
                    ),
                    "staged_nonrouted_experts": staged_nonrouted,
                    "staging_mode": (
                        "jit_exact" if pf is not None and pf.jit_staging
                        else "predictive" if pf is not None else "resident"
                    ),
                    "jit_stage_rows": jit_stage_rows,
                    "jit_staged_experts": (
                        sorted(ids) if pf is not None and pf.jit_staging else []
                    ),
                    "repair_rows": repair_rows,
                    "missing_experts": missing_before_repair,
                    "slot_counts": slot_counts,
                    "acceptance_weighted_demand": demand,
                    "nodes": [
                        {
                            "node": node,
                            "accept_probability": float(accept[node]),
                            "router_probability": row[:cfg.num_experts],
                            "original_top2_experts": [
                                int(value) for value in
                                row[cfg.num_experts:cfg.num_experts + 2]
                            ],
                            "original_top2_probability": row[
                                cfg.num_experts + 2:cfg.num_experts + 4
                            ],
                            "selected_experts": [
                                int(value) for value in
                                row[cfg.num_experts + 4:cfg.num_experts + 6]
                            ],
                            "selected_gates": row[
                                cfg.num_experts + 6:cfg.num_experts + 8
                            ],
                        }
                        for node, row in enumerate(routing_detail)
                    ],
                    "expert_row_bytes": (
                        pf.expert_row_bytes if pf is not None else 0
                    ),
                    })
            phase = tracer.phase(layer_record, "expert_gemm") \
                if tracer else nullcontext()
            with phase:
                out = tree_moe_forward(
                    x, lw.w1, lw.w2, lw.w3, lw.router,
                    accept, layer_budget, routing=routing,
                    deterministic=not args.atomic_moe,
                )
            phase = tracer.phase(layer_record, "cold_cpu") \
                if tracer and cold_x else nullcontext()
            with phase:
                for e, dev_toks, gates, xc in cold_x:
                    hw = host_layers[layer_idx]
                    y = ((F.silu(xc @ hw.w1[e].t()) * (xc @ hw.w3[e].t()))
                         @ hw.w2[e].t())
                    out.index_add_(
                        0, dev_toks,
                        (gates.unsqueeze(1) * y.float()).to(out.dtype).to(x.device),
                    )
            return out

        moe_fn.current_accept_prob = torch.ones(tree_size, device="cuda")
        moe_fn.expert_trace = []
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
                objective=args.layer_budget_objective,
            )
        return SpecDecodeEngine(
            target, draft, tree_size=tree_size, expert_budget=budget,
            layer_budget_allocator=allocator,
            performance_tracer=tracer,
        ), pf
    if args.check_lossless:
        print(f"routing objective: {args.routing_objective}", flush=True)
        run_lossless_check(
            factory, args.tree_sizes[0], prompts, args.max_new_tokens,
            args.output_json,
        )
        return
    if args.ar_baseline:
        print(f"routing objective: {args.routing_objective}", flush=True)
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
        if pf is not None and pf.routed_total and not pf.jit_staging:
            hit = 1.0 - pf.repair_misses / pf.routed_total
        print(f"{'AR':>3} {'-':>5} {wall / total * 1e3:>10.2f} {'-':>11} "
              f"{hit:>9.3f} {'-':>6}", flush=True)
        return

    trace_rows = []
    execution_trace_rows = []
    staging_mode = "predictive" if args.predictive_prefetch else "jit_exact"
    if args.warmup_new_tokens > 0:
        print(
            f"warming up {args.warmup_new_tokens} generated tokens", flush=True,
        )
        warm_engine, warm_prefetcher = factory(
            budget=args.budgets[0], tree_size=args.tree_sizes[0],
            routing_objective=args.routing_objective, enable_trace=False,
        )
        warm_engine.generate(
            prompts[0], max_new_tokens=args.warmup_new_tokens,
        )
        torch.cuda.synchronize()
        del warm_engine, warm_prefetcher
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    print(f"staging mode: {staging_mode}", flush=True)
    print(
        f"moe reduction: {'atomic' if args.atomic_moe else 'deterministic'}",
        flush=True,
    )
    stage_rows_label = "planR/s" if args.predictive_prefetch else "jitR/s"
    objectives = (
        ["mass", "critical_path"]
        if args.compare_routing_objectives else [args.routing_objective]
    )
    print(f"{'B':>3} {'N':>5} {'TPOT(ms)':>10} {'accept_len':>11} {'steps':>6} "
            f"{'expert_hit':>10} {'budget_hist':>18} {stage_rows_label:>8} "
          f"{'repairR/s':>9} {'H2DGiB/tok':>10} {'cold':>6}", flush=True)
    for objective in objectives:
        print(f"routing objective: {objective}", flush=True)
        configured_factory = partial(factory, routing_objective=objective)
        for n in args.tree_sizes:
            for b in args.budgets:
                baseline = None
                if args.execution_trace_baseline_first:
                    baseline = run_config(
                        partial(configured_factory, enable_trace=False),
                        b, n, prompts, max_new_tokens=args.max_new_tokens,
                    )
                r = run_config(configured_factory, b, n, prompts,
                               max_new_tokens=args.max_new_tokens,
                               profiler_dir=args.torch_profiler_dir,
                               profiler_label=(
                                   f"{objective}-b{b}-n{n}"
                               ),
                               profiler_memory=args.torch_profiler_memory,
                               profiler_with_stack=args.torch_profiler_with_stack)
                if args.budget_trace_json is not None:
                    trace_rows.append({
                        "routing_objective": objective,
                        "budget": b,
                        "tree_size": n,
                        "target_steps": r["target_steps"],
                        "mean_accept_len": r["accept_len"],
                        "budget_trace": r["budget_trace"],
                        "demand_trace": r["demand_trace"],
                        "expert_trace": r["expert_trace"],
                    })
                if args.execution_trace_json is not None:
                    execution_trace_rows.append({
                        "routing_objective": objective,
                        "staging_mode": staging_mode,
                        "moe_reduction": (
                            "atomic" if args.atomic_moe else "deterministic"
                        ),
                        "trace_detail": args.execution_trace_detail,
                        "num_layers": cfg.num_layers,
                        "num_experts": cfg.num_experts,
                        "budget": b,
                        "tree_size": n,
                        "num_prompts": args.num_prompts,
                        "max_new_tokens": args.max_new_tokens,
                        "tpot_ms": r["tpot_ms"],
                        "baseline_tpot_ms": (
                            baseline["tpot_ms"] if baseline is not None else None
                        ),
                        "mean_accept_len": r["accept_len"],
                        "generated_tokens": (
                            baseline["generated_tokens"]
                            if baseline is not None else r["generated_tokens"]
                        ),
                        "target_steps": r["target_steps"],
                        "trace": r["execution_trace"],
                        "profiler_artifacts": r["profiler_artifacts"],
                    })
                histogram = r["budget_histogram"]
                if histogram is None:
                    plan_hist = "-"
                else:
                    plan_hist = "/".join(
                        f"{budget}x{count}"
                        for budget, count in enumerate(histogram) if count
                    )
                print(
                    f"{r['budget']:>3} {r['tree_size']:>5} {r['tpot_ms']:>10.2f} "
                    f"{r['accept_len']:>11.2f} {r['target_steps']:>6} "
                    f"{r['hit_rate']:>10.3f} {plan_hist:>18} "
                    f"{r['stage_rows_per_step']:>8.1f} "
                    f"{r['repair_rows_per_step']:>9.1f} "
                    f"{r['h2d_gib_per_token']:>10.2f} {r['cold']:>6}",
                    flush=True,
                )
                if r["profiler_artifacts"] is not None:
                    for artifact, path in r["profiler_artifacts"].items():
                        print(f"  profiler {artifact}: {path}", flush=True)
    if args.budget_trace_json is not None:
        args.budget_trace_json.parent.mkdir(parents=True, exist_ok=True)
        args.budget_trace_json.write_text(json.dumps(trace_rows, indent=2) + "\n")
        print(f"budget trace: {args.budget_trace_json}", flush=True)
    if args.execution_trace_json is not None:
        args.execution_trace_json.parent.mkdir(parents=True, exist_ok=True)
        args.execution_trace_json.write_text(
            json.dumps(execution_trace_rows, indent=2) + "\n"
        )
        print(f"execution trace: {args.execution_trace_json}", flush=True)


if __name__ == "__main__":
    main()
