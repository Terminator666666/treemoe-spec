"""Save and compare per-layer HF/TreeMoE hidden states in separate processes.

Usage:
  python benchmarks/diag_layer_parity.py hf
  python benchmarks/diag_layer_parity.py ours
  python benchmarks/diag_layer_parity.py compare
    python benchmarks/diag_layer_parity.py compare --layer 0
"""

from __future__ import annotations

import argparse
import gc
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from treemoe.model.config import MixtralConfig
from treemoe.model.kv_cache import PagedKVCache
from treemoe.model.mixtral import MixtralForward
from treemoe.model.weights import default_model_dir, load_mixtral_weights

PROMPT = "The capital of France is"
DEFAULT_DIR = Path("artifacts/layer_parity")


def snapshot(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def add_derived_checkpoints(trace: dict[str, torch.Tensor], num_heads: int,
                            num_kv_heads: int, head_dim: int) -> None:
    for prefix, input_name in (
        ("attn.norm", "layer.input"),
        ("moe.norm", "attn.residual"),
    ):
        norm_input = trace[input_name]
        variance = norm_input.float().pow(2).mean(-1, keepdim=True)
        normalized = norm_input.float() * torch.rsqrt(variance + 1e-5)
        trace[f"{prefix}.variance"] = variance
        trace[f"{prefix}.normalized"] = normalized.to(norm_input.dtype)

    query = trace["attn.q_rope"].transpose(0, 1)
    key = trace["attn.k_rope"].transpose(0, 1)
    value = trace["attn.value_states"].transpose(0, 1)
    groups = num_heads // num_kv_heads
    key = key.repeat_interleave(groups, dim=0)
    value = value.repeat_interleave(groups, dim=0)
    scores = torch.matmul(query, key.transpose(1, 2)) * (head_dim**-0.5)
    allowed = trace["attn.mask"].unsqueeze(0)
    masked_scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    probs = torch.softmax(masked_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    context = torch.matmul(probs, value).transpose(0, 1).reshape(query.shape[1], -1)
    trace["attn.reference_scores"] = scores
    trace["attn.reference_masked_scores"] = masked_scores
    trace["attn.reference_probs"] = probs
    trace["attn.reference_context"] = context


def save_artifact(path: Path, source: str, input_ids: torch.Tensor,
                  layers: list[torch.Tensor], **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "source": source,
        "prompt": PROMPT,
        "input_ids": input_ids.cpu(),
        "layers": layers,
        **metadata,
    }, path)
    size_mib = path.stat().st_size / 2**20
    traces = metadata.get("traces", [])
    checkpoint_count = sum(len(trace) for trace in traces)
    checkpoint_count += len(metadata.get("global_trace", {}))
    print(f"[{source}] saved {len(layers)} layers / {checkpoint_count} checkpoints "
          f"to {path} ({size_mib:.1f} MiB)", flush=True)


def run_hf(model_dir: str, output: Path) -> None:
    import transformers
    from transformers.models.mixtral import modeling_mixtral

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids[0]
    print("[hf] loading model on CPU", flush=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cpu",
    )

    with torch.inference_mode():
        hidden = model.model.embed_tokens(input_ids.unsqueeze(0)).cuda()
        global_trace = {"embedding": snapshot(hidden[0])}
        position_ids = torch.arange(input_ids.shape[0], device="cuda").unsqueeze(0)
        causal_mask = modeling_mixtral.create_causal_mask(
            config=model.config, inputs_embeds=hidden, attention_mask=None,
            past_key_values=None, position_ids=position_ids,
        )
        model.model.rotary_emb.to("cuda")
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        model.model.rotary_emb.to("cpu")
        layers = []
        traces: list[dict[str, torch.Tensor]] = []
        for layer_idx, layer in enumerate(model.model.layers):
            layer.to("cuda")
            trace: dict[str, torch.Tensor] = {"layer.input": snapshot(hidden[0])}
            live: dict[str, torch.Tensor] = {}

            def capture(name: str, tensor: torch.Tensor, *, squeeze_batch: bool = True) -> None:
                value = tensor[0] if squeeze_batch and tensor.ndim > 0 else tensor
                live[name] = value
                trace[name] = snapshot(value)

            def capture_router(_module, _args, output) -> None:
                capture("moe.router_logits", output[0], squeeze_batch=False)
                capture("moe.topk_weights", output[1], squeeze_batch=False)
                capture("moe.topk_indices", output[2], squeeze_batch=False)

            handles = [
                layer.input_layernorm.register_forward_hook(
                    lambda _m, _a, out: capture("attn.norm", out)),
                layer.self_attn.q_proj.register_forward_hook(
                    lambda _m, _a, out: capture("attn.q_proj", out)),
                layer.self_attn.k_proj.register_forward_hook(
                    lambda _m, _a, out: capture("attn.k_proj", out)),
                layer.self_attn.v_proj.register_forward_hook(
                    lambda _m, _a, out: capture("attn.v_proj", out)),
                layer.self_attn.o_proj.register_forward_pre_hook(
                    lambda _m, args: capture("attn.context", args[0])),
                layer.self_attn.o_proj.register_forward_hook(
                    lambda _m, _a, out: capture("attn.output", out)),
                layer.post_attention_layernorm.register_forward_hook(
                    lambda _m, _a, out: capture("moe.norm", out)),
                layer.mlp.gate.register_forward_hook(capture_router),
            ]

            original_experts_forward = layer.mlp.experts.forward

            def traced_experts_forward(experts, hidden_states, top_k_index, top_k_weights):
                final_hidden_states = torch.zeros_like(hidden_states)
                expert_mask = torch.nn.functional.one_hot(
                    top_k_index, num_classes=experts.num_experts,
                ).permute(2, 1, 0)
                expert_hit = torch.greater(
                    expert_mask.sum(dim=(-1, -2)), 0,
                ).nonzero()

                for expert_index_tensor in expert_hit:
                    expert_index = int(expert_index_tensor[0])
                    if expert_index == experts.num_experts:
                        continue
                    top_k_pos, token_indices = torch.where(expert_mask[expert_index])
                    current_state = hidden_states[token_indices]
                    gate_up = torch.nn.functional.linear(
                        current_state, experts.gate_up_proj[expert_index],
                    )
                    gate, up = gate_up.chunk(2, dim=-1)
                    activated = experts.act_fn(gate) * up
                    down = torch.nn.functional.linear(
                        activated, experts.down_proj[expert_index],
                    )
                    weighted = down * top_k_weights[token_indices, top_k_pos, None]
                    final_hidden_states.index_add_(
                        0, token_indices, weighted.to(final_hidden_states.dtype),
                    )
                    for slot in range(model.config.num_experts_per_tok):
                        selected = top_k_pos == slot
                        if not selected.any():
                            continue
                        prefix = f"moe.expert_{expert_index}.slot_{slot}"
                        trace[f"{prefix}.token_indices"] = snapshot(token_indices[selected])
                        trace[f"{prefix}.input"] = snapshot(current_state[selected])
                        trace[f"{prefix}.gate"] = snapshot(gate[selected])
                        trace[f"{prefix}.up"] = snapshot(up[selected])
                        trace[f"{prefix}.activated"] = snapshot(activated[selected])
                        trace[f"{prefix}.down"] = snapshot(down[selected])
                        trace[f"{prefix}.weighted"] = snapshot(weighted[selected])

                trace["moe.output"] = snapshot(final_hidden_states)
                return final_hidden_states

            layer.mlp.experts.forward = types.MethodType(
                traced_experts_forward, layer.mlp.experts,
            )
            try:
                hidden = layer(
                    hidden, position_embeddings=position_embeddings,
                    attention_mask=causal_mask, position_ids=position_ids,
                    past_key_values=None, use_cache=False,
                )
            finally:
                layer.mlp.experts.forward = original_experts_forward
                for handle in handles:
                    handle.remove()

            query = live["attn.q_proj"].view(
                input_ids.shape[0], model.config.num_attention_heads, -1,
            ).transpose(0, 1).unsqueeze(0)
            key = live["attn.k_proj"].view(
                input_ids.shape[0], model.config.num_key_value_heads, -1,
            ).transpose(0, 1).unsqueeze(0)
            query, key = modeling_mixtral.apply_rotary_pos_emb(
                query, key, *position_embeddings,
            )
            trace["attn.q_rope"] = snapshot(query[0].transpose(0, 1))
            trace["attn.k_rope"] = snapshot(key[0].transpose(0, 1))
            trace["attn.value_states"] = snapshot(
                live["attn.v_proj"].view(
                    input_ids.shape[0], model.config.num_key_value_heads, -1,
                )
            )
            trace["attn.mask"] = snapshot(causal_mask[0, 0] == 0)
            trace["attn.residual"] = snapshot(
                trace["layer.input"] + trace["attn.output"],
            )
            router_probs = torch.softmax(live["moe.router_logits"].float(), dim=-1)
            trace["moe.router_probs"] = snapshot(router_probs)
            trace["layer.output"] = snapshot(hidden[0])
            add_derived_checkpoints(
                trace, model.config.num_attention_heads,
                model.config.num_key_value_heads, layer.self_attn.head_dim,
            )
            layers.append(hidden[0].float().cpu())
            traces.append(trace)
            layer.to("cpu")
            torch.cuda.empty_cache()
            print(f"[hf] layer {layer_idx + 1}/32", flush=True)

        model.model.norm.to("cuda")
        final_hidden = model.model.norm(hidden)
        global_trace["final.norm"] = snapshot(final_hidden[0])
        model.model.norm.to("cpu")
        model.lm_head.to("cuda")
        global_trace["logits"] = snapshot(model.lm_head(final_hidden)[0].float())
        model.lm_head.to("cpu")

    save_artifact(output, "hf", input_ids, layers,
                  traces=traces, global_trace=global_trace,
                  transformers_version=transformers.__version__,
                  attention_implementation=model.config._attn_implementation)


def run_ours(model_dir: str, output: Path) -> None:
    from tqdm import tqdm

    from treemoe import __version__ as treemoe_version

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids[0]
    cfg = MixtralConfig()
    print("[ours] loading offloaded weights", flush=True)
    weights = load_mixtral_weights(
        model_dir, cfg, layout="offload", offload_layers=set(range(cfg.num_layers)),
    )
    kv = PagedKVCache(cfg, num_blocks=64)
    model = MixtralForward(weights, kv)
    layers: list[torch.Tensor] = []
    traces: list[dict[str, torch.Tensor]] = [dict() for _ in range(cfg.num_layers)]
    global_trace: dict[str, torch.Tensor] = {}
    progress = tqdm(total=cfg.num_layers, desc="ours forward", unit="layer")

    def observe(_layer_idx: int, state: torch.Tensor) -> None:
        layers.append(state.float().cpu())
        progress.update(1)

    def observe_trace(layer_idx: int, name: str, tensor: torch.Tensor) -> None:
        target = global_trace if layer_idx < 0 else traces[layer_idx]
        target[name] = snapshot(tensor)

    model.layer_observer = observe
    model.trace_observer = observe_trace
    ids_cuda = input_ids.cuda()
    positions = torch.arange(input_ids.shape[0], device="cuda")
    model.forward(ids_cuda, positions)
    progress.close()
    for trace in traces:
        add_derived_checkpoints(
            trace, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
        )
    save_artifact(output, "ours", input_ids, layers,
                  traces=traces, global_trace=global_trace,
                  treemoe_version=treemoe_version)


def print_tensor_diff(scope: str, name: str, hf_tensor: torch.Tensor,
                      our_tensor: torch.Tensor) -> None:
    if hf_tensor.shape != our_tensor.shape:
        print(f"{scope:>7} {name:<46} shape {tuple(hf_tensor.shape)} != "
              f"{tuple(our_tensor.shape)}")
        return
    if not (hf_tensor.is_floating_point() or our_tensor.is_floating_point()):
        mismatches = int((hf_tensor != our_tensor).sum())
        print(f"{scope:>7} {name:<46} exact mismatches={mismatches}/{hf_tensor.numel()}")
        return

    diff = our_tensor.float() - hf_tensor.float()
    abs_diff = diff.abs()
    ref_rms = hf_tensor.float().square().mean().sqrt()
    rmse = diff.square().mean().sqrt()
    relative_rmse = rmse / ref_rms.clamp_min(torch.finfo(torch.float32).tiny)
    p99 = torch.quantile(abs_diff.flatten(), 0.99)
    print(f"{scope:>7} {name:<46} "
          f"max={float(abs_diff.max()):.6g} p99={float(p99):.6g} "
          f"mean={float(abs_diff.mean()):.6g} rmse={float(rmse):.6g} "
          f"rel={float(relative_rmse):.3e}")


def compare(hf_path: Path, ours_path: Path, selected_layers: list[int] | None) -> None:
    hf = torch.load(hf_path, map_location="cpu", weights_only=True)
    ours = torch.load(ours_path, map_location="cpu", weights_only=True)
    if not torch.equal(hf["input_ids"], ours["input_ids"]):
        raise RuntimeError("HF and ours artifacts use different input_ids")
    if len(hf["layers"]) != len(ours["layers"]):
        raise RuntimeError("HF and ours artifacts contain different layer counts")

    print(f"{'layer':>5} {'max_abs':>12} {'mean_abs':>12} {'rmse':>12}")
    for layer_idx, (hf_state, our_state) in enumerate(
            zip(hf["layers"], ours["layers"], strict=True)):
        diff = our_state.float() - hf_state.float()
        print(f"{layer_idx:>5} {float(diff.abs().max()):>12.6f} "
              f"{float(diff.abs().mean()):>12.6f} "
              f"{float(diff.square().mean().sqrt()):>12.6f}")

    if "traces" not in hf or "traces" not in ours:
        print("\nDetailed traces are absent; rerun both `hf` and `ours` phases.")
        return

    print("\nDetailed checkpoint comparison")
    print("  scope checkpoint                                      metrics")
    for name in sorted(set(hf["global_trace"]) | set(ours["global_trace"])):
        if name not in hf["global_trace"] or name not in ours["global_trace"]:
            print(f" global {name:<46} missing in one artifact")
            continue
        print_tensor_diff("global", name, hf["global_trace"][name],
                          ours["global_trace"][name])

    layer_filter = set(selected_layers) if selected_layers else set(range(len(hf["traces"])))
    for layer_idx, (hf_trace, our_trace) in enumerate(
            zip(hf["traces"], ours["traces"], strict=True)):
        if layer_idx not in layer_filter:
            continue
        print(f"\n[layer {layer_idx}]")
        for name in sorted(set(hf_trace) | set(our_trace)):
            if name not in hf_trace or name not in our_trace:
                print(f"{layer_idx:>7} {name:<46} missing in one artifact")
                continue
            print_tensor_diff(str(layer_idx), name, hf_trace[name], our_trace[name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("hf", "ours", "compare"))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--layer", type=int, action="append",
        help="compare detailed checkpoints only for this layer (repeatable)",
    )
    args = parser.parse_args()
    model_dir = args.model_dir or default_model_dir()
    hf_path = args.artifact_dir / "hf_layers.pt"
    ours_path = args.artifact_dir / "ours_layers.pt"

    if args.phase == "hf":
        run_hf(model_dir, hf_path)
    elif args.phase == "ours":
        run_ours(model_dir, ours_path)
    else:
        compare(hf_path, ours_path, args.layer)
    gc.collect()


if __name__ == "__main__":
    main()