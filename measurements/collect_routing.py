"""Task 0.2: collect per-layer top-2 routing traces from HF Mixtral.

Runs greedy generation on MT-Bench prompts, hooks every MoE gate, and stores
per-token top-2 expert ids + gates. Output feeds analyze.py (paper figs 1-3).

Usage (GPU machine, needs `transformers` + downloaded weights):
    python measurements/collect_routing.py \
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
        --prompts measurements/mtbench_prompts.jsonl \
        --max-new-tokens 256 --out measurements/data/routing_traces.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def register_router_hooks(model, store: dict[int, list]):
    """Hook each MixtralSparseMoeBlock's gate Linear to capture router logits."""
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        gate = layer.block_sparse_moe.gate  # Linear(H, E)

        def hook(_mod, _inp, out, layer_idx=layer_idx):
            # out: [T, E] router logits; keep fp32 for exact top-2 boundaries
            logits = out.detach().float()
            gates = torch.softmax(logits, dim=-1)
            topg, topi = gates.topk(2, dim=-1)
            store[layer_idx].append(
                (topi.cpu().to(torch.int8), (topg / topg.sum(-1, keepdim=True)).cpu())
            )

        handles.append(gate.register_forward_hook(hook))
    return handles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    ap.add_argument("--prompts", default="measurements/mtbench_prompts.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--out", default="measurements/data/routing_traces.pt")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
    prompts = prompts[: args.limit]

    store: dict[int, list] = {i: [] for i in range(len(model.model.layers))}
    handles = register_router_hooks(model, store)

    traces = []  # one entry per prompt: {layer: (ids[T,2] int8, gates[T,2] f32)}
    with torch.inference_mode():
        for p_i, prompt in enumerate(prompts):
            for lst in store.values():
                lst.clear()
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            prompt_len = inputs.input_ids.shape[1]
            model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.eos_token_id,
            )
            per_layer = {}
            for layer_idx, chunks in store.items():
                ids = torch.cat([c[0] for c in chunks], dim=0)
                gates = torch.cat([c[1] for c in chunks], dim=0)
                per_layer[layer_idx] = (ids, gates)
            traces.append({"prompt_len": prompt_len, "layers": per_layer})
            print(f"[{p_i + 1}/{len(prompts)}] tokens={ids.shape[0]}")

    for h in handles:
        h.remove()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(traces, out)
    print(f"saved {len(traces)} traces -> {out}")


if __name__ == "__main__":
    main()
