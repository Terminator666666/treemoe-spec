"""Save and compare per-layer HF/TreeMoE hidden states in separate processes.

Usage:
  python benchmarks/diag_layer_parity.py hf
  python benchmarks/diag_layer_parity.py ours
  python benchmarks/diag_layer_parity.py compare
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from treemoe.model.config import MixtralConfig
from treemoe.model.kv_cache import PagedKVCache
from treemoe.model.mixtral import MixtralForward
from treemoe.model.weights import default_model_dir, load_mixtral_weights

PROMPT = "The capital of France is"
DEFAULT_DIR = Path("artifacts/layer_parity")


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
    print(f"[{source}] saved {len(layers)} layer states to {path} ({size_mib:.1f} MiB)",
          flush=True)


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
        position_ids = torch.arange(input_ids.shape[0], device="cuda").unsqueeze(0)
        causal_mask = modeling_mixtral.create_causal_mask(
            config=model.config, inputs_embeds=hidden, attention_mask=None,
            past_key_values=None, position_ids=position_ids,
        )
        model.model.rotary_emb.to("cuda")
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        model.model.rotary_emb.to("cpu")
        layers = []
        for layer_idx, layer in enumerate(model.model.layers):
            layer.to("cuda")
            hidden = layer(
                hidden, position_embeddings=position_embeddings,
                attention_mask=causal_mask, position_ids=position_ids,
                past_key_values=None, use_cache=False,
            )
            layers.append(hidden[0].float().cpu())
            layer.to("cpu")
            torch.cuda.empty_cache()
            print(f"[hf] layer {layer_idx + 1}/32", flush=True)

    save_artifact(output, "hf", input_ids, layers,
                  transformers_version=transformers.__version__)


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
    progress = tqdm(total=cfg.num_layers, desc="ours forward", unit="layer")

    def observe(_layer_idx: int, state: torch.Tensor) -> None:
        layers.append(state.float().cpu())
        progress.update(1)

    model.layer_observer = observe
    ids_cuda = input_ids.cuda()
    positions = torch.arange(input_ids.shape[0], device="cuda")
    model.forward(ids_cuda, positions)
    progress.close()
    save_artifact(output, "ours", input_ids, layers,
                  treemoe_version=treemoe_version)


def compare(hf_path: Path, ours_path: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("hf", "ours", "compare"))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    model_dir = args.model_dir or default_model_dir()
    hf_path = args.artifact_dir / "hf_layers.pt"
    ours_path = args.artifact_dir / "ours_layers.pt"

    if args.phase == "hf":
        run_hf(model_dir, hf_path)
    elif args.phase == "ours":
        run_ours(model_dir, ours_path)
    else:
        compare(hf_path, ours_path)
    gc.collect()


if __name__ == "__main__":
    main()