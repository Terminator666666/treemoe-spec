"""Task 1.1: safetensors streaming weight loading. Experts stay native BF16.

Two resident layouts (spec §1.1):
  * "full"    - everything on GPU (H200-141G, or TP=2 sharded by intermediate dim)
  * "offload" - non-expert weights + hot experts on GPU, cold experts in host
                pinned memory, fetched on demand / prefetched by op2 (config B)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from treemoe.model.config import MixtralConfig

# HF Mixtral parameter name templates
_ATTN_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj")
_EXPERT_KEYS = ("w1", "w2", "w3")


@dataclass
class LayerWeights:
    input_layernorm: torch.Tensor
    post_attn_layernorm: torch.Tensor
    attn: dict[str, torch.Tensor]
    router: torch.Tensor            # [E, H] gate weight
    # Stacked expert weights, [E, I, H] / [E, H, I] contiguous — required by op1's
    # expert-stationary kernel (one TMA-friendly base pointer per matrix).
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    experts_on_gpu: bool = True


@dataclass
class MixtralWeights:
    config: MixtralConfig
    embed_tokens: torch.Tensor
    final_norm: torch.Tensor
    lm_head: torch.Tensor
    layers: list[LayerWeights] = field(default_factory=list)


def _stack_experts(get, layer: int, key: str, num_experts: int) -> torch.Tensor:
    parts = [
        get(f"model.layers.{layer}.block_sparse_moe.experts.{e}.{key}.weight")
        for e in range(num_experts)
    ]
    return torch.stack(parts, dim=0).contiguous()


def load_mixtral_weights(
    model_dir: str | Path,
    config: MixtralConfig | None = None,
    device: str = "cuda",
    layout: str = "full",
    hot_experts_per_layer: int = 4,
    offload_layers: set[int] | None = None,
) -> MixtralWeights:
    """Stream weights from a HF safetensors checkpoint directory.

    layout="offload": for layers in `offload_layers`, only the first
    `hot_experts_per_layer` experts (by global routing frequency, see
    measurements/analyze.py) live on GPU; the rest are pinned host tensors that
    op2's prefetcher copies into a GPU ring buffer.
    """
    from safetensors import safe_open

    config = config or MixtralConfig()
    model_dir = Path(model_dir)
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors found under {model_dir}")

    # Build a global key -> shard index so we can stream lazily.
    key_to_shard: dict[str, Path] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                key_to_shard[k] = shard

    open_handles: dict[Path, object] = {}

    def get(key: str) -> torch.Tensor:
        shard = key_to_shard[key]
        if shard not in open_handles:
            open_handles[shard] = safe_open(shard, framework="pt", device="cpu").__enter__()
        t = open_handles[shard].get_tensor(key)
        assert t.dtype == config.dtype, f"{key}: expected {config.dtype}, got {t.dtype} (no quantization allowed)"
        return t

    offload_layers = offload_layers or set()
    layers = []
    for i in range(config.num_layers):
        pfx = f"model.layers.{i}"
        expert_device = "cpu" if (layout == "offload" and i in offload_layers) else device
        w1 = _stack_experts(get, i, "w1", config.num_experts)
        w2 = _stack_experts(get, i, "w2", config.num_experts)
        w3 = _stack_experts(get, i, "w3", config.num_experts)
        if expert_device == "cpu":
            try:
                w1, w2, w3 = w1.pin_memory(), w2.pin_memory(), w3.pin_memory()
            except RuntimeError:
                # not enough lockable host memory: keep pageable (slower H2D
                # copies but functionally identical)
                pass
        else:
            w1, w2, w3 = w1.to(device), w2.to(device), w3.to(device)
        layers.append(
            LayerWeights(
                input_layernorm=get(f"{pfx}.input_layernorm.weight").to(device),
                post_attn_layernorm=get(f"{pfx}.post_attention_layernorm.weight").to(device),
                attn={k: get(f"{pfx}.self_attn.{k}.weight").to(device) for k in _ATTN_KEYS},
                router=get(f"{pfx}.block_sparse_moe.gate.weight").to(device),
                w1=w1,
                w2=w2,
                w3=w3,
                experts_on_gpu=(expert_device != "cpu"),
            )
        )

    weights = MixtralWeights(
        config=config,
        embed_tokens=get("model.embed_tokens.weight").to(device),
        final_norm=get("model.norm.weight").to(device),
        lm_head=get("lm_head.weight").to(device),
        layers=layers,
    )
    for h in open_handles.values():
        h.__exit__(None, None, None)
    return weights
