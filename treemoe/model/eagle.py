"""EAGLE-2 draft model: single-layer feature autoregression + dynamic tree scores.

Weights: yuhuili/EAGLE-mixtral-instruct-8x7B (EAGLE-1 checkpoint; EAGLE-2 is an
inference-time dynamic-tree algorithm on top of the same weights, spec §3.5).

Structure (per official EAGLE repo): one decoder layer operating on
concat(embed(token), feature) -> fc -> decoder_layer -> feature', plus the
target model's lm_head reused for draft token distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from treemoe.model.config import MixtralConfig
from treemoe.model.mixtral import apply_rope, build_rope_cache, rms_norm


@dataclass
class EagleWeights:
    fc: torch.Tensor                # [H, 2H] fuse embed+feature
    attn: dict[str, torch.Tensor]   # q/k/v/o proj, Mixtral-shaped single layer
    input_layernorm: torch.Tensor
    post_attn_layernorm: torch.Tensor
    mlp_gate: torch.Tensor          # [I_d, H] dense (draft层是dense FFN,非MoE)
    mlp_up: torch.Tensor
    mlp_down: torch.Tensor


class EagleDraftModel:
    """Feature-space autoregression producing (next_feature, token_logits)."""

    def __init__(self, w: EagleWeights, config: MixtralConfig,
                 embed_tokens: torch.Tensor, lm_head: torch.Tensor):
        self.w = w
        self.cfg = config
        self.embed_tokens = embed_tokens  # shared with target
        self.lm_head = lm_head            # shared with target
        self.rope_cos, self.rope_sin = build_rope_cache(config, embed_tokens.device.type)
        # simple dense KV for the single draft layer (small; rebuilt per step-window)
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []

    def reset(self) -> None:
        self.k_cache.clear()
        self.v_cache.clear()

    @torch.inference_mode()
    def step(self, token_ids: torch.Tensor, features: torch.Tensor,
             positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """token_ids/features/positions: [T] / [T, H] / [T] -> (feat' [T,H], logits [T,V])."""
        cfg = self.cfg
        emb = F.embedding(token_ids, self.embed_tokens)
        x = F.linear(torch.cat([emb, features], dim=-1), self.w.fc)

        h = rms_norm(x, self.w.input_layernorm, cfg.rms_eps)
        t = h.shape[0]
        q = F.linear(h, self.w.attn["q_proj"]).view(t, cfg.num_heads, cfg.head_dim)
        k = F.linear(h, self.w.attn["k_proj"]).view(t, cfg.num_kv_heads, cfg.head_dim)
        v = F.linear(h, self.w.attn["v_proj"]).view(t, cfg.num_kv_heads, cfg.head_dim)
        q = apply_rope(q, self.rope_cos, self.rope_sin, positions)
        k = apply_rope(k, self.rope_cos, self.rope_sin, positions)
        self.k_cache.append(k)
        self.v_cache.append(v)
        fk = torch.cat(self.k_cache, dim=0)
        fv = torch.cat(self.v_cache, dim=0)

        rep = cfg.num_heads // cfg.num_kv_heads
        total = fk.shape[0]
        mask = torch.zeros(t, total, dtype=torch.bool, device=x.device)
        past = total - t
        for i in range(t):
            mask[i, : past + i + 1] = True
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            fk.repeat_interleave(rep, dim=1).transpose(0, 1),
            fv.repeat_interleave(rep, dim=1).transpose(0, 1),
            attn_mask=mask.unsqueeze(0),
        )
        o = o.transpose(0, 1).reshape(t, cfg.num_heads * cfg.head_dim)
        x = x + F.linear(o, self.w.attn["o_proj"])

        h = rms_norm(x, self.w.post_attn_layernorm, cfg.rms_eps)
        feat = x + F.linear(
            F.silu(F.linear(h, self.w.mlp_gate)) * F.linear(h, self.w.mlp_up),
            self.w.mlp_down,
        )
        logits = F.linear(feat, self.lm_head).float()
        return feat, logits


def load_eagle_weights(path: str, device: str = "cuda",
                       dtype: torch.dtype = torch.bfloat16) -> EagleWeights:
    from safetensors.torch import load_file

    sd = load_file(path)

    def g(key: str) -> torch.Tensor:
        return sd[key].to(device=device, dtype=dtype)

    return EagleWeights(
        fc=g("fc.weight"),
        attn={k: g(f"layers.0.self_attn.{k}.weight") for k in ("q_proj", "k_proj", "v_proj", "o_proj")},
        input_layernorm=g("layers.0.input_layernorm.weight"),
        post_attn_layernorm=g("layers.0.post_attention_layernorm.weight"),
        mlp_gate=g("layers.0.mlp.gate_proj.weight"),
        mlp_up=g("layers.0.mlp.up_proj.weight"),
        mlp_down=g("layers.0.mlp.down_proj.weight"),
    )
