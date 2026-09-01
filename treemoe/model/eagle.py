"""EAGLE-2 draft model: single-layer feature autoregression + dynamic tree scores.

Weights: yuhuili/EAGLE-mixtral-instruct-8x7B (EAGLE-1 checkpoint; EAGLE-2 is an
inference-time dynamic-tree algorithm on top of the same weights, spec §3.5).

Structure (per official EAGLE repo): one decoder layer operating on
concat(embed(token), feature) -> fc -> decoder_layer -> feature', plus the
target model's lm_head reused for draft token distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from treemoe.model.config import MixtralConfig
from treemoe.model.mixtral import apply_rope, build_rope_cache, rms_norm


@dataclass
class EagleWeights:
    fc: torch.Tensor                # [H, 2H] fuse embed+feature
    attn: dict[str, torch.Tensor]   # q/k/v/o proj; draft may be full MHA (32 kv
                                    # heads per official config) unlike target GQA
    input_layernorm: torch.Tensor | None  # None = official EAGLE layer-0 (skipped)
    post_attn_layernorm: torch.Tensor
    mlp_gate: torch.Tensor          # [I_d, H] dense (draft层是dense FFN,非MoE)
    mlp_up: torch.Tensor
    mlp_down: torch.Tensor
    fc_bias: torch.Tensor | None = None  # config.json "bias": false for mixtral


class EagleDraftModel:
    """Feature-space autoregression producing (next_feature, token_logits).

    Two-tier KV, mirroring the official EAGLE inference loop:
      * committed KV — the accepted sequence (prompt + committed tokens),
        extended via extend_committed() with TARGET features (accurate);
        rejected branches never enter it.
      * tree KV — scratch for the current draft tree, cleared by begin_tree();
        step() rows attend committed KV + their tree ANCESTORS only (topology
        mask), never siblings or other branches.
    """

    def __init__(self, w: EagleWeights, config: MixtralConfig,
                 embed_tokens: torch.Tensor, lm_head: torch.Tensor,
                 rms_eps: float | None = None,
                 rope_theta: float | None = None):
        self.w = w
        self.cfg = config
        self.embed_tokens = embed_tokens  # shared with target
        self.lm_head = lm_head            # shared with target
        # draft checkpoint facts may differ from the target (official EAGLE
        # mixtral config.json: MHA kv=32, rms_norm_eps=1e-6, Llama-default
        # rope theta=10000) -- head counts derived from the weights, the
        # rest passed by the loader site; defaults keep target semantics
        # for duck-typed tiny-config tests
        self.num_heads = w.attn["q_proj"].shape[0] // config.head_dim
        self.num_kv_heads = w.attn["k_proj"].shape[0] // config.head_dim
        self.rms_eps = config.rms_eps if rms_eps is None else rms_eps
        rope_cfg = config if rope_theta is None else replace(config, rope_theta=rope_theta)
        self.rope_cos, self.rope_sin = build_rope_cache(rope_cfg, embed_tokens.device.type)
        self._ck: torch.Tensor | None = None  # committed K [S, kvh, hd]
        self._cv: torch.Tensor | None = None
        self._tree_k: list[torch.Tensor] = []  # per-level tree scratch
        self._tree_v: list[torch.Tensor] = []

    def reset(self) -> None:
        self._ck = self._cv = None
        self._tree_k.clear()
        self._tree_v.clear()

    def begin_tree(self) -> None:
        """Start a fresh draft tree (drops the previous tree's scratch KV)."""
        self._tree_k.clear()
        self._tree_v.clear()

    def _qkv(self, token_ids: torch.Tensor, features: torch.Tensor,
             positions: torch.Tensor):
        cfg = self.cfg
        emb = F.embedding(token_ids, self.embed_tokens)
        x = F.linear(torch.cat([emb, features], dim=-1), self.w.fc, self.w.fc_bias)
        # official EAGLE skips input_layernorm on layer 0 (cnets.py: only
        # index != 0 normalizes) -- the checkpoint has no such weight
        h = x if self.w.input_layernorm is None else rms_norm(
            x, self.w.input_layernorm, self.rms_eps)
        t = h.shape[0]
        q = F.linear(h, self.w.attn["q_proj"]).view(t, self.num_heads, cfg.head_dim)
        k = F.linear(h, self.w.attn["k_proj"]).view(t, self.num_kv_heads, cfg.head_dim)
        v = F.linear(h, self.w.attn["v_proj"]).view(t, self.num_kv_heads, cfg.head_dim)
        q = apply_rope(q, self.rope_cos, self.rope_sin, positions)
        k = apply_rope(k, self.rope_cos, self.rope_sin, positions)
        return x, q, k, v

    def _finish(self, x: torch.Tensor, q: torch.Tensor, fk: torch.Tensor,
                fv: torch.Tensor, mask: torch.Tensor,
                need_logits: bool = True):
        cfg = self.cfg
        t = q.shape[0]
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), fk.transpose(0, 1), fv.transpose(0, 1),
            attn_mask=mask.unsqueeze(0), enable_gqa=True,
        )
        o = o.transpose(0, 1).reshape(t, self.num_heads * cfg.head_dim)
        x = x + F.linear(o, self.w.attn["o_proj"])
        h = rms_norm(x, self.w.post_attn_layernorm, self.rms_eps)
        feat = x + F.linear(
            F.silu(F.linear(h, self.w.mlp_gate)) * F.linear(h, self.w.mlp_up),
            self.w.mlp_down,
        )
        logits = F.linear(feat, self.lm_head).float() if need_logits else None
        return feat, logits

    @torch.inference_mode()
    def extend_committed(self, token_ids: torch.Tensor, features: torch.Tensor,
                         positions: torch.Tensor) -> None:
        """Append accepted tokens (prompt or verified path) to the committed KV.

        Causal within the batch, full visibility of prior committed KV. No
        logits (nothing is sampled here) — skips the vocab GEMM entirely.
        """
        x, q, k, v = self._qkv(token_ids, features, positions)
        s = 0 if self._ck is None else self._ck.shape[0]
        self._ck = k if self._ck is None else torch.cat([self._ck, k], dim=0)
        self._cv = v if self._cv is None else torch.cat([self._cv, v], dim=0)
        t = k.shape[0]
        cols = torch.arange(s + t, device=x.device)
        mask = cols[None, :] <= (s + torch.arange(t, device=x.device))[:, None]
        self._finish(x, q, self._ck, self._cv, mask, need_logits=False)

    @torch.inference_mode()
    def step(self, token_ids: torch.Tensor, features: torch.Tensor,
             positions: torch.Tensor, tree_mask: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft one tree level: [T] / [T, H] / [T] -> (feat' [T,H], logits [T,V]).

        tree_mask: bool [T, P+T] over (prior tree nodes, this batch) — ancestor
        visibility built by build_eagle2_tree. All committed KV is always
        visible (it is an ancestor of every tree node). Falls back to
        batch-causal over the tree scratch when tree_mask is None.
        """
        x, q, k, v = self._qkv(token_ids, features, positions)
        self._tree_k.append(k)
        self._tree_v.append(v)
        tk = torch.cat(self._tree_k, dim=0)
        tv = torch.cat(self._tree_v, dim=0)
        fk = tk if self._ck is None else torch.cat([self._ck, tk], dim=0)
        fv = tv if self._cv is None else torch.cat([self._cv, tv], dim=0)

        t = k.shape[0]
        s = 0 if self._ck is None else self._ck.shape[0]
        p = tk.shape[0] - t  # prior tree nodes
        if tree_mask is None:
            cols = torch.arange(p + t, device=x.device)
            tree_mask = cols[None, :] <= (p + torch.arange(t, device=x.device))[:, None]
        if s:
            mask = torch.cat(
                [torch.ones(t, s, dtype=torch.bool, device=x.device), tree_mask], dim=1)
        else:
            mask = tree_mask
        return self._finish(x, q, fk, fv, mask)


def load_eagle_weights(path: str, device: str = "cuda",
                       dtype: torch.dtype = torch.bfloat16) -> EagleWeights:
    if path.endswith(".bin") or path.endswith(".pt"):
        sd = torch.load(path, map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file

        sd = load_file(path)

    def g(key: str) -> torch.Tensor:
        return sd[key].to(device=device, dtype=dtype)

    return EagleWeights(
        fc=g("fc.weight"),
        attn={k: g(f"layers.0.self_attn.{k}.weight") for k in ("q_proj", "k_proj", "v_proj", "o_proj")},
        # official EAGLE layer 0 has no input_layernorm (cnets.py index==0)
        input_layernorm=g("layers.0.input_layernorm.weight")
        if "layers.0.input_layernorm.weight" in sd else None,
        post_attn_layernorm=g("layers.0.post_attention_layernorm.weight"),
        mlp_gate=g("layers.0.mlp.gate_proj.weight"),
        mlp_up=g("layers.0.mlp.up_proj.weight"),
        mlp_down=g("layers.0.mlp.down_proj.weight"),
        fc_bias=g("fc.bias") if "fc.bias" in sd else None,
    )
