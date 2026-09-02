"""Task 1.2: minimal Mixtral forward (AR + tree verification paths).

Design goals over speed (this is the M1 correctness anchor, plan Task 1.2):
  * numerics follow HF exactly: RMSNorm fp32, router softmax fp32, BF16 weights
  * MoE FFN is pluggable: `moe_fn=None` uses the naive per-expert loop; the
    engine later injects treemoe.kernels.op1 without touching this file
  * attention: SDPA with explicit masks; tree verification passes a [N, N]
    ancestor mask + full visibility of the committed prefix
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from treemoe.model.config import MixtralConfig
from treemoe.model.kv_cache import PagedKVCache
from treemoe.model.weights import LayerWeights, MixtralWeights

MoEFn = Callable[[torch.Tensor, LayerWeights, int], torch.Tensor]
MoETraceFn = Callable[[str, torch.Tensor], None]
LayerTraceFn = Callable[[int, str, torch.Tensor], None]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    # Match HF MixtralRMSNorm exactly: normalize in FP32, cast the normalized
    # activations back to the input dtype, then multiply by the native weight.
    return weight * x32.to(x.dtype)


def build_rope_cache(config: MixtralConfig, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2, device=device).float() / config.head_dim)
    )
    t = torch.arange(config.max_seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    # HF Llama/Mixtral duplicates the half-width frequencies and rotates the
    # first/second halves of each head (not adjacent even/odd dimensions).
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor):
    # x: [T, heads, head_dim]; positions: [T]
    # HF rotary embeddings cast cos/sin to the activation dtype before the
    # multiply; retaining FP32 here changes BF16 rounding at every layer.
    c = cos[positions].to(x.dtype).unsqueeze(1)
    s = sin[positions].to(x.dtype).unsqueeze(1)  # [T,1,head_dim]
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * c + rotated * s


def naive_moe(
    x: torch.Tensor,
    lw: LayerWeights,
    _layer_idx: int,
    trace: Optional[MoETraceFn] = None,
) -> torch.Tensor:
    """HF-equivalent per-expert loop. x: [T, H] -> [T, H]."""
    # HF MixtralTopKRouter runs the linear in the activation/weight dtype,
    # then promotes the rounded logits to FP32 for softmax.
    logits = F.linear(x, lw.router)
    gates = torch.softmax(logits.float(), dim=-1)
    topg, topi = gates.topk(2, dim=-1)
    topg = topg / topg.sum(-1, keepdim=True)
    if trace is not None:
        trace("moe.router_logits", logits)
        trace("moe.router_probs", gates)
        trace("moe.topk_weights", topg)
        trace("moe.topk_indices", topi)
    out = torch.zeros_like(x)
    for e in range(lw.w1.shape[0]):
        for k in range(2):
            sel = topi[:, k] == e
            if not sel.any():
                continue
            token_indices = sel.nonzero(as_tuple=False).flatten()
            xe = x[sel]
            gate = xe @ lw.w1[e].t()
            up = xe @ lw.w3[e].t()
            activated = F.silu(gate) * up
            down = activated @ lw.w2[e].t()
            weighted = down * topg[sel, k : k + 1].to(x.dtype)
            out[sel] += weighted
            if trace is not None:
                prefix = f"moe.expert_{e}.slot_{k}"
                trace(f"{prefix}.token_indices", token_indices)
                trace(f"{prefix}.input", xe)
                trace(f"{prefix}.gate", gate)
                trace(f"{prefix}.up", up)
                trace(f"{prefix}.activated", activated)
                trace(f"{prefix}.down", down)
                trace(f"{prefix}.weighted", weighted)
    if trace is not None:
        trace("moe.output", out)
    return out


class MixtralForward:
    def __init__(self, weights: MixtralWeights, kv: PagedKVCache,
                 moe_fn: Optional[MoEFn] = None, prefetcher=None):
        self.w = weights
        self.cfg = weights.config
        self.kv = kv
        self.moe_fn: MoEFn = moe_fn or naive_moe
        # op2 LayerPrefetcher (spec §3.2): ahead-of-time side-stream staging of
        # offloaded layers; None falls back to synchronous _stage_experts
        self.prefetcher = prefetcher
        self.trace_observer: Optional[LayerTraceFn] = None
        self.rope_cos, self.rope_sin = build_rope_cache(self.cfg, weights.embed_tokens.device.type)
        # reusable GPU staging buffers for layout="offload" layers (one layer's
        # experts = 2.82GB at Mixtral shapes) -- allocated on first use
        self._staging: Optional[dict[str, torch.Tensor]] = None

    def _trace(self, layer_idx: int, name: str, tensor: torch.Tensor) -> None:
        if self.trace_observer is not None:
            self.trace_observer(layer_idx, name, tensor)

    def _stage_experts(self, lw: LayerWeights) -> LayerWeights:
        """Copy pinned-host expert weights into a reusable GPU staging buffer.

        Correctness-only path for small-VRAM cards (e.g. 4090-24G red-line
        runs): synchronous per-layer PCIe copy, ~6s/forward at Mixtral shapes.
        The performance path is op2's ring-buffer prefetcher (config B)."""
        dev = lw.router.device
        if self._staging is None:
            self._staging = {
                "w1": torch.empty_like(lw.w1, device=dev),
                "w2": torch.empty_like(lw.w2, device=dev),
                "w3": torch.empty_like(lw.w3, device=dev),
            }
        self._staging["w1"].copy_(lw.w1, non_blocking=True)
        self._staging["w2"].copy_(lw.w2, non_blocking=True)
        self._staging["w3"].copy_(lw.w3, non_blocking=True)
        return replace(lw, w1=self._staging["w1"], w2=self._staging["w2"],
                       w3=self._staging["w3"], experts_on_gpu=True)

    # ---------------- attention ----------------

    def _attention(
        self,
        lw: LayerWeights,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        tree_mask: Optional[torch.Tensor],
        is_tree: bool,
        start_pos: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        t = x.shape[0]
        q_proj = F.linear(x, lw.attn["q_proj"])
        k_proj = F.linear(x, lw.attn["k_proj"])
        v_proj = F.linear(x, lw.attn["v_proj"])
        self._trace(layer_idx, "attn.q_proj", q_proj)
        self._trace(layer_idx, "attn.k_proj", k_proj)
        self._trace(layer_idx, "attn.v_proj", v_proj)
        q = q_proj.view(t, cfg.num_heads, cfg.head_dim)
        k = k_proj.view(t, cfg.num_kv_heads, cfg.head_dim)
        v = v_proj.view(t, cfg.num_kv_heads, cfg.head_dim)
        q = apply_rope(q, self.rope_cos, self.rope_sin, positions)
        k = apply_rope(k, self.rope_cos, self.rope_sin, positions)
        self._trace(layer_idx, "attn.q_rope", q)
        self._trace(layer_idx, "attn.k_rope", k)
        self._trace(layer_idx, "attn.value_states", v)

        if is_tree:
            self.kv.write_tree(layer_idx, k, v)
            # fused prefix-gather + tail append: one prefix copy, not two
            full_k, full_v = self.kv.gather_with_tail(layer_idx, k, v)
            prefix_len = full_k.shape[0] - t
            mask = torch.zeros(t, prefix_len + t, dtype=torch.bool, device=x.device)
            mask[:, :prefix_len] = True
            mask[:, prefix_len:] = tree_mask  # [N, N] bool, ancestors incl. self
        else:
            self.kv.append(layer_idx, k, v, start_pos=start_pos)
            full_k, full_v = self.kv.gather(layer_idx)
            total = full_k.shape[0]
            # causal mask without per-row int(positions[i]) D2H syncs
            # (was 32 hidden syncs/step in the AR decode path)
            mask = (torch.arange(total, device=x.device)[None, :]
                    <= positions[:, None])
            self._trace(layer_idx, "attn.mask", mask)

        qh = q.transpose(0, 1)          # [heads, T, hd]
        kh = full_k.transpose(0, 1)     # [kv_heads, S, hd]
        vh = full_v.transpose(0, 1)
        # enable_gqa: SDPA maps q head h -> kv head h // (heads/kv_heads)
        # internally, same grouping as the old repeat_interleave but without
        # materializing a 4x copy of the whole K/V prefix per layer per step.
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask.unsqueeze(0),
                                           enable_gqa=True)
        o = o.transpose(0, 1).reshape(t, cfg.num_heads * cfg.head_dim)
        self._trace(layer_idx, "attn.context", o)
        output = F.linear(o, lw.attn["o_proj"])
        self._trace(layer_idx, "attn.output", output)
        return output

    # ---------------- forward ----------------

    @torch.inference_mode()
    def forward(
        self,
        token_ids: torch.Tensor,           # [T]
        positions: torch.Tensor,           # [T]
        tree_mask: Optional[torch.Tensor] = None,  # [N, N] bool when verifying
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        is_tree = tree_mask is not None
        x = F.embedding(token_ids, self.w.embed_tokens)
        self._trace(-1, "embedding", x)
        # one D2H for the whole step instead of int(positions[0]) per layer
        start_pos = 0 if is_tree else int(positions[0])
        if self.prefetcher is not None:
            self.prefetcher.begin()
        for layer_idx, lw in enumerate(self.w.layers):
            self._trace(layer_idx, "layer.input", x)
            h = rms_norm(x, lw.input_layernorm, cfg.rms_eps)
            self._trace(layer_idx, "attn.norm", h)
            x = x + self._attention(lw, layer_idx, h, positions, tree_mask,
                                    is_tree, start_pos)
            self._trace(layer_idx, "attn.residual", x)
            h = rms_norm(x, lw.post_attn_layernorm, cfg.rms_eps)
            self._trace(layer_idx, "moe.norm", h)
            use_prefetch = self.prefetcher is not None and not lw.experts_on_gpu
            if lw.experts_on_gpu:
                moe_lw = lw
            elif use_prefetch:
                buf = self.prefetcher.acquire(layer_idx)
                moe_lw = replace(lw, w1=buf["w1"], w2=buf["w2"], w3=buf["w3"],
                                 experts_on_gpu=True)
            else:
                moe_lw = self._stage_experts(lw)
            if self.trace_observer is not None and self.moe_fn is naive_moe:
                moe_output = naive_moe(
                    h, moe_lw, layer_idx,
                    trace=lambda name, tensor: self._trace(layer_idx, name, tensor),
                )
            else:
                moe_output = self.moe_fn(h, moe_lw, layer_idx)
            x = x + moe_output
            self._trace(layer_idx, "layer.output", x)
            observer = getattr(self, "layer_observer", None)
            if observer is not None:
                observer(layer_idx, x)
            if use_prefetch:
                self.prefetcher.release(layer_idx)
        x = rms_norm(x, self.w.final_norm, cfg.rms_eps)
        self._trace(-1, "final.norm", x)
        logits = F.linear(x, self.w.lm_head).float()
        self._trace(-1, "logits", logits)
        if return_hidden:
            # EAGLE draft feature = the exact lm_head input (official ea_model:
            # base_model.model(...)[0], i.e. AFTER all layers + final_norm).
            # Feeding an earlier layer's pre-norm hidden collapses acceptance.
            return logits, x
        return logits
