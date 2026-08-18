"""Mixtral-8x7B configuration (numbers frozen per spec §1.1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MixtralConfig:
    num_layers: int = 32
    num_experts: int = 8
    top_k: int = 2
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 32000
    rope_theta: float = 1e6
    rms_eps: float = 1e-5
    max_seq_len: int = 8192
    dtype: torch.dtype = torch.bfloat16

    @property
    def expert_bytes_per_layer(self) -> int:
        """Single expert single layer weight bytes in native BF16 (352 MB)."""
        return 3 * self.intermediate_dim * self.hidden_dim * 2
