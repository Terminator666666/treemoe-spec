"""Shared fixtures: tiny random Mixtral-shaped weights for CPU logic tests."""

from __future__ import annotations

import gc

import pytest
import torch

from treemoe.model.config import MixtralConfig


@pytest.fixture(scope="session")
def tiny_config() -> MixtralConfig:
    """Scaled-down architecture: fast on CPU, same code paths."""
    return MixtralConfig(
        num_layers=2, num_experts=8, top_k=2,
        hidden_dim=64, intermediate_dim=128,
        num_heads=4, num_kv_heads=2, head_dim=16,
        vocab_size=256, max_seq_len=512,
        dtype=torch.float32,  # fp32 on CPU: keeps parity checks tight
    )


@pytest.fixture()
def rng():
    g = torch.Generator().manual_seed(1234)
    return g


@pytest.fixture(autouse=True)
def cleanup_cuda_after_gpu_test(request):
    """Prevent model/offload hooks from starving later kernel tests."""
    yield
    if request.node.get_closest_marker("gpu") is not None and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def make_moe_inputs(n: int, e: int, h: int, i: int, g: torch.Generator, dtype=torch.float32):
    x = torch.randn(n, h, generator=g, dtype=dtype)
    w1 = torch.randn(e, i, h, generator=g, dtype=dtype) * 0.02
    w2 = torch.randn(e, h, i, generator=g, dtype=dtype) * 0.02
    w3 = torch.randn(e, i, h, generator=g, dtype=dtype) * 0.02
    router = torch.randn(e, h, generator=g, dtype=dtype) * 0.1
    accept = torch.rand(n, generator=g)
    return x, w1, w2, w3, router, accept
