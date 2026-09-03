from types import ModuleType, SimpleNamespace

import torch

from benchmarks import bench_op1


def test_resolve_vllm_fused_moe_legacy_export(monkeypatch):
    def fused_moe():
        return None

    package = SimpleNamespace(fused_moe=fused_moe)
    monkeypatch.setattr(
        bench_op1.importlib,
        "import_module",
        lambda name: package,
    )

    assert bench_op1.resolve_vllm_fused_moe() == ("fused_moe", fused_moe)


def test_resolve_current_vllm_fused_experts(monkeypatch):
    def fused_experts():
        return None

    package = SimpleNamespace(
        fused_moe=ModuleType("fused_moe"),
        fused_experts=fused_experts,
    )

    def import_module(name):
        if name == "vllm.model_executor.layers.fused_moe":
            return package
        raise ImportError(name)

    monkeypatch.setattr(bench_op1.importlib, "import_module", import_module)

    assert bench_op1.resolve_vllm_fused_moe() == ("fused_experts", fused_experts)


def test_bench_vllm_current_api_routes_exact_top2(monkeypatch):
    captured = {}

    def fused_experts(x, w13, w2, topk_weights, topk_ids):
        captured["weights"] = topk_weights
        captured["ids"] = topk_ids
        return x

    monkeypatch.setattr(
        bench_op1,
        "resolve_vllm_fused_moe",
        lambda: ("fused_experts", fused_experts),
    )
    monkeypatch.setattr(bench_op1, "timed", lambda fn: fn())
    x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    router = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.bfloat16,
    )
    w1 = torch.zeros(3, 2, 2, dtype=torch.bfloat16)
    w2 = torch.zeros(3, 2, 2, dtype=torch.bfloat16)
    w3 = torch.zeros(3, 2, 2, dtype=torch.bfloat16)

    output = bench_op1.bench_vllm(x, w1, w2, w3, router)

    logits = torch.nn.functional.linear(x, router).float()
    expected_logits, expected_ids = logits.topk(2, dim=-1)
    assert output is x
    assert torch.equal(captured["ids"], expected_ids)
    torch.testing.assert_close(
        captured["weights"], torch.softmax(expected_logits, dim=-1),
    )