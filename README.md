# TreeMoE-Spec

树感知 MoE 推测解码框架：Mixtral-8x7B-Instruct + EAGLE-2，专家权重原生 BF16。

- 设计规格：[docs/specs/treemoe-spec-design.md](docs/specs/treemoe-spec-design.md)
- 实施计划：[docs/plans/2026-08-18-treemoe-spec-implementation.md](docs/plans/2026-08-18-treemoe-spec-implementation.md)

## 布局

```
treemoe/model/    权重加载 / paged KV / Mixtral 前向 / EAGLE-2 草稿
treemoe/engine/   动态树构建 / draft-verify-commit 主循环 / CUDA Graph 捕获
treemoe/kernels/  算子1 树感知专家驻留 MoE（内嵌算子3 预算路由）/ 算子2 预取 / 算子4 融合提交
treemoe/ref/      每个算子的纯 PyTorch 参考实现（数值锚点）
measurements/     Phase 0 观测实验（路由局部性 / 激活膨胀 / 长尾）
benchmarks/       与 vLLM fused_moe / MegaBlocks / DeepGEMM / CUTLASS 的 kernel 对比
```

## 测试

```bash
pytest -m "not gpu and not model"   # CPU 逻辑测试（树/路由/采样参考实现）
pytest -m gpu                       # kernel 数值对齐（需 CUDA）
pytest -m model                     # 端到端一致性（需 Mixtral/EAGLE 权重）
```
