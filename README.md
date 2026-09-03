# TreeMoE-Spec

树感知 MoE 推测解码框架：Mixtral-8x7B-Instruct + EAGLE-2，专家权重原生 BF16。

核心机制是接受概率感知的层内专家选择，以及固定全局 H2D expert-row 预算下的层自适应分配；
缺失权重在 GEMM 前精确修复。系统不训练路由预测器，也未在生产路径使用 CUDA Graph。

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
pytest -m "not gpu and not model and not interpret"  # CPU 逻辑测试
TRITON_INTERPRET=1 pytest -m interpret               # CPU 执行真实 Triton IR
pytest -m "gpu and not model"                        # kernel 数值对齐（需 CUDA）
pytest -m "gpu and model"                            # 单独进程跑真实模型一致性
```
