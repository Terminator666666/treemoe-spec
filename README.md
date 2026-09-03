# TreeMoE-Spec

树感知 MoE 推测解码框架：Mixtral-8x7B-Instruct + EAGLE-2，专家权重原生 BF16。

核心机制是接受概率加权的层内专家选择，以及保护高接受概率树节点原始 top-2 的关键路径路由；
已被实验否定的跨层预算保留为负消融。缺失权重在 GEMM 前精确修复。系统不训练路由预测器，
也未在生产路径使用 CUDA Graph。

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

## 全阶段性能追踪

诊断模式用 host wall clock 和延迟解析的 CUDA Events 记录 prefill、EAGLE 建树、target verification、
verify/commit、draft commit，以及每层 attention、prefetch wait、route、expert-id D2H、repair、MoE GEMM
和侧流 H2D。JSON 同时保存完整草稿树的 token/parent/depth/接受概率、每个节点的根路径、实际接受路径，
以及逐层 staged/routed/missing 专家、slot 数和 demand：

```bash
python3 -u benchmarks/bench_e2e.py \
	--layout offload --budgets 4 --tree-sizes 64 \
	--num-prompts 1 --max-new-tokens 32 --uniform-layer-budget \
	--no-router-hint \
	--execution-trace-json artifacts/execution_trace.json
python3 benchmarks/analyze_execution_trace.py \
	artifacts/execution_trace.json --show-tree
```

trace 模式会增加 CUDA Events 和诊断 D2H，只用于归因；论文 TPOT 必须关闭
`--execution-trace-json` 后单独测量。
