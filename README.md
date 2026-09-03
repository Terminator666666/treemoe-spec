# TreeMoE-Spec

树感知 MoE 推测解码框架：Mixtral-8x7B-Instruct + EAGLE-2，专家权重原生 BF16。

候选研究主线是渐进式精确 target tree verification：先验证前缀闭合核心树，仅在真实 target 决策继续接受时
扩展下一 stage。当前已实现 full-trace 反事实模拟器；运行时仍一次 forward 完整树，只有 B=8 trace 通过
传输与 TPOT 硬门槛后才改造。接受概率专家预算、跨层预算和关键路径保护降为重合基线或负消融。
offload 默认在当前层 route 后仅搬运实际选中的 BF16 专家，旧预测预取加同步 repair 通过
`--predictive-prefetch` 保留为对照。系统不训练路由预测器，也未在生产路径使用 CUDA Graph。

- 设计规格：[docs/specs/treemoe-spec-design.md](docs/specs/treemoe-spec-design.md)
- 实施计划：[docs/plans/2026-08-18-treemoe-spec-implementation.md](docs/plans/2026-08-18-treemoe-spec-implementation.md)

## 布局

```
treemoe/model/    权重加载 / paged KV / Mixtral 前向 / EAGLE-2 草稿
treemoe/engine/   动态树构建 / draft-verify-commit 主循环 / CUDA Graph 捕获
treemoe/kernels/  算子1 树感知专家驻留 MoE（内嵌算子3 预算路由）/ 算子2 JIT staging / 算子4 融合提交
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
verify/commit、draft commit，以及每层 QKV、RoPE、KV/cache mask、SDPA、输出投影、route、
expert-id D2H、JIT staging、repair、MoE GEMM 和可选预测侧流 H2D。JSON 同时保存：

- 完整草稿树的 token、parent、depth、接受概率、根路径和实际接受路径；
- 每层级全部 frontier、候选 token/logprob 和下一层入选节点；
- 每个 target 节点的 top-8、提议 token 的精确 rank/probability 及接受、拒绝、未访问状态；
- 每层每节点的八专家 router 概率、原始 top-2、预算后 top-2、JIT staged/routed/missing 专家和 demand；
- 每次 JIT H2D、可选 planned H2D/repair、ring-slot wait、GPU 已分配/保留/空闲/峰值显存及运行环境。

同时打开 PyTorch profiler 会生成全部 CPU/CUDA 算子、kernel、memcpy、stream 和输入形状：

```bash
python3 -u benchmarks/bench_e2e.py \
	--layout offload --budgets 4 --tree-sizes 64 \
	--num-prompts 1 --max-new-tokens 32 --uniform-layer-budget \
	--execution-trace-json artifacts/execution_trace.json \
	--torch-profiler-dir artifacts/profiler
python3 benchmarks/analyze_execution_trace.py \
	artifacts/execution_trace.json --show-all \
	> artifacts/execution_trace_report.txt
```

`artifacts/profiler` 下的 `*.chrome.json` 用 `chrome://tracing` 或 Perfetto 打开，
`*.operators.txt` 按 self CUDA time 列出前 500 个 shape-aware 算子。逐分配显存事件和 Python 栈
分别用 `--torch-profiler-memory`、`--torch-profiler-with-stack` 打开；部分 PyTorch/Kineto 版本对这两个
选项存在非 UTF-8 事件名缺陷，所以默认关闭，逐步显存快照仍保存在 execution JSON 中。任一 profiler
产物导出失败时会生成对应的 `*.error.txt`，benchmark 会继续完成并写出其余结果。
`--show-all` 可能输出数万行，完整报告应重定向到文件。

同一输入下追加 `--predictive-prefetch --no-router-hint` 可复现旧 temporal prediction + repair 对照。

op1 默认使用 fixed-order deterministic GEMM2 partial+combine，以守住 B=8 与 AR 的逐 token 对齐。
`--atomic-moe` 可测量直接 scatter atomic-add 的性能上限，但其 FP32 加法顺序不确定，禁止与
`--check-lossless` 同时使用，也不能把该模式的结果写成严格无损配置。

渐进式验证的离线准入只接受无专家裁剪的 B=8 trace：

```bash
PYTHONPATH=. python3 benchmarks/simulate_progressive_verification.py \
	artifacts/execution_trace_b8.json \
	--stage-budgets 8 16 32 --pcie-gbps 25.03
```

报告中的 `progressive` 按当前 staging ring 计入跨 stage 重传，是可实现估计；
`persistent_lower_bound` 假设每层专家可跨整次 verification 永久驻留，仅作为乐观下界。进入运行时开发的门槛
是可实现估计至少减少 20% H2D bytes，且加入额外 stage 固定开销后的预计 TPOT 恶化不超过 10%。

RTX 4090 上可直接运行完整准入流程：

```bash
NUM_PROMPTS=4 MAX_NEW_TOKENS=64 \
	EXTRA_STAGE_OVERHEAD_MS=5.0 \
	bash benchmarks/run_progressive_gate.sh
```

这会固定采集 B=8、N=64、exact-JIT trace，比较 probability/depth 两种策略与四组 stage 网格的八个候选，
并写入
`artifacts/progressive_gate/progressive_gate.json`。退出码 0 表示至少一组同时通过双门槛，退出码 2 表示
收益不足。正式判定使用 `NUM_PROMPTS=20 MAX_NEW_TOKENS=128`；若需相对当前 B4 而非 B8 one-shot 判断，
额外设置 `REFERENCE_TPOT_MS=<同机无 trace 的 B4 TPOT>`。trace 模式本身有诊断开销，因此报告中的绝对
TPOT 不是论文性能数字，门槛只用于决定是否值得实现 partial target runtime。

脚本先用独立 engine warmup 8 个 token，再在同一次 93GB 权重加载内依次运行无 trace B8 baseline 和
`--execution-trace-detail progressive` 瘦 trace。gate 使用前者的 TPOT，并按 benchmark 实际返回的
`generated_tokens` 归一化；瘦 trace 不保存 draft candidates、target top-8、完整 router 概率或逐层 CUDA
Events。需要完整性能归因时仍使用默认 `--execution-trace-detail full`，不要用瘦 trace 替代热点报告。

trace/profiler 模式会增加 CUDA Events、log-softmax、精确 rank 计算、诊断 D2H、栈采集和大量 I/O，
只用于归因；论文 TPOT 必须同时关闭 `--execution-trace-json` 和 `--torch-profiler-dir` 后单独测量。
