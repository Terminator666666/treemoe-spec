# TreeMoE-Spec 实施计划与最终状态（2026-08-18，2026-09-02 更新）

> 本文保留 2026-08-18 的任务拆分，并按 2026-09-02 的代码状态标注取舍。最终系统没有使用训练式
> 路由预测器，也没有接入整步 CUDA Graph。论文只陈述已经进入 `bench_e2e.py` 端到端路径的功能。
>
> **Spec**: [docs/specs/treemoe-spec-design.md](../specs/treemoe-spec-design.md)
>
> **Current Architecture**: Python draft/tree loop → tree-aware MoE 验证；专家 H2D 在侧流预取；
> greedy verify/KV commit 使用 Triton，随后由主机读回接受结果并维护序列状态。
> 论文将接受概率感知的层内专家选择，以及全局传输约束下的层自适应专家预算列为
> 两项核心贡献；MoE Triton kernel 和 GPU verify/KV commit 作为系统实现优化。
>
> **Tech Stack**: Python 3.12 + PyTorch + Triton + transformers + safetensors + pytest。实际硬件为
> RTX 4090 24GB，专家权重原生 BF16，经 host pinned memory offload。
>
> **Global Constraints**:
> - 每个 kernel 必须先有 PyTorch 参考实现 + 数值对齐测试（BF16 rtol=1e-3），测试先行（TDD）；
> - kernel 使用静态输出缓冲；端到端允许 CPU 读回接受结果；
> - 每完成一个任务 git commit 一次；每阶段末跑全量 pytest；
> - 基线 kernel（vLLM fused_moe / DeepGEMM / HPC-Ops / MegaBlocks）只作为 pip 依赖引入 benchmark，
>   不 vendored 进代码库。

## 仓库结构

```
treemoe-spec/
├── docs/{specs,plans}/
├── treemoe/
│   ├── model/          # mixtral.py, eagle.py, kv_cache.py, weights.py
│   ├── kernels/        # op1_tree_moe.py, op2_prefetch.py, op3_budget.py(内嵌op1), op4_commit.py
│   ├── engine/         # loop.py（生产路径）, tree.py；graph.py 为未接入原型
│   └── ref/            # 每个算子的 PyTorch 纯参考实现（测试锚点）
├── measurements/       # Phase 0 观测脚本与图
├── benchmarks/         # op1/op2 微基准与 4090 端到端实验
└── tests/
```

---

## Phase 0 — 观测实验（论文第 3 章数据，1 人周）

**先于一切 kernel 开发：如果路由局部性/激活膨胀不成立，方案止损重定向。**

### Task 0.1 环境与模型就位
- `uv init`；安装 torch/triton/transformers/safetensors/pytest；
- 下载 `mistralai/Mixtral-8x7B-Instruct-v0.1` 与 `yuhuili/EAGLE-mixtral-instruct-8x7B`；
- 验收：`tests/test_env.py::test_mixtral_forward_one_token` 通过（HF 加载、单 token 前向不 OOM，
  device_map=auto 双卡或 CPU offload 均可，此阶段不追求速度）。

### Task 0.2 路由采集 hook
- 文件 `measurements/collect_routing.py`：对 HF Mixtral 注册 forward hook，
  抓取每层 `router_logits[T, 8]`，输入用 MT-Bench 80 条 prompt 的真实生成轨迹（greedy，256 new tokens）；
- 输出 `measurements/data/routing_traces.pt`：`{layer: LongTensor[T, 2]}` top-2 ids + gates；
- 测试：`test_collect.py` 校验 shape 与 gate 归一化。

### Task 0.3 三张观测图
- `measurements/analyze.py` 产出 spec §1.2 的图 1/2/3（模拟树：从 trace 按 EAGLE-2 树形状采样节点组）；
- **决策门**：若图 2 显示父子 top-2 Jaccard ≥ 0.5 且图 3 长尾明显 → 继续；否则重议算子 1/3 设计。

---

## Phase 1 — 最小可跑框架（无推测、无自研核，2 人周）

### Task 1.1 权重加载与显存布局
- `treemoe/model/weights.py`：safetensors 流式读取，专家权重保持原生 BF16 不量化；
  最终实验使用 4090 offload：专家权重常驻 host pinned memory，按层送入 GPU staging buffer；
- 测试：单层 FFN 输出 vs HF 实现逐元素一致（同 dtype，零量化损失）。

### Task 1.2 极简 Mixtral 前向（AR 路径）
- `treemoe/model/mixtral.py`：RMSNorm/RoPE/GQA-attention(SDPA)/MoE-FFN(先用朴素 for-expert 循环)；
- `treemoe/model/kv_cache.py`：paged KV，block_size=64；
- 验收：`test_parity.py::test_ar_logits_match_hf`——同 prompt 32 步 greedy 与 HF 输出 token 完全一致（BF16 配置）。
  **这是全项目最重要的测试，后续任何优化不得破坏。**

### Task 1.3 EAGLE-2 草稿 + 树扩展 + 朴素验证
- `treemoe/engine/tree.py`：EAGLE-2 动态树（top-K 扩展 + 全局接受概率重排序，填充到 N=64）；
- `treemoe/engine/loop.py`：draft → 目标前向（tree mask SDPA）→ PyTorch 参考版 rejection sampling → 提交；
- 验收：`test_spec_lossless.py`——推测输出与 AR greedy 逐 token 一致；
  记录基线 TPOT 与接受长度 τ（预期 τ≈2.5–3.5）。
- **里程碑 M1**：有一个能跑、正确、但慢的完整推测解码系统 → 之后每个算子都能立即测端到端收益。

---

## Phase 2 — 核心贡献：树级专家预算与支撑 MoE kernel（4 人周）

### Task 2.1 参考实现与测试先行
- `treemoe/ref/tree_moe_ref.py`：spec §3.1 签名的纯 PyTorch 实现（含预算路由逻辑）；
- `tests/test_op1.py`：随机路由/极端路由（全 token 同专家、专家空载）/B∈{2,4,8} 共 12 个 case，
  先对 ref 自测通过。

### Task 2.2 Kernel A（budget_and_bucket, Triton）
- router 使用与 HF 同源的 BF16 `F.linear` + FP32 softmax，避免 Triton 与 cuBLAS reduction 顺序不同导致
  top-k 翻转；`_budget_bucket_fused_kernel` 将需求统计、预算路由和稳定 bucket 融合为单 CTA；
- 输出 `sorted_token_ids/expert_offsets` 全 GPU 侧；
- 测试：与 ref 的 bucket 结果逐元素一致（排序稳定性用 (expert, dfs_order) 双键）。

### Task 2.3 expert-stationary GEMM1/GEMM2（Triton）
- BF16 权重直读（主线唯一精度，无量化 epilogue）；GEMM1 融合 w1/w3、SiLU 和逐元素乘法，
  中间激活写入预分配 workspace 后由 GEMM2 读取；
- 性能路径用 atomic add；确定性路径把 split-k partial 写入 FP32 workspace，再由 `_combine_kernel` 固定顺序归约；
- 测试：BF16 vs ref rtol=1e-3。

### Task 2.4 microbenchmark 与调优
- `benchmarks/bench_op1.py`：N∈{32,64,128} 的确定性/atomic 路径，vLLM 可用时追加同边界对照；
- autotune 空间：BK∈{64,128}, SPLIT_K∈{4,8,16}, num_warps∈{4,8}；
- 当前 4090 容器无 Nsight Compute 权限，使用权重流量估算有效带宽，并以 torch.profiler 做内核归因。

### Task 2.5 接入端到端 + B 扫描（已实现，最终实验待补）
- loop.py 换用算子 1；跑 B∈{2,3,4,5,6,8} 的 τ-TPOT Pareto；
- `test_spec_lossless.py` 在 B=8 下必须仍然通过。

---

## Phase 3 — greedy 验证与提交（部分完成）

### Task 3.1 greedy 验证与 KV 提交（已实现）
- `treemoe/kernels/op4_commit.py`：temperature=0 使用 argmax、树验证和 KV commit 三个 Triton 阶段；
- temperature>0 保留 PyTorch 参考路径，不计入性能实验；
- CPU、Triton interpreter 和真机 kernel commit 均有回归测试。

### Task 3.2 全步 CUDA Graph 捕获（未实现，移入未来工作）
- `treemoe/engine/graph.py` 只有原型，生产入口没有实例化 `StepGraph`；
- 当前 Python 建树、动态 `children` 列表、KV 元数据和 `.tolist()` 不可直接捕获；
- 论文不报告 graph launch 时延或 CUDA Graph 加速比。

---

## Phase 4 — 全局传输约束的层自适应预算（代码已实现，当前为负结果）

### Task 4.1 训练式跨层路由预测器（取消）
- `RouterPredictor` 和 `measurements/train_predictor.py` 保留为实验原型；
- 未采集 ShareGPT 训练集，未生成或加载 predictor checkpoint；
- EAGLE feature hint 已完成 pilot，但相对 temporal-only 只有 0.74% TPOT 差异且命中率相同，不再作为贡献。

### Task 4.2 可修复预取执行路径（已实现）
- offload 配置使用 host pinned BF16 权重、深度缓冲和侧流 H2D；
- `repair()` 在真实路由后补齐漏预测专家，因此预测错误不影响输出正确性；
- 累计报告 staged/repair expert rows 与真实字节，不能用表面 hit rate 代替传输成本。

### Task 4.3 层自适应预算分配（已实现，4090 pilot 未通过）
- `treemoe/engine/layer_budget.py` 从上一轮完整 target router demand 构造 EMA；
- 在可配置信赖域 $B_{min}\le B_l\le B_{max}$、$\sum_lB_l=L B_{avg}$ 下最大化各层 retained mass
  的乘积，按对数边际收益分配并输出预算和预取位图；线性 mass 目标只保留为消融；
- prefill 强制 B=8/full-copy，首轮 verification 使用 uniform/full-copy，prompt 间重置历史；
- `bench_e2e.py --adaptive-layer-budget` 与 `--uniform-layer-budget` 在相同 $B_{avg}$ 下比较
  TPOT、接受长度、质量及
  staged/repair GiB；CPU 与 Triton interpreter 回归已覆盖；RTX 4090 上 mass 与 log-mass 都降低
  repair/step，但因接受长度下降而显著慢于 uniform，暂不列为核心贡献；
- 增加逐 verification 的逐层预算轨迹，下一步诊断敏感层降配和预算振荡，不继续盲调目标或 EMA。

---

## Phase 5 — 论文实验与收尾（进行中）

- RTX 4090 上完成 AR、B∈{2,3,4,5,6,8} uniform 主实验、tree-size，以及固定总传输预算下的
  adaptive-vs-uniform 消融；
- 使用 MT-Bench 验证 B<8 质量，B=8 作为 lossless 对照；
- 所有最终配置和结果维护在 `measurements/final_experiments.csv`。

---

## 里程碑摘要

| 里程碑 | 内容 | 累计工期 |
|---|---|---|
| M0 | 观测三图 + 决策门 | 1 周 |
| M1 | 正确的最小推测解码系统 | 3 周 |
| M2 | 树级专家预算与支撑 MoE kernel 落地 | 7 周 |
| M3 | greedy Triton verify/KV commit；CUDA Graph 取消 | 已完成（不含 Graph） |
| M4 | 全局约束层预算与精确可修复预取接入，正式消融待测 | 进行中 |
| M5 | 4090 论文实验齐全 | 进行中 |
