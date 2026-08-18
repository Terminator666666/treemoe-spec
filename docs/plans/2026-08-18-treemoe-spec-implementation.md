# TreeMoE-Spec 实施计划（2026-08-18）

> **Goal**: 从零构建独立的 Mixtral-8x7B + EAGLE-2 MoE 推测解码框架，实现 4 个树感知推理算子，
> 产出可复现的论文实验数据。
>
> **Spec**: [docs/specs/treemoe-spec-design.md](../specs/treemoe-spec-design.md)
>
> **Architecture**: draft(Stream0) ∥ prefetch(Stream1) → tree-aware MoE 验证 → 融合提交，整步单 CUDA Graph。
>
> **Tech Stack**: Python 3.11 + PyTorch 2.5 + Triton 3.x（算子 1/3/4 v1）+ CUDA/CuTe（v2 可选）
> + safetensors + pytest。硬件 H200-141G 或 2×H100-80G TP=2（A100 复验）。专家权重原生 BF16 不量化。
>
> **Global Constraints**:
> - 每个 kernel 必须先有 PyTorch 参考实现 + 数值对齐测试（BF16 rtol=1e-3），测试先行（TDD）；
> - 所有 kernel 输出形状静态、不依赖 CPU 读回（CUDA Graph 红线）；
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
│   ├── engine/         # loop.py（draft-verify-commit）, graph.py（CUDA Graph 捕获）, tree.py
│   └── ref/            # 每个算子的 PyTorch 纯参考实现（测试锚点）
├── measurements/       # Phase 0 观测脚本与图
├── benchmarks/         # 与 4 个基线 kernel 的对比
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
  支持两种布局：全常驻（H200 / TP=2 按 I 维切分）与冷层专家 host pin 内存 offload（80G 配置 B）；
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

## Phase 2 — 算子 1 + 算子 3（核心，4 人周）

### Task 2.1 参考实现与测试先行
- `treemoe/ref/tree_moe_ref.py`：spec §3.1 签名的纯 PyTorch 实现（含预算路由逻辑）；
- `tests/test_op1.py`：随机路由/极端路由（全 token 同专家、专家空载）/B∈{2,4,8} 共 12 个 case，
  先对 ref 自测通过。

### Task 2.2 Kernel A（route_and_bucket, Triton）
- `treemoe/kernels/op1_tree_moe.py::_route_bucket_kernel`：router GEMM(FP32 acc) + top-2 +
  预算路由（spec §3.3 的 4 步）+ SMEM radix bucket，单 CTA；
- 输出 `sorted_token_ids/expert_offsets` 全 GPU 侧；
- 测试：与 ref 的 bucket 结果逐元素一致（排序稳定性用 (expert, dfs_order) 双键）。

### Task 2.3 Kernel B（expert_stationary_fused_ffn, Triton）
- BF16 权重直读（主线唯一精度，无量化 epilogue），grid=(E, SPLIT_K)，K-tile 内 w1/w3→SiLU⊙→w2 累加；
- split-k partial 用 `_combine_kernel` 归约；
- 测试：BF16 vs ref rtol=1e-3。

### Task 2.4 microbenchmark 与调优
- `benchmarks/bench_op1.py`：N∈{32,64,128} vs vLLM fused_moe / DeepGEMM BF16 masked /
  CUTLASS grouped GEMM / MegaBlocks（均 BF16 同精度，pip 版本锁定）；Nsight 抓 `dram__bytes_read`；
- autotune 空间：BK∈{64,128}, SPLIT_K∈{4,8,16}, num_warps∈{4,8}；
- **验收门**：N=64 时延 ≤ vLLM fused_moe 的 0.8x，或 HBM 读字节数下降 ≥ 30%（两者满足其一才进 Phase 3）。

### Task 2.5 接入端到端 + B 扫描
- loop.py 换用算子 1；跑 B∈{3,4,5,6,8} 的 τ-TPOT Pareto（论文主图初版）；
- `test_spec_lossless.py` 在 B=8 下必须仍然通过。

---

## Phase 3 — 算子 4 + CUDA Graph（2 人周）

### Task 3.1 融合提交核
- `treemoe/kernels/op4_commit.py`：spec §3.4 六步单 kernel（Triton；vocab 归约先两 kernel、
  后合并）；Philox 随机数；
- 测试：固定种子下与 ref rejection sampling 的接受路径完全一致（1000 随机树）。

### Task 3.2 全步 CUDA Graph 捕获
- `treemoe/engine/graph.py`：静态张量池 + `torch.cuda.graph` 捕获 draft→verify→commit；
- 陷阱清单：树形状填充恒定、KV 页指针经 indirection buffer、B 值放 graph 外可写标量；
- 验收：nsys 显示每步 CPU launch 间隙 < 20 μs；端到端 TPOT 相对 M1 的累计加速 ≥ 2x。

---

## Phase 4 — 算子 2 预取（2 人周，独立可并行）

### Task 4.1 跨层路由预测器
- `measurements/train_predictor.py`：ShareGPT 5 万样本采集 (倒数第二层 hidden → 各层 top-2 label)，
  训 Wp[4096,256]；验收 recall@4 ≥ 70%（不达标走 spec 风险表回退：上一步激活集启发式）。

### Task 4.2 预取执行路径
- `treemoe/kernels/op2_prefetch.py`：HBM 配置 L2-warm kernel；offload 配置（BF16 专家 + host pin 内存
  + 环形缓冲 + cudaMemcpyAsync on Stream 1；单专家单层 352MB@PCIe Gen5≈5.5ms → 预取提前 ≥4 层流水）；
- 验收：offload 配置下 TPOT 改善 ≥ 20%（这是算子 2 的主战场）；HBM 配置如实报告（哪怕 ~0）。

---

## Phase 5 — 论文实验与收尾（3 人周）

- 全量基线表（spec §4 的 4 基线 × 3 数据集 × 指标）；消融表；A100 复验；
- 质量验证：B∈{4,5,6} 的 GSM8K/HumanEval/MT-Bench vs B=8；
- （可选加分）算子 1 v2 持久化巨核（CUDA + PDL），只在时间富余时做；
- 整理复现脚本 `benchmarks/reproduce_all.sh`。

---

## 里程碑摘要

| 里程碑 | 内容 | 累计工期 |
|---|---|---|
| M0 | 观测三图 + 决策门 | 1 周 |
| M1 | 正确的最小推测解码系统 | 3 周 |
| M2 | 算子 1+3 落地，kernel 级验收门通过 | 7 周 |
| M3 | 单 CUDA Graph 整步，端到端 ≥2x | 9 周 |
| M4 | 预取消融完成 | 11 周 |
| M5 | 论文实验齐全 | 14 周 |
