# TreeMoE-Spec 设计规格（v2，细化版）

> 一个不依赖 vLLM/SGLang 的独立 MoE 推测解码推理框架，目标模型 Mixtral-8x7B-Instruct + EAGLE-2 草稿模型，
> 核心贡献是 4 个树感知（tree-aware）推理算子。
> 参考库：Tencent HPC-Ops、DeepSeek DeepGEMM、Databricks MegaBlocks、FlashInfer、vLLM fused_moe。

---

## 1. 问题定义与量化分析

### 1.1 模型参数（精确数字）

| 项 | 值 |
|---|---|
| 层数 L | 32 |
| 每层专家数 E | 8 |
| 路由 | top-2, softmax gating |
| hidden dim H | 4096 |
| FFN intermediate dim I | 14336 |
| 单专家单层权重 | w1[I,H] + w3[I,H] + w2[H,I] = 3×58.7M = 176M 参数 |
| 单专家单层权重体积 | BF16: 352 MB；FP8: 176 MB；INT4: 88 MB |
| 总参数 | 46.7B（BF16 ≈ 93 GB） |
| 每 token 激活参数 | 12.9B |

**精度决策**：专家权重保持原生 BF16 **不量化**——规避量化误差与接受率 τ 的耦合干扰，
且 BF16 下权重读字节数是 FP8 方案的 2 倍，"验证期权重读放大"矛盾更尖锐，树感知算子的收益上限更高。

**显存预算（关键落地约束）**：BF16 全模型 ≈93 GB。
- 主配置 A：**单卡 H200-141G**（93 GB 权重 + KV + 激活仍有富余）；
  或 **2×H100-80G TP=2**——专家 FFN 按 I 维 Megatron 式切分（每卡 I/2=7168，w1/w3 切行、w2 切列），
  每层一次 all-reduce，算子 1 的 kernel 结构不变、仅 I 减半；
- 配置 B（offload，单卡 80G）：非专家权重 + 每层热专家常驻 GPU（~60 GB），
  其余专家 BF16 常驻 host pin 内存按需 PCIe 拉取——算子 2 预取的主战场；
- FP8/INT4 量化：**降级为附录消融项**，不在主线。

### 1.2 核心矛盾（论文第 3 章观测，需实测填数）

自回归 decode 每步激活 2/8 专家/层，读专家权重 ≈ 2×32×0.352 ≈ 22.5 GB（BF16）。
EAGLE-2 树验证一次前向送入 N=64 个 token，**64 个 token 的 top-2 路由并集在每层几乎覆盖全部 8 个专家**
（Cascade, arXiv:2506.20675 报告 2–3x 权重读放大）。此时：

- 验证一步读权重 ≈ 90 GB（全专家、BF16），是 AR 单步的 ~4x，H200 上仅权重读就 ≈ 19 ms/步；
- 每专家平均只分到 64×2/8 = **16 个 token**，M 极小 → GEMM 是典型 flat/skinny 形状，
  现有 fused MoE kernel（vLLM Triton、CUTLASS grouped GEMM）按大 batch 吞吐设计，
  此形状下带宽利用率低（HPC-Ops 用 split-k 在同类 decode 形状上比 DeepGEMM 快 1.88x，证明有很大空间）；
- 树内 token 是父子链，**路由高度相关**（兄弟/父子节点大概率命中相同专家）——这是可利用而未被利用的结构先验。

**观测实验（Phase 0 产出，论文图 1–3）**：
1. 图1：树大小 N ∈ {8,16,32,64,128} vs 每层激活专家数期望（HF Mixtral + 真实 MT-Bench 前缀）；
2. 图2：树内父子/兄弟节点 top-2 专家 Jaccard 相似度分布（验证路由局部性假设）；
3. 图3：专家 gating 权重长尾分布（验证预算路由 B<8 的可行性，对齐 MoE-Spec 的 drop 长尾观察）。

### 1.3 与现有工作的差异定位

| 工作 | 做了什么 | 没做什么（我们的空间） |
|---|---|---|
| Cascade (NVIDIA) | 发现 MoE 验证膨胀，动态调 K | 纯调度策略，无 kernel 创新 |
| MoE-Spec | 验证期专家预算（drop 长尾） | 未与 kernel 融合；未用树结构概率 |
| SP-MoE / MoE-SpeQ | 专家 offload 预取 | 需额外训练预测器；未用 EAGLE 特征白嫖 |
| HPC-Ops | split-k decode GroupGEMM、fused sampler、Route GEMM | 无推测解码/树感知；Megakernel 在 roadmap 未实现 |
| DeepGEMM | masked grouped GEMM（CUDA Graph 友好） | 面向 EP 大 batch；无树重排、无 FFN 全融合 |

---

## 2. 系统架构

```
┌────────────────────────── 1 个 CUDA Graph 捕获整步 ──────────────────────────┐
│  Stream 0（主）                          Stream 1（预取）                     │
│  ┌──────────────┐                       ┌──────────────────┐                │
│  │ EAGLE-2 草稿  │──draft features──────▶│ 算子2: 路由预执行  │                │
│  │ + GPU 树扩展  │                       │  + 专家预取       │                │
│  └──────┬───────┘                       └────────┬─────────┘                │
│         │ tree tokens[64], tree_mask             │ expert bitmap[32,8]      │
│         ▼                                        ▼                          │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │ 目标模型验证前向 ×32 层：                                   │               │
│  │   attention（tree mask, paged KV）                        │               │
│  │   算子1: 树感知专家驻留 MoE 核（内嵌 算子3: 预算路由）        │               │
│  └──────────────────────────┬──────────────────────────────┘               │
│                             │ logits[64, 32000]                            │
│                             ▼                                              │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │ 算子4: 融合 验证采样-KV压缩-提交 核（单 kernel）             │               │
│  └──────────────────────────┬──────────────────────────────┘               │
│                             │ accepted_len, next_root_feature（全留 GPU）    │
│                             └──────────────▶ 回到草稿阶段，零 CPU 同步        │
└─────────────────────────────────────────────────────────────────────────────┘
```

静态形状约定（CUDA Graph 前提）：树节点数 N=64（可编译 32/128 变体）、最大深度 D=6、
top-k 子节点数按 EAGLE-2 动态树但**填充到固定 N**（无效节点 mask 掉）。

---

## 3. 算子详细设计

### 3.1 算子 1：树感知专家驻留 MoE 核（核心贡献）

**签名（v1 两 kernel 方案，Triton）**

```python
def tree_moe_forward(
    x: Tensor,               # [N=64, H=4096] bf16，树节点 hidden states（DFS 序）
    w1: Tensor,              # [E=8, I=14336, H] bf16（原生精度，不量化）
    w3: Tensor,              # [E, I, H] bf16
    w2: Tensor,              # [E, H, I] bf16
    router_weight: Tensor,   # [E, H] bf16（router 用 fp32 累加，借鉴 HPC-Ops Route GEMM）
    node_accept_prob: Tensor,# [N] fp32，EAGLE-2 给出的节点接受概率（算子3 用）
    expert_budget: int,      # B ∈ [2, 8]，本层保留专家数上限
    out: Tensor,             # [N, H] bf16
) -> None
```

**Kernel A — route_and_bucket（一个 CTA 完成，~5 μs 级）**
1. router GEMM：x[64,4096] @ router_weight.T[4096,8] → logits[64,8]，**FP32 累加**
   （HPC-Ops 单列 Route GEMM 证明 router 对精度敏感，BF16 累加会翻转 top-2 边界样本）；
2. softmax + top-2 → topk_ids[64,2], topk_gates[64,2]；
3. **算子 3 内嵌于此**：预算路由（见 3.3），产出修正后的 topk_ids/gates；
4. 片上 radix bucket（SMEM 内计数 + 前缀和）：产出
   - `sorted_token_ids[128]`：按 (expert, DFS序) 排序的 token 槽位
   - `expert_offsets[E+1]`：每专家 token 段的前缀和
   - `num_tokens_per_expert[8]`：**由 GPU 写出，CPU 不读**（DeepGEMM masked layout 思路，
     保证整步可被 CUDA Graph 捕获——这是与 vLLM fused_moe 需要 CPU 侧 moe_align 的本质区别）。

**Kernel B — expert_stationary_fused_ffn（主计算核）**

数学恒等式（中间激活不落 HBM 的关键）：

$$\mathrm{FFN}(x) = \big(\mathrm{SiLU}(xW_1^\top) \odot xW_3^\top\big) W_2^\top = \sum_{k\text{-tile}} \big(\mathrm{SiLU}(xW_1[k]^\top) \odot xW_3[k]^\top\big) W_2[k]^\top$$

对 intermediate 维 I=14336 分块（BK=128），每块算出 h_blk[M, BK] 后**立即**乘 W2[BK, 4096] 累加，
14336 维中间激活永不物化到 HBM。

网格与调度（细节）：
- grid = (E=8, SPLIT_K=14336/BK_SPLIT)。**split-k 是小 M 场景的占用率关键**：
  M_e≈16 时若只按 M×N 切 tile，只有 8×2=16 个 CTA，H100 132 个 SM 大量闲置；
  按 K 维再切 8 份 → 128 CTA 打满。partial 结果用第二个轻量 combine kernel 归约
  （HPC-Ops "sm90 dynamic decode with split-k combine" 的同款结构）；
- **专家驻留（weight-stationary）**：每个 CTA 绑定一个专家，w1/w3/w2 的 K-tile 经 TMA/cp.async
  流过 SMEM 一次，对该专家名下所有 token（≤64）复用——权重读一遍服务多 token，
  而 token-stationary 布局（vLLM 默认）在小 M 下会重复读权重；
- token 段按 DFS 序排列 → 同一专家内的树节点父子相邻，x 的加载有 L2 局部性；
- 权重 BF16 直读，无反量化 epilogue（kernel 更简单、数值路径与 HF 完全对齐；
  FP8/INT4 变体仅作附录消融）；
- 空专家（num_tokens_per_expert[e]==0）：CTA 读到 0 立即退出（masked 语义，形状恒定）。

**v2（可选，论文加分项）**：持久化巨核，Kernel A/B 融为单 kernel，用 cooperative groups
grid.sync() 或 PDL（programmatic dependent launch，HPC-Ops 教程中使用）消除 kernel 间隙。
风险高，v1 已构成完整贡献，v2 作为附加优化章节。

**验收基准**（对照组固定为 4 个，均 BF16 同精度对比）：
vLLM `fused_moe` Triton、MegaBlocks dMoE、DeepGEMM BF16 grouped GEMM（masked layout）+独立激活核、
CUTLASS grouped GEMM+独立激活核；HPC-Ops FP8 GroupGEMM 仅作低精度 skyline 参考、不进主对比表。
指标：N∈{32,64,128} 下 kernel 时延、HBM 读字节数（Nsight `dram__bytes_read`）、SM 占用率。

### 3.2 算子 2：草稿引导的路由预执行与专家预取

**问题的诚实表述**：第 l 层 router 的真实输入是第 l 层的 hidden state，草稿阶段拿不到。
EAGLE-2 的 draft feature f ≈ 目标模型**倒数第二层** hidden state 的近似。
因此需要一组轻量跨层预测器：

```
P_l: f[4096] → expert_logits_l[8]     l = 0..31
合并为一个 GEMM：f[64, 4096] @ Wp[4096, 256] → [64, 32, 8]   （Wp 仅 1M 参数）
```

- 训练：ShareGPT 上跑 Mixtral 前向，采集 (倒数第二层 hidden, 各层 top-2 label)，
  32 个 8 分类头，交叉熵，1 张卡半天可训完。**这是与 SP-MoE/MoE-SpeQ 的差异**：
  它们为预取训练独立预测网络，我们复用 EAGLE-2 已有特征，预测器只有 1M 参数；
- 输出聚合成每层专家位图 `prefetch_bitmap[32, 8]`（树内 64 节点做 OR + top-B 截断）；
- 预取执行（Stream 1）：
  - **权重全在 HBM**（80G 主配置）：预取退化为 L2 warm——对预测命中的专家权重段发
    `cp.async.bulk.prefetch.L2`（或读 1 byte/128B 的 dummy load kernel），
    收益预期温和（~5-10%），作为消融项；
  - **offload 配置**（BF16 专家 + host pin 内存）：`cudaMemcpyAsync` H2D 拉取预测专家到 GPU 环形缓冲。
    注意时延量级：单专家单层 352 MB @ PCIe Gen5 ≈5.5 ms，无法在层内隐藏 →
    预取必须**提前 ≥4 层流水**（第 l 层验证时拉第 l+4 层的预测专家），且只 offload 冷层；
    命中免 PCIe 等待——这是预取的主战场，论文报告 recall@B 与 TPOT 的关系曲线；
- 评估指标：预测 recall@2 / recall@4（每层）、预取命中率、错误预取带宽浪费比。

### 3.3 算子 3：预算约束的树验证路由（内嵌于算子 1 Kernel A）

逐层执行，输入本层真实 router 输出（非预测）：

1. 聚合分数：$s_e = \sum_{n=0}^{63} \; p_{\text{accept}}(n) \cdot g_{n,e}$，
   其中 $p_{\text{accept}}(n)$ 是 EAGLE-2 树构建时的节点全局接受概率（现成的，无需额外计算），
   $g_{n,e}$ 是 token n 对专家 e 的 gating 权重。**用接受概率加权是与 MoE-Spec（均匀计数）的差异**：
   深层低概率分支的 token 反正大概率被拒绝，它们的专家需求不值得付权重读取；
2. 保留 $\text{TopB}(s)$ 专家集合 $\mathcal{K}$；
3. 重路由：token n 的 top-2 中被逐出的专家 → 替换为该 token 路由分布中 $\mathcal{K}$ 内得分最高者，
   gating 权重重新归一化（保证 $\sum g = 1$，避免输出幅值漂移）；
4. 低概率分支降级：$p_{\text{accept}}(n) < \tau$（默认 0.05）的节点直接 top-1 路由。

B 的选择：静态扫 B∈{3,4,5,6,8} 出接受率-时延 Pareto 曲线（论文主图）；
进阶：按上一步实际接受长度反馈自适应 B（简单 PI 控制器，CPU 侧更新，graph 外参数）。

**正确性红线**：预算路由改变了目标模型输出分布，严格的 speculative sampling 无损性不再成立。
论文处理方式（与 MoE-Spec 相同）：报告下游任务分数（GSM8K/HumanEval/MT-Bench judge）证明无统计显著退化，
并提供 B=8 无损模式作为对照。

### 3.4 算子 4：融合 验证-采样-KV压缩-提交 核

现状痛点：HF/朴素实现里这一段是 ~10 个小 kernel + 多次 GPU→CPU 同步（读接受长度），
每步浪费 50–200 μs 且阻断 CUDA Graph。HPC-Ops fused sampler（8.5x）证明融合采样类算子收益巨大，
但它不处理树。本算子 = fused sampler 的树验证扩展：

单 kernel（grid = 树路径数并行 + vocab 归约维），内部顺序：
1. logits 后处理：temperature / repetition penalty（融合，HPC-Ops 同款）；
2. online softmax（vocab=32000，两遍 max/sum 归约，SMEM 缓存热区）；
3. 树 rejection sampling：从根 DFS，逐节点 accept/reject（Philox 计数器随机数，
   种子固定可复现），拒绝时按残差分布 $\max(0, p-q)$ 采 bonus token；
4. 选出最长接受路径，写 `accepted_len`（GPU 标量，不回读 CPU）；
5. KV 压缩：接受路径的树槽位 KV → 按 `kv_remap_index` 写回主 KV cache 连续区
   （paged KV，block 内 index remap，避免整块搬运）；
6. 写 `next_root_feature[4096]`（接受路径末端 hidden，供下一步 EAGLE-2 起草）。

全部输出留在 GPU，配合固定树形状 → **整个 draft-verify-commit 循环单 CUDA Graph 重放**，
每步 CPU 开销 ≈ 一次 graph launch（~10 μs）。

### 3.5 非贡献组件的选型（不重造轮子）

| 组件 | 选型 | 理由 |
|---|---|---|
| tree attention | PyTorch SDPA + 显式 64×64 tree mask（prefill 用 flash-attn varlen） | N=64 的验证 attention 非瓶颈（<5% 时延），不值得自研 |
| KV cache | 自实现极简 paged KV（block=64，正好一棵树） | 结构需配合算子 4 的 remap，第三方难嵌入 |
| 权重加载 | safetensors 直读原生 BF16（无量化步骤） | 无框架依赖，数值与 HF 严格对齐 |
| EAGLE-2 草稿权重 | `yuhuili/EAGLE-mixtral-instruct-8x7B`（官方已发布；EAGLE-2 是推理时动态树，复用 EAGLE-1 权重无需重训） | 省 2–4 周训练 |
| 随机数 | Philox4x32（counter-based） | CUDA Graph 重放安全 |

---

## 4. 实验设计

**硬件**：主实验 H200-141G ×1（备选 2×H100-80G TP=2，专家 I 维切分）；A100 复验（无 TMA 路径回退 cp.async）。

**基线**：
1. HF transformers Mixtral AR（正确性锚点 + 最慢基线）；
2. 本框架 AR（无推测，隔离框架本身开销）；
3. 本框架 + EAGLE-2 + vLLM fused_moe kernel（隔离算子 1 的贡献）；
4. vLLM 官方 EAGLE+Mixtral、SGLang EAGLE（端到端外部基线，注明版本）。

**指标**：TPOT、每步接受长度 τ、端到端加速比、`dram__bytes_read`（Nsight）、
预取 recall/命中率、GSM8K/HumanEval/MT-Bench 分数（B<8 时的质量验证）。

**消融**（每算子独立开关）：+算子1 / +算子3(B扫描) / +算子4+CUDA Graph / +算子2(offload场景)；
树大小 32/64/128；（附录）FP8/INT4 量化对 τ 与 TPOT 的影响。

**风险与回退**：

| 风险 | 概率 | 回退 |
|---|---|---|
| Kernel B split-k 在 M=16 仍打不满带宽 | 中 | 双专家/CTA 绑定（E=8→4 CTA 组）+ 增大 BK |
| 预测器 recall@4 < 70%，预取无收益 | 中 | 算子 2 降级为"上一验证步激活集"启发式（成本零，论文改为对比两种信号） |
| 预算路由伤接受率（τ 掉 >15%） | 低 | τ 阈值分支降级关闭，只保留 top-B |
| BF16 全模型显存吃紧（H200 不可得 / TP=2 通信开销大） | 中 | 冷层专家 offload（配置 B）+ 算子 2 预取兜底；TP=2 时 all-reduce 与算子 4 重叠 |
