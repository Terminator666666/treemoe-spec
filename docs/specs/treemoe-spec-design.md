# TreeMoE-Spec 当前实现规格（v4）

> 一个不依赖 vLLM/SGLang 的 MoE 推测解码原型，目标模型为 Mixtral-8x7B-Instruct，草稿模型使用
> EAGLE-2 动态树。当前论文贡献收敛为两个已接入端到端路径的机制：接受概率感知的层内专家选择，
> 以及全局传输约束下的层自适应专家预算。树感知 MoE 核、可修复权重流送、greedy 树验证和 KV
> 提交是承载上述机制的系统实现优化，不作为独立算法创新。
> 参考库：Tencent HPC-Ops、DeepSeek DeepGEMM、Databricks MegaBlocks、FlashInfer、vLLM fused_moe。

### 实现边界

| 模块 | 当前状态 | 论文口径 |
|---|---|---|
| 接受概率感知的层内专家选择 | 已内嵌路由核，B=8 可回到原始 top-2 | 核心贡献 1 |
| 全局传输约束下的层自适应专家预算 | 已实现在线分配、EMA、计划位图和传输计数；4090 对照实验待跑 | 核心贡献 2 |
| EAGLE feature router hint | 已完成 pilot，但相对 temporal-only 仅改善 0.74% TPOT，且未改变聚合命中率 | 失败消融，不作为贡献 |
| 树感知 expert-stationary MoE | 已实现并接入，承载预算后的专家分桶和计算 | 系统实现优化，不单列创新 |
| greedy 树验证与 KV 提交 | 已实现 Triton 三核路径 | 系统实现优化 |
| 训练式 `RouterPredictor` | 仅有实验类和训练脚本，未训练、未加载到运行时 | 不作为论文贡献或实验配置 |
| 全步 CUDA Graph | `StepGraph` 仅为原型，未接入；当前循环仍有 Python 建树和主机读回 | 不纳入性能结论，列为未来工作 |

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

**实际硬件与显存布局**：主实验平台为单张 RTX 4090 24GB。BF16 模型约 93GB，专家权重常驻
host pinned memory；每层按预算预取到 GPU 环形缓冲，缺失专家在 GEMM 前同步修复。项目未实现 TP、
H200 常驻权重路径或 FP8/INT4 量化，论文不报告这些配置的实测结果。

### 1.2 核心矛盾（论文第 3 章观测，需实测填数）

自回归 decode 每步激活 2/8 专家/层，读专家权重 ≈ 2×32×0.352 ≈ 22.5 GB（BF16）。
EAGLE-2 树验证一次前向送入 N=64 个 token，**64 个 token 的 top-2 路由并集在每层几乎覆盖全部 8 个专家**
（Cascade, arXiv:2506.20675 报告 2–3x 权重读放大）。此时：

- 验证一步最多搬运约 90 GB 专家权重（全专家、BF16）。在 4090 offload 配置中，瓶颈是 PCIe H2D，
  因而预算 B 和预取命中率直接决定 TPOT；
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
| SP-MoE / MoE-SpeQ | 专家 offload 预取 | 本项目联合决定各层预算与预取位图，并以同步 repair 保证权重读取正确 |
| HPC-Ops | split-k decode GroupGEMM、fused sampler、Route GEMM | 无推测解码/树感知；Megakernel 在 roadmap 未实现 |
| DeepGEMM | masked grouped GEMM（CUDA Graph 友好） | 面向通用 grouped GEMM；不负责本项目的树节点路由、预算与修复 |

---

## 2. 系统架构

```text
Python 控制循环
  EAGLE-2 分层扩展 + 动态树构建
    └── tree tokens / mask / accept_prob
                    ↓
  上一轮目标路由需求 → EMA → 全局约束分配 {B_l, prefetch bitmap}
        ↓
  目标模型 tree forward（主流） || 专家 H2D 预取（侧流）
    attention + 接受概率感知 MoE（逐层 B_l，缺失权重同步 repair）
                    ↓
  Triton argmax → greedy 树验证 → KV commit
                    ↓
  D2H 取回接受数、accepted tokens 和 bonus token，进入下一轮
```

树节点数 N 和最大深度 D 固定，便于 Triton kernel 专门化与缓冲复用。当前实现没有捕获整步 CUDA Graph：
建树使用 Python 容器，`SpecDecodeEngine.step()` 还会把接受结果转成主机列表。

---

## 3. 算子详细设计

### 3.1 算子 1：树感知专家驻留 MoE 核（系统实现）

**签名（当前多阶段 Triton 方案）**

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

**阶段 A — exact router + fused budget/demand/bucket**
1. router 使用与 HF 相同的 BF16 `F.linear`，其 BF16 logits 转 FP32 后 softmax。不能在单 CTA 内另算
  router GEMM：Triton tensor-core 与 cuBLAS 的合法 reduction 顺序不同，在边界样本上会翻转 top-k；
2. `_budget_bucket_fused_kernel` 读取 FP32 gates，在单 CTA 内聚合完整 demand 并执行算子 3 的预算路由；
3. 同一 CTA 完成稳定 bucket，产出
   - `sorted_token_ids[128]`：按 (expert, DFS序) 排序的 token 槽位
   - `expert_offsets[E+1]`：每专家 token 段的前缀和
   - `num_tokens_per_expert[8]`：由 GPU 写出，MoE 路由和分桶阶段不需要逐专家 CPU 读回。

**阶段 B/C — expert-stationary GEMM1/GEMM2（主计算核）**

当前实现分两阶段计算原生 Mixtral FFN：GEMM1 同时读取 w1/w3，融合 SiLU 与逐元素乘法后把 BF16
中间激活写入预分配的 `h` workspace；GEMM2 紧接着读取 `h` 与 w2。代码通过 L2
`evict_last` 提示促进该短生命周期 workspace 的复用，但不声称中间激活完全驻留片上。

网格与调度（细节）：
- grid = (E=8, SPLIT_K=14336/BK_SPLIT)。**split-k 是小 M 场景的占用率关键**：
  M_e≈16 时仅按 token 维切分会产生太少 CTA；当前 4090 配置采用 `SPLIT_K=2`，partial 结果用
  轻量 combine kernel 归约；
- **专家驻留（weight-stationary）**：每个 CTA 绑定一个专家，w1/w3/w2 以 BF16 packed load 流式读取，
  对该专家名下所有 token（≤64）复用——权重读一遍服务多 token，
  而 token-stationary 布局（vLLM 默认）在小 M 下会重复读权重；
- token 段按 DFS 序排列 → 同一专家内的树节点父子相邻，x 的加载有 L2 局部性；
- 权重 BF16 直读，无反量化 epilogue（kernel 更简单、数值路径与 HF 完全对齐；
  FP8/INT4 变体仅作附录消融）；
- 空专家（num_tokens_per_expert[e]==0）：CTA 读到 0 立即退出（masked 语义，形状恒定）。

确定性路径把 split-K 和 top-2 槽位 partial 写入 FP32 workspace，再由 combine kernel 按固定顺序归约；
性能路径使用 atomic add。论文分别报告两条路径，不把尚未实现的持久化巨核列为贡献。

**验收基准**：`benchmarks/bench_op1.py` 测 N∈{32,64,128} 的确定性与 atomic 路径，报告时延、
按权重字节估算的有效带宽和峰值利用率。安装 vLLM 时额外报告同一输入输出边界下的 `fused_moe`；
当前环境没有完成 MegaBlocks、DeepGEMM 或 CUTLASS 的公平实测，不把它们列入最终结果表。

### 3.2 核心贡献：全局传输约束下的层自适应专家预算

统一的标量 B 默认假设 32 个 MoE 层具有相同的预算收益曲线。实际上一些层的需求集中在少数专家，继续增加
预算收益很小；另一些层分布平坦，少一个专家就会丢掉较多接受概率质量。因此系统在固定总预取量下在线决定
$B_l$，而不是给每层相同预算。

第 t 轮 verification 在预算裁剪前直接复用 Kernel A 已算出的完整需求，不增加 router GEMM：

$$d^{(t)}_{l,e}=\sum_n p_{\mathrm{accept}}(n)g^{(t)}_{l,n,e},\qquad
q^{(t)}_{l,e}=d^{(t)}_{l,e}/\sum_jd^{(t)}_{l,j}.$$

对 $q$ 做 EMA 后，将每层专家按需求降序记为 $q_{l,(1)}\ge\cdots\ge q_{l,(8)}$。给定平均预算
$B_{avg}$ 和下界 $B_{min}=2$，分配器求解：

$$\max_{B_1,\ldots,B_L}\sum_l\sum_{k=1}^{B_l}\bar q_{l,(k)},\quad
B_{min}\le B_l\le8,\quad\sum_lB_l=L B_{avg}.$$

实现先给每层 $B_{min}$，再按归一化边际收益 $\bar q_{l,(k)}$ 全局降序分配剩余 expert row。该离散问题
具有前缀收益递减结构，因此贪心分配得到最优整数解，并且严格满足总计划传输量。上一轮需求形成第 t+1 轮
的 `LayerBudgetPlan`：`budgets[l]` 控制本层接受概率感知路由，需求排名前 $B_l$ 的专家形成同一计划的
prefetch bitmap。每次只需回传 $L\times E=256$ 个 FP32 数，即 1 KiB。

**因果与边界规则**：当前层需求只有执行目标 router 后才真实可知，因此本轮不能用本轮所有层需求反过来
决定自身预算。首个 verification 使用统一 $B_{avg}$ 且全量预取；之后使用上一轮计划。不同 prompt 之间
重置 EMA。prompt prefill 始终使用 B=8 和全量权重，预算近似只作用于 verification。

**精确流送**：计划预取量在稳态严格为 $\sum_lB_l=L B_{avg}$ 个 expert row。host pinned 权重在侧流送入
深度为 2 的环形缓冲；若当前真实路由集合与上一轮计划不一致，`repair()` 在 expert GEMM 前同步补拷贝。
因此实际 H2D 成本为“计划 staged bytes + repair bytes”，后者是优化是否有效的关键指标。repair 保证 GEMM
不读取陈旧权重，但 B<8 的路由近似仍会改变模型 logits。

旧 EAGLE feature hint 将同一个 draft final-like feature 输入 32 个 target 中间层 router，分布语义并不匹配；
而且 temporal 集合已占满 B 时，temporal-first capped merge 不会改变 staged 集合。两 prompt pilot 中它相对
temporal-only 只改善 0.74% TPOT，聚合命中率同为 0.650，因此降为失败消融。系统不训练预测器，也不使用
CUDA Graph。

### 3.3 核心贡献：接受概率感知的层内专家选择（内嵌于算子 1 Kernel A）

逐层执行，输入本层真实 router 输出（非预测）：

1. 聚合分数：$s_e = \sum_{n=0}^{63} \; p_{\text{accept}}(n) \cdot g_{n,e}$，
   其中 $p_{\text{accept}}(n)$ 是 EAGLE-2 树构建时的节点全局接受概率（现成的，无需额外计算），
   $g_{n,e}$ 是 token n 对专家 e 的 gating 权重。**用接受概率加权是与 MoE-Spec（均匀计数）的差异**：
   深层低概率分支的 token 反正大概率被拒绝，它们的专家需求不值得付权重读取；
2. 保留 $\text{TopB}(s)$ 专家集合 $\mathcal{K}$；
3. 重路由：token n 的 top-2 中被逐出的专家 → 替换为该 token 路由分布中 $\mathcal{K}$ 内得分最高者，
   gating 权重重新归一化（保证 $\sum g = 1$，避免输出幅值漂移）；
4. 可选近似：$p_{\text{accept}}(n) < \tau$ 的节点退化为 top-1。正式主实验默认 $\tau=0$；
  $\tau=0.05$ 只作为消融，因为它会改变验证 logits，却不能减少已 staged 的 PCIe 字节。

B 的选择：静态扫描统一 B∈{2,3,4,5,6,8} 得到基线 Pareto 曲线；自适应模式在相同
$L B_{avg}$ 计划传输量下比较，不通过多搬专家换取质量或接受长度。

**正确性红线**：预算路由改变了目标模型输出分布，严格的 speculative sampling 无损性不再成立。
论文处理方式（与 MoE-Spec 相同）：报告下游任务分数（GSM8K/HumanEval/MT-Bench judge）证明无统计显著退化，
并提供 B=8 无损模式作为对照。

### 3.4 算子 4：greedy 树验证与 KV 提交

正式实验使用 temperature=0。GPU 路径由三个阶段组成：

1. 每个树节点对 logits 做 Triton argmax；greedy 模式不计算 softmax；
2. 单 program 沿树执行串行 greedy 验证，输出接受槽位、bonus token 和接受数；
3. KV commit kernel 将根节点和接受路径从 tree scratch block 写入 paged KV tail。

temperature>0 的 rejection sampling 仍走 PyTorch 参考实现，不属于当前性能实验。引擎随后用一次 `.tolist()`
取回接受 token 和 bonus token，以便 Python 处理 EOS 和输出列表。因此本模块减少了逐节点同步，但没有做到
零 CPU 同步，也没有形成单 kernel 或整步 CUDA Graph。

### 3.5 非贡献组件的选型（不重造轮子）

| 组件 | 选型 | 理由 |
|---|---|---|
| tree attention | PyTorch SDPA + 显式 64×64 tree mask | 沿用 PyTorch 后端分派；本项目不把自研 attention kernel 列为贡献 |
| KV cache | 自实现极简 paged KV（block=64，正好一棵树） | 结构需配合算子 4 的 remap，第三方难嵌入 |
| 权重加载 | safetensors 直读原生 BF16（无量化步骤） | 无框架依赖，数值与 HF 严格对齐 |
| EAGLE-2 草稿权重 | `yuhuili/EAGLE-mixtral-instruct-8x7B`（官方已发布；EAGLE-2 是推理时动态树，复用 EAGLE-1 权重无需重训） | 省 2–4 周训练 |
| 随机数 | PyTorch generator（仅 temperature>0 参考路径） | 正式实验使用 greedy，不进入采样核 |

---

## 4. 实验设计

**硬件**：主实验为 RTX 4090 24GB ×1，专家权重通过 PCIe Gen4 ×16 从 host pinned memory 按层加载。

**基线**：
1. HF transformers Mixtral AR（正确性锚点 + 最慢基线）；
2. 本框架 AR（无推测，隔离框架本身开销）；
3. 本框架 + EAGLE-2，关闭预取位图并全量搬运专家（隔离预取收益）。

**指标**：TPOT、每步接受长度 τ、端到端加速比、每层预算分布、计划 staged rows/bytes、同步
repair rows/bytes，以及 B<8 时的
MT-Bench 质量分数。当前 AutoDL 容器没有 Nsight Compute 计数器权限，不把 `dram__bytes_read` 作为必填实测项。

**消融**：在 $B_{avg}\in\{3,4,6\}$ 下比较 `--uniform-layer-budget` 与
`--adaptive-layer-budget`，二者使用相同的 demand bitmap、首轮策略和 repair，且总计划预算完全相同；扫描
EMA decay 和 $B_{min}$；树大小 16/32/64；full-copy 作为传输上界；top-1 阈值 0/0.05。旧 hint 结果只作为
失败 pilot。项目不报告 CUDA Graph、训练预测器或量化消融。最终配置和待测结果统一维护在
`measurements/final_experiments.csv`。

**风险与回退**：

| 风险 | 概率 | 回退 |
|---|---|---|
| Kernel B split-k 在 M=16 仍打不满带宽 | 中 | 双专家/CTA 绑定（E=8→4 CTA 组）+ 增大 BK |
| 相邻 verification 的层需求漂移导致 repair 过多 | 中 | `repair()` 保证权重正确；报告额外字节并扫描 EMA，收益不足则回退 uniform |
| 预算路由伤接受率（τ 掉 >15%） | 低 | τ 阈值分支降级关闭，只保留 top-B |
| BF16 全模型超出 4090 显存 | 已发生 | 全部专家权重放在 host pinned memory，按层预取和修复 |
