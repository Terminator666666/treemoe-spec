# TreeMoE-Spec 当前实现规格与候选方案（v7）

> 一个不依赖 vLLM/SGLang 的 MoE 推测解码原型，目标模型为 Mixtral-8x7B-Instruct，草稿模型使用
> EAGLE-2 动态树。新的候选主线是**渐进式精确 target tree verification**：把完整草稿树拆成嵌套、
> 前缀闭合的 stage tree，先精确执行高接受概率核心，仅当真实 target 决策沿边界继续接受时才扩展。
> 它不裁专家、不改变验证规则。当前已完成 full-trace 反事实模拟器，运行时改造须在 RTX 4090 的 B=8 trace
> 通过传输收益门槛后才启动。接受概率加权专家预算与现有 MoE-Spec/EcoSpec 的技术边界不足，降为对照。
> route 后精确 JIT 权重流送、树感知 MoE 核、greedy 树验证和 KV 提交均按系统实现报告。
> 参考库：Tencent HPC-Ops、DeepSeek DeepGEMM、Databricks MegaBlocks、FlashInfer、vLLM fused_moe。

### 实现边界

| 模块 | 当前状态 | 论文口径 |
|---|---|---|
| 渐进式精确 target tree verification | 已实现离线 stage planner 与 full-trace 成本重放；运行时尚未接入 | 候选核心贡献，先过收益门槛 |
| 接受概率感知的层内专家选择 | 已内嵌路由核，B=8 可回到原始 top-2 | 与 MoE-Spec/EcoSpec 边界不足，降为对照 |
| 关键路径风险保护 | 参考实现与 fused Kernel A 已接入，但真机未改善接受长度且增加 H2D 等待 | 失败消融，不作为贡献 |
| 全局传输约束下的层自适应专家预算 | 计算预算版本恶化接受长度；固定计算的预取回放也未降低 repair | 失败消融，不作为贡献 |
| EAGLE feature router hint | 已完成 pilot，但相对 temporal-only 仅改善 0.74% TPOT，且未改变聚合命中率 | 失败消融，不作为贡献 |
| route 后精确 JIT staging | 当前层 router 确定实际专家集后在计算流只搬这些 BF16 行；将承载多 stage 的新增节点 | 系统实现优化，不单列创新 |
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
host pinned memory；每层 route 后将实际选中的专家精确搬入 GPU 环形缓冲。项目未实现 TP、
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
| EcoSpec | 按边际专家并集代价选择草稿节点，构造专家足迹较小的树 | 构树后仍一次执行 target；没有依据真实 target 决策渐进扩展精确验证 |
| Staged Speculative Decoding | 分阶段投入 draft 计算 | stage 位于 draft 侧，不是 target tree 的部分执行与续算 |
| SP-MoE / MoE-SpeQ | 专家 offload 预取 | 本项目在低重叠环境中放弃跨步预测，route 后只搬当前层实际专家集 |
| HPC-Ops | split-k decode GroupGEMM、fused sampler、Route GEMM | 无推测解码/树感知；Megakernel 在 roadmap 未实现 |
| DeepGEMM | masked grouped GEMM（CUDA Graph 友好） | 面向通用 grouped GEMM；不负责本项目的树节点路由、预算与修复 |

---

## 2. 系统架构

```text
Python 控制循环（候选运行时；当前实现仍一次 forward 完整树）
  EAGLE-2 分层扩展 + 动态树构建
    └── tree tokens / mask / accept_prob
                    ↓
  T1 ⊂ T2 ⊂ ... ⊂ TK = T（每个 stage 前缀闭合）
    对 ΔTk 续算 target layers，复用祖先 tree KV
      → 当前层 exact router(B=8) → 实际专家 ID D2H
      → 当前流精确 H2D staging → expert GEMM
                    ↓
  partial greedy verify：停止，或由真实 target 决策触发下一 stage
                    ↓
  最终 KV commit
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

生产 fused 路径覆盖 N≤64。N=128 的二次稳定 rank 矩阵在 sm_89 上产生约 0.5KB/thread spill，并观察到
跨 warp argmax 错选，因此明确回退到同 gates 的 PyTorch bucket；正式端到端实验只使用 N≤64。

**阶段 B/C — expert-stationary GEMM1/GEMM2（主计算核）**

当前实现分两阶段计算原生 Mixtral FFN：GEMM1 同时读取 w1/w3，融合 SiLU 与逐元素乘法后把 BF16
中间激活写入预分配的 `h` workspace；GEMM2 紧接着读取 `h` 与 w2。代码通过 L2
`evict_last` 提示促进该短生命周期 workspace 的复用，但不声称中间激活完全驻留片上。

网格与调度（细节）：
- Kernel A 将各专家 slot block 紧致排列，不再为每个专家预留 `ceil(2N/BM)` 私有区域。对 S=2N 个 slot、
  最多 B 个非空专家，block 数上界为 $B+\lfloor(S-B)/BM\rfloor$，再向上取 2 的幂以满足 Triton 静态
  `arange`。buffer capacity 与 GEMM launch 上界分别保存：N=64、BM=16 时 B2/B4/B8 的 launch 上界为
  9/11/15，B8 buffer capacity 为 16；相对旧实现的 64 个 block-row，h workspace 从 1024 行降到
  256 行，deterministic partial 同比例缩小；
- `bench_op1.py --block-m 32` 保留为负消融。N=64、B8 真机上 active block 从 13 降至 8，但
  atomic 总时延从 3065.8 μs 恶化至 3119.8 μs（+1.8%），GEMM1 从 2034 μs 恶化至 2078 μs。
  BM32 使用 16 warps，sm90a 静态检查为 25% occupancy、0 spill，故退化不是 spill 所致；结果表明
  BM16 重复 block 的权重 load 已由 L2/并发复用，减少 logical load 并未减少 compulsory DRAM bytes。
  默认保持 BM16；
- GEMM2 采用 `SPLIT_K=2`。**split-k 是小 M 场景的占用率关键**：M_e≈16 时仅按 token 维切分会产生
  太少 CTA；partial 结果用轻量 combine kernel 归约；
- **专家驻留（weight-stationary）**：每个 CTA 绑定一个专家，w1/w3/w2 以 BF16 packed load 流式读取，
  对该专家名下所有 token（≤64）复用——权重读一遍服务多 token，
  而 token-stationary 布局（vLLM 默认）在小 M 下会重复读权重；
- token 段按 DFS 序排列 → 同一专家内的树节点父子相邻，x 的加载有 L2 局部性；
- 权重 BF16 直读，无反量化 epilogue（kernel 更简单、数值路径与 HF 完全对齐；
  FP8/INT4 变体仅作附录消融）；
- 空专家（num_tokens_per_expert[e]==0）：CTA 读到 0 立即退出（masked 语义，形状恒定）。

确定性路径把 split-K 和 top-2 槽位 partial 写入 FP32 workspace，再由 combine kernel 按固定顺序归约；
性能路径使用 atomic add。端到端默认保持确定性路径；`--atomic-moe` 仅用于测量性能上限，并禁止与
lossless red-line 同时使用。论文分别报告两条路径，不把尚未实现的持久化巨核列为贡献。

**验收基准**：`benchmarks/bench_op1.py` 测 N∈{32,64,128} 的确定性与 atomic 路径，报告时延、
active expert/block/cap、每个活跃专家读一次的权重流量估计，以及按 active block 统计的逻辑 load 流量。
后者包含可命中 L2 的重复加载，因此不得解释为 DRAM 带宽或峰值利用率；物理 HBM bytes 以 Nsight Compute
计数器为准。安装 vLLM 时额外报告同一输入输出边界下的 `fused_moe`；
当前环境没有完成 MegaBlocks、DeepGEMM 或 CUTLASS 的公平实测，不把它们列入最终结果表。

### 3.2 候选贡献：渐进式精确 target tree verification

给定 EAGLE-2 已构造的完整树 $T$，规划嵌套且前缀闭合的节点集合：

$$T_1\subset T_2\subset\cdots\subset T_K=T,\qquad
n\in T_k\Rightarrow\operatorname{parent}(n)\in T_k.$$

候选 planner 依节点接受概率降序加入节点及其尚未加入的祖先，stage node budget 默认为 8/16/32，最后强制
补全整树。breadth-first 基线按 `(depth, node_id)` 扩展；二者共用相同 budgets、停止规则和成本模型，用于区分
“渐进执行本身”的收益与接受概率排序的增益。该规则只决定 target **何时执行节点**，不改变草稿树、候选
token、target router 或验证规则。依赖未执行节点真实 target route 的 resident-aware 选择只能作为离线 oracle；
不能在线实现，也不能把 EcoSpec 已使用的边际专家并集重新表述成独立创新。

运行语义如下：第一 stage 只把 $T_1$ 的节点送入全部 target layers。每层保存这些节点的 tree K/V；后续
stage 仅对 $\Delta T_k=T_k\setminus T_{k-1}$ 续算，并令新节点注意其已经计算的祖先。partial verifier 使用
已计算父节点 logits 和完整树中已知的 child token 判断下一条 greedy 边：若发生拒绝，立即停止；若下一接受
child 尚未执行，扩展到包含它的最小后续 stage；若接受路径已覆盖，则提交。不能在每个 stage 提前 commit KV，
因为后续扩展与拒绝仍可能改变最终提交长度。

**精确性条件**：正式路径固定 B=8，不做专家预算、量化或 logits 近似。树 attention 中节点只依赖已提交前缀
及其祖先。由层数归纳，祖先 K/V 与一次性完整树 forward 相同，因此每个新增节点的各层 hidden state 和最终
logits 也相同；partial verifier 只延迟观察相同 logits，最终 accepted path、bonus token 和 committed KV
必须逐 token/逐槽位等于 one-shot B=8。任何 B<8 trace 的后层 hidden state 已被预算路由改变，只能调试，
不得用于这一精确性或收益结论。

对层 $l$、节点 $n$ 的自然 top-2 专家集合记作 $E_l(n)$，单专家权重字节为 $W_l$。one-shot 与当前可实现
的 exact-JIT progressive 传输估计分别为：

$$C_{one}=\sum_l W_l\left|\bigcup_{n\in T}E_l(n)\right|,$$

$$C_{prog}=\sum_{k=1}^{K^*}\sum_l W_l
\left|\bigcup_{n\in\Delta T_k}E_l(n)\right|.$$

$K^*$ 由 trace 中的真实接受路径决定。第二式允许同一层专家跨 stage 重传，符合当前深度为 2 的 staging ring；
把已执行 stage 的同层专家永久驻留所得并集只报告为不可直接实现的 optimistic lower bound。模拟器还报告
重复行、平均 stage 数、完整树触达率、按实测 PCIe 带宽折算的节省时间，以及每个额外 stage 可容忍的
break-even 固定开销。

离线准入命令：

```bash
PYTHONPATH=. python3 benchmarks/simulate_progressive_verification.py \
  artifacts/execution_trace_b8.json \
  --strategies probability depth \
  --stage-grid 8,16,32 --stage-grid 8,24,40 \
  --pcie-gbps 25.03 --extra-stage-overhead-ms 5
```

进入运行时实现前必须同时满足：在足量 MT-Bench B=8 full trace 上，$C_{prog}$ 相对 $C_{one}$ 至少降低 20%；
按 trace 的 target 非 H2D 时间加 stage break-even 开销估算，TPOT 相对当前 one-shot exact JIT 不恶化超过 10%。
若未通过，保留模拟器和负结果，不引入 partial-forward/partial-commit 复杂度。通过后才修改 `MixtralForward`
以接收增量节点和祖先 KV、为 verifier 增加可恢复状态，并做 one-shot/progressive 的 logits、路径、KV 三重对齐。

### 3.3 失败消融：全局传输约束下的层自适应专家预算

统一的标量 B 默认假设 32 个 MoE 层具有相同的预算收益曲线。实际上一些层的需求集中在少数专家，继续增加
预算收益很小；另一些层分布平坦，少一个专家就会丢掉较多接受概率质量。因此系统在固定总预取量下在线决定
$B_l$，而不是给每层相同预算。

第 t 轮 verification 在预算裁剪前直接复用 Kernel A 已算出的完整需求，不增加 router GEMM：

$$d^{(t)}_{l,e}=\sum_n p_{\mathrm{accept}}(n)g^{(t)}_{l,n,e},\qquad
q^{(t)}_{l,e}=d^{(t)}_{l,e}/\sum_jd^{(t)}_{l,j}.$$

对 $q$ 做 EMA 后，将每层专家按需求降序记为 $q_{l,(1)}\ge\cdots\ge q_{l,(8)}$。给定平均预算
$B_{avg}$ 和信赖域 $[B_{min},B_{max}]$，分配器求解：

$$\max_{B_1,\ldots,B_L}\sum_l\log\left(\sum_{k=1}^{B_l}\bar q_{l,(k)}\right),\quad
B_{min}\le B_l\le B_{max},\quad\sum_lB_l=L B_{avg}.$$

该目标等价于最大化各层 retained router mass 的乘积，避免线性总质量允许“一层严重受损、另一层过量补偿”
这一不符合串行网络误差传播的行为。实现先给每层 $B_{min}$，再按对数边际收益
$\log(R_l+q)-\log R_l$ 全局降序分配剩余 expert row；边际收益单调递减，因此贪心仍得到精确整数解。
旧线性 mass 目标通过 `--layer-budget-objective mass` 保留为失败消融。默认
$B_{max}=8$；保守配置可限制 $B_l\in[B_{avg}-1,B_{avg}+1]$，避免少数层过度增配并迫使大量层降至
低预算。该离散问题
具有前缀收益递减结构，因此贪心分配得到最优整数解，并且严格满足总计划传输量。上一轮需求形成第 t+1 轮
的 `LayerBudgetPlan`：`budgets[l]` 控制本层接受概率感知路由，需求排名前 $B_l$ 的专家形成同一计划的
prefetch bitmap。每次只需回传 $L\times E=256$ 个 FP32 数，即 1 KiB。

**真机判定**：RTX 4090 的单提示 B=4 对照中，uniform 达到 985.87 ms TPOT、3.10 接受长度和
46.5 repair rows/step。mass 信赖域 `[3,5]` 虽将 repair 降至 40.1 rows/step，接受长度降至 2.54，
TPOT 增至 1182.39 ms；log-mass 进一步得到 41.4 repair rows/step、2.43 接受长度和 1282.44 ms TPOT。
因此“更高 retained router mass 会保持 draft-target 一致性”这一代理假设被否定。当前实现保留用于负结果复现，
不进入正式主实验。进一步在完全固定 uniform B4 计算与真实专家集合的 trace 上离线回放，仅重分配预取行数：
uniform 预测得到 46.50 repair rows/step，与真机计数完全一致；全局 `[3,5]` 方案得到 46.60 rows/step，
恶化 0.2%，且每个预测步平均改变 19.8 层预算。这排除了“只需将计算预算与预取预算解耦”的解释，也说明
上一轮 demand 的跨层边际不可预测下一轮的 repair 收益。至此停止 demand-only 层预算路线，不再作为候选创新。

**因果与边界规则**：当前层需求只有执行目标 router 后才真实可知，因此不能用本轮所有层需求反过来决定
自身预算。不同 prompt 之间重置 EMA。prompt prefill 始终使用 B=8；预算近似只作用于 verification。

**精确流送**：默认路径不再猜测下一轮专家。每层 attention 和 router 完成后，`prepare_experts()` 将实际
路由集合对应的 host pinned BF16 行在当前 CUDA stream 搬入深度为 2 的环形缓冲，随后同一 stream 启动
expert GEMM。流顺序同时保证 use-after-copy 和 overwrite-after-use，无预测错行和同步 repair；稳态最多搬运
$\sum_l B_l$ 个 expert row，若预算集合中有专家未被任何节点实际使用则更少。该系统路径保持给定 B 下的
路由和数值计算不变，但 B<8 本身仍会改变模型 logits。旧 temporal/hint 侧流预取加 `repair()` 仅通过
`--predictive-prefetch` 保留，用于同输入 A/B 和历史结果复现。

旧 EAGLE feature hint 将同一个 draft final-like feature 输入 32 个 target 中间层 router，分布语义并不匹配；
而且 temporal 集合已占满 B 时，temporal-first capped merge 不会改变 staged 集合。两 prompt pilot 中它相对
temporal-only 只改善 0.74% TPOT，聚合命中率同为 0.650，因此降为失败消融。系统不训练预测器，也不使用
CUDA Graph。

### 3.4 重合基线：接受概率感知的层内专家选择（内嵌于算子 1 Kernel A）

逐层执行，输入本层真实 router 输出（非预测）：

1. 聚合分数：$s_e = \sum_{n=0}^{63} \; p_{\text{accept}}(n) \cdot g_{n,e}$，
   其中 $p_{\text{accept}}(n)$ 是 EAGLE-2 树构建时的节点全局接受概率（现成的，无需额外计算），
   $g_{n,e}$ 是 token n 对专家 e 的 gating 权重。**用接受概率加权是与 MoE-Spec（均匀计数）的差异**：
   深层低概率分支的 token 反正大概率被拒绝，它们的专家需求不值得付权重读取；
2. mass 基线保留 $\text{TopB}(s)$ 专家集合 $\mathcal{K}$；
3. 重路由：token n 的 top-2 中被逐出的专家 → 替换为该 token 路由分布中 $\mathcal{K}$ 内得分最高者，
   gating 权重重新归一化（保证 $\sum g = 1$，避免输出幅值漂移）；
4. 可选近似：$p_{\text{accept}}(n) < \tau$ 的节点退化为 top-1。正式主实验默认 $\tau=0$；
  $\tau=0.05$ 只作为消融，因为它会改变验证 logits，却不能减少已 staged 的 PCIe 字节。

B 的选择：静态扫描统一 B∈{2,3,4,5,6,8} 得到基线 Pareto 曲线。已否定的跨层自适应模式
不进入正式主实验。

**正确性红线**：预算路由改变了目标模型输出分布，严格的 speculative sampling 无损性不再成立。
论文处理方式（与 MoE-Spec 相同）：报告下游任务分数（GSM8K/HumanEval/MT-Bench judge）证明无统计显著退化，
并提供 B=8 无损模式作为对照。

### 3.5 失败消融：关键路径风险保护

mass 目标优化所有树节点的加权平均，却可能让大量外围节点的累计需求挤掉根节点或高概率路径的原始 top-2
专家。对每个专家定义尾部风险：

$$c_e=\max_{n:e\in\operatorname{Top2}(g_n)}p_{\mathrm{accept}}(n).$$

若节点低于 top-1 降级阈值，其第二专家不计入 $c_e$。`critical_path` 策略按 $(c_e,s_e)$ 词典序降序选择
B 个专家：先覆盖可能被接受的最高概率节点，再用贡献 1 的 acceptance-weighted mass 打破同风险专家的平局。
因为根节点 $p_{\mathrm{accept}}=1$ 且 B≥2，其原始 top-2 必然保留；B=8 时严格退化为原始 Mixtral top-2。
该选择与 demand、稳定 bucket 一起内嵌在 Kernel A，`CRITICAL_PATH` 是 Triton `constexpr`，因此 mass 对照
不会执行 criticality 逻辑。

真机采用同一 prompt、B=4、N=64、uniform 预算和 temporal prefetch，只切换
`--routing-objective mass|critical_path`。critical-path 未提高接受长度，并把逐步 repair 等待从
2075.3 ms 增至 2148.5 ms；target 总时间从 2764.8 ms 增至 2774.7 ms。因此它未通过保留门槛，
不进行 20-prompt 正式实验。

### 3.6 算子 4：greedy 树验证与 KV 提交

正式实验使用 temperature=0。GPU 路径由三个阶段组成：

1. 每个树节点对 logits 做 Triton argmax；greedy 模式不计算 softmax；
2. 单 program 沿树执行串行 greedy 验证，输出接受槽位、bonus token 和接受数；
3. KV commit kernel 将根节点和接受路径从 tree scratch block 写入 paged KV tail。

temperature>0 的 rejection sampling 仍走 PyTorch 参考实现，不属于当前性能实验。引擎随后用一次 `.tolist()`
取回接受 token 和 bonus token，以便 Python 处理 EOS 和输出列表。因此本模块减少了逐节点同步，但没有做到
零 CPU 同步，也没有形成单 kernel 或整步 CUDA Graph。

### 3.7 非贡献组件的选型（不重造轮子）

| 组件 | 选型 | 理由 |
|---|---|---|
| tree attention | PyTorch SDPA + 显式 64×64 tree mask | 沿用 PyTorch 后端分派；本项目不把自研 attention kernel 列为贡献 |
| KV cache | 自实现极简 paged KV（block=64，正好一棵树） | 结构需配合算子 4 的 remap，第三方难嵌入 |
| 权重加载 | safetensors 直读原生 BF16（无量化步骤） | 无框架依赖，数值与 HF 严格对齐 |
| EAGLE-2 草稿权重 | `yuhuili/EAGLE-mixtral-instruct-8x7B`（官方已发布；EAGLE-2 是推理时动态树，复用 EAGLE-1 权重无需重训） | 省 2–4 周训练 |
| 随机数 | PyTorch generator（仅 temperature>0 参考路径） | 正式实验使用 greedy，不进入采样核 |

### 3.8 全阶段可观测性

`--execution-trace-json` 是默认关闭的诊断路径。它以延迟解析的 CUDA Events 记录引擎级和逐层 GPU 阶段，
并单独记录 host wall time；JIT H2D 位于 current stream，预测消融的侧流 prefetch 仍在自身 stream 上计时。
每步保存完整树拓扑、所有根路径、节点接受概率、实际接受路径，以及逐层 JIT/planned/staged/routed/missing
专家和 repair 字节。`benchmarks/analyze_execution_trace.py` 输出逐步路径、阶段均值
和 32 层热点。该模式包含额外事件与 D2H，不能用其 TPOT 作为正式性能数字。

渐进验证准入使用 `--execution-trace-detail progressive`：只保存树、接受路径、逐层每节点 natural top-2、
expert row bytes 和每步总时间，跳过 draft candidates、target top-8、完整 router 概率与逐层 CUDA Events。
同一进程先用独立 engine warmup，再用 `--execution-trace-baseline-first` 跑无 trace TPOT，最后采集瘦 trace；
两次运行共享已加载权重但不共享 KV、stats 或 staging ring。模拟器使用无 trace TPOT，并以实际截断后返回的
`generated_tokens` 归一化 verification 节省，避免最后一步超发 token 造成分母偏差。

---

## 4. 实验设计

**硬件**：主实验为 RTX 4090 24GB ×1，专家权重通过 PCIe Gen4 ×16 从 host pinned memory 按层加载。

**基线**：
1. HF transformers Mixtral AR（正确性锚点 + 最慢基线）；
2. 本框架 AR（无推测，隔离框架本身开销）；
3. 本框架 + EAGLE-2，旧 temporal prediction + repair（`--predictive-prefetch`）；
4. 本框架 + EAGLE-2，route 后精确 JIT staging（默认）。

**指标**：TPOT、每步接受长度 τ、target verification 次数、端到端加速比、每层预算分布、每个
verification 的实际 JIT rows、旧路径的计划/repair rows，以及包含 prefill 的总 H2D GiB/token。
总 staged GiB 会随接受长度导致的 verification 次数变化，不能单独用来判断固定全局预算是否成立。
另外报告 B<8 时的
MT-Bench 质量分数。当前 AutoDL 容器没有 Nsight Compute 计数器权限，不把 `dram__bytes_read` 作为必填实测项。

**消融**：在 $B_{avg}\in\{3,4,6\}$ 下比较 `--uniform-layer-budget` 与
`--adaptive-layer-budget`，二者使用相同的 demand bitmap、首轮策略和 repair，且总计划预算完全相同；扫描
EMA decay 和 $B_{min}$；树大小 16/32/64；predictive、full-copy 作为传输对照；top-1 阈值 0/0.05。
旧 hint 结果只作为失败 pilot。项目不报告 CUDA Graph、训练预测器或量化消融。最终配置和待测结果统一维护在
`measurements/final_experiments.csv`。

**风险与回退**：

| 风险 | 概率 | 回退 |
|---|---|---|
| Kernel B split-k 在 M=16 仍打不满带宽 | 中 | 双专家/CTA 绑定（E=8→4 CTA 组）+ 增大 BK |
| 精确 JIT 放弃侧流重叠后反而变慢 | 低 | 与 `--predictive-prefetch` 做同输入 A/B；保留旧路径作为回退 |
| 预算路由伤接受率（τ 掉 >15%） | 低 | τ 阈值分支降级关闭，只保留 top-B |
| BF16 全模型超出 4090 显存 | 已发生 | 全部专家权重放在 host pinned memory，按层精确 JIT staging |
