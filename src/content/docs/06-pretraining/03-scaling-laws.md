---
title: "6.3 Scaling Law：Kaplan / Chinchilla / DeepSeek + over-training 反思"
description: "LLM 时代最重要的\"信仰\"——训练 loss 是 compute / params / data 的可预测函数——从 Kaplan 2020 的 model-heavy 配方出发，被 Chinchilla 2022 的 IsoFLOP 实验颠覆为\"params : data ≈ 1 : 20\"，又被 DeepSeek 2024 的 batch-size-aware 修正、以及 LLaMA-3 /"
---

> ⏱ 预计阅读 50 分钟 ｜ 难度 ★★★ ｜ 前置：6.1（训练目标 = CLM）、6.2（数据 pipeline 与 quality 取舍）

## 一句话本节讲什么

LLM 时代最重要的"信仰"——**训练 loss 是 compute / params / data 的可预测函数**——从 Kaplan 2020 的 model-heavy 配方出发，被 Chinchilla 2022 的 IsoFLOP 实验颠覆为"params : data ≈ 1 : 20"，又被 DeepSeek 2024 的 batch-size-aware 修正、以及 LLaMA-3 / Qwen2.5 系列的 over-training 实战推翻成"train-time optimal ≠ deploy-time optimal"——本节把这条主线连同"emergent ability 是真是假"和"test-time compute 第二轴"一次讲清。

---

## 1. Mental model（直觉）

### 1.1 为什么 LLM 时代相信 scaling law

深度学习上半场（CNN / RNN 时代）一直有个困扰：**你不知道把 model 调大、数据加多、训练再久，效果会变成什么样**。每个新 size 都要重新调超参，sweep 成本巨大；很多结构创新拿到大模型上反而退化。

LLM 时代第一次给出了**可预测**的答案——给定参数量 $N$、训练 token 数 $D$、训练 compute $C$，你**能算出**最终的训练 loss $L$ 大约是多少，误差在百分之几以内。Kaplan 2020 的论文标题就是《Scaling Laws for Neural Language Models》——"law"这个词在 ML 里很少见，物理学色彩非常重。这条 law 的形式典型如下：

$$L(N) \approx \frac{a}{N^{\alpha}} + L_\infty$$

含义：随着模型参数量 $N$ 增大，loss 沿一条 **power law** 平滑下降，渐近到一个不可降的下限 $L_\infty$（数据自身的不可压缩熵）。在 log-log 坐标上，$\log(L - L_\infty)$ vs $\log N$ 是一条直线——这正是"law"的来源。同样形式也成立于 $L(D)$ 和 $L(C)$。

为什么这件事重要到值得用一节来讲？因为它**让 LLM 训练从工艺变成工程**：

- **预算决策可计算**——给老板报"再给我 1000 万美金算力，能训一个 loss 降 0.X 的 model"
- **小模型外推大模型**——花 1% 的算力跑一系列 1B / 3B / 7B 实验，**外推**到 70B 时的最优配比
- **架构改动可量化**——任何号称"涨点"的新结构，必须证明它把整条 scaling 曲线**整体下移**，否则很可能只是在某个 size 偶然好

如果没有 scaling law，"GPT-4 大约要花 X 亿美金"、"DeepSeek-V3 671B 选 14.8T token 是最优"这种决策都做不了——只能凭直觉。

### 1.2 三个时代的"配比哲学"

LLM scaling 史可以粗略分三个阶段：

```
═══════════════════════════════════════════════════════════════════
Kaplan 时代 (2020-2021)：model-heavy
   N ∝ C^0.73, D ∝ C^0.27
   "compute 翻倍，model 多放 7 倍、data 只多 3 倍"
   代表：GPT-3 175B + 300B token  (D/N ≈ 1.7)
═══════════════════════════════════════════════════════════════════
Chinchilla 时代 (2022-2023)：1:1 平衡
   N ∝ C^0.5, D ∝ C^0.5
   "compute 翻倍，model 和 data 各放 √2 倍"
   经验法则：D ≈ 20 N
   代表：Chinchilla 70B + 1.4T token  (D/N = 20)
        LLaMA-1 65B + 1.4T token    (D/N = 22)
═══════════════════════════════════════════════════════════════════
Over-training 时代 (2024-)：deploy-aware
   "train-time optimal 不等于 deploy-time optimal"
   推理是亿级 query，训完只一次 → 砸 data 喂小 model
   代表：LLaMA-3 8B + 15T token   (D/N ≈ 1875，远超 Chinchilla 7-8 倍)
        Qwen2.5 7B + 18T token   (D/N ≈ 2570)
═══════════════════════════════════════════════════════════════════
```

三段历史一句话总结：**Kaplan 教我们 scaling 是 power law，Chinchilla 教我们 model 和 data 同等重要，2024 年之后我们意识到——选 model size 必须把"训完之后要服务多少 query"算进去**。

### 1.3 本节关键概念地图

```
                 scaling law 主轴：训练 compute → loss
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
   Kaplan 2020              Chinchilla 2022         DeepSeek 2024
   model-heavy 配比         1:1 配比 + 20 token/p   修正 batch/lr 偏差
                                 │
                                 │
                       ┌─────────┴─────────┐
                       │                   │
                  train-time           deploy-time
                  optimal              optimal (over-training)
                                            │
                                       LLaMA-3 / Qwen2.5
                                       Sardana 2024 系统化
                                            │
                       ┌────────────────────┴────────────────────┐
                  emergent ability                        test-time scaling
                  (Wei 2022 提出)                          (R1 / o1 之后的第二轴)
                  (Schaeffer 2023 反驳)                    详见 Module 10
```

---

## 2. 公式与原理

### 2.1 scaling law 的基本形式

定义：$N$ = 非 embedding 参数量、$D$ = 训练 token 数、$C$ = 训练 compute（FLOPs）、$L$ = test cross-entropy loss（nats / token）。

**Kaplan 2020 的核心 fit**：在固定 data 充足、固定模型形状下，$L$ 关于 $N$ 是 power law：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N} \quad (\alpha_N \approx 0.076)$$

类似地有 $L(D) = (D_c / D)^{\alpha_D}$ 和 $L(C) = (C_c / C)^{\alpha_C}$。Kaplan 发现这三条曲线在很大动态范围（$N$ 跨 6 个数量级、$D$ 跨 3 个数量级、$C$ 跨 9 个数量级）内拟合精度都很好，**整套 scaling law 可以从几十个小实验外推到 GPT-3 大小并做到几个百分点的预测精度**——这是 OpenAI 当年敢直接砸钱训 175B 的依据。

**联合形式**（同时变 $N$ 与 $D$）：

$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}$$

直觉：loss 由两部分组成——"模型容量限制"（$N$ 不够）+ "数据不够"（$D$ 不够），二者独立加性。当某一项远大于另一项时，**整体 loss 由瓶颈决定**——所以模型和数据要"配套"，否则其中一个浪费。

### 2.2 Compute = 6 N D：训练 FLOPs 的经验估计

要做 scaling law 必须能量化 compute。LLM 预训的 FLOPs 经验估计：

$$C \approx 6 \cdot N \cdot D$$

其中 $N$ 是模型参数量、$D$ 是训练 token 数。系数 **6** 怎么来？dense Transformer 的每个 token 大致经过：

- **Forward 一次**：$2N$ FLOPs（每个参数参与一次乘 + 一次加，共 2 FLOPs；这是 matmul-dominant 假设）
- **Backward 一次**：$4N$ FLOPs（gradient w.r.t. weights + gradient w.r.t. activations，约是 forward 的 2 倍）
- 合计 **$6N$ FLOPs / token**

乘以总 token 数 $D$ → $C \approx 6 N D$。这是 Kaplan / Chinchilla 都用的 baseline 估计。

**注意事项**：

1. **不含 embedding**：$N$ 是非 embedding 参数。embedding 在 forward / backward 中只是 lookup，不参与 matmul，FLOPs 极低
2. **不含 attention 的 $O(T^2)$ 部分**：上式假设 $T \ll d_{\text{model}}$，attention 的 $T^2$ 贡献可忽略。当 $T = 32k$、$d = 4096$ 时 attention 已不能忽略，需要 $C \approx 6ND + 12 L T^2 d$（$L$ 是层数）这样的修正
3. **Activation recomputation 不算**：如果开了 gradient checkpointing，backward 多跑一次 forward，$6 \to 8$
4. **MoE 不一样**：每个 token 只激活一部分 expert，要用激活参数 $N_{\text{active}}$ 而不是总参数 $N$

简化记忆：**dense LLM 训一遍 ≈ 6 × 模型参数 × token 数 FLOPs**。LLaMA-3 70B 训 15T token：$6 \times 7\times10^{10} \times 1.5\times10^{13} \approx 6.3 \times 10^{24}$ FLOPs ≈ 6.3 ZettaFLOPs（确实是公开报道的数量级）。

### 2.3 Kaplan 配方：model-heavy

Kaplan 2020 用上述联合形式 fit 完后，在固定 compute $C = 6ND$ 约束下解最优 $(N^*, D^*)$。结果（论文 §6）：

$$N^* \propto C^{0.73}, \qquad D^* \propto C^{0.27}$$

**含义**：compute 翻倍，最优 model 大小放约 $2^{0.73} \approx 1.66$ 倍，data 只放 $2^{0.27} \approx 1.21$ 倍——**model 增长远快于 data**。Kaplan 给出的实操结论："don't train small models for too long"——多余 compute 优先放给更大的 model。

GPT-3 175B + 300B token 就是按这套设计的——D/N 只有 1.7。在 Kaplan 的 worldview 里这是 compute-optimal。

**Kaplan 的隐患**（Chinchilla 后才发现）：

- 实验中**LR schedule 对小模型欠优化**——cosine LR 的 warmup 周期没按 model size 缩放，小模型还没收敛就用大 LR 训
- 实验中 D/N 探索的范围窄——主要在 D/N ∈ [1, 5]，没充分探索 D/N > 10 的区域
- 数据 quality 没控制——不同实验用不同 data subset，干扰 fit

这些问题让 Kaplan 系统性低估了"加 data"的边际收益。

### 2.4 Chinchilla 配方：1:1 平衡

Hoffmann et al. 2022（DeepMind）重做实验，**用三种独立方法**交叉验证：

**方法 1：固定 model size 变 data size**——对每个 $N \in \{75M, 150M, ..., 70B\}$ 训一系列不同 $D$ 的 model，看 loss vs $D$ 的曲线在哪里"弯曲"（loss 不再下降）

**方法 2：IsoFLOP curve**——固定 $C$（比如 $10^{19}$ FLOPs），训一系列不同 $(N, D)$ 但 $6ND = C$ 的 model（小模型多 token vs 大模型少 token），找曲线最低点

**方法 3：parametric form fit**——直接拟合三参数形式：

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

其中 $E$ 是不可约 entropy。Chinchilla 拟合得到 $\alpha \approx 0.34$、$\beta \approx 0.28$、$E \approx 1.69$、$A \approx 406.4$、$B \approx 410.7$。

**三种方法殊途同归**——都得出 $N^* \propto C^{a}$、$D^* \propto C^{1-a}$ with $a \approx 0.5$。即：

$$\boxed{N^* \propto C^{0.5}, \qquad D^* \propto C^{0.5}}$$

经验法则：**$D \approx 20 N$**——每 1B 参数最优配 20B token。

**实证打脸 Kaplan**——Chinchilla 70B + 1.4T token 同 compute 下击败 Gopher 280B + 300B token：

| Model | Params | Tokens | Compute (FLOPs) | MMLU |
|---|---|---|---|---|
| Gopher | 280B | 300B | $5.0 \times 10^{23}$ | 60.0 |
| Chinchilla | 70B | 1.4T | $5.7 \times 10^{23}$ | **67.6** |

同样的 compute 预算，Kaplan 风格（大 model 少 token）输给 Chinchilla 风格（小 model 多 token）7.6 个点——这是 LLM 史上最重要的实证之一，**直接改写了之后两年的 model 设计哲学**。

### 2.5 Compute-optimal 的严格推导

Chinchilla 的核心数学并不复杂，值得自己推一遍——这是面试常考。

**Setup**：$L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$，约束 $C = 6 N D$（固定 compute 预算）。求 $L$ 最小化的 $(N^*, D^*)$。

**用 Lagrange multiplier**——构造 $\mathcal{L} = L + \lambda(C - 6ND)$，对 $N, D$ 求偏导置零：

$$\frac{\partial L}{\partial N} = -\frac{\alpha A}{N^{\alpha+1}} = 6 \lambda D$$

$$\frac{\partial L}{\partial D} = -\frac{\beta B}{D^{\beta+1}} = 6 \lambda N$$

两式相除消去 $\lambda$：

$$\frac{\alpha A / N^{\alpha+1}}{\beta B / D^{\beta+1}} = \frac{D}{N} \quad\Rightarrow\quad \frac{\alpha A}{\beta B} \cdot \frac{D^{\beta+1}}{N^{\alpha+1}} = \frac{D}{N}$$

$$\Rightarrow\quad D^{\beta} = \frac{\beta B}{\alpha A} \cdot N^{\alpha} \quad\Rightarrow\quad D \propto N^{\alpha/\beta}$$

代入约束 $C = 6 N D$：

$$C \propto N \cdot N^{\alpha/\beta} = N^{1 + \alpha/\beta}$$

$$\Rightarrow\quad N^* \propto C^{\beta / (\alpha + \beta)}, \qquad D^* \propto C^{\alpha / (\alpha + \beta)}$$

代入 Chinchilla 拟合的 $\alpha \approx 0.34, \beta \approx 0.28$：

$$\frac{\beta}{\alpha + \beta} \approx \frac{0.28}{0.62} \approx 0.45, \qquad \frac{\alpha}{\alpha + \beta} \approx 0.55$$

实证再化简到 $0.5$ / $0.5$——这就是著名的"50/50 配比"。再算 $D^*/N^*$：

$$\frac{D^*}{N^*} \propto C^{(\alpha - \beta)/(\alpha+\beta)} \approx C^{0.097}$$

随 $C$ 弱依赖（指数 0.1 以内），所以"$D / N \approx 20$"在很宽 $C$ 范围内近似成立。**严格说 $D/N$ 不是常数，而是缓慢随 $C$ 增长的弱 power law**——这是为什么 70B 的 Chinchilla optimal 和 1B 的 Chinchilla optimal 都说"$D/N \approx 20$"——指数 0.1 在两个数量级 $C$ 范围内只变化 $10^{0.1} \approx 1.3$ 倍。

### 2.6 DeepSeek scaling law（2024）

Shao et al. 2024（DeepSeekMath paper §6）做了第三轮 scaling law 修正。核心观察：

**Chinchilla 的 IsoFLOP 实验在做"固定 $C$ 变 $(N, D)$"时，没有同步调整 batch size 和 learning rate**。DeepSeek 发现：

- 不同 model size 的最优 batch size 不同（大 model 适合大 batch）
- 不同 batch size 下的最优 LR 不同（大 batch 配大 LR）
- **如果 IsoFLOP 实验里所有 model 用同一组 $(B, lr)$**，小 model 因为 batch 太大被欠训（gradient noise 不够）、大 model 因为 batch 太小被过训——结果**fit 出的最优 $N^*$ 偏大**

DeepSeek 的修正方法：在每个 IsoFLOP 点上**同步 sweep $(B, lr)$**，选每个 $(N, D)$ 下的真正最优 $(B, lr)$，再 fit scaling law。结果：

- 拟合出的 $\alpha$、$\beta$ 与 Chinchilla 略有不同
- $D^*/N^*$ 随 $C$ 的增长指数比 Chinchilla 的 0.1 略大——意味着**大 model 应该比 Chinchilla 法则建议的多吃 data**
- 给出更精确的 (N, D, batch, lr) 联合预测

DeepSeek-V3 671B（37B 激活）+ 14.8T token 就是按这套修正后的 scaling law 设计的。

**这里的洞察**：scaling law 不是"宇宙常数"，**它依赖于你怎么训**——LR schedule、batch size、data quality 都会移动 fit 出的指数。所以"同行 paper 报的 scaling law 数字不可直接拷贝"——你必须在自己的 training stack / data 上 refit。

### 2.7 Over-training：deploy-time optimal ≠ train-time optimal

Chinchilla 的所有公式都隐含一个假设：**目标是最小化训练 loss / compute 比**。但工业上还有另一个目标——**最小化总的 (training + serving) 成本**。

**核心洞察**：训练只发生一次（一次性 sunk cost），而 inference 要服务亿级 query（持续 cost）。如果一个 model 训完之后要被调用 1 万亿次，把 model size 减一半（哪怕训得更累）、推理便宜一半，**总账可能赚翻**。

形式化（Sardana et al. 2024 的 framework）：

$$C_{\text{total}} = \underbrace{6 N D_{\text{train}}}_{\text{training FLOPs}} + \underbrace{2 N \cdot Q \cdot \bar{T}}_{\text{inference FLOPs}}$$

其中 $Q$ = 总服务 query 数、$\bar{T}$ = 每 query 平均 token 数（input + output）。最小化 $C_{\text{total}}$ 关于 $N$ 时——如果 $Q \bar{T}$ 比 $D_{\text{train}}$ 大几个数量级（典型 production scenario），最优 $N$ 显著小于 Chinchilla optimal、最优 $D_{\text{train}}/N$ 显著大于 20。

**LLaMA-3 8B 的实战**：Chinchilla optimal 的 8B 应当配 ~160B token，但 LLaMA-3 8B 训了 **15T token**——$D/N \approx 1875$，是 Chinchilla 的 90 倍多。结果：

- 训练 loss 比 Chinchilla optimal 略好（但边际收益递减，多花 90x 数据只换 0.1-0.3 nats loss）
- 模型质量大幅提升——MMLU、GSM8K、HumanEval 全面接近甚至超过同一时期的 70B Chinchilla optimal model
- **推理便宜**——8B inference 的 GPU 内存、TTFT、throughput 全部远好于 70B
- 总账：训练多花 90x 算力、推理省 ~9x 算力（按 $N$ 比例），如果 $Q$ 足够大 → **赚**

这种"明知 Chinchilla 法则告诉你不该这么训，但因为推理收益就是要这么训"的策略叫 **over-training** 或 **inference-aware scaling**。Sardana 2024 的论文 *Beyond Chinchilla-Optimal* 系统化了这条路线。

### 2.8 现代主流模型 scaling 速览表（必背）

| Model | Params | Training Tokens | Tokens/Params | Scaling 哲学 |
|---|---|---|---|---|
| GPT-3 175B | 175B | 300B | 1.7 | Kaplan, model-heavy |
| Gopher | 280B | 300B | 1.1 | Kaplan-style（被 Chinchilla 打脸） |
| Chinchilla 70B | 70B | 1.4T | 20 | **Chinchilla optimal 的标杆** |
| LLaMA-1 65B | 65B | 1.4T | 22 | 几乎等同 Chinchilla |
| LLaMA-2 70B | 70B | 2T | 28 | 略 over-train |
| Mistral 7B | 7B | ~7-8T | ~1000+ | over-train（具体未公开） |
| LLaMA-3 8B | 8B | **15T** | **1875** | **极度 over-train** |
| LLaMA-3 70B | 70B | 15T | 215 | 重 over-train |
| DeepSeek-V3 | 671B（37B 激活）| 14.8T | 22（按总参） / 400（按激活） | 按修正 scaling law |
| Qwen2.5 7B | 7B | **18T** | **2570** | **极度 over-train** |
| Qwen2.5 72B | 72B | 18T | 250 | 重 over-train |
| Gemma 2 9B | 9B | 8T | 890 | over-train |

一眼能读出几条规律：

1. **Chinchilla 时代（2022-23）的旗舰模型**（LLaMA-1/2、Chinchilla）D/N 在 20-30 之间，老老实实跟 Chinchilla
2. **2024 之后的小模型**（LLaMA-3 8B、Qwen2.5 7B、Gemma 2 9B）D/N 飙到 800-2570，全面 over-train
3. **大 MoE 模型**（DeepSeek-V3）按总参数看 D/N 在 Chinchilla 范围（22），但**按激活参数算 D/N ≈ 400**——MoE 等价于"训练时享受大模型容量、推理时享受小模型成本"，本身就是另一种"over-train"
4. **同一代旗舰系（如 LLaMA-3）大小 model 用同一份 corpus**——15T token 训 8B / 70B / 405B 全用，工程上简化（一份数据训三遍 model）但配比哲学完全不同

### 2.9 Emergent ability：scaling 之外的不连续

Wei et al. 2022 在 *Emergent Abilities of Large Language Models* 提出："**有些能力在小模型上几乎为零，过某个临界 scale 突然出现**"——典型例子：

- **few-shot in-context learning**：< 13B 几乎不会，~62B 才稳定出现
- **多步算术**：< 70B 几乎随机，过 70B 突然能做
- **指令理解 / chain-of-thought**：scale 临界点更高

这种"非平滑跃迁"是 scaling law 不能预测的——scaling law 只 fit pretraining loss（一个连续可降的指标），但 downstream task accuracy 在小 scale 上可能"踩不到 threshold"长期是 0，过 threshold 后突然飙到合理水平。如果 emergent ability 是真的，意味着**只有训到足够大才能解锁某些能力**，model 选型有不可预测的"阶跃风险"。

**Schaeffer et al. 2023 的反驳**——*Are Emergent Abilities of Large Language Models a Mirage?*——他们论证 emergence 大部分是**度量选择**问题：

- 用"严格匹配"（exact match）类指标 → accuracy 在小 scale 是 0、大 scale 突然非零，看起来 emergent
- 用"token-level cross-entropy"或"部分匹配"指标 → 同样数据下能力是**渐进**提升的，没有跃迁

例如，多位数加法用"完全正确"评分，要求每一位都对——任何一位错就 0 分——小 model 偶尔猜对一位但极少全对，accuracy 长期是 0；改用"位精度"（每位独立评分）→ accuracy 平滑随 scale 上升。

**实战影响**：

- Schaeffer 的反驳被很多人接受——**多数 emergence 是 metric artifact**，本质上能力是渐进的
- 但少数 emergence（比如 chain-of-thought 在小 model 上可能因为 trace 长度有限根本写不完整推理）确实有阶跃成分
- 工程结论：**不要赌 emergence**——选 model 时按"已经在我目标 scale 验证过的能力"决策，不要假设"scale 上去了能力会自然出现"

### 2.10 第二轴：test-time compute scaling

2024 年 OpenAI o1 / 2025 年 DeepSeek-R1 引入了 LLM scaling 的**第二条独立轴**——**test-time compute**（推理时的 reasoning 长度）。一个固定的 model，让它生成更长的 chain-of-thought（甚至 best-of-N、tree search），下游 acc 也能持续提升——这是不同于"训练 compute scaling"的另一个维度。

简短结论（详见 Module 10）：

- 训练 compute（本节主题）：决定 base model 的"原始能力"
- 测试 compute：决定推理时能从这个 base 里"榨"出多少能力
- 二者**乘性叠加**——同 base model 上，128k thinking budget 比 1k thinking budget 在 AIME / GPQA 等数学题上能差 30+ 个点

本节只用作 awareness——**scaling law 的世界从 2024 年后从单轴变双轴**。Module 10 会展开 RLVR / R1 / o1 的具体技术。

---

## 3. 最小代码示例

### 3.1 Compute = 6 N D 的快速估算函数

```python
def training_flops(n_params: float, n_tokens: float,
                   include_recompute: bool = False) -> float:
    """
    估算 dense Transformer 训练 FLOPs。
    n_params: 非 embedding 参数量
    n_tokens: 训练 token 数
    """
    coef = 8.0 if include_recompute else 6.0   # gradient checkpointing 时是 8
    return coef * n_params * n_tokens

# LLaMA-3 8B 训 15T token
flops = training_flops(8e9, 15e12)
print(f"LLaMA-3 8B: {flops:.2e} FLOPs ≈ {flops/1e21:.2f} ZettaFLOPs")
# LLaMA-3 8B: 7.20e+23 FLOPs ≈ 720 ZettaFLOPs（与公开数据一致）

# H100 算力 ≈ 1e15 FLOPs/s（FP16），算需要多少 GPU-时
gpu_seconds = flops / 1e15
gpu_hours = gpu_seconds / 3600
print(f"≈ {gpu_hours:.0f} H100 hours (理论峰值)")
# 实际利用率 ~40% MFU，再乘 2.5 ≈ 50 万 GPU 时
```

`include_recompute=True` 时系数变 8 是因为 backward 多跑一次 forward。生产估算还要除以 MFU（Model FLOPs Utilization，typically 30-50%）才是真实墙钟时间。

### 3.2 Chinchilla compute-optimal 计算

```python
def chinchilla_optimal(compute_budget_flops: float,
                       a: float = 0.5):
    """
    给定 compute 预算，按 Chinchilla 法则返回最优 (N, D)。
    a 是 N 的指数（Chinchilla a≈0.5；Kaplan a≈0.73）
    """
    # 由 C = 6 N D 与 N ∝ C^a, D ∝ C^(1-a)，且 D = 20 N（经验）
    # 联立：C = 6 N * 20 N = 120 N^2  →  N = sqrt(C / 120)
    import math
    N_opt = math.sqrt(compute_budget_flops / 120)
    D_opt = 20 * N_opt
    return N_opt, D_opt

# 假设你有 1e22 FLOPs 预算
N, D = chinchilla_optimal(1e22)
print(f"Chinchilla optimal: N = {N/1e9:.1f}B params, D = {D/1e9:.0f}B tokens")
# Chinchilla optimal: N = 9.1B params, D = 183B tokens
# → "如果只有 1e22 FLOPs，最划算的是 9B model + 183B token"

# 验证：6 N D ≈ 1e22
print(f"check: 6ND = {6 * N * D:.2e}")
```

注意这只用了"$D = 20N$"经验近似——更严格要按 §2.5 的精确推导（用 $\alpha, \beta$ 拟合值）来算，结果略有差异（在 50-100B model size 范围内 $D/N$ 实际略大于 20）。

### 3.3 Scaling law 拟合 demo（合成数据）

```python
# pip install numpy scipy matplotlib
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 真实 scaling law（用合成数据模拟实验）
def true_law(N, D):
    return 1.69 + 406.4 / N**0.34 + 410.7 / D**0.28   # Chinchilla fit 值

# 模拟 IsoFLOP 实验：固定 C，扫不同 (N, D)
C_budget = 1e19           # 一个小 budget 演示
N_grid = np.logspace(7, 9.5, 12)              # 10M ~ 3B params
D_grid = C_budget / (6 * N_grid)              # 由 6ND=C 反推 D
losses = true_law(N_grid, D_grid) + np.random.normal(0, 0.01, N_grid.shape)

# 拟合 IsoFLOP 曲线，找最优 N
def iso_loss(N, a, b, c):
    return a + b / N**0.34 + c * N**0.28 / (C_budget / 6)**0.28

popt, _ = curve_fit(iso_loss, N_grid, losses, p0=[1.7, 400, 400])
N_fit = np.logspace(7, 9.5, 200)
loss_fit = iso_loss(N_fit, *popt)
N_optimal = N_fit[np.argmin(loss_fit)]

# 可视化
plt.loglog(N_grid, losses, 'o', label='IsoFLOP experiments')
plt.loglog(N_fit, loss_fit, '-', label=f'fit, N* ≈ {N_optimal/1e6:.0f}M')
plt.axvline(N_optimal, ls='--', color='red')
plt.xlabel('Model size N'); plt.ylabel('Loss')
plt.title(f'IsoFLOP curve at C = {C_budget:.0e} FLOPs')
plt.legend(); plt.show()
```

实际跑 Chinchilla-style 实验是这套流程的工业版本——同一个 IsoFLOP 实验要跑数十个 $(N, D)$ 点，每个 $C$ 跑一条曲线，最后 fit 拐点位置。Chinchilla paper 跑了 $C \in [10^{18}, 10^{21}]$ 的 9 条曲线交叉验证。

---

## 4. 工程踩坑与经验

- ❗ **Chinchilla 的"20 token/param"是 train-time optimal，不是 deploy-time optimal**。Production model 几乎所有都 over-train——LLaMA-3 8B 用 15T token (D/N=1875) 是 Chinchilla 的 90×。**面试或工程报告里写"我按 Chinchilla 选了 D=20N"基本表明你只读了 2022 年的论文**——现代正确答案是"先估算 inference budget $Q\bar{T}$，再按 Sardana 2024 求 $C_{\text{total}}$ 最小化的 $(N, D)$"。

- ❗ **scaling law 的 $N$ 在不同 paper 里不一致**——Kaplan 包含 embedding，Chinchilla 不包含 embedding。读 paper / 复现实验时**第一件事是核对 $N$ 的定义**。embedding 在小 model（< 1B）里占比可能 30-50%，定义不对会让所有数字偏一个数量级。Chinchilla paper 在 §A.2 明确写了"$N$ excludes embeddings"——一定要读细。

- ❗ **scaling law 不可跨 data quality 移植**。FineWeb 上 fit 出的 scaling 指数与 RedPajama 不同——同 $(N, D)$ 下 FineWeb 数据训出来的 loss 系统性更低，但**指数 $\alpha, \beta$ 也会移动**，不是简单的常数偏移。**自己换数据要重新 fit scaling law**——直接抄 Chinchilla 的数字会预测错。

- ❗ **scaling law 不预测 downstream accuracy，只预测 loss**。loss 和 acc 高度相关但不严格单调——某些任务（特别是 emergent task）loss 持续下降而 acc 长期 0 然后突然飙升。**给老板做"再花 X 钱能涨多少 MMLU"的预测，不能直接套 scaling law**——只能预测 perplexity，accuracy 要单独 fit "loss → acc"映射，而且这个映射在不同任务上完全不同。

- ❗ **MoE 的 scaling law 不能套 dense 的**。DeepSeek-V3 671B 总参 + 37B 激活——按总参算 D/N ≈ 22（看似 Chinchilla），按激活算 D/N ≈ 400（极度 over-train）。**MoE 应该用激活参数 $N_{\text{active}}$ fit scaling law 的"能力侧"，用总参数 fit "memory cost 侧"**——两条法则分别管 quality 和 infra cost。Krajewski 2024 *Scaling Laws for Fine-grained MoE* 给了 MoE-specific 的修正。

- ❗ **Over-training 的边际收益快速衰减**。从 Chinchilla optimal（D/N=20）扩到 D/N=200（10x）能涨 5-10 个 MMLU 点；从 D/N=200 扩到 D/N=2000（再 10x）只能涨 1-3 个点。Sardana 2024 的曲线非常清楚——**continuing-to-over-train 的 ROI 是次线性**。LLaMA-3 8B 用 15T 已经接近 plateau，再扩到 30T 多半只能涨 < 1 点 MMLU——这可能是为什么 LLaMA-3 没继续扩的工程原因。

- ❗ **$C \approx 6ND$ 是粗略经验**——精确算还要加 attention 的 $O(T^2)$ 部分。当 context length $T \geq 32k$ 时 attention 已经占 10-30% FLOPs，不能忽略。完整公式：$C \approx 6ND + 12 L T^2 d \cdot D / T$（$L$ 层数、$d$ hidden dim、$T$ context length）。**做 long-context model（详见 6.5）的 budget 估算时不能用 6ND**，要用完整公式。

- ❗ **Chinchilla 论文当年公开时业界震动**——Chinchilla 70B + 1.4T token 在 2022 年 3 月公开，**同年 2 月 LLaMA-1 65B + 1.4T token 已经基本按这套配方训完了**——Meta 内部应该独立得出了类似结论。Chinchilla 公开后，原本按 Kaplan 设计的下一代 model（Gopher 系、PaLM 系列后续）全部重训；社区从此公认"data 不够 compute-optimal" → 直接催生了 RedPajama / FineWeb 这类大规模开源 corpus 的需求。**这是 LLM 史上最重要的"算法→开源生态"传导事件之一**。

- ❗ **小模型上 fit 的 scaling law 外推到大模型可能失败**。Chinchilla / Kaplan 都是用 $\leq$ 100B 数据点 fit 后外推到更大 size——**外推距离越远风险越大**。GPT-4 据信用了类似方法外推到 ~1T 量级，但 OpenAI 内部 paper 提过"在 1T 量级 fit 误差显著大于小 size"。**自己做 scaling 实验，不要用 1B 直接外推到 100B**——跨度 > 1 个数量级时务必加 ablation 验证。

---

## 5. 经典 paper

- **Kaplan et al., 2020 — Scaling Laws for Neural Language Models** — LLM scaling law 的"原典"。读 §3 三种独立 fit 形式（$L(N), L(D), L(C)$）+ §6 联合形式与最优 $(N, D)$ 推导 + Figure 1（power law 主图，本节所有直觉的来源）。Take-away：理解 power law 形式 + Kaplan 的 model-heavy 配比 $N \propto C^{0.73}$（虽被 Chinchilla 修正但公式形式仍是 baseline）。
- **Hoffmann et al., 2022 — Training Compute-Optimal Large Language Models (Chinchilla)** — 必读必背。读 §3 三种方法（IsoFLOP、固定 N 变 D、parametric fit）三角验证 + Table 3 Chinchilla 70B 击败 Gopher 280B 的实证 + §A 的完整推导。Take-away：$N^* \propto C^{0.5}, D^* \propto C^{0.5}$、$D \approx 20 N$ 经验法则、以及"为什么 model 和 data 必须同等增长"。这是过去 5 年 LLM 设计哲学的转折点。
- **Shao et al., 2024 — DeepSeekMath** §6 scaling law section — DeepSeek 修正版。读 §6 "Comparison with previous work"——他们如何论证 Chinchilla IsoFLOP 没同步 batch/lr 而系统性偏差。Take-away：scaling law 不是宇宙常数，依赖 training stack；自己做 scaling 实验必须 sweep $(B, lr)$。
- **Wei et al., 2022 — Emergent Abilities of Large Language Models** — emergence 提出 paper。读 §3 多个 task 的 phase transition 曲线（in-context learning、multi-step reasoning、math）。Take-away：scaling 不能预测 downstream acc 的不连续跃迁——这是 model 选型的"阶跃风险"来源。
- **Schaeffer et al., 2023 — Are Emergent Abilities of Large Language Models a Mirage?** (NeurIPS 2023 best paper) — emergence 反驳。读 §3 的 metric 实验——同样数据用不同 metric 看到的 emergence 完全不同。Take-away：多数 "emergence" 是 metric artifact——本质上能力是渐进的。读完这两篇形成完整观点。
- 加分阅读：**Sardana et al., 2024 — Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws** — over-training 系统化讨论，给出 inference-aware $C_{\text{total}}$ 的最优 $(N, D)$ 推导、复现 LLaMA-3 这种 D/N=1875 的合理性。**这篇可能是 2024 年最重要的 scaling law 论文**，理解现代 over-training 的所有动机都在这里。

---

## 6. 自测与面试题

**Q1（公式）**：写出 Chinchilla 的 compute-optimal 配比经验法则；按 Chinchilla 算 LLaMA-2 7B 的最优训练 token 数（与实际 2T 对比）。

<details>
<summary>Answer sketch</summary>

经验法则：$D \approx 20 N$，即每 1B 参数最优配 20B 训练 token。

**LLaMA-2 7B 的 Chinchilla optimal**：
- $N = 7 \times 10^9$ params
- $D^* = 20 \times 7\text{B} = 140 \text{B}$ tokens
- 对应 $C \approx 6 N D = 6 \times 7\text{e9} \times 1.4\text{e11} \approx 5.9 \times 10^{21}$ FLOPs

**LLaMA-2 7B 实际**用了 **2T token**（2000B）= **D/N = 286**，是 Chinchilla 的 14× over-train。所以 LLaMA-2 7B 已经偏离 Chinchilla 一个数量级——证明"over-training" 在 2023 年就已经开始（虽然不像 LLaMA-3 那么极端）。

加分点：

- 能指出 Chinchilla 是 train-time optimal、不考虑 inference cost
- 能指出 Sardana 2024 的 inference-aware $C_{\text{total}}$ framework 解释为什么要 over-train
- 能算出对应 compute：6 N D 公式
- 能引用对比表，知道 LLaMA-3 8B 已经是 D/N=1875，比 LLaMA-2 7B 还要极端 7×

</details>

**Q2（推导 + 实战）**：解释为什么 LLaMA-3 8B 用 15T token（远超 Chinchilla 的 ~160B optimal），从 deploy 视角给出至少 2 个理由 + 边际收益的限制。

<details>
<summary>Answer sketch</summary>

**deploy 视角的核心理由**：

1. **Inference cost 主导总成本**——一个 model 训完后要服务亿~万亿 query。8B inference 比 70B 在 FLOPs / GPU 内存 / TTFT / throughput 上都便宜约 9 倍。如果生命周期内 query 数 $Q$ 足够大（典型 Meta 部署量），把 model 缩小、推理便宜的省钱量 >> 训练多花的 compute。$C_{\text{total}} = 6 N D_{\text{train}} + 2 N Q \bar{T}$，当 $Q\bar{T} \gg D_{\text{train}}$ 时最优 $N$ 显著小于 Chinchilla optimal、$D/N$ 显著大于 20。

2. **小模型部署门槛低**——8B 可以塞进单张 24GB GPU（量化后能塞 12GB），70B 需要至少 80GB GPU 或多卡部署。开源生态接受度（开发者、终端设备）也强烈偏好小 model——"能在本地跑"是产品差异化。

3. **同 corpus 训三个 size 工程上简化**——LLaMA-3 8B / 70B / 405B 都用同一份 15T token corpus，data pipeline 投资可以摊销，不需要为每个 size 重新做 IsoFLOP 实验。

4. **LLaMA-3 8B 在 MMLU 等通用 benchmark 上接近 LLaMA-2 23B 量级**——Sardana 2024 系统化地证明：对小 model over-train 能"换"出比 Chinchilla optimal 更高的能力，相当于"用训练 compute 换 inference compute"。

**边际收益限制**：

- Over-training 的收益是次线性——D/N 从 20→200 涨 5-10 MMLU 点，从 200→2000 只涨 1-3 点
- 15T → 30T 不会有 15T → 7.5T 那样大的提升
- LLaMA-3 8B 用 15T 已经接近 plateau，可能是为什么没继续扩
- 极端 over-train 还有 risk——data quality 下降（数据池见底重复 token）、specific benchmark 上可能 overfit

加分点：

- 能引用 Sardana 2024 的 $C_{\text{total}}$ 公式
- 能区分"训练 compute 优化"与"deploy compute 优化"两个目标
- 能指出 over-training 的边界条件（data 不够、quality 衰减）
- 能指出 LLaMA-3 70B / 405B 也用同 15T corpus，这是工程考量

</details>

**Q3（前沿）**：emergent ability 是真实存在还是 metric 错觉？给出两种观点 + 你的判断。

<details>
<summary>Answer sketch</summary>

**观点 1（Wei 2022 — emergent 是真实的）**：

- 在 in-context learning、多步算术、CoT、指令理解等任务上，accuracy 在小 model（< 临界 scale）上几乎是 0，过临界 scale 后突然飙升到合理水平
- 这种 phase transition 在多种任务上独立观察到，不是 single-task 偶然
- 如果 emergence 真实存在，意味着 LLM 选型有"阶跃风险"——某些能力只能通过训到足够大解锁
- 实战意义：BigBench-Hard、emergent task 类 benchmark 必须用 SOTA 大 model 评测

**观点 2（Schaeffer 2023 — emergent 是 metric artifact）**：

- 用"严格匹配"指标（如 exact match）→ 看到 emergence；改用 token-level perplexity 或部分匹配 → 能力是连续渐进的
- 多步算术：要求每位都对的指标在小 model 长期 0、突然飙升；用"位精度"（每位独立评分）→ 平滑随 scale 上升
- 本质上 emergence = 你选的指标在小 scale 是 saturated zero、过 threshold 后才 detectable
- 实战意义：用合理的 graded metric / 软指标可以提前预测大模型能力，不必等"涌现"

**我的判断**：

- **多数 emergence 是 metric artifact——Schaeffer 的解释更可信**。能力本质上是连续渐进的，"看起来 emergent"是因为我们用了 0/1 类硬指标
- 但**少数 emergence 有 capacity 阶跃成分**——比如 chain-of-thought 在小 model 上可能因为 trace 长度有限根本写不完整推理，这有"模型必须有足够 capacity hold 长 reasoning"的硬约束
- **工程结论**：不要赌 emergence——选 model 时按"已经在你目标 scale 验证过的能力"决策；scaling law 只能预测 loss，不能预测 emergent task 的 accuracy
- **研究意义**：emergence 这个 framing 改变了人们对"为什么需要大模型"的直觉——从 "smooth gain" 到 "phase transition"——即使 metric artifact 的解释成立，它在公众认知和投资决策上的影响已经形成

加分点：

- 能引用 Schaeffer 拿 NeurIPS 2023 best paper 的事实
- 能区分"loss 是连续的，downstream metric 选择决定看不看到 emergence"
- 能联系到 scaling law 的局限——loss 可预测、downstream acc 不可直接预测
- 能联系到 Module 10 的 RLVR—— R1 的 "aha moment" 也常被讨论是真 emergence 还是逐步渐进

</details>

---

## 7. 延伸阅读

- [Kaplan et al. 2020 — Scaling Laws for Neural Language Models (arXiv)](https://arxiv.org/abs/2001.08361) — 原典，必读 §3、§6
- [Hoffmann et al. 2022 — Training Compute-Optimal Large Language Models / Chinchilla (arXiv)](https://arxiv.org/abs/2203.15556) — 必背的核心 paper，§3 三方法验证 + Table 3
- [Sardana et al. 2024 — Beyond Chinchilla-Optimal (arXiv)](https://arxiv.org/abs/2401.00448) — over-training 系统化讨论，理解 LLaMA-3 这种 D/N=1875 的所有动机
- [DeepSeekMath paper § scaling law (arXiv)](https://arxiv.org/abs/2402.03300) — DeepSeek 对 Chinchilla 的修正，IsoFLOP 必须同步 sweep $(B, lr)$
- [Wei et al. 2022 — Emergent Abilities of LLMs (arXiv)](https://arxiv.org/abs/2206.07682) — emergence 提出
- [Schaeffer et al. 2023 — Are Emergent Abilities a Mirage? (arXiv)](https://arxiv.org/abs/2304.15004) — emergence 反驳，NeurIPS 2023 best paper
- [LLaMA-3 paper (arXiv)](https://arxiv.org/abs/2407.21783) — 8B / 70B / 405B 同一 15T corpus 的 over-training 实战，§3 数据 + §4 训练详情
- [Krajewski et al. 2024 — Scaling Laws for Fine-Grained Mixture of Experts (arXiv)](https://arxiv.org/abs/2402.07871) — MoE 专属 scaling law，DeepSeek-V3 这类模型的设计依据
- [Tom Henighan 的 Anthropic blog 《How Anthropic thinks about scaling》](https://www.anthropic.com/research) — 工业实操视角的 scaling law 决策（如何用小实验 forecast 大 model）
- 推荐继续读本教程的 **6.4 节《训练曲线诊断与超参》**——scaling law 给你"应该训多大"，6.4 给你"训不动 / 训歪了怎么 debug"
- 推荐继续读本教程的 **5.4 节《MoE》** + **6.6 节《经典开源 LLM 解读》**——把本节的 scaling law 与具体 model 设计选择对应起来
- 推荐 **Module 10 第 10.3 节《RLVR 与 DeepSeek-R1》**——理解 scaling 第二轴 test-time compute 的具体技术
