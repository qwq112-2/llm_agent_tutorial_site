---
title: "5.2 GQA / MQA / MLA：KV Cache 压缩"
description: "4.7 已经算清楚 LLaMA-3 70B 在 8k context 下单 sample KV cache 高达 ~20 GB——KV cache 是 LLM 推理 OOM 第一杀手；本节讲三种架构层面直接压缩 KV cache 的方案：MQA（所有 head 共享 1 组 K/V，cache ÷ $h$ 但质量掉点）、GQA（h 个 Q head 分 g 组共享 K/V，cache ÷ $h/g"
---

> ⏱ 预计阅读 55 分钟 ｜ 难度 ★★★ 🔥 必考 ｜ 前置：4.2 multi-head、4.3 RoPE、4.7 KV cache

## 一句话本节讲什么

4.7 已经算清楚 LLaMA-3 70B 在 8k context 下单 sample KV cache 高达 ~20 GB——KV cache 是 LLM 推理 OOM 第一杀手；本节讲三种**架构层面**直接压缩 KV cache 的方案：**MQA**（所有 head 共享 1 组 K/V，cache ÷ $h$ 但质量掉点）、**GQA**（h 个 Q head 分 g 组共享 K/V，cache ÷ $h/g$ 且质量几乎无损，**已成现代 LLM 标配**）、**MLA**（把 K/V 投影到低维 latent space 缓存，cache ÷ 16+ 但需要与 RoPE 解耦的精巧设计，DeepSeek-V2/V3/R1 用）。本节会把三方案的公式、cache size 算账、PyTorch 手撕 GQA 的 30 行实现、与 RoPE 的兼容性、与 PagedAttention / KV 量化的协同、以及业界主流模型选型对照表全部讲透——这是当代 LLM 推理优化里**最重要的架构设计选择题**，也是后训练 / 推理工程师面试 90% 会考的高频点。

---

## 1. Mental model（直觉）

### 1.1 回顾 4.7：KV cache 凭什么是 OOM 第一杀手

把 4.7 §2.4 的算账重新拎出来——LLaMA-3 70B（$L = 80$ 层、$h = 64$ head、$d_k = 128$、$d = h \cdot d_k = 8192$）在 8k context 下单 sample 的 bf16 KV cache：

$$M_{\text{KV}} = 2 \cdot L \cdot d \cdot T \cdot 2 \text{ B} = 2 \times 80 \times 8192 \times 8192 \times 2 \approx 21 \text{ GB}$$

**单个 user**就要 21 GB。80 GB 的 H100 装完 70B 权重（fp16 即 140 GB，已经需要 TP 切到 2 卡）后，剩下的 KV cache 预算只够装 ~3 个 8k context 的 user——这意味着 throughput 极低，单卡服务不了几个并发。如果 batch = 16，KV cache 直接 336 GB，比模型本身还大 2.4 倍。

这个数字告诉我们一件事：**70B+ 模型用标准 MHA 推理在经济上根本不可行**——必须从架构层把 KV cache 砍下来。本节讲的 GQA / MQA / MLA 就是三个一刀切的解法，分别对应"激进 / 折中 / 极致"三档。

### 1.2 三方案的核心 idea 一句话

| 方案 | 一句话 idea | 类比 |
|---|---|---|
| **MQA** | 所有 $h$ 个 Q head 共享**同一组** K, V 投影 | "$h$ 个 detective 共用 1 套档案库" |
| **GQA** | 把 $h$ 个 Q head 分成 $g$ 组，每组共享 1 组 K, V | "$g$ 个小组，每组 1 个档案柜" |
| **MLA** | 把 K, V 投影到一个低维 latent 空间存储，attention 时再升回 | "档案柜里只存压缩包，要看时再解压" |

这三个 idea 解决的都是同一个问题——**KV head 数与 Q head 数解耦**——但激进程度不同。MQA 最激进（KV head 数 = 1），GQA 折中（KV head 数 = g），MLA 最激进同时最优雅（不再以 head 为单位、直接投影到 latent 维度）。

### 1.3 为什么"减 KV head"比"减 Q head"更有效

回到 4.2 §2.6 的核心算式：每层每 token 的 KV cache 大小 $\propto h_{\text{kv}} \cdot d_k$（$h_{\text{kv}}$ 是 KV head 数）。**只要砍 KV head 数，cache 就线性减小**——而 Q 在每步生成里都是新的、不缓存，砍 Q head 对显存毫无帮助。

更关键的是，4.2 §2.5 引用过 Michel et al. 2019 的 head pruning 实证——**大部分 head 是冗余的**，16 个 head 砍到只剩 1-2 个仍然 work。这给了一个非常自然的工程问题：

> 既然 head 多数冗余，为什么每个 Q head 都要配一组独占的 K, V 投影？

MQA / GQA / MLA 是这个问题的三种答案：
- **MQA**：极端答案——所有 Q head 共享 1 套 KV，KV cache 直接 ÷ $h$
- **GQA**：折中答案——分组共享，质量与 KV cache 之间找平衡
- **MLA**：另起炉灶——抛弃"以 head 为单位"，把 KV 整体投影到一个低维 latent

### 1.4 直觉图：MHA / MQA / GQA / MLA 的 head 拓扑

```
   MHA (h=8)                       MQA (h=8)
   Q: [1][2][3][4][5][6][7][8]     Q: [1][2][3][4][5][6][7][8]
       │  │  │  │  │  │  │  │             ↘ ↘ ↘ ↘ ↓ ↙ ↙ ↙
   K: [1][2][3][4][5][6][7][8]     K:        [shared K]
   V: [1][2][3][4][5][6][7][8]     V:        [shared V]
   KV cache: 8 份                   KV cache: 1 份  → cache ÷ 8


   GQA-2 (h=8, g=2)                MLA (DeepSeek-V2/V3)
   Q: [1][2][3][4] [5][6][7][8]    Q: [1][2][3][4][5][6][7][8] (rope + nope 拆分)
       \  |  |  /   \  |  |  /            ↓
   K:  [shared] [shared]            K, V 都不直接缓存
   V:  [shared] [shared]            cache 只存一个低维 latent c_t
   KV cache: 2 份  → cache ÷ 4      attention 时从 c_t 升回 K, V
```

记住三句口诀：
- **MQA**：1 = $h$ 极致压缩、质量掉点
- **GQA**：$g$ 折中、$g = 8$ 是甜点
- **MLA**：跳出 head 框架、压到 latent

### 1.5 为什么 GQA 几乎成了现代 LLM 标配

历史时间线一句话：**MQA → 太激进掉点 → GQA 提出折中 → 几乎所有 2023 年后开源 LLM 都用 GQA → DeepSeek 觉得 GQA 还不够激进 → 再走 MLA**。

具体路径：
- **2019** Shazeer 提出 MQA，但当时 LLM 还没大到 KV cache 痛——只在 PaLM、StarCoder、Falcon-180B 等少数模型上用
- **2022-2023** LLaMA-2 70B 推理 KV cache 痛苦，Ainslie 等人提出 GQA：MQA 与 MHA 之间插值，**g = 8 时质量几乎无损 + cache ÷ 8**
- **LLaMA-2 70B** 直接采用 GQA-8（$h = 64, g = 8$），质量上与同期 MHA 模型持平、推理显存压力锐减
- **2024 后** Qwen2/2.5、Mistral、Gemma、Phi、GLM 全用 GQA——成为事实标配
- **2024.05** DeepSeek-V2 提出 MLA：把 KV cache 进一步压到 ~576 维 latent，128k context 也能装下；2024.12 V3、2025.01 R1 沿用

**面试一句话总结**：现代 LLM 选 GQA 是默认选项，选 MLA 表示"我对 long context + 推理成本极致敏感（DeepSeek 路线）"，选 MHA 通常表示"模型小（< 7B）或者历史遗留"。

---

## 2. 公式与原理

### 2.1 标准 MHA 回顾（baseline）

设 hidden dim $d$、head 数 $h$、每 head 维 $d_k = d/h$。每个 head $i \in \{1, \dots, h\}$ 独立维护三组投影：

$$W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d \times d_k}$$

每 token 在每 layer 缓存的 K, V：

$$\text{KV cache (per token, per layer)} = 2 \cdot h \cdot d_k \cdot \text{bytes} = 2 d \cdot \text{bytes}$$

LLaMA-3 70B（$d = 8192$、bf16）即 $2 \times 8192 \times 2 = 32$ KB/token/layer，80 层 = 2.6 MB/token，8k token = 21 GB——本节 §1.1 算过的数字。

### 2.2 MQA — Multi-Query Attention（Shazeer 2019）

**核心定义**：保留 $h$ 个独立 Q 投影，但所有 head 共享**唯一的一组** K, V 投影：

$$Q_i = X W_Q^{(i)}, \quad i = 1, \dots, h$$

$$K = X W_K, \quad V = X W_V \quad (\text{只有 1 组})$$

其中 $W_Q^{(i)} \in \mathbb{R}^{d \times d_k}$ 仍是 $h$ 个独立矩阵，但 $W_K, W_V \in \mathbb{R}^{d \times d_k}$ **只有 1 个**。每个 head 算 attention 时 K, V 都用同一份：

$$\text{head}_i = \text{softmax}\!\left(\frac{Q_i K^\top}{\sqrt{d_k}}\right) V$$

**KV cache 大小**：

$$\text{KV cache (MQA, per token, per layer)} = 2 \cdot 1 \cdot d_k \cdot \text{bytes} = \frac{2d}{h} \cdot \text{bytes}$$

**直接 ÷ $h$**——LLaMA-3 70B 的 21 GB 直接降到 21 / 64 ≈ 0.33 GB。

**质量代价**：MQA 让 $h$ 个 head 共享同一份 "key 检索空间"，等于强行减少了 attention pattern 的多样性。Shazeer 2019 的实测：MQA 在 6B 量级模型上 BLEU 降 ~1-2%、perplexity 升 ~3-5%。**小模型（< 7B）掉点比大模型明显**——大模型有更多冗余 head 可以"吸收"这种压缩。

**用户列表**：
- PaLM（Google，2022）—— 540B，第一个公开宣传用 MQA 的大模型
- StarCoder / StarCoder2（HuggingFace + ServiceNow，代码模型）
- Falcon-180B（TII）—— 大参数 + MQA 的极端组合

但 MQA 在 7B-70B 中等模型上的质量下降不可忽视，所以**纯 MQA 已不是主流**——GQA 出来后基本被取代。

### 2.3 GQA — Grouped-Query Attention（Ainslie 2023）

**核心 idea**：把 $h$ 个 Q head 平均分成 $g$ 组，每组共享 1 套 K, V 投影（共 $g$ 套，而不是 1 套或 $h$ 套）。

设 $h$ 个 Q head 分成 $g$ 组（要求 $g \mid h$），每组 $h/g$ 个 head 共享 K, V：

$$Q_i = X W_Q^{(i)}, \quad i = 1, \dots, h \quad (h \text{ 个独立 Q 投影})$$

$$K^{(j)} = X W_K^{(j)}, \quad V^{(j)} = X W_V^{(j)}, \quad j = 1, \dots, g \quad (g \text{ 组 KV 投影})$$

第 $i$ 个 Q head 对应的 KV 组索引 $j(i) = \lceil i \cdot g / h \rceil$（即 head $i$ 属于第 $j$ 组）：

$$\text{head}_i = \text{softmax}\!\left(\frac{Q_i (K^{(j(i))})^\top}{\sqrt{d_k}}\right) V^{(j(i))}$$

**KV cache 大小**：

$$\text{KV cache (GQA, per token, per layer)} = 2 \cdot g \cdot d_k \cdot \text{bytes} = \frac{2 d \cdot g}{h} \cdot \text{bytes}$$

**典型 $g = 8$**：LLaMA-2 70B 用 $h = 64, g = 8$，KV cache 直接 ÷ 8——21 GB 降到 2.6 GB。

**两个边界条件**：
- $g = h$：退化为标准 MHA（每个 head 都有独立 KV）
- $g = 1$：退化为 MQA（所有 head 共享 1 组 KV）

GQA 在 MHA 与 MQA 之间提供了一个连续的 trade-off 旋钮。Ainslie 2023 的实测最关键的发现：

| 配置 | KV cache | BLEU / Perplexity | 备注 |
|---|---|---|---|
| MHA | 8 (baseline) | 100% | baseline |
| MQA ($g = 1$) | 1 | ~98% | 掉点 1-2% |
| **GQA-8** ($g = 8$) | **8** | **~99-100%** | **几乎无损 + cache ÷ 8** |
| GQA-2 ($g = 2$) | 2 | ~98.5% | 接近 MQA |

**关键 take-away**：$g = 8$ 是 sweet spot——KV cache ÷ 8 + 质量损失 < 1%。这就是为什么后来几乎所有 LLaMA / Qwen / Mistral 70B+ 模型默认用 GQA-8。

**Upcycling 技巧**：把已训好的 MHA 模型转成 GQA 时，标准做法是**把每组的 $h/g$ 个 K, V head 求平均**作为新的 KV 投影初始化，再短 fine-tune（论文称 ~5% pretrain compute 即可恢复性能）。这是 LLaMA-2 70B 的实际做法，比从头训省钱很多。

**用户列表（几乎所有现代开源 LLM）**：
- LLaMA-2 70B（$h = 64, g = 8$）
- LLaMA-3 8B / 70B / 405B（全系 $g = 8$）
- Mistral 7B / Mixtral 8×7B（$g = 8$）
- Qwen2 / 2.5 系列（$g = 2$ 至 8 不等，参考各模型 config）
- Gemma / Gemma-2、Phi-3、GLM-4 等

GQA 已成为**事实标配**——2023 后新出的开源 decoder-only LLM 几乎清一色用 GQA。

### 2.4 MLA — Multi-head Latent Attention（DeepSeek-V2 2024）

GQA 已经把 KV cache 压了 4-8×，但 DeepSeek 觉得不够——他们想把 KV cache 再压一个数量级，目标是 128k context 仍能装下 batch。

**核心 idea**：抛弃"以 head 为单位缓存 K, V"的路线，把 K, V **联合**投影到一个低维 latent space，**只缓存 latent**；attention 时再 up-project 回 K, V。

设 latent 维度 $d_c \ll h \cdot d_k$（DeepSeek-V2 取 $d_c = 512$，相比 MHA 的 $h \cdot d_k = 16384$ 缩小 32×）。

**Down-projection（compress）**：

$$c_t^{KV} = X_t W_{DKV} \in \mathbb{R}^{d_c}$$

其中 $W_{DKV} \in \mathbb{R}^{d \times d_c}$ 是一个共享的 down-projection 矩阵——把 hidden state $X_t$ 直接压到 $d_c$ 维 latent。

**Up-projection（decompress）**：在 attention 时把 latent 升回 K, V：

$$K_t = c_t^{KV} W_{UK}, \quad V_t = c_t^{KV} W_{UV}$$

其中 $W_{UK}, W_{UV} \in \mathbb{R}^{d_c \times (h \cdot d_k)}$。

**关键**：cache 里只存 $c_t^{KV} \in \mathbb{R}^{d_c}$，不存 K, V。

**KV cache 大小**：

$$\text{KV cache (MLA, per token, per layer)} = d_c \cdot \text{bytes}$$

DeepSeek-V2 取 $d_c = 512$、bf16，每 token 每层 1024 B；相比 LLaMA-3 70B MHA 的 32 KB/token/layer 缩小 32×。

### 2.5 MLA 与 RoPE 解耦：最 tricky 的设计

简单地把上面的 MLA 套到现代 LLM 会立刻撞到一个问题：**RoPE 不能直接应用在 latent space**。

回顾 4.3：RoPE 是在 q, k 投影**之后**对它们做位置相关的旋转，让 attention 内积只依赖相对位置。如果我们要从 latent $c_t^{KV}$ 通过 $W_{UK}$ up-project 出 K，再对 K 做 RoPE 旋转——那么旋转后的 K **不能再写成 $c_t \cdot W$ 的形式**（因为旋转矩阵 $R_t$ 依赖位置 $t$，不能与 $W_{UK}$ 合并），等于**每次 attention 都要现场 up-project + 旋转**——失去了 cache 的意义。

DeepSeek-V2 的解决方案是 **decoupled RoPE**：把 query 与 key 各拆成两半——一半走 RoPE、一半走 nope（no positional encoding），用 nope 部分走 latent 压缩，用 rope 部分单独缓存。

**具体拆分**（简化版描述）：

每个 head 的 query / key 维度 $d_k$ 拆成：
- $d_k^{\text{nope}}$：non-positional 部分（不带 RoPE），从 latent up-project
- $d_k^{\text{rope}}$：positional 部分（带 RoPE），单独投影 + 缓存

DeepSeek-V2 取 $d_k^{\text{nope}} = 128, d_k^{\text{rope}} = 64$，每 head 总维度 192（比标准 128 略大，但只是 q/k 维不是 cache 维）。

**KV cache 实际占用**（DeepSeek-V2/V3 实际数字）：

$$\text{cache} = \underbrace{d_c}_{\text{latent KV}} + \underbrace{d_k^{\text{rope}}}_{\text{rope key 单独存}} = 512 + 64 = 576 \text{ 维/token/layer}$$

bf16 即 $576 \times 2 = 1152$ B/token/layer。相比 LLaMA-3 70B 的 32 KB 缩小 ~28×；128k context 单 sample $\approx$ 5 GB（vs 朴素 MHA 的 ~220 GB）。

**Q 端的处理**：DeepSeek-V2 给 query 也加了一层 down-projection（$X \to c_t^Q$ 维度 $d'_c$，再 up-project 出 Q）——这部分主要是减小训练的 attention 投影矩阵参数量、与 KV cache 关系不大，本节不展开。详见原 paper §2.1.3。

### 2.6 MLA 的 forward 流程伪代码

```
# === 训练 / prefill ===
c_kv = X @ W_DKV                          # (B, T, d_c)        compress
k_rope_full = X @ W_KR                    # (B, T, d_k_rope)   单独的 rope key
k_rope_full = apply_rotary(k_rope_full)   # 加 RoPE

# decode 时只需要缓存这两个：
cache_c_kv.append(c_kv)                   # (B, T_cur, d_c)
cache_k_rope.append(k_rope_full)          # (B, T_cur, d_k_rope)

# === Attention（每步重新 up-project）===
K_nope = c_kv @ W_UK    # → (B, T_cur, h * d_k_nope)，再 reshape 到 (B, h, T_cur, d_k_nope)
V       = c_kv @ W_UV    # → (B, T_cur, h * d_k_v),    再 reshape 到 (B, h, T_cur, d_k_v)
K = concat([K_nope, k_rope_broadcast], dim=-1)  # 拼回完整 K (B, h, T_cur, d_k_nope+d_k_rope)

# Q 端类似处理（含 nope + rope 拆分）
out = attention(Q, K, V)                  # 标准 SDPA
```

**核心要点**：
- cache 里只存 `c_kv` 与 `k_rope`，不存完整 K, V
- attention 时**每步都要现场 up-project**——多了一次 matmul
- 但 cache 显存压力锐减——典型 trade-off：用一点 compute 换大量 memory

DeepSeek-V2 / V3 / R1 全用 MLA。**工程复杂度明显高于 GQA**——尤其 RoPE 解耦那部分自己实现非常容易出错，强烈建议直接读 DeepSeek-V2 paper §2.1.3 + 官方 modeling code。

### 2.7 三方案对比表（必背）

| 方案 | KV cache size (per token, per layer) | 质量（vs MHA） | 工程复杂度 | 代表模型 |
|---|---|---|---|---|
| **MHA**（标准） | $2 \cdot h \cdot d_k$ bytes (bf16) | 100% (baseline) | 简单 | GPT-2 / 3、LLaMA-1、LLaMA-2 7B/13B |
| **MQA** | $2 \cdot d_k$ (1 KV) | 98-99% | 简单 | PaLM、StarCoder、Falcon-180B |
| **GQA** ($g = 8$) | $2 \cdot \frac{h}{g} \cdot d_k$ | 99-100% | 简单 | LLaMA-2/3 70B、Mistral、Qwen2/2.5、Gemma |
| **MLA** | $\sim 2 \cdot (d_c + d_k^{\text{rope}})$ (典型 ~1152 B) | 100%（DeepSeek 实证） | 复杂（与 RoPE 解耦） | DeepSeek-V2 / V3 / R1 |

### 2.8 现代 LLM 选择速览（必背）

| 模型 | 注意力方案 | $h_q$ | $h_{kv}$ / $g$ | $d_k$ | KV cache (per token per layer, bf16) |
|---|---|---|---|---|---|
| LLaMA-2 7B | MHA | 32 | 32 | 128 | 8192 B |
| LLaMA-2 13B | MHA | 40 | 40 | 128 | 10240 B |
| LLaMA-2 70B | GQA-8 | 64 | 8 | 128 | 2048 B |
| LLaMA-3 8B | GQA-8 | 32 | 8 | 128 | 2048 B |
| LLaMA-3 70B | GQA-8 | 64 | 8 | 128 | 2048 B |
| LLaMA-3 405B | GQA-8 | 128 | 8 | 128 | 2048 B |
| Mistral 7B | GQA-8 | 32 | 8 | 128 | 2048 B |
| Mixtral 8×7B | GQA-8 | 32 | 8 | 128 | 2048 B |
| Qwen2.5 7B | GQA-4 | 28 | 4 | 128 | 1024 B |
| Qwen2.5 72B | GQA-8 | 64 | 8 | 128 | 2048 B |
| Gemma-2 27B | GQA-2 | 32 | 16 | 128 | 4096 B |
| **DeepSeek-V2** | **MLA** | 128 | (n/a) | (latent + rope) | **~1152 B**（含 RoPE 部分） |
| **DeepSeek-V3** | **MLA** | 128 | (n/a) | (latent + rope) | **~1152 B** |

**眼睛眯一下能看出三件事**：
1. 7B-13B 旧模型还有 MHA，但 30B+ 几乎全 GQA
2. GQA-8 是事实主流（KV head 数固定 8）
3. DeepSeek 走 MLA 路线，KV cache 比 GQA-8 还小 ~2× 但工程更复杂

### 2.9 前沿：Cross-layer KV 共享（YOCO / CLA）

GQA / MLA 都是在**同一层内**减少 KV cache。一个更激进的方向是**跨层共享**——既然相邻层的 attention 模式有相关性，能不能让相邻几层共享同一份 KV cache？

代表方案：
- **YOCO**（Sun et al. 2024，You Only Cache Once）：模型分前后两半，前半正常算 KV，后半所有层共享前半最后一层的 KV cache
- **CLA**（Cross-Layer Attention，Brandon et al. 2024）：相邻 $k$ 层共享同一组 KV cache，$k$ 通常取 2-4

预期收益：在 GQA 之上再 ÷ 2-4 倍 KV cache。但这一方向**还在试验阶段、未成主流**——质量影响仍在 calibration 中。本节只做提及，详细实证看 6.5 / 11.x。

### 2.10 与 PagedAttention / 量化的协同

现代 LLM 推理优化有**三个层次**叠加：

| 层次 | 解决什么 | 代表方案 | 收益 |
|---|---|---|---|
| **架构层** | 让每 token 每层 KV 数据少 | GQA / MQA / MLA | ÷ 4-32× |
| **内存管理层** | 减少 padding 浪费 + 多 user 共享前缀 | PagedAttention（vLLM） / RadixAttention（SGLang） | ÷ 2-3× 碎片浪费 |
| **数值精度层** | 降低每个数的字节数 | INT8 / FP8 KV cache | ÷ 2-4× |

**三层互相正交**——可以叠加使用。例如 LLaMA-3 70B production 部署的典型组合：
- GQA-8（架构层）→ KV cache ÷ 8
- vLLM PagedAttention（内存层）→ 碎片率从 60% 降到 < 5%
- FP8 KV cache（精度层）→ 再 ÷ 2

总效果：相比朴素 MHA + naive batching + bf16，KV cache 显存压力降 ~30-50×——这才是 70B 模型在 80 GB H100 上能服务**千 user**的根本原因。

本节只讲架构层（GQA/MLA）；内存管理层（11.2 PagedAttention、11.3 RadixAttention）与数值精度层（11.4 量化）在 Module 11 详讲。

---

## 3. 最小代码示例

### 3.1 手撕 GQA forward（30 行 PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """GQA：h 个 Q head + g 组 KV head；g | h 必须成立。"""

    def __init__(self, d_model: int, n_head: int, n_kv_head: int):
        super().__init__()
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_head = n_head             # h_q：Q head 数
        self.n_kv_head = n_kv_head       # g：KV head 数
        self.n_rep = n_head // n_kv_head # 每组的 Q head 数（用于 expand）
        self.d_k = d_model // n_head

        # Q 仍是 h_q × d_k 维输出；K, V 只是 g × d_k 维输出（缩小 h/g 倍）
        self.W_q = nn.Linear(d_model, n_head * self.d_k,  bias=False)         # (d, h_q*d_k) = (d, d)
        self.W_k = nn.Linear(d_model, n_kv_head * self.d_k, bias=False)       # (d, g*d_k) ← 关键
        self.W_v = nn.Linear(d_model, n_kv_head * self.d_k, bias=False)       # (d, g*d_k)
        self.W_o = nn.Linear(n_head * self.d_k, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        # 1) 投影 + reshape
        Q = self.W_q(x).view(B, T, self.n_head,    self.d_k).transpose(1, 2)  # (B, h_q, T, d_k)
        K = self.W_k(x).view(B, T, self.n_kv_head, self.d_k).transpose(1, 2)  # (B, g,   T, d_k)
        V = self.W_v(x).view(B, T, self.n_kv_head, self.d_k).transpose(1, 2)  # (B, g,   T, d_k)

        # 2) 关键一步：把 K, V 从 g 组 expand 到 h_q 组（不是真复制，repeat_interleave 会复制）
        #    PyTorch 2.5+ SDPA 与 FlashAttention 2 已直接支持 GQA，不必 expand
        K = K.repeat_interleave(self.n_rep, dim=1)                            # (B, h_q, T, d_k)
        V = V.repeat_interleave(self.n_rep, dim=1)                            # (B, h_q, T, d_k)

        # 3) 标准 attention
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)         # (B, h_q, T, d_k)

        # 4) 拼回 (B, T, d) + W_O
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_k)
        return self.W_o(out)
```

**关键 4 处**：
1. `W_k = nn.Linear(d, g * d_k)`——只投影 $g \cdot d_k$ 维而不是 $d$ 维。这是 GQA 与 MHA 在权重层面的唯一区别
2. `K.repeat_interleave(n_rep, dim=1)` 把 K 从 $(B, g, T, d_k)$ 扩成 $(B, h_q, T, d_k)$——逻辑上每组的 $h_q/g$ 个 Q head 共享同一份 K
3. **cache 时仍只存 $(B, g, T, d_k)$，不要把 expand 后的 $(B, h_q, T, d_k)$ 存进去**——expand 是 attention 时临时做的，cache 才有 ÷ $h/g$ 的好处（详见 §4 第 1 条）
4. PyTorch 2.5+ 的 `F.scaled_dot_product_attention` 与 FlashAttention v2+ 已**原生支持 GQA**，可以直接传 `Q: (B, h_q, T, d_k)` 与 `K, V: (B, g, T, d_k)`，无需手工 expand——内部 kernel 会自动处理 broadcast，性能更优（详见 §4 第 2 条）

### 3.2 KV cache 大小对比（MHA / MQA / GQA / MLA 各算一次）

```python
def kv_cache_per_token_per_layer(scheme: str, h: int = 64, d_k: int = 128,
                                  g: int = 8, d_c: int = 512, d_k_rope: int = 64,
                                  dtype_bytes: int = 2) -> int:
    """返回单 token 单层的 KV cache 字节数（bf16=2, fp8/int8=1）。"""
    if scheme == "MHA":   return 2 * h * d_k * dtype_bytes
    if scheme == "MQA":   return 2 * 1 * d_k * dtype_bytes
    if scheme == "GQA":   return 2 * g * d_k * dtype_bytes
    if scheme == "MLA":   return (d_c + d_k_rope) * dtype_bytes  # 不×2: latent KV 共享 + rope key 单独
    raise ValueError(scheme)

# === LLaMA-3 70B 配置 (h=64, d_k=128, 80 layer, T=8k) 对比 ===
configs = ["MHA", "MQA", "GQA", "MLA"]
for s in configs:
    bytes_per = kv_cache_per_token_per_layer(s)
    total_gb = bytes_per * 80 * 8192 / (1024 ** 3)
    print(f"{s:5s}: {bytes_per:>6d} B/token/layer  →  total {total_gb:6.2f} GB @ 80 layer × 8k context")
# 输出（bf16）:
#   MHA  :  32768 B/token/layer  →  total  20.00 GB
#   MQA  :    512 B/token/layer  →  total   0.31 GB
#   GQA  :   4096 B/token/layer  →  total   2.50 GB
#   MLA  :   1152 B/token/layer  →  total   0.70 GB
```

注意 MLA 的 `2 * (d_c + d_k_rope)` 算法略不同——KV 在 latent 里**共享一个 $c_t$**（不是 K, V 各一份），加上 RoPE key 单独的部分。具体见 §2.5。

### 3.3 MLA forward 伪代码（≤ 20 行）

```python
# pseudo-code，省略 reshape / norm / Q 端 down-projection 等细节
# 完整实现见 DeepSeek-V2 官方 modeling_deepseek.py

class MLAttention(nn.Module):
    def __init__(self, d, h, d_k_nope, d_k_rope, d_v, d_c):
        # d_c = latent dim (e.g. 512); d_k_nope=128, d_k_rope=64
        self.W_DKV = nn.Linear(d, d_c, bias=False)              # down: KV → latent
        self.W_UK  = nn.Linear(d_c, h * d_k_nope, bias=False)   # up: latent → K_nope
        self.W_UV  = nn.Linear(d_c, h * d_v,    bias=False)     # up: latent → V
        self.W_KR  = nn.Linear(d, d_k_rope, bias=False)         # 单独的 rope key（共享，无 head 维）
        # Q 端类似（W_DQ, W_UQ_nope, W_UQ_rope）省略

    def forward(self, x, cache_c=None, cache_kr=None):
        # 1) compress：cache 只存 c_kv 与 k_rope
        c_kv = self.W_DKV(x)                                    # (B, T, d_c)        ← 缓存这个
        k_rope = apply_rotary(self.W_KR(x))                      # (B, T, d_k_rope)  ← 也缓存这个
        if cache_c is not None:
            c_kv  = torch.cat([cache_c,  c_kv ], dim=1)         # 与历史 latent 拼接
            k_rope = torch.cat([cache_kr, k_rope], dim=1)

        # 2) up-project：现场展开成完整 K, V（不缓存）
        K_nope = self.W_UK(c_kv).view(B, T_full, h, d_k_nope)
        V      = self.W_UV(c_kv).view(B, T_full, h, d_v)
        # k_rope 在 head 维 broadcast（所有 head 共享同一个 rope key）
        K = torch.cat([K_nope, k_rope.unsqueeze(2).expand(-1, -1, h, -1)], dim=-1)

        # 3) Q 端类似处理（含 rope + nope 拆分），略
        # 4) 标准 attention
        out = scaled_dot_product_attention(Q, K, V)
        return self.W_o(out), (c_kv, k_rope)                    # 返回 cache
```

**关键三处**：
1. cache 只存 `c_kv` 与 `k_rope`，**不存** K, V
2. attention 时每步现场 up-project（多一次 matmul，trade-off）
3. `k_rope` 是所有 head 共享的（unsqueeze + expand），不是每 head 一个 rope key——这是 MLA 节省 cache 的关键技巧

完整实现强烈建议直接读 DeepSeek-V2 官方 [`modeling_deepseek.py`](https://github.com/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py) 中的 `DeepseekV2Attention`——本节伪代码省略了大量细节（layer norm 位置、Q 端 down-projection、cache shape 优化等）。

---

## 4. 工程踩坑与经验

- ❗ **GQA 实现时 K/V 用 `repeat_interleave` 复制到 $h_q$ 个 head，不要把 expand 后的 K/V 存进 cache**。如果你 `K.repeat_interleave(n_rep, dim=1)` 之后再把 K 塞进 KV cache，cache 大小立刻退化为 MHA（$h_q$ 份）——GQA 的 ÷ $h/g$ 收益完全消失。**正确**：cache 只存 $(B, g, T, d_k)$ 的原始 K, V；attention 时再 expand。本节 §3.1 的实现就是这个顺序。Production 推理代码（vLLM、TGI）做得更彻底——根本不 expand，直接调支持 GQA 的 attention kernel。

- ❗ **FlashAttention v2+ 与 PyTorch SDPA（2.5+）原生支持 GQA，不必 expand 即可调用**。可以直接传 `Q: (B, h_q, T, d_k)` 与 `K, V: (B, g, T, d_k)`，kernel 内部会处理 broadcast。这比手工 `repeat_interleave` 后再调 SDPA 性能好 10-30%（避免临时 expand 的内存拷贝）。**判断方法**：升级 `torch >= 2.5` + `flash-attn >= 2.5`，看 SDPA 文档说明 GQA 支持；若运行报错"expected K, V to have same head count as Q"，说明 backend 太旧、必须手工 expand。

- ❗ **MQA / GQA 训练时 K/V 投影矩阵的初始化不当 → 比 MHA 掉点 1-2 个百分点；upcycling 时把每组 head 的 K/V 平均**。从头训 GQA 模型时，$W_K, W_V$ 的初始化方差应当与 MHA 的 per-head 投影一致（不是按 $g \cdot d_k$ 维度算 $1/d$ 方差，而是按 $d_k$ 维度——否则信号尺度不对）。**Upcycling 经验**（LLaMA-2 70B 论文做法）：从已训好的 MHA ckpt 转 GQA 时，把每组的 $h/g$ 个 K, V head 求平均作为新 K, V 投影的初始化，再短 fine-tune（~5% pretrain compute），可恢复几乎全部性能。直接随机初始化 GQA 投影会显著掉点。

- ❗ **MLA 与 RoPE 解耦设计极其 tricky，自己实现容易错；强烈建议直接读 DeepSeek-V2 paper §2.1.3 + 官方 modeling code**。MLA 把 q, k 拆成 nope（走 latent）+ rope（单独缓存）两半，rope 部分的维度选择（$d_k^{\text{rope}} = 64$）、rope key 是否在 head 维共享、down-projection 矩阵的初始化、attention 时 nope+rope 拼接顺序——任何一处搞错都会让模型不收敛。**没有现成框架支持时不要从零实现**——LLaMA Factory / vLLM / SGLang 的 MLA 支持都是直接 port DeepSeek 官方代码，有微妙差异都要测对齐。

- ❗ **切换 MHA → GQA 时 KV cache shape 改变，预训练 ckpt 不能直接 load**。$W_K, W_V$ 矩阵从 $(d, d)$ 变成 $(d, g \cdot d_k)$，参数量从 $d^2$ 减到 $d \cdot g \cdot d_k$——形状不兼容、`load_state_dict` 会报错。**正确做法**：(1) 用 upcycling 脚本把 MHA ckpt 的 $W_K, W_V$ 按组求平均、保存成新的 GQA shape；(2) 短 fine-tune 恢复性能。HuggingFace 在 LLaMA-2 7B → GQA 实验时给过 reference 脚本，社区也有 `mha-to-gqa` 工具。

- ❗ **MQA 在小模型（< 7B）上掉点比大模型明显，70B+ 上几乎无损（一种 emergence？）**。Shazeer 2019 与后续 ablation 都观察到：MQA 在 < 1B 模型上 perplexity 升 5-10%，在 6B 量级升 2-3%，在 70B+ 量级几乎不掉点。**直觉解释**：大模型有更多冗余 head，MQA 的"压缩到 1 组 KV"对大模型相对小损失；小模型 head 已经少且每个都"工作量饱满"，强行共享 KV 损伤大。**实战建议**：< 7B 模型不要直接用 MQA、用 GQA-8 即可；如果想更激进、用 MLA。

- ❗ **MLA 训练时的 compute 略高于 MHA（多一次 up-projection），但推理 KV cache 降 16-32×——是值得的 trade-off**。MLA 在 forward 里多一次 $c_t \to K, V$ 的 up-projection matmul，训练 step 时间增加 ~5-10%；但 production 推理时 KV cache 显存压力降 16-32×，长 context 与高并发场景的 throughput 提升远超这点训练 overhead。**DeepSeek 团队的算账**：MLA 多花的训练 compute 在第一个月 production 部署里就赚回来了。

- ❗ **GQA 的 group 数 $g$ 选择有 modality 经验**：纯文本 LLM 通常 $g = 8$（LLaMA-3）；多模态 / 长 context 优化倾向 $g = 4$ 或更小（Qwen2.5 7B 取 $g = 4$）；代码模型可以更激进（StarCoder 直接用 MQA）。**没有"一刀切"答案**——需要在自己的数据 + 任务上做 ablation。一个反直觉的发现：增大 $g$（更接近 MHA）有时反而不涨点，说明 GQA-8 已经接近天花板。

- ❗ **不同 GQA 配置的 ckpt 不能跨架构 LoRA / merge**：LLaMA-3 8B（GQA-8）的 LoRA 不能直接套到 LLaMA-2 7B（MHA）上——$W_K, W_V$ shape 不同。同理 MLA 与 GQA 之间也不能。**实战规则**：LoRA / 量化 / 蒸馏 / 权重 merge 必须在**KV 投影结构完全相同**的模型族内进行。

- ❗ **MLA 的 cache 形状与 GQA 完全不同，推理引擎要单独适配**：vLLM / SGLang 早期版本只支持 MHA / GQA、不支持 MLA；DeepSeek-V2 / V3 上线初期社区花了几个月才把 MLA 支持加进 vLLM 主线。**production 选型 tip**：如果你确定要跑 DeepSeek 系，要么用 DeepSeek 官方 inference engine、要么用支持 MLA 的最新版 vLLM / SGLang；不要用 HuggingFace `transformers.generate`（性能太差）。

---

## 5. 经典 paper

- **Shazeer, 2019 — Fast Transformer Decoding: One Write-Head is All You Need** — MQA 必读。Google 内部短 paper（5 页），直接给出"所有 head 共享 1 组 KV"的设计，并实证在 6B WMT 翻译模型上 perplexity 略升但 decode 速度大幅提升。读它能体会"KV cache 压缩这件事在 2019 年就被想到了，只是当时模型还没大到痛"——MQA 是 GQA / MLA 的精神祖师爷。
- **Ainslie et al., 2023 — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** — GQA 必读。提出"$g$ 组分组共享"的中庸方案，并给出 upcycling 脚本（从 MHA ckpt 转 GQA、~5% pretrain compute 恢复性能）。读它能看到 GQA-8 是 sweet spot 的实证依据，以及为什么 LLaMA-2 70B 选了这个配置。这篇是过去 3 年最广引用的"小论文大影响"代表作之一。
- **DeepSeek-AI, 2024 — DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** — MLA 必读。§2.1 详细描述 MLA 设计——latent down-projection、与 RoPE 的 decoupled 处理、Q 端的 low-rank。读 §2.1.1 / §2.1.3 即可（30-45 分钟），会理解为什么 MLA 比 GQA 更激进 + 为什么必须把 RoPE 拆出来。配合官方 [`modeling_deepseek.py`](https://github.com/deepseek-ai/DeepSeek-V2) 的 `DeepseekV2Attention` class 阅读。
- **加分阅读：Touvron et al., 2023 — LLaMA-2** — §2.2 简短描述 70B 用 GQA 的工程动机与配置，是 GQA 在工业大模型上落地的范例。可以与 Ainslie 2023 paper 对照，看"学术提出 → 工业采用"的完整路径。

---

## 6. 自测与面试题

**Q1（公式 + 算账）**：写出 GQA 与 MHA 的参数对比；并算出 LLaMA-3 70B（$n_{\text{layer}} = 80, h = 64, g = 8, d_k = 128, T = 8\text{k}, \text{bf16}$）在 GQA 下的单 sample KV cache 大小，与 MHA 对比缩小几倍。

<details>
<summary>Answer sketch</summary>

**参数对比**（每层）：

| 组件 | MHA | GQA |
|---|---|---|
| $W_Q$ | $(d, h \cdot d_k) = (d, d)$ | $(d, h \cdot d_k) = (d, d)$（不变） |
| $W_K$ | $(d, h \cdot d_k) = (d, d)$ | $(d, g \cdot d_k)$ ← 缩 $h/g$ |
| $W_V$ | $(d, h \cdot d_k) = (d, d)$ | $(d, g \cdot d_k)$ ← 缩 $h/g$ |
| $W_O$ | $(d, d)$ | $(d, d)$（不变） |
| 总参数 | $4 d^2$ | $2 d^2 + 2 d \cdot g \cdot d_k = 2 d^2 + 2 d^2 g/h$ |

LLaMA-3 70B 数字：$h = 64, g = 8$，$g/h = 1/8$，每层 attention 参数相对 MHA 减少 $2 d^2 \cdot (1 - 1/8) = 1.75 d^2$（约 22%）。整模型参数减少不大（FFN 占大头），但 **KV cache 减少巨大**。

**KV cache 算账**：

公式：$M_{\text{KV}} = 2 \cdot L \cdot h_{\text{kv}} \cdot d_k \cdot T \cdot \text{bytes}$

- **MHA** ($h_{\text{kv}} = h = 64$)：
$$M = 2 \times 80 \times 64 \times 128 \times 8192 \times 2 \text{ B} \approx 21.5 \text{ GB}$$

- **GQA-8** ($h_{\text{kv}} = g = 8$)：
$$M = 2 \times 80 \times 8 \times 128 \times 8192 \times 2 \text{ B} \approx 2.7 \text{ GB}$$

**缩小 8×**——与 $h/g = 64/8 = 8$ 一致。

加分：能指出 GQA 对模型总参数量影响很小（FFN 是参数大头）、对推理 KV cache 影响巨大；能算 GQA-8 + INT8 KV cache 再 ÷ 2 = 1.35 GB。

</details>

**Q2（实现）**：用 PyTorch 写一个最小 GQA forward（≤ 15 行），包含 K/V 从 g 组 expand 到 $h_q$ 组的关键逻辑。

<details>
<summary>Answer sketch</summary>

核心 ≤ 15 行：

```python
def gqa_forward(self, x):
    B, T, _ = x.shape
    h, g, d_k = self.n_head, self.n_kv_head, self.d_k
    n_rep = h // g
    # 1) 投影 + reshape：注意 K, V 用 g 组而非 h 组
    Q = self.W_q(x).view(B, T, h, d_k).transpose(1, 2)              # (B, h, T, d_k)
    K = self.W_k(x).view(B, T, g, d_k).transpose(1, 2)              # (B, g, T, d_k) ← 关键
    V = self.W_v(x).view(B, T, g, d_k).transpose(1, 2)              # (B, g, T, d_k)
    # 2) expand：每组 K/V 复制 n_rep 份给该组的 h/g 个 Q head 用
    K = K.repeat_interleave(n_rep, dim=1)                           # (B, h, T, d_k)
    V = V.repeat_interleave(n_rep, dim=1)                           # (B, h, T, d_k)
    # 3) 标准 attention
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)   # (B, h, T, d_k)
    out = out.transpose(1, 2).contiguous().view(B, T, h * d_k)
    return self.W_o(out)
```

考核要点：
- `W_k = nn.Linear(d, g * d_k)` 而不是 `(d, d)` —— GQA 与 MHA 在权重上的唯一区别
- K, V reshape 到 `(B, g, T, d_k)` 而不是 `(B, h, T, d_k)`
- `repeat_interleave(n_rep, dim=1)` 把 K, V 在 head 维 expand 到 $h_q$
- **cache 时仍存 $(B, g, T, d_k)$**，attention 时才 expand——这是 KV cache ÷ $h/g$ 的关键
- PyTorch 2.5+ / FlashAttention 2 原生支持 GQA，可以省略 expand 直接传，性能更优

加分：能说出 `repeat_interleave` 与 `expand` 的差异（一个真复制 + 占内存、一个 stride 重排不复制）；能解释为什么生产代码不 expand；能指出 GQA-8 是甜点（质量几乎无损）。

</details>

**Q3（trade-off）**：MLA 比 GQA 工程复杂得多（需与 RoPE 解耦、cache 形状不同推理引擎要单独适配），DeepSeek 为什么坚持用？至少给 2 个理由。

<details>
<summary>Answer sketch</summary>

至少要点到 2-3 个理由：

**理由 1：长 context 经济性（核心动机）**
- DeepSeek-V2/V3 主打 128k context（V3 部分版本 1M+），long context 下 KV cache 是绝对 OOM 杀手
- MLA 把 KV cache 压到 ~1152 B/token/layer，比 GQA-8 的 2048 B/token/layer 再小 ~2×、比朴素 MHA 小 ~28×
- 128k context 下：MHA ~220 GB / sample（不可能上线）、GQA-8 ~10 GB、**MLA ~5 GB**——只有 MLA 能在 80 GB H100 上 batch ~10+ 同时服务多 user

**理由 2：高并发服务的 throughput**
- production 部署的 throughput ∝ batch_size，batch_size ∝ 1 / 单 sample KV cache
- MLA 相比 GQA-8 把单 sample KV cache 再降 ~2× → 同显存下 batch ~2× → throughput ~2×
- 对 to-C 商业模型（DeepSeek-Chat / R1 API）来说就是**直接 ÷ 2 推理成本**——这是 DeepSeek API 价格能比 GPT-4 便宜数十倍的关键之一

**理由 3：质量未损（甚至略好）**
- DeepSeek-V2 论文实证：MLA 在 perplexity / benchmark 上与 MHA 持平甚至略优
- 因为 MLA 的 latent down-projection 引入了一个 low-rank bottleneck，类似一种"被迫 regularization"——不像 MQA 那种暴力压缩会显著掉点
- 也与 head pruning 文献一致——KV head 大量冗余，MLA 的"取消 head 维"反而让模型更紧凑

**理由 4（加分）：训练 compute 增加可控**
- MLA forward 多一次 up-projection matmul，训练 step 时间增加 ~5-10%
- 但 DeepSeek 算账：production 第一个月就赚回（推理 throughput 提升远超训练 overhead）

**理由 5（加分）：一致性 + 路径依赖**
- DeepSeek 从 V2 → V3 → R1 一直用 MLA，所有训练管线、infra、推理引擎都围绕 MLA 优化
- 切回 GQA 反而是更大的工程切换成本——路径依赖锁定

加分：
- 能指出 MLA 的代价不是"质量不行"而是"工程门槛 + 推理引擎适配成本"
- 能说出"如果你不是 DeepSeek、不需要 128k context + 千 user 高并发，GQA-8 已经够用"——这是工程师的成本意识
- 能引用具体数字：DeepSeek-V3 671B 在 128k context 上的 cache 仅 ~5 GB / sample

</details>

---

## 7. 延伸阅读

- [Shazeer 2019 — Fast Transformer Decoding (arXiv:1911.02150)](https://arxiv.org/abs/1911.02150) — MQA 原 paper，5 页短文必读
- [Ainslie et al. 2023 — GQA (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245) — GQA 原 paper，含 upcycling 实证
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — §2.1 MLA 设计完整描述
- [DeepSeek-V3 paper (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — V3 沿用 MLA + FP8 训练的工程报告
- [DeepSeek-V2 官方 modeling code](https://github.com/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py) — `DeepseekV2Attention` 是 MLA 工程实现的标准答案
- [LLaMA-2 paper (arXiv:2307.09288)](https://arxiv.org/abs/2307.09288) — §2.2 GQA 在 70B 上的工程报告
- [Sebastian Raschka — Understanding Multi-Head Latent Attention](https://magazine.sebastianraschka.com/p/understanding-multi-head-latent-attention) — MLA 的可视化讲解
- [Hugging Face Blog — GQA 解读](https://huggingface.co/blog/gqa) — 配 §2.3 食用
- 推荐继续读本教程的 **5.3 节《FlashAttention 1/2/3：IO-aware kernel》**——架构层之外的 attention 内核优化；FlashAttention 2.5+ 原生支持 GQA 是本节代码运行的底层基础
- 推荐继续读本教程的 **11.2 节《PagedAttention 与 Continuous Batching（vLLM）》**——KV cache 的内存管理优化，与本节架构层压缩正交叠加
- 推荐继续读本教程的 **11.4 节《量化：GPTQ / AWQ / GGUF / FP8 / SmoothQuant》**——KV cache 数值精度优化（FP8 / INT8），与本节叠加
