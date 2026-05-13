---
title: "4.2 Multi-head attention 与 head 的物理意义"
description: "把 4.1 的单头 self-attention 沿 hidden 维切成 $h$ 个 $d_k = d/h$ 维子空间并行做 attention，最后用 $W_O$ 拼回——这样模型可以同时学多种注意力 pattern（句法 / 语义 / 共指 / 局部 / 全局），而总参数量与单头基本相同。本节把多头公式、$d_k = d/h$ 的参数等价性、head 的可解释性实证、\"3 个大 Linear"
---

> ⏱ 预计阅读 40 分钟 ｜ 难度 ★★ 🔥 必考 ｜ 前置：4.1 self-attention

## 一句话本节讲什么

把 4.1 的单头 self-attention 沿 hidden 维切成 $h$ 个 $d_k = d/h$ 维子空间并行做 attention，最后用 $W_O$ 拼回——这样模型可以**同时学多种注意力 pattern**（句法 / 语义 / 共指 / 局部 / 全局），而总参数量与单头基本相同。本节把多头公式、$d_k = d/h$ 的参数等价性、head 的可解释性实证、"3 个大 Linear + reshape"的高效实现、KV cache 按 head 线性增长的工程后果（GQA / MQA / MLA 的根本动机）一次讲透——是 4.6 完整 decoder、4.7 KV cache、5.2 KV cache 压缩的共同基础。

---

## 1. Mental model（直觉）

### 1.1 单头的根本局限

回顾 4.1：一组 $W_Q, W_K, W_V$ 把每个 token 投影成一个 query、一个 key、一个 value，然后做一次 $\text{softmax}(QK^\top/\sqrt{d_k})V$。这意味着**整个序列只能学到一种"相似度"概念**——比如"哪些 token 是当前 token 的句法父节点"。

但语言里同一个 token 同时与多个其他 token 有多种关系：

```
"The cat that the dog chased ate the fish."
                                  ↑
                       ate 这个 token 同时关心:
   ① 主语   The cat        (主谓关系，跨 5 个 token 的长程依赖)
   ② 宾语   the fish       (动宾关系，紧跟其后)
   ③ 时态   过去时          (与从句 chased 时态一致)
   ④ 修饰   that-clause    (从句嵌套层级)
```

让一组 $W_Q, W_K, W_V$ 同时编码这 4 种关系——它们想要的"相似度"度量完全不同——本质是过载。从信号处理角度类比：单头 attention 像只有一个 IIR 滤波器，必须在所有频率上做 trade-off；multi-head 像把信号送进一个 filter bank，每个滤波器专注一段频率。

### 1.2 multi-head 的核心想法

Vaswani 2017 的 fix 极其朴素：

> 与其用 1 组 $d \to d$ 的高维投影学 1 种 attention，不如用 $h$ 组 $d \to d/h$ 的低维投影**并行**学 $h$ 种 attention，最后把 $h$ 个结果拼起来。

关键设计：

- **$d_k = d/h$**：每个 head 在 $d/h$ 维子空间做 attention，总参数量 $h \cdot 3 \cdot d \cdot (d/h) = 3 d^2$，与单头完全一样
- **并行**：$h$ 个 head 之间没有依赖，一次大 matmul + reshape 全做完，wall-clock 时间几乎与单头持平
- **末尾 $W_O$ 投影**：拼接 $h$ 个 head 的输出后过一个 $d \to d$ 的线性层，让 head 之间**信息混合**

直觉图——4 个 head 各看一种关系：

```
           input X (B, T, d)
                │
   ┌───────┬────┴────┬───────┐         切成 4 份 (沿 d 维)
   ▼       ▼         ▼       ▼
 head_1  head_2   head_3   head_4      每个在 d/4 维子空间做 attention
 (句法)  (邻居)   (共指)   (长程)        ← 实证发现的 head 分工
   │       │         │       │
   └───────┴────┬────┴───────┘         concat 回 (B, T, d)
                │
              W_O                       d×d 线性混合
                │
                ▼
           output (B, T, d)
```

### 1.3 一个常见的误解

> "multi-head 是把同一份 Q/K/V 复制 $h$ 份分别做 attention 然后平均。"

**错**。multi-head 的每个 head 有**自己独立的 $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$**，三个投影都不一样——所以每个 head 看到的 Q/K/V 都不同，本质是在 $h$ 个**不同子空间**分别做 attention。这点是面试经典坑。

### 1.4 为什么 multi-head 有用（直觉版）

- **多视角并行**：$h$ 个独立 head 可以学不同的 attention pattern，整体表达力 $\gg$ 单头
- **正则化效果**：把高维 $d$ 切成 $h$ 个低维 $d/h$ 子空间，每个 head 的 $QK^\top$ 是低秩的（rank $\le d/h$），这种"被迫低秩"反而像 inductive bias，不容易过拟合到某一种模式
- **集成思想**：$h$ 个 head 类似 $h$ 个 weak learner，concat + $W_O$ 类似 ensemble；后续 head pruning 实证（Michel 2019）发现很多 head 是冗余的——这反过来印证了"集成视角"是有价值的

---

## 2. 公式与原理

### 2.1 公式（完整推导）

设输入 $X \in \mathbb{R}^{n \times d}$（先忽略 batch 维），头数 $h$，每头维度

$$d_k = d_v = d / h$$

要求 **$h \mid d$**（$h$ 整除 $d$），否则切不齐。

每个 head $i \in \{1, \dots, h\}$ 有自己的三组投影矩阵：

$$W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d \times d_k}$$

第 $i$ 个 head 的输出（直接复用 4.1 的 single-head attention）：

$$\text{head}_i = \text{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)}) = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i \in \mathbb{R}^{n \times d_k}$$

把 $h$ 个 head 沿最后一维拼接，再经一个输出投影 $W_O \in \mathbb{R}^{(h \cdot d_k) \times d} = \mathbb{R}^{d \times d}$：

$$\boxed{\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)\, W_O \in \mathbb{R}^{n \times d}}$$

形状全程：$X: (n, d) \to \text{每个 head}: (n, d/h) \to \text{Concat}: (n, d) \to \text{MultiHead}: (n, d)$。带 batch 时全部前面加 $B$。

### 2.2 $d_k = d/h$ 的参数等价性

为什么强制 $d_k = d/h$ 而不是别的？因为这样能让"$h$ 个小头"和"1 个大头"的**总参数量相等**，于是模型容量持平、提升完全归功于"多视角"而非"参数变多"。

参数量逐项算：

| 组件 | 单头 ($h = 1, d_k = d$) | 多头 ($h$ heads, $d_k = d/h$) |
|---|---|---|
| 一头的 $W_Q$ | $d \cdot d = d^2$ | $d \cdot (d/h) = d^2 / h$ |
| 一头的 $W_K$ | $d^2$ | $d^2 / h$ |
| 一头的 $W_V$ | $d^2$ | $d^2 / h$ |
| 所有 head 的 Q+K+V | $3 d^2$ | $h \cdot 3 \cdot d^2 / h = 3 d^2$ |
| $W_O$ | 单头时常省略 | $d \cdot d = d^2$（不能省） |
| **合计** | $3 d^2$ | $\mathbf{4 d^2}$ |

注意：multi-head 比单头多了一个 $W_O$（$d^2$ 参数）。所以严格说 multi-head 不是"白嫖容量"——多了 $\sim 33\%$ 的参数（$W_O$ 那一份）。但即使把单头也加上一个 $d \to d$ 的 output projection 做 fair compare，multi-head 在 BLEU / perplexity 上仍然显著领先（Vaswani 2017 §5.4）。

### 2.3 为什么不用一个大 d 单头

直觉上"我用 $d_k = d$ 的单头，参数量更多还更简单，效果应该更好吧？"——错。Vaswani 2017 §5.4 的 ablation（base Transformer 在 WMT14 EN-DE）：

| 配置 | $h$ | $d_k$ | BLEU |
|---|---|---|---|
| single-head | 1 | 512 | 24.1 |
| **multi-head** | **8** | **64** | **25.1** |
| many-head | 16 | 32 | 24.9 |
| extreme-head | 32 | 16 | 24.2 |

观察三件事：

- **8 head 比 1 head 涨 1.0 BLEU**，比把模型换成 LSTM 还显著
- **过多 head（32）反而下降**——每个 head 的 $d_k = 16$ 太小，承载不下足够信息
- **存在最优 head 数**——经验上 $d_k$ 落在 32-128 范围最稳

为什么单头不行？单头在 $d$ 维空间里只能做**一种**线性投影后的 attention，被迫把所有"想关注的关系"塞进一组 Q/K/V；多头允许每组 Q/K/V 学一种独立的 attention pattern，本质是在表达力（更多 attention 模式）与每个 pattern 容量（每个子空间维度）之间找平衡。$d_k = d/h$ 让两者大致折中。

### 2.4 head 数的工程选择

业界 LLM 的 $(d, h, d_k)$ 配置：

| 模型 | $d$ | $h$ | $d_k$ | 注释 |
|---|---|---|---|---|
| GPT-2 small (124M) | 768 | 12 | 64 | 经典配置 |
| GPT-3 (175B) | 12288 | 96 | 128 | 大模型 $h$ 多 |
| LLaMA-2 7B | 4096 | 32 | 128 | $d_k = 128$ 是甜点 |
| LLaMA-2 13B | 5120 | 40 | 128 | $d_k$ 不变，$h$ 跟 $d$ 涨 |
| LLaMA-2 70B | 8192 | 64 | 128 | KV head 已用 GQA = 8 |
| Qwen-2.5 72B | 8192 | 64 | 128 | 同 LLaMA |
| DeepSeek-V2 (236B) | 5120 | 128 | 128 | MLA 进一步压缩 KV |
| DeepSeek-V3 (671B) | 7168 | 128 | 128 | MLA + FP8 |

经验规律：

- **$d_k$ 几乎所有现代 LLM 都固定在 64 或 128**——再小信息不够，再大冗余且 $\sqrt{d_k}$ scaling 不够压
- **$h$ 跟着 $d$ 线性涨**：$h = d / d_k$，比如 $d = 8192, d_k = 128 \Rightarrow h = 64$
- **GQA / MLA 让 KV head 数 $\ll$ Q head 数**：详见 5.2，本节按"标准 multi-head（$h_{kv} = h_q$）"处理

### 2.5 head 的可解释性（必考）

Vaswani 2017 paper 末尾就给了 attention map 可视化（Figure 3-5）显示某些 head 确实学到了 anaphora（指代）。后续工作把这件事系统化——最经典的是 **Clark et al. 2019 《What Does BERT Look At?》**：

- **不同 head 学到不同 syntactic / semantic 关系**：
  - 浅层（layer 1-3）：很多 head 关注**邻近 token**（前一个 / 后一个）
  - 中层（layer 5-8）：head 开始学到**句法依赖**（直接宾语 → 动词、形容词 → 名词）
  - 深层（layer 9-12）：head 学到**长程语义关系**（指代消解、共指、coreference）
- **separator head**：相当一部分 head 强烈关注 `[CLS]` / `[SEP]` / 句号——这些 head 可能相当于"no-op"（没找到要关注的，就 fall back 到 separator）
- **head 之间高度冗余**：很多 head 学的 pattern 几乎一样

更激进的实证：**Michel et al. 2019 《Are Sixteen Heads Really Better than One?》**

- 在训好的 BERT / Transformer-MT 上做 head pruning：每次去掉一个 head 看性能下降
- 结论：**大部分 head 可以剪掉而不掉点**，某些 layer 16 个 head 剪到只剩 1-2 个仍然 work
- 但少数 head 是"关键 head"，剪掉立刻崩

实战 take-away：

- **head 不是严格"一个 head 一个角色"**——大部分 head 重复或几乎不工作，少数 head 承担主要功能
- 这是**后续 GQA / MQA / MLA 的根本动机**：既然多数 head 冗余，那 KV 投影不必每个 Q head 都配一份
- 可解释性研究（mechanistic interpretability）现在主要工作之一就是定位"功能 head"（如 induction head、name mover head），Anthropic 的 circuits 系列就是这个方向

### 2.6 KV cache 占用预告（与 4.7 / 5.2 衔接）

推理时每生成一个 token，K/V 都要保存下来供后续 step 复用（避免重算前缀，详见 4.7）。每 layer 缓存的 KV 量：

$$\text{KV size per layer} = 2 \times h \times T \times d_k \times \text{bytes per scalar} = 2 \times T \times d \times \text{bytes}$$

（用了 $h \cdot d_k = d$，2 是 K + V 两份）

**LLaMA-2 70B 的具体数字**：80 layer、$d = 8192$、$T = 8\text{k}$、bf16（2 byte）：

$$80 \times 2 \times 8192 \times 8192 \times 2 \approx 21 \text{ GB}$$

**单个 sample** 的 KV cache 就是 21 GB——70B 模型 weights 本身 bf16 也只有 140 GB。如果 batch = 16，KV cache 就是 336 GB——已经超过模型本身。

这是 **GQA / MQA / MLA 出现的根本动机**（5.2 详讲）：让 KV head 数 $h_{kv} \ll h_q$（或者投影到 latent 空间），KV cache 直接降 $h_q / h_{kv}$ 倍。LLaMA-2 70B 用 $h_{kv} = 8$（vs $h_q = 64$），KV cache 直接缩小 8×。

### 2.7 Multi-head 的工程实现：天真 vs 高效

**天真实现**：$3h$ 个独立的 `nn.Linear(d, d_k)`，循环 $h$ 次做 $h$ 次 attention，最后 concat。

```python
# 不要这么写！
self.W_qs = nn.ModuleList([nn.Linear(d, d_k, bias=False) for _ in range(h)])
self.W_ks = nn.ModuleList([nn.Linear(d, d_k, bias=False) for _ in range(h)])
self.W_vs = nn.ModuleList([nn.Linear(d, d_k, bias=False) for _ in range(h)])

heads = []
for i in range(h):
    q_i = self.W_qs[i](x)                        # (B, T, d_k)
    # ... 单独算每个 head
    heads.append(out_i)
out = torch.cat(heads, dim=-1)                   # (B, T, d)
```

慢的原因：$h$ 次小 matmul + 显式 Python for loop。GPU 上小 kernel launch overhead 会被放大几十倍。

**高效实现（业界标准）**：3 个大 `nn.Linear(d, d)` 一次产 $h$ 个 head 的 Q/K/V，再 reshape + permute 到 `(B, h, T, d_k)`，做 batched attention。

```python
# 这才是工业写法
self.W_q = nn.Linear(d, d, bias=False)           # 一个大投影，等价于 h 个 Linear(d, d_k)
self.W_k = nn.Linear(d, d, bias=False)
self.W_v = nn.Linear(d, d, bias=False)

Q = self.W_q(x)                                  # (B, T, d)
Q = Q.view(B, T, h, d_k).transpose(1, 2)         # (B, h, T, d_k)  ← 关键
# K, V 同理
# 然后 batched attention
attn = (Q @ K.transpose(-2, -1)) / d_k ** 0.5    # (B, h, T, T)
out = (softmax(attn) @ V).transpose(1, 2).reshape(B, T, d)
```

为什么等价？$W_q \in \mathbb{R}^{d \times d}$ 可以**视为** $h$ 个 $\mathbb{R}^{d \times d_k}$ 投影沿列方向 concat 的结果——一次 matmul 后 reshape 把这 $h$ 段拆出来即可。两种实现数学上等价，但高效实现把 $h$ 次小 matmul 合成 1 次大 matmul，GPU 利用率高几十倍。

PyTorch SDPA / FlashAttention 等内置 kernel 都期望输入是 `(B, h, T, d_k)` 这种"head 维与 batch 维并列"的形状——后两维 `(T, d_k)` 才是 attention 的"工作维"。

---

## 3. 最小代码示例

### 3.1 手撕 multi-head causal self-attention（35 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_len: int = 1024):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        # 三个大 Linear，一次产 h 个 head 的 Q/K/V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # 输出投影 W_O —— multi-head 的灵魂，不能省
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        B, T, _ = x.shape
        h, d_k = self.n_head, self.d_k

        # Step 1: 投影 + reshape 成 (B, h, T, d_k)
        Q = self.W_q(x).view(B, T, h, d_k).transpose(1, 2)   # (B, h, T, d_k)
        K = self.W_k(x).view(B, T, h, d_k).transpose(1, 2)   # (B, h, T, d_k)
        V = self.W_v(x).view(B, T, h, d_k).transpose(1, 2)   # (B, h, T, d_k)

        # Step 2: SDPA 自动用 FlashAttention backend，自动处理 causal mask
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)   # (B, h, T, d_k)

        # Step 3: 拼回 (B, T, d) + W_O 混合
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d)
        return self.W_o(out)                                  # (B, T, d)
```

关键点逐条：

- `assert d_model % n_head == 0`：$h \mid d$ 是硬约束，否则 reshape 会 silent 错位
- 用 3 个 `nn.Linear(d, d)`（不是 $3h$ 个 `Linear(d, d_k)`）—— GPU 友好
- `view(B, T, h, d_k).transpose(1, 2)` 把 head 维换到 batch 旁边——SDPA 期望 `(B, h, T, d_k)`
- `F.scaled_dot_product_attention(..., is_causal=True)` 自动选 backend（FlashAttention 优先），不用自己造 mask
- `transpose(1, 2).contiguous().view(B, T, d)` 把 head 维拼回最后一维——`contiguous()` 必须加，否则 `view` 报错
- `self.W_o(out)`：**最关键的一步**。少了它，head 之间没有任何 mixing，每个 head 输出"各管各的 d/h 维"，效果显著掉

### 3.2 用 PyTorch 内置 `nn.MultiheadAttention`

```python
import torch
import torch.nn as nn

class MHACausalNN(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        # 注意 batch_first=True 不是默认！默认是 (T, B, d) 老 RNN 风格
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, batch_first=True, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        T = x.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)  # (T, T) -inf 上三角
        out, _ = self.mha(x, x, x, attn_mask=causal, need_weights=False, is_causal=True)
        return out                                            # (B, T, d)
```

要点：

- `batch_first=True` 必传，否则输入要 `(T, B, d)`——**新手最常踩**
- `nn.MultiheadAttention` 内部已经包了 $W_O$，所以输出直接就是 `(B, T, d)`
- `is_causal=True` 是 PyTorch 1.13+ 的优化提示——配合 SDPA backend 可以走更快路径

### 3.3 数值一致性验证

```python
torch.manual_seed(42)
B, T, d, h = 2, 16, 64, 4
x = torch.randn(B, T, d)

m_manual = MultiHeadCausalSelfAttention(d_model=d, n_head=h, max_len=T)
m_builtin = MHACausalNN(d_model=d, n_head=h)

# 把 builtin MHA 的 in_proj_weight 拆成 W_q, W_k, W_v 与手撕版本对齐
with torch.no_grad():
    in_proj = m_builtin.mha.in_proj_weight                    # (3d, d) 默认拼接
    m_manual.W_q.weight.copy_(in_proj[:d])
    m_manual.W_k.weight.copy_(in_proj[d:2*d])
    m_manual.W_v.weight.copy_(in_proj[2*d:])
    m_manual.W_o.weight.copy_(m_builtin.mha.out_proj.weight)

with torch.no_grad():
    out_manual = m_manual(x)
    out_builtin = m_builtin(x)

print((out_manual - out_builtin).abs().max())                 # 期望 < 1e-5
assert torch.allclose(out_manual, out_builtin, atol=1e-5)
print("PASS: 手撕 multi-head 与 nn.MultiheadAttention 数值一致")
```

跑一次能确认两点：

- 手撕版本数学上正确（与官方实现 bit-level 等价）
- `nn.MultiheadAttention` 内部就是 `in_proj_weight` 拼成 $(3d, d)$ + `out_proj` 这种结构——理解了这个，看 HF transformers 源码不再蒙

### 3.4 工业级参考

Karpathy 的 **nanoGPT `model.py`** 里的 `CausalSelfAttention`（multi-head 版）几乎就是 §3.1 的工业精炼版：加了 dropout、可选 Flash backend 切换、可选 bias，约 50 行内全搞定。强烈推荐对照阅读。

---

## 4. 工程踩坑与经验

- ❗ **reshape 时 transpose 错维度，把 head 与 token 维搞反**。最经典 bug：`Q.view(B, T, h, d_k)` 之后忘了 `.transpose(1, 2)`，直接拿 `(B, T, h, d_k)` 去做 `Q @ K.transpose(-2, -1)`——结果在 `h` 与 `d_k` 维做 matmul 而不是在 `T` 与 `d_k` 维。shape 算出来 `(B, T, h, h)` 看起来"对"，但 attention 跨 head 算成了跨 token 算，loss 不收敛或 NaN。**永远写 shape 注释 `# (B, h, T, d_k)`**，每行 reshape 后人肉验一遍。
- ❗ **`nn.MultiheadAttention` 默认 `batch_first=False`**。PyTorch 这个 API 历史上为了与 RNN 对齐，默认输入 `(T, B, d)`——与现代 `(B, T, d)` 风格冲突。漏传 `batch_first=True` 会得到完全错的输出而**不会报错**（shape 都能对上）。**所有 MHA 实例化都应当传 `batch_first=True`**。
- ❗ **multi-head 的 $W_O$ 投影不能省**。新手手撕时常常觉得"反正 concat 已经把 head 拼回 $d$ 维了，加 $W_O$ 是冗余"——错。少了 $W_O$，head 之间没有 mixing，每个 head 的输出"只贡献最后 $d/h$ 维的部分维度"，下游 FFN 看到的是被切碎的特征，效果会显著掉（Vaswani 2017 ablation 把 $W_O$ 去掉 BLEU 直接掉 0.5+）。
- ❗ **head 数必须能整除 $d$**。`d = 768, h = 10` 这种配置直接 reshape 报错。代码里务必 `assert d_model % n_head == 0`。如果非要用 $h \nmid d$，要么先用 padding 把 $d$ 补到能整除，要么换架构（如 GQA 把 KV head 数与 Q head 数解耦）。
- ❗ **KV cache 占用按 head 数线性增长 → 70B 模型推理 KV cache 经常超过模型本身**。$\text{KV cache} = 2 \cdot L \cdot h \cdot T \cdot d_k \cdot \text{bytes}$——LLaMA-2 70B 用 standard MHA 时 8k context 单 sample 约 21 GB，batch 16 就要 336 GB，比 weights 还大。这是 GQA（让 $h_{kv} = 8$ 而不是 64，KV cache 直接 ÷8）和 MLA（latent 投影后再展开，KV cache ÷16+）出现的根本动机。如果你做推理优化，KV cache 通常是**第一个 OOM 的地方**而不是 weights。
- ❗ **head pruning 论文（Michel 2019）发现 16 head 大部分可剪到只剩 1-2 head 而不掉点 → 暗示标准 multi-head 有冗余**。这个 finding 是 GQA / MQA 的间接动机：既然多数 head 冗余，KV 投影完全没必要每个 Q head 都配一份。但 prune Q head 比较难（涉及 $W_O$ 重新拟合），所以业界更激进的方向是直接砍 KV head（GQA）或砍 KV 维度（MLA）。如果你训自己的 LLM，head 数可以从 small（如 8）开始扫，不必默认 32。
- ❗ **多头实现里 Q/K/V 的 reshape 顺序**：建议**必须**走 `(B, T, d)` → `(B, T, h, d_k)` → `(B, h, T, d_k)`，后两 dim 才是 SDPA 的 batch dim。如果反过来 `(B, T, d) → (B, h, T, d_k)` 用 view 直接 reshape，**结果是错的**——view 不能重排内存，必须先 view 再 transpose。同理输出回来要 `transpose(1, 2).contiguous().view(B, T, d)`，缺 `contiguous()` view 会报错。
- ❗ **fp16 训练 multi-head 时的 $-\infty$ 数值问题与 4.1 一致**。multi-head 不引入新坑，但因为有 $h$ 个 attention map 同时算，fp16 下任何一个出 NaN 都会传染到所有 head。建议长 context + multi-head + fp16 的组合直接换 bf16，或用 SDPA 内部处理。

---

## 5. 经典 paper

- **Vaswani et al., 2017 — Attention Is All You Need** — Transformer 原典，本节就是它 §3.2.2 "Multi-Head Attention" 的复述与扩展。必读 §3.2.2 完整公式 + §5.4 ablation table（$h = 1/4/8/16/32$ 的 BLEU 对比）。整篇 paper 11 页应当反复重读。
- **Clark et al., 2019 — What Does BERT Look At? An Analysis of BERT's Attention** — head 可解释性的经典。系统性分析 BERT 144 个 attention head 学到了什么——syntactic dependency / coreference / separator-attending head 等分类。读完会建立"head 不是均匀干活"的具体直觉，是理解 5.2 GQA 动机的前置。
- **Michel et al., 2019 — Are Sixteen Heads Really Better than One?** — head pruning 的开山之作。结论是**大部分 head 可剪而不掉点**。直接催生了"既然 Q head 冗余 KV head 更冗余" → GQA / MQA 的研究方向。读 abstract + section 3 "Head Importance" 就够。

---

## 6. 自测与面试题

**Q1（公式）**：写出 multi-head attention 的完整公式，并解释为什么 $d_k = d/h$（参数等价性）。

<details>
<summary>Answer sketch</summary>

完整公式：

$$\text{head}_i = \text{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)}) = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i$$

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O$$

其中 $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d \times d_k}$、$W_O \in \mathbb{R}^{d \times d}$、$d_k = d/h$。

为什么 $d_k = d/h$（参数等价性论证）：

- 单头 ($h = 1, d_k = d$) 的 Q/K/V 投影参数量：$3 d^2$
- 多头 ($h$ heads, $d_k = d/h$) 的 Q/K/V 投影参数量：$h \cdot 3 \cdot d \cdot (d/h) = 3 d^2$
- **总参数量相同**——所以 multi-head 的提升完全归功于"$h$ 个独立子空间各学一种 attention pattern"，而不是参数量增加
- 严格说 multi-head 多了一个 $W_O$（$d^2$），所以总参数比单头多 33%——但即使把单头也加 output projection 做 fair compare，multi-head 仍然显著领先

加分：能说出 Vaswani 2017 §5.4 的 ablation 数字（$h = 8$ 比 $h = 1$ 涨约 1 BLEU），能指出 $h$ 过大（$d_k$ 太小）反而下降。

</details>

**Q2（实现）**：用 3 个 `nn.Linear(d, d)` 实现 multi-head Q/K/V 投影，给出 reshape + permute 到 `(B, h, T, d_k)` 的代码（≤ 6 行）。

<details>
<summary>Answer sketch</summary>

核心 6 行：

```python
B, T, d = x.shape
h, d_k = self.n_head, d // self.n_head
Q = self.W_q(x).view(B, T, h, d_k).transpose(1, 2)   # (B, h, T, d_k)
K = self.W_k(x).view(B, T, h, d_k).transpose(1, 2)
V = self.W_v(x).view(B, T, h, d_k).transpose(1, 2)
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)  # (B, h, T, d_k)
```

考核要点：

- 用 3 个 `nn.Linear(d, d)`（不是 $3h$ 个），等价于 $h$ 个 head 的 $W_Q^{(i)}$ 沿列拼接
- reshape 顺序必须是 `(B, T, d) → (B, T, h, d_k) → (B, h, T, d_k)`，先 view 再 transpose
- 直接 `(B, T, d) → (B, h, T, d_k)` view 是错的（view 不能重排内存）
- 输出回到 `(B, T, d)` 需要 `.transpose(1, 2).contiguous().view(B, T, d)` + `W_O`

加分：能解释为什么需要 `contiguous()`；能说出 PyTorch SDPA 期望 `(B, h, T, d_k)` 的形状约定；能指出 head 维放在 batch 旁边是为了 batched matmul 一次算完所有 head。

</details>

**Q3（trade-off）**：head 数取大（如 32）vs 小（如 8）的优缺点；如果你的 head 太多导致 KV cache 爆，业界有哪些解决方案？

<details>
<summary>Answer sketch</summary>

head 数大 vs 小的 trade-off：

| 维度 | $h$ 大（32+） | $h$ 小（8） |
|---|---|---|
| 表达力 | 强（多种 pattern 并行） | 弱（只能学少数 pattern） |
| 单 head 容量 | 小（$d_k = d/h$ 更小） | 大（$d_k$ 大，单 head 信息量足） |
| KV cache | 大（线性 $\propto h$） | 小 |
| 计算并行度 | 高 | 中 |
| 冗余度 | 高（很多 head 重复） | 低 |
| 训练难度 | 略难（每 head $d_k$ 小，方差归一压力大） | 易 |

**经验区间**：$d_k$ 落在 64-128，$h = d / d_k$；GPT-3 取 $h = 96, d_k = 128$；LLaMA-2 7B 取 $h = 32, d_k = 128$；过小 $d_k < 32$ 通常不 work。

KV cache 爆的解决方案（多答更好）：

- **GQA (Grouped-Query Attention)**：把 $h_q$ 个 Q head 分组，每组共享一对 K/V。LLaMA-2 70B 用 $h_q = 64, h_{kv} = 8$，KV cache 直接 ÷8。详见 5.2
- **MQA (Multi-Query Attention)**：极端版 GQA，所有 Q head 共享 1 对 K/V，KV cache ÷$h$。但容量损失较大，纯 MQA 模型已较少
- **MLA (Multi-head Latent Attention)**：DeepSeek-V2/V3 提出，把 KV 投影到一个 latent 空间（维度比 $h \cdot d_k$ 小很多），cache latent 而非 KV 本身。KV cache ÷16+，且性能不掉
- **量化 KV cache**：把 KV 从 bf16 压到 fp8 / int8 / int4，cache 直接 ÷2/4/8（11.4 详讲）
- **PagedAttention（vLLM）**：不直接压缩单个 sample 的 KV，而是用分页内存管理消除碎片，让同等显存能装更多 sample（11.2 详讲）
- **head pruning**：训练后剪掉冗余 head，KV cache 也对应缩小，但精度通常掉一点（Michel 2019）

加分：能区分"减 Q head"与"减 KV head"两条路（GQA/MQA/MLA 都是后者）；能指出 KV cache 通常比 weights 还大是 long-context 推理的核心痛点。

</details>

---

## 7. 延伸阅读

- [Vaswani et al. 2017 — Attention Is All You Need (arXiv)](https://arxiv.org/abs/1706.03762) — §3.2.2 "Multi-Head Attention" 与 §5.4 ablation 必读
- [Clark et al. 2019 — What Does BERT Look At? (arXiv)](https://arxiv.org/abs/1906.04341) — head 可解释性经典，配大量 attention map 可视化
- [Michel et al. 2019 — Are Sixteen Heads Really Better than One? (arXiv)](https://arxiv.org/abs/1905.10650) — head pruning 开山，引出 GQA/MQA 的动机
- [Karpathy — nanoGPT `model.py` `CausalSelfAttention`](https://github.com/karpathy/nanoGPT/blob/master/model.py) — 工业级 multi-head 实现 50 行精炼版
- [Anthropic — A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) — mechanistic interpretability 视角理解 multi-head 的"功能 head"
- [PyTorch — `nn.MultiheadAttention` 文档](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — 注意 `batch_first` 与 `is_causal` 参数
- 推荐继续读本教程的 **4.3 节《位置编码：绝对 / 相对 / RoPE / ALiBi》**——multi-head 仍然是置换等变的，必须显式注入位置信息；以及 **5.2 节《GQA / MQA / MLA》**——multi-head 在长 context 推理时代的演化形态
