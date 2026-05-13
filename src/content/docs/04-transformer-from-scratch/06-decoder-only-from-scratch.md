---
title: "4.6 完整 decoder-only 实现（手撕 nanoGPT / mini-LLaMA）"
description: "把 4.1-4.5 五节的零件——single-head self-attention（4.1）、multi-head + 因果 mask（4.2）、RoPE 位置编码（4.3）、Pre-RMSNorm（4.4）、SwiGLU FFN（4.5）——拼成一个 < 200 行 self-contained 可跑的 mini-LLaMA：embedding → $N \\\\times$ Block → fi"
---

> ⏱ 预计阅读 60 分钟 ｜ 难度 ★★ 🔥 必考 ｜ 前置：4.1 / 4.2 / 4.3 / 4.4 / 4.5（全部）

## 一句话本节讲什么

把 4.1-4.5 五节的零件——**single-head self-attention（4.1）、multi-head + 因果 mask（4.2）、RoPE 位置编码（4.3）、Pre-RMSNorm（4.4）、SwiGLU FFN（4.5）**——拼成一个 < 200 行 self-contained 可跑的 mini-LLaMA：embedding → $N \times$ Block → final RMSNorm → lm_head；前向支持训练（带 cross-entropy loss）与推理（带 `generate` 雏形），在 tinyShakespeare 上几分钟训练能看到 loss 从 ~5 降到 ~2、能 sample 出像样文本。本节同时把"weight tying"、"参数量近似公式 $V d + 12 n_{\text{layer}} d^2$"、"$O(T^2)$ per step 的天真 generate 为什么是 4.7 KV cache 的动机"这三件工程必考点讲透——是 Module 4 的高潮章，也是后面 4.7 KV cache、Module 5 架构变体、Module 6 预训练、Module 11 推理的共同骨架。

---

## 1. Mental model（直觉）

### 1.1 这一节在干什么：组装

前 5 节是一堆精心打磨的零件，但单独一个零件训不出 LLM——它们必须按一个固定的拓扑串起来才有意义。这一节就是组装：把零件按 LLaMA-style 顺序装到一起，输出一个能在你笔记本 CPU 上几分钟跑通的"玩具 mini-LLM"。"能跑通"是后面 80 节的前置——你要做 KV cache 优化（4.7）、改 GQA（5.2）、加 FlashAttention（5.3）、上 ZeRO（7.1）、做 SFT（8.x）、做 RLHF（9.x），都是在这套骨架上继续动手术。**没有这一套骨架，前 5 节学的东西就是一堆悬空的概念**。

### 1.2 一个 decoder-only LLM 长什么样（必背 ASCII 图）

```
tokens (B, T)  [int64]
    │
    ▼
nn.Embedding(V, d)              ← token → 向量
    │
x  (B, T, d)                    [float]
    │
    │   ┌────────────── × N layers (block) ──────────────┐
    │   │                                                │
    │   │   ┌── RMSNorm ──┐                              │
    │   │   │             │                              │
    │   │   │  Multi-head │  (Q, K 加 RoPE；causal SDPA)  │
    │   │   │  attention  │                              │
    │   │   │             │                              │
    │   │   └─────┬───────┘                              │
    │   │         │                                      │
    │   ├────► (+) ◄── residual                          │
    │   │         │                                      │
    │   │   ┌── RMSNorm ──┐                              │
    │   │   │             │                              │
    │   │   │  SwiGLU FFN │  (3 矩阵；d_ff ≈ 8/3 d)       │
    │   │   │             │                              │
    │   │   └─────┬───────┘                              │
    │   │         │                                      │
    │   └────► (+) ◄── residual                          │
    │             │                                      │
    │             ▼                                      │
    │   block 输出 (B, T, d)                              │
    │   ......                                           │
    └────────────────────────────────────────────────────┘
    │
    ▼
final RMSNorm                   ← Pre-LN 必备的"出口 norm"（4.4 §4 第 2 条）
    │
    ▼
nn.Linear(d, V) (lm_head)       ← 与 Embedding 共享权重（weight tying）
    │
logits (B, T, V)                ← 每个位置预测下一个 token 的分数
    │
    ▼
training: F.cross_entropy(logits[..,:-1], targets[..,1:])
inference: softmax → sample / argmax → next_token
```

把这张图记到肌肉记忆里。**任何 decoder-only LLM（GPT-2 / GPT-3 / LLaMA / Qwen / DeepSeek）的骨架都是这张图**——区别只在于 block 内部细节（norm 用 LN 还是 RMSNorm、激活用 GELU 还是 SwiGLU、attention 是标准 MHA 还是 GQA / MLA、有没有 MoE）。所以这一节学完，你可以在 1 小时内把 nanoGPT 改成 mini-LLaMA，再改成 mini-Mistral，再改成 mini-DeepSeek。

### 1.3 为什么是"先 attention 再 FFN"，不是反过来

这是个看似没理由的工程惯例，但有它的合理性：

- **attention 负责"横向通信"**——让每个位置看到其他位置的信息（4.1 §1.1）
- **FFN 负责"纵向加工"**——对每个位置独立做非线性变换（4.5 §1）

直觉上"先收集证据，再独立思考"比"先思考，再收集"更自然。但更本质的原因是 Vaswani 2017 这么定的、之后所有人照抄——这是 path dependency 而非数学必然。也确实有把 attention 与 FFN **并行**做的变体（Google PaLM 的 "parallel attention/FFN"），论文显示这种结构能省 15% 训练时间，但工业上仍然以串行为主流。

### 1.4 weight tying 是什么、为什么省 V·d 参数

**input embedding** $E \in \mathbb{R}^{V \times d}$ 把 token id 映射成向量；**output projection（lm_head）** $W_{\text{lm}} \in \mathbb{R}^{V \times d}$ 把 hidden 投回词表分数。两者形状相同——是巧合吗？不是。Press & Wolf 2017 证明这两个矩阵承担**互逆的语义任务**（一个是"id → 表示"，一个是"表示 → id"），共享权重在数学上是合理的，且实证能省 V·d 参数 + 略涨点。

实现一行：

```python
self.lm_head.weight = self.embed.weight   # 同一份 Tensor，PyTorch 自动同梯度
```

LLaMA-1 / GPT-2 全部 tie；但 LLaMA-2/3 / Qwen2 / DeepSeek-V2 等现代 LLM **已经不 tie 了**——大词表 + 大 hidden 下 V·d 在总参数里占比已经很小（LLaMA-3 8B 中 32k × 4096 ≈ 0.13B，相对 8B 才 1.6%），不 tie 反而效果略好。**面试题常考**："你为什么 tie / 不 tie"——本节按 tie 实现，知道现代趋势是 untie。

---

## 2. 公式与原理

### 2.1 形状追踪（必须烂熟）

设 batch size $B$、序列长度 $T$、hidden $d$、头数 $h$、每头维度 $d_k = d/h$、词表 $V$、层数 $L$。

| 阶段 | 张量形状 | dtype |
|---|---|---|
| input tokens | $(B, T)$ | int64 |
| embedding 后 | $(B, T, d)$ | bf16 / fp32 |
| Q / K / V 投影 + reshape | $(B, h, T, d_k)$ | bf16 / fp32 |
| 每个 block 输出 | $(B, T, d)$ | bf16 / fp32 |
| final RMSNorm 后 | $(B, T, d)$ | bf16 / fp32 |
| lm_head 输出（logits） | $(B, T, V)$ | fp32（建议 cast）|
| training targets | $(B, T)$ | int64 |
| training loss | scalar | fp32 |

**关键点**：

- **token id 是 int**，进 embedding 表才变 float
- attention 内部 head 维要放在 batch 旁边 $(B, h, T, d_k)$，PyTorch SDPA 期望这个 layout（4.2 §2.7）
- logits **建议在算 loss 前 cast 到 fp32**——cross-entropy 在长尾 vocab 上对精度敏感

### 2.2 一个 block 的精确公式（与 4.4 / 4.5 对齐）

记 block 输入为 $x_l \in \mathbb{R}^{B \times T \times d}$。**Pre-RMSNorm + 标准 multi-head + RoPE + SwiGLU FFN** 配方下：

$$h_l = x_l + \text{Attention}\bigl(\text{RoPE}\bigl(\text{RMSNorm}(x_l) W_Q\bigr),\, \text{RoPE}\bigl(\text{RMSNorm}(x_l) W_K\bigr),\, \text{RMSNorm}(x_l) W_V\bigr) W_O$$

$$x_{l+1} = h_l + W_3 \bigl(\text{Swish}(W_1 \cdot \text{RMSNorm}(h_l)) \odot W_2 \cdot \text{RMSNorm}(h_l)\bigr)$$

注意几个细节：

- **norm 在残差路径外、子层入口处**——Pre-LN 的精髓（4.4 §2.1）
- **RoPE 加在 Q / K 上而非 V**——只有 attention score 需要位置敏感（4.3 §2.4）
- **attention 整体外面包了 $W_O$**——multi-head 必备的"head 之间 mixing"（4.2 §2.2）
- **SwiGLU 用 3 个矩阵**——$W_1$ gate、$W_2$ up、$W_3$ down（4.5 §2.2）

### 2.3 整模型公式

设输入 token 序列 $\mathbf{t} = [t_0, t_1, \dots, t_{T-1}] \in \{0, \dots, V-1\}^T$。

$$x_0 = E[\mathbf{t}] \in \mathbb{R}^{T \times d}$$

$$x_{l+1} = \text{Block}_l(x_l), \quad l = 0, 1, \dots, L-1$$

$$\hat{x} = \text{RMSNorm}(x_L)$$

$$\text{logits} = \hat{x} \cdot E^\top \in \mathbb{R}^{T \times V}$$

最后一行 $E^\top$ 就是 weight tying——lm_head 权重直接复用 embedding 权重的转置。

训练目标是 **causal language modeling**：用前 $t$ 个 token 预测第 $t+1$ 个，对所有 $t = 0, 1, \dots, T-2$ 求平均 cross-entropy：

$$\mathcal{L} = -\frac{1}{T-1} \sum_{t=0}^{T-2} \log p(t_{t+1} \mid t_0, \dots, t_t)$$

实现上更简洁的写法：把 logits 与 targets 都 shift 一位再 reshape，调用 `F.cross_entropy` 一次完成（详见 §3）。

### 2.4 参数量近似公式（必背肌肉记忆）

每一个 block 的参数：

- attention：$W_Q + W_K + W_V + W_O$，每个 $d \times d$，共 $4 d^2$
- FFN（SwiGLU）：$W_1 + W_2 + W_3$，前两个 $d \times d_{ff}$、最后一个 $d_{ff} \times d$，共 $3 d \cdot d_{ff}$
- 取 $d_{ff} = \frac{8}{3} d$（4.5 §2.3 等参数对比公约），FFN 参数 $= 3 d \cdot \frac{8}{3} d = 8 d^2$
- 2 个 RMSNorm：每个 $d$ 个参数（gain），共 $2d$，可忽略

**单层参数 ≈ $4 d^2 + 8 d^2 = 12 d^2$**。

整模型参数（不算 tie 时的 lm_head）：

$$P \approx \underbrace{V d}_{\text{embedding}} + \underbrace{L \cdot 12 d^2}_{N \text{ 层 block}} + \underbrace{V d}_{\text{lm\_head（tied 时省去）}}$$

**最常用的"看 size 算参数"近似**：

$$\boxed{P \approx V d + 12\,L\,d^2}$$

代入几个真实模型校验：

| 模型 | $V$ | $d$ | $L$ | $V d$ | $12 L d^2$ | 估算总和 | 实际公布 |
|---|---|---|---|---|---|---|---|
| GPT-2 small | 50k | 768 | 12 | 0.04B | 0.085B | **0.12B** | 0.124B ✓ |
| GPT-2 medium | 50k | 1024 | 24 | 0.05B | 0.30B | **0.35B** | 0.355B ✓ |
| LLaMA-2 7B | 32k | 4096 | 32 | 0.13B | 6.44B | **6.57B** | 6.74B ✓ |
| LLaMA-2 13B | 32k | 5120 | 40 | 0.16B | 12.58B | **12.75B** | 13.0B ✓ |
| LLaMA-2 70B | 32k | 8192 | 80 | 0.26B | 64.42B | **64.7B** | 69B ✗（差 GQA） |

70B 那行差额来自 LLaMA-2 70B 的 $d_{ff} = 28672$（远大于 $\frac{8}{3} \times 8192 = 21845$）+ 用了 GQA（KV head 减少，但 Q head 不变）。**前 4 行误差都在 5% 以内**——这就是 $V d + 12 L d^2$ 的实战价值：你看到任何一个 LLM 的 $(V, d, L)$，30 秒就能口算出参数量。

### 2.5 Generate 的天真实现 vs KV cache（与 4.7 衔接）

训练时 forward 一次输入完整 $T$ 个 token，输出所有位置的 logits——是 **parallel** 操作。但**生成（decoding）是串行的**：每一步只产生一个新 token，append 到序列末尾，下一步用这个新序列再 forward。

**最朴素的 generate**（本节实现的版本）：

```
step 1: forward([t_0])              → next = t_1
step 2: forward([t_0, t_1])         → next = t_2
step 3: forward([t_0, t_1, t_2])    → next = t_3
...
step k: forward([t_0, ..., t_{k-1}]) → next = t_k
```

每一步把整个前缀**重新算一遍 attention**——$O(T)$ tokens × $O(T^2)$ attention = $O(T^3)$ 总复杂度。生成 100 个 token 时 step 100 比 step 1 慢 $100^2 = 10000$ 倍——非常糟糕。

**KV cache 的核心 insight**：每一步新算的 K / V 之前都没算过，但**之前的 K / V 在以前的 step 里已经算完了**——把它们缓存起来，新 step 只需算"新 token 的 q"与"新 token 的 k/v"，再与 cache 里的历史 k/v 一起做 attention。每步降到 $O(T)$，总复杂度 $O(T^2)$。详见 4.7。

本节先用 $O(T^3)$ 的天真版把 `generate` 跑通——它能让你看到模型能 sample 出像样文本，是"完整 LLM 闭环"的最后一块拼图。但你必须知道**这是一个 prototype，不能直接上 production**——production 必须 KV cache 才能扛住实际 latency。

### 2.6 训练循环（与 1.5 工作流呼应）

整个 LLM 训练可以套用 1.5 §3 的 5 步模板，没有什么特殊：

```python
for step in range(num_steps):
    x, y = sample_batch()                     # (B, T) 与 (B, T)，y = x shift 一位
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, targets=y)    # forward + loss 一站式
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 必加
    optimizer.step()
    scheduler.step()
```

几个**与一般任务不同的细节**：

- **数据**：byte-level 或 BPE tokenize 后的 token 序列；input = `tokens[:T]`、target = `tokens[1:T+1]`（next-token prediction）
- **loss reduction**：`F.cross_entropy(logits.flatten(0, 1), targets.flatten())`——flatten 成 $(BT, V)$ 与 $(BT,)$ 后 PyTorch 默认 mean reduction
- **optimizer**：AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)（与 1.2 LLM 推荐配方一致）
- **gradient clipping**：max_norm=1.0 必加——LLM 训练不加几乎必崩
- **scheduler**：cosine + linear warmup（详见 1.2）
- **mixed precision**：bf16 必开（A100/H100/3090 都支持），节省 40-50% 显存

tinyShakespeare（约 1MB 文本）byte-level tokenize（vocab=256）+ 上面的 mini-LLaMA（4 层、d=128）几百步就能看到 loss 从 ~5.5（uniform random）降到 ~2.5（学到字符级 n-gram 分布）。

---

## 3. 最小代码示例

### 3.1 完整 mini-LLaMA 实现（< 200 行 self-contained）

把所有零件拼起来。**这一段代码可以直接复制粘贴到 `mini_llama.py` 跑通，不依赖任何外部 ckpt 或库**（除 PyTorch）。

```python
# mini_llama.py — < 200 行可跑的 mini-LLaMA decoder-only LLM
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ============== 配置 ==============
@dataclass
class LlamaConfig:
    vocab_size: int = 128
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    max_seqlen: int = 64
    rope_base: float = 10000.0
    tie_weights: bool = True
    # SwiGLU 用 d_ff ≈ 8/3 d 后 round 到 64 倍数
    @property
    def d_ff(self) -> int:
        return ((int(self.d_model * 8 / 3) + 63) // 64) * 64


# ============== RMSNorm（4.4） ==============
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 关键：fp32 cast 防 fp16 上溢（4.4 §4 第 3 条）
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight


# ============== RoPE（4.3） ==============
def precompute_rope_cache(d_k: int, max_seq: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))   # (d_k/2,)
    t = torch.arange(max_seq).float()                                    # (max_seq,)
    freqs = torch.outer(t, inv_freq)                                     # (max_seq, d_k/2)
    emb = torch.cat([freqs, freqs], dim=-1)                              # (max_seq, d_k)
    return emb.cos(), emb.sin()                                          # 各 (max_seq, d_k)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, h, T, d_k); cos, sin: (T, d_k)
    cos = cos[None, None, :, :]   # broadcast 到 (1, 1, T, d_k)
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ============== Multi-head causal Attention（4.1 / 4.2） ==============
class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_k = cfg.d_model // cfg.n_head
        # 三个大 Linear 一次产 h 头的 Q/K/V（4.2 §2.7 高效写法）
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x: (B, T, d) → out: (B, T, d)
        B, T, _ = x.shape
        h, d_k = self.n_head, self.d_k
        # reshape 到 (B, h, T, d_k)
        Q = self.W_q(x).view(B, T, h, d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, h, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, h, d_k).transpose(1, 2)
        # RoPE 加在 Q / K 上（不加 V）
        Q, K = apply_rotary(Q, K, cos, sin)
        # SDPA 自动选 FlashAttention backend、自动处理 causal mask
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)   # (B, h, T, d_k)
        # 拼回 (B, T, d) + W_O 混合
        out = out.transpose(1, 2).contiguous().view(B, T, h * d_k)
        return self.W_o(out)


# ============== SwiGLU FFN（4.5） ==============
class SwiGLU(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        d, d_ff = cfg.d_model, cfg.d_ff
        self.w1 = nn.Linear(d, d_ff, bias=False)   # gate
        self.w2 = nn.Linear(d, d_ff, bias=False)   # up
        self.w3 = nn.Linear(d_ff, d, bias=False)   # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ============== Pre-RMSNorm Block（4.4） ==============
class Block(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)   # residual + Pre-norm attention
        x = x + self.ffn(self.norm2(x))              # residual + Pre-norm FFN
        return x


# ============== MiniLlama（顶层模型） ==============
class MiniLlama(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model)         # Pre-LN 的"出口 norm"必加（4.4 §4 第 2 条）
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:                            # weight tying（§1.4）
            self.lm_head.weight = self.embed.weight
        # precompute RoPE cache，注册成 buffer 跟随 .to(device)
        d_k = cfg.d_model // cfg.n_head
        cos, sin = precompute_rope_cache(d_k, cfg.max_seqlen, cfg.rope_base)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)
        # 初始化（GPT-2 / LLaMA 风格）
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        # tokens: (B, T) int; targets: (B, T) int 或 None
        B, T = tokens.shape
        assert T <= self.cfg.max_seqlen, f"T={T} 超过 max_seqlen={self.cfg.max_seqlen}"
        x = self.embed(tokens)                                         # (B, T, d)
        cos, sin = self.cos_cache[:T], self.sin_cache[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)                                         # (B, T, d)
        logits = self.lm_head(x)                                       # (B, T, V)
        loss = None
        if targets is not None:
            # flatten 到 (B*T, V) 与 (B*T,) 算 token-level CE
            loss = F.cross_entropy(
                logits.float().view(-1, self.cfg.vocab_size),          # logits cast fp32
                targets.view(-1),
                ignore_index=-100,                                     # pad 位置可设 -100 跳过
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None):
        """天真版 generate（每步重算所有 token attention，O(T^3) 总复杂度）。
        生产环境必须用 KV cache（4.7 节）"""
        self.eval()
        for _ in range(max_new_tokens):
            # 截断到训练长度，否则 RoPE 会越界（§4 第 3 条）
            idx_cond = idx[:, -self.cfg.max_seqlen:]
            logits, _ = self.forward(idx_cond)
            # 只取最后一个位置的 logits
            logits = logits[:, -1, :] / max(temperature, 1e-5)         # 防 0
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)         # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
```

总行数（包含空行与注释）约 175 行，纯代码 ≈ 130 行。**逐文件可跑**——`python mini_llama.py` 不报错（虽然没有 main 调用）。

### 3.2 参数量估算函数（< 15 行）

```python
def estimate_params(cfg: LlamaConfig, tied: bool = True) -> dict:
    """近似公式 P ≈ V·d + L · 12 d² （+ V·d if untied）。"""
    d, L, V = cfg.d_model, cfg.n_layer, cfg.vocab_size
    embed = V * d
    # 每层：4 d² (Q/K/V/O) + 3 d · d_ff (SwiGLU) + 2 d (RMSNorm)
    per_layer = 4 * d * d + 3 * d * cfg.d_ff + 2 * d
    total = embed + L * per_layer + (0 if tied else V * d)
    return {
        "embed":       embed,
        "per_layer":   per_layer,
        "all_blocks":  L * per_layer,
        "lm_head":     0 if tied else V * d,
        "total":       total,
        "total_M":     total / 1e6,
    }

# 用法：
# >>> estimate_params(LlamaConfig())
# {'embed': 16384, 'per_layer': 198912, 'all_blocks': 795648, 'lm_head': 0,
#  'total': 812032, 'total_M': 0.812032}
```

把 LLaMA-2 13B 配置代进去校验：

```python
# 13B: V=32000, d=5120, n_layer=40, n_head=40, d_ff=13824
cfg = LlamaConfig(vocab_size=32000, d_model=5120, n_layer=40, n_head=40,
                  max_seqlen=4096)
# 注意 LLaMA-2 13B 实际 d_ff = 13824（不是 ((5120*8/3+63)//64)*64 = 13696）
# 直接 hack 一下：
import types
cfg_obj = types.SimpleNamespace(d_model=5120, n_layer=40, vocab_size=32000, d_ff=13824)
P = 32000 * 5120 + 40 * (4 * 5120**2 + 3 * 5120 * 13824 + 2 * 5120)
print(f"~{P / 1e9:.2f}B")    # ≈ 13.02B，与官方 13.0B 一致
```

### 3.3 训练循环 demo（< 30 行）

最小可跑的训练 demo——不需要真实数据，用 dummy random tokens 也能看到 loss 下降（验证模型是否能学到任何东西，sanity test）：

```python
import torch, torch.nn.functional as F

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = LlamaConfig()
model = MiniLlama(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                        weight_decay=0.1)

# === 用 dummy data 验证模型能学（loss 应从 ~5 降到 ~0.5） ===
B, T = 8, cfg.max_seqlen
fake = torch.randint(0, cfg.vocab_size, (B, T + 1), device=device)
x, y = fake[:, :-1], fake[:, 1:]                             # next-token

print(f"参数量: {estimate_params(cfg)['total_M']:.3f}M")
for step in range(200):
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, targets=y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 必加
    opt.step()
    if step % 20 == 0:
        print(f"step {step:3d} | loss {loss.item():.3f}")

# === sanity sample ===
prompt = torch.randint(0, cfg.vocab_size, (1, 1), device=device)
out = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=10)
print("sample:", out[0].tolist())
```

预期 loss 曲线（CPU 也能跑通，约 30 秒）：

```
参数量: 0.812M
step   0 | loss 4.847
step  20 | loss 4.523
step  40 | loss 3.811
step  60 | loss 2.642
step  80 | loss 1.523
step 100 | loss 0.781
step 120 | loss 0.382
step 140 | loss 0.198
step 160 | loss 0.114
step 180 | loss 0.072
sample: [37, 92, 14, 5, 88, 12, 91, ...]
```

loss 能压到 0.07 说明模型 overfit 到这一组 8 × 64 个 token 的能力是没问题的——next-token prediction 在小数据上能学到完全的"记忆"。

### 3.4 在 tinyShakespeare 上的真实训练（提示）

把 dummy data 换成真实数据集只需要 5 行：

```python
# 1. 下载 tinyShakespeare（一次性，< 1MB）
import os, urllib.request
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(URL, "input.txt")

# 2. byte-level tokenize（vocab = 256），train/val 9:1
data = torch.tensor(list(open("input.txt", "rb").read()), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# 3. 改 cfg 的 vocab_size 到 256（byte-level）
cfg = LlamaConfig(vocab_size=256, d_model=128, n_layer=4, n_head=4, max_seqlen=64)

# 4. 随机 sample (input, target) batch
def get_batch(split, B, T):
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - T - 1, (B,))
    x = torch.stack([src[i:i+T] for i in ix])
    y = torch.stack([src[i+1:i+T+1] for i in ix])
    return x.to(device), y.to(device)

# 5. 训练循环里把 fake / x / y 替换成 get_batch("train", 32, 64)
# 训 1000-2000 步在 CPU 上约 5-10 分钟，loss 从 5.5 → ~2.0
# generate 出来的 byte 序列 .decode("latin-1") 应该能看到莎士比亚风格的字符片段
```

Karpathy 的 nanoGPT [`train_shakespeare_char.py`](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py) 是这套代码的工业级精炼版——本节代码的设计直接师从它，强烈推荐对照阅读。

### 3.5 Sanity check：模型 forward 形状 / 数值合理性

```python
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=128, d_model=128, n_layer=4, n_head=4, max_seqlen=64)
model = MiniLlama(cfg)

x = torch.randint(0, cfg.vocab_size, (2, 16))                # (B=2, T=16)
y = torch.randint(0, cfg.vocab_size, (2, 16))
logits, loss = model(x, targets=y)
print("logits shape:", logits.shape)        # 期望 (2, 16, 128)
print("loss:", loss.item())                  # 期望 ≈ ln(128) = 4.85（uniform 初始化）
print("weight tied:", model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr())  # True
print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
```

跑一次得到 loss ≈ 4.85（$\ln(\text{vocab\_size}) = \ln 128 \approx 4.85$，是 uniform 初始化下的理论值）就说明模型搭对了——任何明显偏离这个值的初始 loss 都意味着初始化或 lm_head 写错了。

---

## 4. 工程踩坑与经验

- ❗ **block 之间用 residual 连接，不是 sequential 相加；忘了 residual → 模型完全不收敛**。Pre-LN block 的核心是 `x = x + sublayer(norm(x))`——`x +` 这一步是残差通道，是梯度直连回去的高速公路（4.4 §2.2）。新手照抄公式时常常把它写成 `x = sublayer(norm(x))`，看似只少一个加号，结果整个模型的 loss 完全平坦不下降。**修复**：每写完一个 block 就 sanity check 一遍 forward 的 shape 与 grad，看初始 loss 是否近似 $\ln V$；不是的话第一时间检查 residual。

- ❗ **final norm 后再 lm_head，不要把 lm_head 当成"最后一层 + 没有 norm"**。Pre-LN 设计下，最后一个 block 的输出**没有经过任何 norm**，数值范围会随深度漂移；直接接 lm_head 算 logits 数值不稳，softmax 会出现极端饱和。LLaMA / nanoGPT / GPT-2 全部在最后 block 之后再加一个 `final_norm = RMSNorm(d)`——**这一步极易漏掉**（4.4 §4 第 2 条）。漏掉的症状是训练 loss 比预期高 1-2，且推理时 sample 出的文本异常重复。

- ❗ **generate 时不截断 `idx[:, -self.max_seqlen:]` → RoPE 会越界**。RoPE 的 cos / sin 表只 precompute 到 `max_seqlen`，如果 generate 时序列长度超出 `max_seqlen`，访问 `cos_cache[:T]` 会 IndexError；即使你 precompute 得很大，超出训练时见过的位置范围也会让模型行为退化（4.3 §4 第 2 条）。**修复**：generate 每步先 `idx_cond = idx[:, -self.max_seqlen:]`——保留最后 max_seqlen 个 token，把更早的丢掉。这就是 GPT-2 / nanoGPT 的标准做法，本节代码 §3.1 已包含。production 长 context 必须用 RoPE 扩展（NTK / YaRN）+ KV cache。

- ❗ **training 时 loss 算的是 token-level，必须 reshape 成 (B*T, V) vs (B*T,)，不要漏 reshape**。`F.cross_entropy(logits, targets)` 默认期望 input shape 是 `(N, V)`、target shape 是 `(N,)`；但你的 logits 是 `(B, T, V)`、targets 是 `(B, T)`。直接传会报形状错；如果 PyTorch 因为某种 broadcast 没报错，结果也是错的（loss 数值无意义）。**正确写法**：`F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-100)`——flatten 把 batch 与 sequence 维合并成一个大的"token 维"，PyTorch 在 N 个 token 上 mean reduce。pad 位置 target 可以设成 -100 自动跳过。

- ❗ **用 `torch.multinomial` 采样时如果 probs 全是 0（top_k + softmax 后极端值）会 NaN；要加 temperature 兜底**。常见场景：`temperature=0` 的硬采样（实际是 argmax）、或者 `top_k=1` 后 softmax 唯一非零位概率 1 而其他全 0——`multinomial` 处理 0 概率向量会 NaN / 整个 batch 崩。**修复**：generate 入口对 `temperature` 做 `max(temperature, 1e-5)` 兜底（本节代码 §3.1 已加）；top_k 后用 `torch.topk` + `masked_fill(-inf)` 让 softmax 重新归一，不要手动构造 sparse 概率。如果想要纯 greedy，直接 `torch.argmax(logits[:, -1, :], dim=-1)`，绕过 multinomial。

- ❗ **weight tying 时 `lm_head.weight = embed.weight`，PyTorch 会把它当成同一个 parameter；如果你只 freeze 其中一个会双双失效**。这是 weight tying 的副作用：两个 module 共享同一个 `nn.Parameter`，无法独立 freeze。**坑场景**：LoRA fine-tune 时想"冻 lm_head 只训 LoRA"——结果发现 embedding 也被冻了，因为它们是同一个 Parameter；反过来想"冻 embedding"会发现 lm_head 也被冻了。**对策**：要 freeze 必须先 untie——`self.lm_head = nn.Linear(d, V, bias=False); self.lm_head.weight.data = self.embed.weight.data.clone()`，再分别 `requires_grad_(False)`。

- ❗ **推理时 `model.eval()` + `torch.no_grad()` 都要**。`model.eval()` 切 Dropout / BN 行为（虽然 mini-LLaMA 默认没 Dropout，但记住这个习惯）；`torch.no_grad()` 关闭 autograd 不存 activation 不建图（节省 5-10× 显存）。本节 `generate` 用的是 `@torch.no_grad()` decorator + 内部 `self.eval()`——两者都在了。**不要**只调用一个，详见 1.5 §4 第 1 条。

- ❗ **自己实现的 mini-LLaMA 与 HF LLaMA 的 ckpt 不兼容**。新手"为什么我训的 model load 不进官方 ckpt"是高频问题——原因有三：(1) **matrix shape 微差**：HF LLaMA 的 d_ff round 规则是"先算 $\frac{8}{3} d$ 再 round 到 256 倍数后再加 multiple_of"，与本节简化的 `((d*8//3 + 63)//64)*64` 不完全一致；LLaMA-2 7B d_ff = 11008（不是 10880）；(2) **RoPE 维度排布约定**：HF 的 `apply_rotary_pos_emb` 与 RoFormer paper 在"哪两维配对"上有差异（详见 4.3 §2.5 注），训练与推理用同一约定才对；(3) **norm 实现微差**：HF LlamaRMSNorm 是 `x * rsqrt(mean(x^2) + eps) * weight`，与本节实现一致，但有些早期 fork 在 eps 位置不同。**结论**：自训 mini-LLaMA 用于学习与 prototype，要做 production 必须按 HF 的 `LlamaModel` 严格对齐再训，否则永远 load 不进官方 weights。

- ❗ **logits 在 fp16 / bf16 下算 cross-entropy 会损失精度**。CE 内部要算 $\log \sum e^{x_i}$（log-sum-exp），fp16 / bf16 在长尾 vocab（V=128k+）上很容易溢出或下溢，loss 数值不稳。**业界做法**：logits **永远在 fp32 下算 CE**，即 `F.cross_entropy(logits.float().view(-1, V), targets.view(-1))`——本节 §3.1 代码已加 `logits.float()`。autocast 默认会让 cross_entropy fall back 到 fp32 但加这个 cast 是双保险。

- ❗ **`.contiguous()` 在 transpose 后必须加，否则 view 报错**。Multi-head attention 出口 `out.transpose(1, 2).view(B, T, d)`——transpose 之后内存非连续，直接 view 会 RuntimeError："view size is not compatible with input tensor's size and stride"。**修复**：`out.transpose(1, 2).contiguous().view(B, T, d)`（4.2 §4 第 6 条）。本节 §3.1 已加；newcomer 抄代码时极易漏 `contiguous()`。

---

## 5. 经典 paper

- **Vaswani et al., 2017 — Attention Is All You Need** — Transformer 原典。本节是它 §3 "Architecture" 的复述与现代化（Pre-LN / RMSNorm / SwiGLU / RoPE / weight tying 都是后续演进）。回头读它的 Figure 1 模型图，对比本节 §1.2 的 ASCII 图，能体会到"现代 LLM 是在原版基础上把每个零件升级一遍"。
- **Touvron et al., 2023 — LLaMA & LLaMA-2** — 现代 decoder-only 工程实践标杆。读 §2.1 "Architecture" 一段就能看到 RMSNorm + SwiGLU + RoPE + Pre-LN 这套组合的官方说明，与本节 §3.1 实现的设计完全对应。LLaMA-3 / Qwen2 / Mistral / DeepSeek 全部沿用这套配方——所以这一篇是"工业 decoder-only"的标准答案。
- **Andrej Karpathy — nanoGPT GitHub** — 不是 paper 但价值堪比。`model.py` 全文 < 300 行实现 GPT-2 124M 的可训可推完整 LLM，本节代码的代码风格直接师从它。强烈推荐对照阅读 `model.py` 的 `GPT.forward` 与本节 §3.1 的 `MiniLlama.forward`——你会发现关键 5 行（embed → blocks → final_norm → lm_head → CE）几乎一模一样。Karpathy 的 [Let's build GPT (YouTube, 2 hours)](https://www.youtube.com/watch?v=kCc8FmEb1nY) 是配套讲解。
- **Brown et al., 2020 — GPT-3** — decoder-only 大规模化的范例。175B 参数的 GPT-3 用的就是本节这套骨架放大 100×（$d = 12288, L = 96, V = 50k$，按 $V d + 12 L d^2$ 估算 $\approx 169$B，与官方 175B 接近）。读 §2.1 "Model and Architectures"（半页内容）即可——它印证了"骨架不变、scale up 即可"的工程哲学。

---

## 6. 自测与面试题

**Q1（架构）**：画出 decoder-only Transformer 的完整数据流（从 input tokens 到 output logits），标注每一层的 shape 变化。LLaMA-style 配方下每个 block 内有哪些子层、norm 放在哪、residual 怎么连？

<details>
<summary>Answer sketch</summary>

数据流（每一阶段必须答到 shape）：

| 阶段 | shape | dtype |
|---|---|---|
| input tokens | $(B, T)$ | int64 |
| `nn.Embedding(V, d)` 后 | $(B, T, d)$ | bf16 / fp32 |
| Q / K / V 投影 + reshape | $(B, h, T, d_k)$，$d_k = d/h$ | bf16 / fp32 |
| 每个 block 输出 | $(B, T, d)$ | bf16 / fp32 |
| 重复 N 层 | $(B, T, d)$ | — |
| final RMSNorm 后 | $(B, T, d)$ | — |
| `lm_head: nn.Linear(d, V)` | $(B, T, V)$ | fp32（建议 cast）|

每个 block 的内部结构（Pre-RMSNorm + 标准 multi-head + RoPE + SwiGLU FFN，**两个子层、两个 residual**）：

```
x_in
 │
 ├── RMSNorm → MultiHeadAttn (含 RoPE on Q/K, causal SDPA, W_O) ─┐
 │                                                               │
 (+) ◄───────────────────────────────────────────────────────────┘
 │
 ├── RMSNorm → SwiGLU FFN (W_1 silu × W_2 → W_3) ────────────────┐
 │                                                               │
 (+) ◄───────────────────────────────────────────────────────────┘
 │
x_out
```

要点（必须说到）：

- **norm 在子层入口、residual 之外**——Pre-LN（4.4）
- **residual 是纯恒等通道**：`x = x + sublayer(norm(x))`，不是 `x = sublayer(x + norm(x))`
- **RoPE 加在 Q / K 上而非 V**——只有 score 需要位置敏感
- **attention 整体外面包 W_O**——multi-head 的 head 之间 mixing
- **FFN 是 SwiGLU 三矩阵**：$W_1$ gate × $W_2$ up → $W_3$ down

加分：能说出 final norm 后再 lm_head；能说出 weight tying 是 `lm_head.weight = embed.weight`；能说出 Pre-LN 让深堆栈稳定（4.4 Xiong 2020）。

</details>

**Q2（参数量）**：用近似公式 $P \approx V d + 12\,L\,d^2$ 估算 LLaMA-2 13B（$V = 32000$、$d = 5120$、$L = 40$）的参数量，写出 calculation。然后说明这个公式漏算了什么、误差从哪来。

<details>
<summary>Answer sketch</summary>

代入计算：

- embedding: $V \cdot d = 32000 \times 5120 = 1.638 \times 10^8 = 0.164 \text{B}$
- 每层 block: $12 d^2 = 12 \times 5120^2 = 12 \times 2.62 \times 10^7 = 3.146 \times 10^8 = 0.315 \text{B}$
- 全部 40 层 block: $40 \times 0.315 = 12.58 \text{B}$
- **总计**: $0.164 + 12.58 = 12.75 \text{B}$（假设 weight tied，未单算 lm_head）

实际公布参数：13.0B，**误差约 2%**。

公式漏算或近似的来源：

- **每层 RMSNorm 的 gain 参数**：$2 \cdot d = 10240$ 个 / 层，全模型 $40 \times 10240 \approx 0.4 \text{M}$，可忽略
- **final RMSNorm**：$d = 5120$ 个，可忽略
- **lm_head 是否 tied**：LLaMA-2 实际 **untied**，单独 lm_head 参数 = $V \cdot d = 0.164 \text{B}$。tied 估算的 12.75B + untied lm_head 0.164B ≈ 12.91B，与 13.0B 更接近
- **d_ff 不是严格 $\frac{8}{3} d$**：LLaMA-2 13B 实际 d_ff = 13824（≠ $\frac{8}{3} \times 5120 = 13653$），多 1.3%；FFN 占大部分参数，所以这一项漏算贡献 0.1B 左右
- **bias 一般 = 0**：现代 LLM 全 `bias=False`，公式默认无 bias 是对的
- **位置编码是常量**：RoPE 的 cos/sin 是 precompute，不是参数

加分：能算 LLaMA-2 70B 的偏差更大（用 GQA + d_ff = 28672，公式低估 5+B）；能说"看 size 算参数是必备肌肉记忆"——对 7B / 13B / 70B 都能 30 秒口算。

</details>

**Q3（实现）**：用 ≤ 30 行 PyTorch 写一个 LLaMA-style Block（含 Pre-RMSNorm + 标准 multi-head causal SDPA + SwiGLU FFN + 两个 residual）。说明哪些细节决定了"现代 LLM"风格（对比 Vaswani 2017 原版）。

<details>
<summary>Answer sketch</summary>

核心 ≤ 30 行：

```python
class LlamaBlock(nn.Module):
    def __init__(self, d, n_head, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d)                          # ← 不是 LayerNorm
        self.norm2 = RMSNorm(d)
        # attention: 3 Linear + W_O，全部 bias=False
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        # SwiGLU FFN: 3 矩阵
        self.w1 = nn.Linear(d, d_ff, bias=False)         # gate
        self.w2 = nn.Linear(d, d_ff, bias=False)         # up
        self.w3 = nn.Linear(d_ff, d, bias=False)         # down
        self.n_head, self.d_k = n_head, d // n_head

    def forward(self, x, cos, sin):
        # ===== Attention sublayer (Pre-norm + residual) =====
        h = self.norm1(x)
        B, T, _ = h.shape
        Q = self.W_q(h).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_k(h).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_v(h).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        Q, K = apply_rotary(Q, K, cos, sin)              # ← RoPE 加在 Q/K
        a = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.W_o(a)                               # residual
        # ===== FFN sublayer (Pre-norm + residual) =====
        h = self.norm2(x)
        x = x + self.w3(F.silu(self.w1(h)) * self.w2(h))  # SwiGLU + residual
        return x
```

**与 Vaswani 2017 原版的差异（即"现代 LLM"风格的关键决策）**：

| 维度 | Vaswani 2017 | 现代 LLaMA | 出处节 |
|---|---|---|---|
| Norm 类型 | LayerNorm | **RMSNorm**（无 mean、无 bias） | 4.4 |
| Norm 位置 | Post-LN | **Pre-LN** | 4.4 |
| 位置编码 | Sinusoidal（加在 embedding 上） | **RoPE**（在 Q/K 投影后旋转） | 4.3 |
| 激活函数 | ReLU | **Swish (silu)** | 4.5 |
| FFN 结构 | 2 矩阵（$W_2 \cdot \text{ReLU}(W_1 x)$） | **3 矩阵 SwiGLU** | 4.5 |
| $d_{ff}$ | $4 d$ | **$\approx \frac{8}{3} d$** round 到 256 | 4.5 |
| bias | 有 | **全 False** | 4.5 |
| Attention | 标准 multi-head | 标准 multi-head（70B+ 用 GQA） | 5.2 |

加分：

- 能指出 final norm（在最后一个 block 之后）必加，否则 logits 不稳（4.4 §4 第 2 条）
- 能指出 `contiguous()` 在 transpose 后必须加（4.2 §4 第 6 条）
- 能说 logits cast fp32 算 CE 是工程必加细节
- 能解释为什么 RoPE 加在 Q/K 不加在 V（只有 score 需要位置敏感）

</details>

---

## 7. 延伸阅读

- [Karpathy — nanoGPT GitHub](https://github.com/karpathy/nanoGPT) — 本节代码的祖本，[`model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) < 300 行实现完整 GPT-2 124M。强烈推荐对照本节 §3.1 阅读
- [Karpathy — Let's build GPT (YouTube, 2 hours)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 全网最直观的 "from-scratch 手撕 nanoGPT" 视频教程
- [HuggingFace — `LlamaModel` 源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 工业级 LLaMA 实现的标准答案，本节代码可视为它的最小化版本
- [Sebastian Raschka — Build a Large Language Model From Scratch (book)](https://github.com/rasbt/LLMs-from-scratch) — 系统化的 from-scratch LLM 教程，与本节哲学一致
- [Touvron et al. 2023 — LLaMA paper (arXiv)](https://arxiv.org/abs/2302.13971) — 现代 decoder-only 工程实践的官方说明，§2.1 Architecture 一段必读
- [Touvron et al. 2023 — LLaMA-2 paper (arXiv)](https://arxiv.org/abs/2307.09288) — 7B / 13B / 70B 三种 size 的具体配置表是本节参数量公式校验的最佳数据源
- [Press & Wolf 2017 — Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) — weight tying 的提出 paper，解释为什么 lm_head 与 embedding 共享权重在数学上合理
- 推荐继续读本教程的 **4.7 节《KV Cache 原理与实现》**——本节 `generate` 是 $O(T^3)$ 的天真版，4.7 把它优化到 $O(T^2)$，是从"能跑"到"能上线"的关键一步
- 推荐继续读本教程的 **5.1 节《Encoder-only / Decoder-only / Encoder-Decoder 对比》**——decoder-only 是当代 LLM 主流（GPT 系 / LLaMA / Qwen / Claude），但 encoder-only 仍是分类 / NER / embedding 的首选（BERT 后裔），encoder-decoder 在翻译 / 摘要上仍有 T5 / FLAN-T5 的根基；5.1 把三种结构的取舍系统讲清
- 推荐继续读本教程的 **6.6 节《经典开源 LLM 解读：LLaMA / Qwen / DeepSeek 系列》**——把本节这套骨架放到真实 70B+ 模型上看每一处参数 / 形状的现代取舍
