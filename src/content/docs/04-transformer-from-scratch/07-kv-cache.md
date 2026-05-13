---
title: "4.7 KV Cache 原理与实现"
description: "4.6 的 `generate` 每生成一个 token 都把整个前缀重新跑一遍 forward——其中前 N 个 token 的 K, V 在之前的 step 里早就算过，每步重算就是把同一份矩阵乘法白白重复几十次。KV cache 的核心 idea 一句话：推理时缓存历史 K, V，新 token 只算自己的 q，再与缓存里的 K, V 拼起来做一次 $1 \\\\times $ 的 attenti"
---

> ⏱ 预计阅读 50 分钟 ｜ 难度 ★★★ 🔥 必考 ｜ 前置：4.6 完整 decoder-only 实现

## 一句话本节讲什么

4.6 的 `generate` 每生成一个 token 都把整个前缀重新跑一遍 forward——其中**前 N 个 token 的 K, V 在之前的 step 里早就算过**，每步重算就是把同一份矩阵乘法白白重复几十次。KV cache 的核心 idea 一句话：**推理时缓存历史 K, V，新 token 只算自己的 q，再与缓存里的 K, V 拼起来做一次 $1 \times (T{+}1)$ 的 attention**。每步 attention 复杂度从 $O(T^2)$ 降到 $O(T)$、生成 M 个 token 的总复杂度从 $O(T^3)$ 降到 $O(T^2)$，实测 5-50× 加速。本节把 prefill / decode 两阶段心智模型、KV cache 的形状与拼接逻辑、与 RoPE 的 `position_ids` 配合、显存占用算账（**LLaMA-3 70B 在 8k context 下 ~20 GB / sample**——LLM 推理 OOM 第一杀手）、以及优化方向（GQA / PagedAttention / 量化 / prefix cache）讲透——是从 4.6 "能跑通"到 Module 11 "能上线"的关键一步，也是 5.2、11.2 之后所有推理优化章的共同前置。

---

## 1. Mental model（直觉）

### 1.1 朴素 generate 在浪费什么

回到 4.6 §2.5 的天真生成循环：

```
step 1: forward([t_0])                   → next = t_1
step 2: forward([t_0, t_1])              → next = t_2
step 3: forward([t_0, t_1, t_2])         → next = t_3
...
step k: forward([t_0, t_1, ..., t_{k-1}]) → next = t_k
```

每一步把整个前缀重新过一遍 model：embedding、所有层的 attention、所有层的 FFN、最后一个位置的 lm_head。

注意一件事：**step k 的 attention 里，前 $k-1$ 个 token 的 K 与 V 与 step $k-1$ 算出来的完全一样**——它们是 token $i$ 的 hidden state 经 $W_K, W_V$ 投影得到的，token $i$ 的 hidden state 又是由更早的 token $0..i-1$ 经 self-attention（causal mask 下只看自己与更早）得到的。当我们在 step k 把 token $0..k-2$ 喂进去重算，输出与 step $k-1$ 那一次完全一致——只是 PyTorch 不知道这件事，它老老实实把所有矩阵乘法又重算了一遍。

step k 真正"新"的东西只有一个：**token $k-1$ 的 q, k, v**。新 token 的 q 需要跟历史 0..$k-1$ 的 k 全部点积出 score，新 token 的 k, v 需要被未来的 token 看到——但**前 $k-1$ 个 token 的 q 不再被需要**（causal mask 下未来的 token 才看历史的 k, v；历史的 q 不会去看新来的 k）。

这就解释了为什么 cache 只缓存 K, V，不缓存 Q：**Q 是"主动方"，每步都换新的；K, V 是"被动方"，一旦被算出来就不再变**。

### 1.2 KV Cache 的核心 idea（一句话画图）

把每一层 attention 想成一个不断"长高"的两个表格：

```
                    K_cache (按 token 维不断 append)
              ┌─────┬─────┬─────┬─────┬─────┐
   token id:  │  0  │  1  │  2  │  3  │  4  │   ←  step 5 时，cache 里已有 0..3
              └─────┴─────┴─────┴─────┴─────┘
                    V_cache  (同步 append)
              ┌─────┬─────┬─────┬─────┬─────┐
              │  0  │  1  │  2  │  3  │  4  │
              └─────┴─────┴─────┴─────┴─────┘

每一步 decode：
  1. 取新 token 的 hidden x  (1 token)
  2. 算它的 q, k, v          (Linear 投影)
  3. K_cache = concat(K_cache, k);  V_cache = concat(V_cache, v)
  4. attn = softmax(q · K_cache^T / √d_k) · V_cache  ← 一行 1×(T+1) 的 attention
  5. 输出 1 个新 hidden state，过 FFN，最后一个位置过 lm_head 采样
```

step k 不再做 $k \times k$ 的 attention，而是做 $1 \times k$——降一阶。整个生成循环里，每个 token 的 K, V 只算一次（被算的那一步），之后无限次复用。

### 1.3 Prefill vs Decode：两个完全不同的计算 profile

Production 推理引擎（vLLM / SGLang / TGI）把生成分成两个截然不同的阶段：

**Prefill（prompt processing）**：
- 用户给的 prompt（比如 1024 token）一次性送入 model
- **所有 token 的 attention 可以并行**——与训练 forward 完全一样
- 同时把所有 token 的 K, V 一次性写入 cache
- attention 计算是 $O(T^2 d)$——**compute-bound**，GPU 的 matmul 算力是瓶颈
- 这一步的延迟决定了 **TTFT（time to first token）**

**Decode（generation）**：
- 一次只生成 1 个 token
- attention 是 "1 个新 q × T 个历史 k"——计算量极小（$O(T d)$）
- 但每步都要把 GB 级的 KV cache 从 HBM 读出来——**memory-bound**，GPU 带宽是瓶颈
- GPU 的 ALU 大量闲置（算 1 行 attention 几乎瞬间，但等数据从 HBM 来要好几毫秒）
- 这一步的延迟决定了 **TBT（time between tokens, 或称 ITL）**

**关键直觉**：prefill 是"批量并行的训练 forward"，decode 是"单 token 流的高频小操作"。两者的优化方向完全不同——prefill 关心 matmul 利用率（FlashAttention、TP），decode 关心带宽利用率（KV cache 量化、GQA 减小 cache、batch 多个 user 一起跑摊薄读 cache 成本）。**Continuous Batching**（vLLM 的招牌特性）正是为了让多 user 的 decode step 拼成一个大 batch、让带宽不浪费。

现代推理引擎还有一招叫 **chunked prefill**：把超长 prompt 的 prefill 拆成小块（比如每块 512 token），与其他 user 的 decode step 混在一个 batch 里跑，平衡 TTFT 与 TBT——长 prompt 用户不会因为他自己的 prefill 把所有人的 decode 卡住。这套调度逻辑会在 11.2 详讲。

### 1.4 为什么 KV cache 是 LLM 推理 OOM 第一杀手

直觉算账：每一层、每一个 token，要存 $K_t \in \mathbb{R}^{d}$ 与 $V_t \in \mathbb{R}^{d}$ 共 $2d$ 个数。**LLaMA-2 7B**（$L = 32, d = 4096$）每个 token 的 KV 是 $32 \times 4096 \times 2 = 262144$ 个数，bf16（2 字节）就是 **0.5 MB**——一个 token，半 MB。

2k context 的 single sample 就是 1 GB。如果你想 batch 32 个 user 同时聊 4k context，KV cache 直接 64 GB——比模型权重（7B × 2 bytes = 14 GB）还重 4 倍。**70B 模型在 32 GB 显存的 A100 上只能跑 batch 1 × 4k context**——这不是夸张，是实测。这就是为什么 5.2 的 GQA / MQA / MLA 对推理是革命性的：直接把 KV cache 缩 4-8 倍（GQA-8）甚至 16 倍（MQA）甚至更多（MLA）。详细算账见 §2.4。

---

## 2. 公式与原理

### 2.1 复杂度推导：$O(T^3) \to O(T^2)$

设已有 $N$ 个 prompt token，要生成 $M$ 个 token。每一层每个头每个 token 的 attention 内部主要开销：

**朴素 generate（无 cache）**：

第 $t$ 步（$t = 1, 2, \dots, M$），输入是 $N + t - 1$ 个 token，forward 整个 model：
- attention 矩阵 $QK^\top$ 是 $(N+t-1) \times (N+t-1)$，乘法量 $O((N+t-1)^2 d_k)$
- attention output 是 $(N+t-1) \times (N+t-1) \cdot V$，再 $O((N+t-1)^2 d_k)$
- 单层每头 $O((N+t-1)^2 d_k)$，所有头所有层 $O(L h (N+t-1)^2 d_k) = O(L (N+t-1)^2 d)$

生成 M 个 token 的总 attention 计算：

$$C_{\text{naive}} = \sum_{t=1}^{M} O\bigl(L (N+t-1)^2 d\bigr) = O\bigl(L d \cdot (M N^2 + N M^2 + M^3)\bigr)$$

当 $N \approx M$ 时，三项同量级，总 $O(L d M^3)$——三次方。

**举例**：$N = 2k, M = 2k, L = 32, d = 4096$，单看 attention score 计算总量 $\approx 32 \times 4096 \times (4 \times 10^9) \approx 5 \times 10^{14}$ FLOPs——5e14 次浮点乘加，A100 算力 312 TFLOPS（bf16）下要 1.6 秒**只是 attention**，加上 FFN 与 IO 实际更慢。

**KV cache 版**：

第 $t$ 步只算新 token 的 q, k, v，attention 是 $1 \times (N+t-1)$：
- $q \in \mathbb{R}^{1 \times d_k}$，$K_{\text{cache}} \in \mathbb{R}^{(N+t-1) \times d_k}$
- $q K_{\text{cache}}^\top$ 是 $1 \times (N+t-1)$，乘法量 $O((N+t-1) d_k)$
- 加权 $V_{\text{cache}}$ 也是 $O((N+t-1) d_k)$
- 单步所有头所有层 $O(L (N+t-1) d)$

生成 M 个 token 总：

$$C_{\text{cache}} = \sum_{t=1}^{M} O\bigl(L (N+t-1) d\bigr) = O\bigl(L d \cdot (N M + M^2)\bigr) = O\bigl(L d \cdot M(N + M)\bigr)$$

**降一阶**——朴素 $O(M^3)$、cache $O(M^2)$。$N = M = 2k$ 时 attention FLOPs 从 $\sim 5 \times 10^{14}$ 降到 $\sim 5 \times 10^{11}$——**1000× 加速**（实测因为 FFN、读写 cache 等开销，end-to-end 加速通常 5-50×）。

**别忘了还有 prefill**：prefill 阶段是一次性算 $N$ 个 token 的 attention（$O(N^2 d L)$），加上 decode 的总：

$$C_{\text{total}} = \underbrace{O(L d N^2)}_{\text{prefill}} + \underbrace{O(L d M(N+M))}_{\text{decode with cache}}$$

prefill 这一项无法被 cache 优化（第一次见到 prompt，必须算）——所以**长 prompt + 短 generation 的场景，瓶颈在 prefill**（chunked prefill 优化此处）；**短 prompt + 长 generation 的场景，瓶颈在 decode**（GQA / KV 量化优化此处）。

### 2.2 KV Cache 的形状

每一层每个 sample 维护两个 tensor：

$$K_{\text{cache}}^{(l)}, V_{\text{cache}}^{(l)} \in \mathbb{R}^{B \times h \times T_{\text{cur}} \times d_k}$$

其中 $T_{\text{cur}}$ 是当前已 cache 的 token 数（每生成一个新 token，$T_{\text{cur}} \mathrel{+}= 1$）。

更精确的形状追踪（与 4.6 的 layout 一致）：

| 阶段 | 张量 | 形状 | 说明 |
|---|---|---|---|
| Prefill 输入 | x | $(B, N, d)$ | N 个 prompt token |
| Prefill 后 cache | $K_{\text{cache}}$ | $(B, h, N, d_k)$ | 一次性写入 N 个 |
| Decode 输入 | x | $(B, 1, d)$ | 1 个新 token |
| Decode 内 q, k, v | q, k, v | $(B, h, 1, d_k)$ | 只 1 个 token |
| Decode concat 后 cache | $K_{\text{cache}}$ | $(B, h, T_{\text{cur}}+1, d_k)$ | append 1 个 |
| Decode attention score | $q K^\top$ | $(B, h, 1, T_{\text{cur}}+1)$ | 1 行 attention |
| Decode 输出 | out | $(B, 1, d)$ | 1 个 hidden |

**关键点**：
- **K, V cache 的 head 维 $h$ 与 batch 维 $B$ 必须明确分开**——4.2 的 layout `(B, h, T, d_k)` 在 KV cache 里继续沿用
- **decode 时的 q 是 1 个 token、cache 里的 K 是历史所有 token**——注意 `is_causal=True` 在 decode 这一步**反而不能用**，因为 SDPA 的 causal mask 是针对方阵的；decode 时 attention shape 是 $1 \times (T_{\text{cur}}+1)$ 矩形，新 q 本来就只看到所有历史 + 自己（新 k, v 已经 append 进去），不需要额外 mask
- **prefill 时 q, k, v 三者都是 N 个 token、是方阵 attention，必须 `is_causal=True`**——和训练 forward 一样

### 2.3 与 RoPE 的配合（必踩坑点）

4.3 的 RoPE 把位置信息通过对 q, k 做旋转编码进去——**位置 id 决定了旋转角度**。训练时位置 id 是 `0, 1, 2, ..., T-1`；prefill 时一样（输入 N 个 token，位置 id 是 `0..N-1`）；**decode 时容易出错**——新 token 的位置 id 不是 0，而是 `past_len`（即 cache 里已有的 token 数）。

设 `past_len = T_cur`（当前 cache 长度），decode 一步输入 1 个新 token：

$$\text{position\_ids}_{\text{decode}} = [\,T_{\text{cur}}\,] \quad (\text{长度 } 1)$$

prefill 一次输入 N 个 token：

$$\text{position\_ids}_{\text{prefill}} = [0, 1, \dots, N-1]$$

通用写法（兼容 prefill 与 decode）：

$$\text{position\_ids} = [\text{past\_len}, \text{past\_len}+1, \dots, \text{past\_len}+T-1]$$

代码上从 RoPE cache 取对应 slice：

```python
cos = cos_cache[past_len : past_len + T]
sin = sin_cache[past_len : past_len + T]
```

**最常见的 bug**：忘了 past_len，每步 decode 都用 `cos_cache[:1]`（位置 0 的旋转）——结果新 token 的 q 永远被当成第 0 个 token，attention score 算到错的位置，模型输出乱码。这是新手实现 KV cache 时的高频踩坑（详见 §4）。

### 2.4 KV Cache 显存占用算账（必背）

每一层每个 token 的 KV：$2 \cdot h \cdot d_k = 2 d$ 个数（$h \cdot d_k = d$）。

**Per-sample 总 KV cache 显存**：

$$M_{\text{KV}} = 2 \cdot L \cdot d \cdot T \cdot \text{dtype\_bytes}$$

| 模型 | $L$ | $d$ | $T$ | bf16 (2B) | 算式 |
|---|---|---|---|---|---|
| LLaMA-2 7B | 32 | 4096 | 2k | **1.0 GB** | $2 \times 32 \times 4096 \times 2048 \times 2$ B |
| LLaMA-2 7B | 32 | 4096 | 4k | **2.0 GB** | $\times 2$ |
| LLaMA-2 13B | 40 | 5120 | 4k | **3.1 GB** | $2 \times 40 \times 5120 \times 4096 \times 2$ B |
| LLaMA-2 70B (MHA, 朴素) | 80 | 8192 | 8k | **20.0 GB** | $2 \times 80 \times 8192 \times 8192 \times 2$ B |
| LLaMA-3 70B (GQA-8) | 80 | 8192 | 8k | **2.5 GB** | KV head 8 个而非 64 个 → 1/8 |
| Qwen2 72B (GQA-8) | 80 | 8192 | 32k | **10.0 GB** | GQA 之后仍然惊人 |
| DeepSeek-V3 (MLA) | 61 | 7168 | 128k | **~5 GB** | MLA 把 KV cache 投影到 576 维 latent |
| DeepSeek-V3 (朴素 MHA, 假设) | 61 | 7168 | 128k | **220 GB** | 比 671B 模型权重还大 → 必须 MLA |

**几个必背的口算技巧**：

1. **bf16 下 KV cache 每 token 每层 ≈ $2d$ bytes**（dtype 2 bytes 抵消 $2d$ 里的 2，所以是 $2d$ bytes per token per layer 这个口诀更好记）。这里口诀：**每 token 每层 = $2d$ bytes（bf16）**——LLaMA-2 7B：$2 \times 4096 = 8192$ B = 8 KB / token / layer；32 层 = 256 KB / token；2k token = 512 MB ≈ 0.5 GB（与上表对齐）。
2. **GQA-g 把 KV head 数从 $h$ 缩到 $g$**，KV cache 直接 ÷ $h/g$；MQA（$g=1$）÷ $h$。
3. **量化 KV cache 到 INT8 或 FP8**，显存再 ÷ 2。
4. **70B 模型 H100（80 GB）下 batch × context 的容量**：weights ~140 GB（fp16），单卡装不下（需 TP）；KV cache 在 8k context 下 ~2.5 GB（GQA-8）/ sample——单 H100（去除 weights 后）可装 batch ≈ 30 个 user 的 8k context。

**应试速算公式**（30 秒口算）：

$$\boxed{M_{\text{KV}}^{\text{GB}} \approx 4 \times L \times d \times T \times 10^{-9} \text{(bf16, 单位 GB)}}$$

代入 LLaMA-2 70B 8k：$4 \times 80 \times 8192 \times 8192 \times 10^{-9} \approx 21.5$ GB——与上表 20 GB 一致（差异来自 8192 ≈ 8000 的近似）。

### 2.5 与训练 attention 的等价性（数学保证）

KV cache 不是近似，是 **bit-exact 等价**——decode 时第 $t$ 步算的 attention output，与"把前 $N+t$ 个 token 一次性 forward 取最后一个位置"的输出**应当数值相同**（除了浮点误差）。

简单证明：causal mask 下，token $t$ 的输出只依赖于 token $0..t$ 的 K, V 与 token $t$ 自己的 Q。

$$\text{out}_t = \sum_{i \le t} \text{softmax}\bigl((q_t k_i^\top) / \sqrt{d_k}\bigr) v_i$$

无论 K, V 是"现场算"还是"从 cache 读"，只要数值一致，结果就一致。**这一性质是 KV cache 能成立的根本——没有这一点，KV cache 就是近似优化而不是 lossless 优化**。可用本节 §3.4 的 sanity test 验证：cache 版与 naive 版输出 logits 的最大差异应在 1e-4 以内（bf16 误差量级）。

---

## 3. 最小代码示例

### 3.1 给 4.6 的 Attention 加 KV cache（< 60 行）

直接在 4.6 §3.1 `Attention` 的基础上扩 `forward(x, past_kv=None) → (out, new_kv)` 接口：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionWithKVCache(nn.Module):
    """与 4.6 的 Attention 接口兼容，多一个 past_kv 参数。"""
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_k = cfg.d_model // cfg.n_head
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, past_kv=None):
        # x: (B, T, d)；T = N (prefill) 或 1 (decode)
        # past_kv: None (prefill) 或 (K_cache, V_cache) 形状 (B, h, T_cur, d_k)
        # 返回: (out (B, T, d), new_kv = (K, V) 全量缓存)
        B, T, _ = x.shape
        h, d_k = self.n_head, self.d_k

        # 1) 算当前 chunk 的 q, k, v —— 与 4.6 一致
        Q = self.W_q(x).view(B, T, h, d_k).transpose(1, 2)   # (B, h, T, d_k)
        K = self.W_k(x).view(B, T, h, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, h, d_k).transpose(1, 2)

        # 2) RoPE（cos, sin 已按 past_len 切好；调用方负责，见 §3.2）
        Q, K = apply_rotary(Q, K, cos, sin)                  # 4.6 同名函数

        # 3) 与 cache 拼接（核心一步）
        if past_kv is not None:
            K_past, V_past = past_kv
            K = torch.cat([K_past, K], dim=2)                # (B, h, T_cur + T, d_k)
            V = torch.cat([V_past, V], dim=2)
        new_kv = (K, V)                                      # 给上层存回去

        # 4) attention：prefill 用 causal mask；decode 不需要（1×T_cur+1 矩形天然满足）
        is_causal = past_kv is None and T > 1                # 仅 prefill 多 token 才 causal
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

        # 5) 拼回 (B, T, d) + W_O 输出
        out = out.transpose(1, 2).contiguous().view(B, T, h * d_k)
        return self.W_o(out), new_kv
```

**关键 5 处**：

1. `past_kv` 是 `(K_cache, V_cache)`，prefill 时为 `None`、decode 时为非空——一个接口同时支持两阶段
2. `cos, sin` 由调用方按 `past_len` 切好传进来——RoPE 的 position 由上层管理，attention 内只负责旋转
3. `torch.cat([K_past, K], dim=2)` 在 T 维拼接——prefill 后 K 形状 $(B, h, N, d_k)$；之后每步 decode K 形状 $(B, h, T_{\text{cur}}+1, d_k)$
4. `is_causal` 只在 prefill 多 token 时为 True；decode 单 token 时 q 是 1 行、K 已含历史 + 自己，自然满足 causal
5. `new_kv` 必须返回——上层 `MiniLlama` 要把它存到 `past_kv_cache[layer]` 字典里供下一步使用

### 3.2 `MiniLlama.forward` + `generate_with_cache`（< 30 行）

外层模型把每一层的 `past_kv` 串成一个 list（每层一个），外加 `past_len` 用来切 RoPE：

```python
def forward_with_cache(self, tokens, past_kvs=None):
    # tokens: (B, T)；past_kvs: List[ (K, V) ] 长度 = n_layer 或 None
    B, T = tokens.shape
    past_len = 0 if past_kvs is None else past_kvs[0][0].shape[2]
    assert past_len + T <= self.cfg.max_seqlen, "超过 RoPE precompute 长度"

    x = self.embed(tokens)                                       # (B, T, d)
    # 关键：cos, sin 从 past_len 起切 T 行
    cos = self.cos_cache[past_len : past_len + T]
    sin = self.sin_cache[past_len : past_len + T]

    new_kvs = []
    for i, block in enumerate(self.blocks):
        past = None if past_kvs is None else past_kvs[i]
        # block.forward 也要改造成支持 past_kv（详见 §3.3）
        x, new_kv = block(x, cos, sin, past_kv=past)
        new_kvs.append(new_kv)

    x = self.final_norm(x)
    logits = self.lm_head(x)                                     # (B, T, V)
    return logits, new_kvs


@torch.no_grad()
def generate_with_cache(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    self.eval()
    # 1) Prefill 整个 prompt → 拿到初始 cache
    logits, kvs = self.forward_with_cache(idx, past_kvs=None)
    # 2) 之后每步 decode 1 个 token + update cache
    for _ in range(max_new_tokens):
        last_logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(last_logits, top_k)
            last_logits[last_logits < v[:, [-1]]] = -float("inf")
        next_tok = torch.multinomial(F.softmax(last_logits, dim=-1), 1)  # (B, 1)
        idx = torch.cat([idx, next_tok], dim=1)
        # 只送 1 个新 token、带上 cache
        logits, kvs = self.forward_with_cache(next_tok, past_kvs=kvs)
    return idx
```

**注意三处**：

- 第一次调用 `forward_with_cache(idx, past_kvs=None)` 是 prefill：把整个 prompt 一次跑 forward
- 之后每次 `forward_with_cache(next_tok, past_kvs=kvs)` 只送 1 个 token，但 cache 在长，attention 仍然能看全历史
- `max_seqlen` 必须够大（precompute RoPE cache 时给足，比如 4096）；超出会越界

`Block.forward` 也要小改一行：

```python
def forward(self, x, cos, sin, past_kv=None):
    # Pre-norm attention, with cache
    a, new_kv = self.attn(self.norm1(x), cos, sin, past_kv=past_kv)
    x = x + a
    # FFN 不需要 cache（无跨 token 依赖）
    x = x + self.ffn(self.norm2(x))
    return x, new_kv
```

FFN 是 token-wise 的（4.5），不存在 token 间通信，因此**只 attention 需要 KV cache，FFN 不需要任何 cache**。

### 3.3 显存占用估算函数（< 15 行）

```python
def kv_cache_size_gb(n_layer: int, d_model: int, seq_len: int,
                      dtype_bytes: int = 2,                       # bf16 = 2
                      n_kv_head: int | None = None,
                      n_head: int | None = None) -> float:
    """估算 single-sample KV cache 显存占用（GB）。
    n_kv_head / n_head 用来表达 GQA：g 个 KV head 而非 h 个 → cache × g/h。
    朴素 MHA 时不传，等价于 g = h。"""
    # bytes per token per layer per (K or V) = h * d_k * dtype_bytes = d * dtype_bytes
    # K 与 V 各一份 → 2 * d * dtype_bytes
    # GQA: KV 投影维度变成 g * d_k = (g/h) * d
    if n_kv_head is None or n_head is None:
        kv_dim = d_model
    else:
        kv_dim = d_model * n_kv_head // n_head
    bytes_per_token = 2 * n_layer * kv_dim * dtype_bytes
    total_bytes = bytes_per_token * seq_len
    return total_bytes / (1024 ** 3)


# 用法 / 校验：
# >>> kv_cache_size_gb(32, 4096, 2048)             # LLaMA-2 7B @ 2k bf16
# 1.0   # GB
# >>> kv_cache_size_gb(80, 8192, 8192)             # LLaMA-2 70B MHA @ 8k
# 20.0  # GB ← OOM 主因
# >>> kv_cache_size_gb(80, 8192, 8192, n_kv_head=8, n_head=64)  # GQA-8 → 1/8
# 2.5   # GB
# >>> kv_cache_size_gb(80, 8192, 8192, n_kv_head=8, n_head=64, dtype_bytes=1)  # FP8
# 1.25  # GB
```

### 3.4 Sanity check：cache 版 vs naive 版应当数值一致

KV cache 是 lossless 优化，可以与 4.6 的 naive `generate` 输出的 logits 直接比较：

```python
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=128, d_model=128, n_layer=4, n_head=4, max_seqlen=64)
model = MiniLlamaWithCache(cfg).eval()                # 上面 §3.1+§3.2 改造后的模型
prompt = torch.randint(0, cfg.vocab_size, (1, 8))     # 8-token prompt

# 1) Naive: 一次 forward 整 16 token，取最后一个 logits
full = torch.cat([prompt, torch.randint(0, cfg.vocab_size, (1, 8))], dim=1)
logits_naive, _ = model.forward(full)                 # 4.6 原版接口
ref = logits_naive[:, -1, :]                          # (1, V)

# 2) Cache: prefill 8 + decode 8
logits, kvs = model.forward_with_cache(prompt, past_kvs=None)
for i in range(8):
    next_tok = full[:, prompt.shape[1] + i].unsqueeze(1)   # 用相同 token，控变量
    logits, kvs = model.forward_with_cache(next_tok, past_kvs=kvs)
out = logits[:, -1, :]                                # (1, V)

print("max diff:", (ref - out).abs().max().item())    # 期望 < 1e-4 (bf16 精度内)
```

**这个 sanity test 是实现 KV cache 时的金标准**——任何超过 1e-3 的偏差都意味着实现有 bug（最常见的两种：RoPE 的 position_ids 没用 past_len、cache 的 head 维拼错了）。

### 3.5 速度对比 demo（可选）

```python
import time

cfg = LlamaConfig(vocab_size=512, d_model=512, n_layer=8, n_head=8, max_seqlen=2048)
model = MiniLlamaWithCache(cfg).cuda().eval()
prompt = torch.randint(0, cfg.vocab_size, (1, 256), device="cuda")

# warmup
_ = model.generate(prompt, max_new_tokens=10)
_ = model.generate_with_cache(prompt, max_new_tokens=10)
torch.cuda.synchronize()

# Naive
t0 = time.time()
_ = model.generate(prompt, max_new_tokens=200)
torch.cuda.synchronize()
naive_t = time.time() - t0

# Cache
t0 = time.time()
_ = model.generate_with_cache(prompt, max_new_tokens=200)
torch.cuda.synchronize()
cache_t = time.time() - t0

print(f"naive: {naive_t:.2f}s, cache: {cache_t:.2f}s, speedup: {naive_t/cache_t:.1f}×")
# 典型输出: naive: 8.43s, cache: 0.47s, speedup: 17.9× (单卡 V100, prompt 256, gen 200)
```

加速比与 prompt 长度成正比——prompt 越长、加速越显著（因为 naive 每步重算的"历史 attention"就越多）。

---

## 4. 工程踩坑与经验

- ❗ **KV cache 必须按 `(layer, head)` 严格分别维护，搞错维度会 attention 算到错的 head**。新手常见错法：把 `past_kv` 弄成全局一个大 tensor，或者把 batch 与 head 维拍扁——结果某一层的 K 串到另一层、某个 head 的 K 串到另一个 head，模型输出乱码却不报错。**正确**：`past_kvs` 是长度 `n_layer` 的 list，每个元素是 `(K, V)` tuple，K, V 形状严格为 `(B, h, T_cur, d_k)`。本节 §3.2 的实现就是这套结构。

- ❗ **用 `torch.cat` 拼接 KV cache 每步都重新分配显存——慢；prealloc 一个 max_seqlen tensor + index_copy 更快**。`torch.cat` 每次都 alloc 一块新显存、把旧 cache + 新 k/v 拷贝过去——decode M 步累计 $O(M^2)$ 拷贝量，TBT 实测劣化 20-40%。**Production 写法**：在 prefill 时一次性 `torch.empty(B, h, max_seqlen, d_k)`、用一个 `cur_len` 指针记位置，decode 时 `K_cache[:, :, cur_len:cur_len+1] = k_new`（in-place 写）。但这又引入"为了 max_seqlen 而预留显存"的浪费——这正是 vLLM **PagedAttention**（11.2）出现的原因：把 KV cache 按 16-token 的 block 分页管理，按需分配，碎片率从 60-80% 降到 < 5%。

- ❗ **与 RoPE 配合时 `position_ids` 必须从 `past_len` 起算，不能从 0**。这是新手实现 KV cache 的高频 bug：decode 一步只输入 1 个 token，调用 `cos_cache[:1]` 拿位置 0 的旋转角——结果新 token 的 q 永远被旋转成"位置 0"的 q，attention 对历史 token 的相对位置全错，模型连"主谓宾"都说不利索。**正确**：`cos = cos_cache[past_len : past_len + T]`、`sin = sin_cache[past_len : past_len + T]`（本节 §3.2 已含）。**自检方法**：用 §3.4 的 sanity test 对比 cache 与 naive 版输出——bug 时偏差会到 1e-1 量级（远超 1e-4 浮点误差）。

- ❗ **batch 推理时不同 sample 长度不一 → 朴素 KV cache padding 浪费严重**。例如 batch=4，4 个 user 的 prompt 长度分别是 128 / 256 / 512 / 1024——naive 实现要按最长 1024 padding，前 3 个 sample 各浪费 75%-87% 的 KV cache 槽位。再叠加每个 user 的 generation 长度也不一（有人 50 token、有人 500 token），同 batch 早结束的 sample 仍占着 cache 不释放。这就是 **Continuous Batching**（vLLM 的招牌）+ **PagedAttention** 解决的问题：iteration-level scheduling，已结束的 sample 立即释放 cache 槽给新 user；不同长度的 KV 用 block 表索引而不必 padding。详见 11.2。

- ❗ **KV cache 是推理 OOM 第一杀手——70B 模型 32 GB 显存只能跑 ~1 batch × 4k context**。新人常以为"模型权重 = 显存瓶颈"，实际**长 context + 大 batch 下 KV cache 经常超过 weights**：LLaMA-2 70B（fp16）weights 140 GB（需 TP 切到多卡），单卡 80 GB H100 加载部分权重 + KV cache @ 32k context @ batch 8 ≈ 80+ GB——直接 OOM。**应对路径**（按性价比排序）：(1) GQA / MLA 压缩 KV cache 4-16×；(2) FP8 / INT8 KV cache 再 ÷ 2；(3) PagedAttention 消除碎片浪费；(4) 长 context 用 chunked prefill + sliding window。任何号称"上 100k context"的 production 部署，必然组合用了这 4 条以上。

- ❗ **Multi-turn chat 时不要每轮重新 prefill 整个 history → 应当增量 append（这是 SGLang RadixAttention 的优化点）**。简单实现：用户每发一条新消息，server 把历史 + 新消息整段拼起来当 prompt 重新 prefill——长聊天时 prefill 算力浪费严重（每轮都重算前面所有 turn 的 attention）。**正确**：保留上一轮的 KV cache，只对"用户的新消息 + 模型即将开始回的部分"做增量 prefill；上一轮的 KV cache 直接复用。SGLang 的 **RadixAttention**（11.3）把这一点系统化——以 token 序列为 key 用 radix tree 索引 KV cache，不同 user 的相同前缀（如 system prompt）也能共享。多 user 共享 system prompt 的场景下，prefill 算力可以省 50%+。

- ❗ **KV cache 量化（INT8 / FP8）现已成熟，质量几乎无损但显存减半（vLLM 支持 `kv-cache-dtype=fp8`）**。KV cache 是 GB 级数据、每步 decode 都要从 HBM 读出来——**memory-bound 阶段，量化是最直接的 speedup**：FP8 比 bf16 快 ~30%（带宽减半），质量在大多数 benchmark 上下降 < 0.5 分。INT8 略激进些但仍可用。**注意**：要量化的是 cache 本身，不是模型权重——这与 GPTQ / AWQ（11.4 权重量化）是两件事，可以叠加。

- ❗ **Prefix caching 在多用户共享 system prompt 时收益巨大（节省 50%+ prefill 算力）**。production 场景里成千上万个请求共享同一段 system prompt（"You are a helpful assistant..."等等，常上千 token）。**朴素**：每个请求独立 prefill 这一段——纯重复劳动。**RadixAttention（SGLang）** / **vLLM 的 Automatic Prefix Caching**：把 system prompt 的 KV cache 算一次缓存到全局 pool，新请求从 pool 直接借用 → prefill 部分跳过 system prompt token 的重算，TTFT 与算力双降。Anthropic Claude 的 Prompt Caching feature、OpenAI 的 Prompt Caching API 商业化卖的就是这个底层能力。详见 11.3。

- ❗ **decode 时**不要**对 cache 后的 K, V 用 `is_causal=True`**。SDPA 的 causal mask 只对方阵 attention 合法（$T \times T$），但 decode 时 q 是 1 行、K 已包含历史 + 新 token 共 $T_{\text{cur}}+1$ 列——这是 $1 \times (T_{\text{cur}}+1)$ 的矩形 attention，新 q 本来就只看到不超过自己位置的 K（cache 里所有 K 都在新 q 之前或就是新 q 自己）。强行 `is_causal=True` 在某些 backend 下会 crash 或静默错误（mask 不对齐）。**本节 §3.1** 用 `is_causal = past_kv is None and T > 1` 仅在 prefill 多 token 时启用 causal——这是正确的判断。

- ❗ **KV cache 在 multi-turn / agent 场景的累积问题**：chat 越长 → KV cache 越大 → 长对话推理 OOM；agent 多轮 tool call 累计的 trace（observation + thought + action）作为上下文不断喂入 → 单 trajectory 几十 turn 后 KV cache 上 GB 级。**对策**：(1) 滑动窗口截断早期 turn（容易丢上下文，慎用）；(2) summary memory（让模型自己总结上文压缩）；(3) RAG 把长文档 retrieval 而非塞 context；(4) MemGPT 风格的"内存分层"。这与 multi-turn RL 训练里 trajectory 越长 KV cache 越爆的问题是同源——Module 14 / 15 会再触及。

---

## 5. 经典 paper

- **Vaswani et al., 2017 — Attention Is All You Need** — 原典在 §3.2 倒数几行简短讨论了"在推理时缓存 K, V"的可能（一笔带过、未展开）；KV cache 的概念在 Transformer 诞生那一刻就已隐含。回头读这段 + 本节 §2.1，能看出"KV cache 不是后人发明，是 Transformer 推理时的自然推论"。
- **Pope et al., 2022 — Efficiently Scaling Transformer Inference** — Google 系统化讨论 LLM 推理 scaling 的开山之作。系统总结了 KV cache 显存模型、TP / PP / batching 对推理的影响、prefill / decode 的 latency 模型。本节 §1.3 / §2.4 的两阶段 + 显存账，这篇是源头。production 推理工程师必读。
- **Kwon et al., 2023 — Efficient Memory Management for Large Language Model Serving with PagedAttention** — vLLM 的 founding paper。揭示 production KV cache 的两大痛点（外部碎片 + 内部碎片，可达 60-80% 浪费），用 OS 风格的 page table 思路把 KV cache 切成 block 管理，吞吐相对 HuggingFace TGI 提升 2-4×。本节 §4 多次预告的"PagedAttention 解决碎片"在 11.2 详讲；现在先读 §3 + §4 建立直觉即可。
- **Karpathy — nanoGPT `model.py`** — 纯教学的最简 KV cache 实现可参考 nanoGPT 的 `generate` + 各种 fork（如 [karpathy/llama2.c](https://github.com/karpathy/llama2.c) 的 `run.c`）——不到 50 行 C 完成 LLaMA-2 推理含 KV cache，是理解"KV cache 极简骨架长什么样"的最佳样本。

---

## 6. 自测与面试题

**Q1（复杂度）**：朴素 generate 与 KV cache 的复杂度差异是什么？分别给出**每 step**与**总 generate**的复杂度，并解释为什么是降一阶。

<details>
<summary>Answer sketch</summary>

设 prompt 长 $N$、生成 $M$ 个 token、模型 $L$ 层、hidden $d$。

**朴素 generate**（每步重算所有 token attention）：
- 第 $t$ 步：输入是 $N+t-1$ 个 token，attention 是 $(N+t-1) \times (N+t-1)$ 方阵，每 step 复杂度 $O\bigl(L (N+t-1)^2 d\bigr)$
- 总：$\sum_{t=1}^{M} O(L (N+t-1)^2 d) = O\bigl(L d (M N^2 + N M^2 + M^3)\bigr)$
- $N \approx M$ 时退化为 $O(L d M^3)$——三次方

**KV cache 版**（缓存历史 K, V，新 token 只算自己的 q）：
- 第 $t$ 步：q 是 1 行、K, V 已含历史 + 新 token 共 $N+t-1$ 行，attention 是 $1 \times (N+t-1)$ 矩形，每 step 复杂度 $O\bigl(L (N+t-1) d\bigr)$
- 总：$\sum_{t=1}^{M} O(L (N+t-1) d) = O\bigl(L d M (N+M)\bigr)$
- $N \approx M$ 时为 $O(L d M^2)$——二次方

**为什么降一阶**：每步从"$O(T^2)$ 全 attention 重算"降到"$O(T)$ 只算新行"，因为：
- 历史 K, V 已经在前面 step 算过、缓存可用
- 新 q 只与历史 K 做 1 行 dot product（causal mask 下未来 token 的 q 不看历史 q）
- Q 不缓存（每步换新的，缓存无意义）

加分：能指出 prefill 仍是 $O(N^2 d L)$（必须算一次）；能说"短 prompt 长 generation"瓶颈在 decode、"长 prompt 短 generation"瓶颈在 prefill；能说实测加速 5-50×（因 FFN / IO 不受 cache 影响、Amdahl 限制）。

</details>

**Q2（显存）**：算一下 LLaMA-3 70B（$n_{\text{layer}} = 80, d_{\text{model}} = 8192$）在 $T = 32k$ context 下，**朴素 MHA** 的 bf16 KV cache 占多少 GB？再算 **GQA-8**（KV head 数 8 而非 64）的 bf16 KV cache 占多少？再算 **GQA-8 + INT8 KV cache 量化**后多少？

<details>
<summary>Answer sketch</summary>

公式：$M_{\text{KV}} = 2 \cdot L \cdot d_{\text{kv}} \cdot T \cdot \text{dtype\_bytes}$

其中 $d_{\text{kv}}$ 是 KV 投影维度——朴素 MHA 时 $= d$；GQA-g 时 $= d \cdot g / h$（KV head 数从 $h$ 缩到 $g$，每个 head 维 $d_k = d/h$ 不变）。

**朴素 MHA, bf16, T = 32k**：
- $d_{\text{kv}} = 8192$
- $M_{\text{KV}} = 2 \times 80 \times 8192 \times 32768 \times 2$ B $= 8.59 \times 10^{10}$ B $\approx \mathbf{80 \text{ GB}}$
- **结论**：单 H100（80 GB）放完 KV cache 没法放权重——必须 TP 多卡

**GQA-8, bf16**（KV head 8 个、Q head 64 个，KV cache ÷ 8）：
- $d_{\text{kv}} = 8192 \times 8/64 = 1024$
- $M_{\text{KV}} = 2 \times 80 \times 1024 \times 32768 \times 2$ B $\approx \mathbf{10 \text{ GB}}$
- **结论**：相比 MHA 缩小 8×，但 32k context 仍占用相当显存

**GQA-8, INT8 KV cache（dtype_bytes = 1）**：
- $M_{\text{KV}} = 10 / 2 = \mathbf{5 \text{ GB}}$
- **结论**：再缩一半，几乎无损质量；production 必开

加分：
- 能算 batch=8 的总 KV cache（5 × 8 = 40 GB）——单卡可装
- 能说 LLaMA-3 70B 实际就是 GQA-8（n_kv_head = 8），所以 32k context 在 80 GB H100 上 batch ~8 是合理上限
- 能说 MLA（DeepSeek-V3）压得更狠（KV cache 投影到 ~576 维 latent + 解压时再升），128k context 也能 ~5 GB

</details>

**Q3（实战）**：你部署一个 70B 模型，发现 batch=2 就 OOM，列出 3 个优化方向（按性价比排序）+ 每个方向的预期收益与可能 trade-off。

<details>
<summary>Answer sketch</summary>

**方向 1：换 GQA / MLA 模型 or 用 KV head 共享版本**
- 收益：KV cache ÷ 4-16×（取决于 GQA-g 还是 MQA / MLA）
- Trade-off：必须换模型权重（如 Qwen2 / LLaMA-3 已经是 GQA-8）；MLA 只有 DeepSeek 系，需重新训不能直接转
- 适用场景：选型阶段、能换模型时——首选

**方向 2：KV cache 量化（FP8 / INT8）**
- 收益：KV cache ÷ 2，附带 decode 阶段带宽利用率提升 ~30%
- Trade-off：MMLU / GSM8K 等 benchmark 通常掉 < 0.5 分（FP8 几乎无损，INT8 略激进）；需要推理引擎支持（vLLM `--kv-cache-dtype fp8` / TensorRT-LLM）
- 适用场景：已选定模型、不能改架构——次选

**方向 3：上 vLLM PagedAttention + Continuous Batching**
- 收益：KV cache 碎片浪费从 60-80% 降到 < 5%；同显存能装 2-4× 更多并发 batch
- Trade-off：需要换推理 framework（HF generate → vLLM）；首次工程接入有学习成本
- 适用场景：production 部署的标配——必做

**方向 4（加分项）：Prefix caching（多 user 场景）**
- 收益：共享 system prompt 时省 50%+ prefill 算力，间接腾出 KV cache 空间给更多 batch
- Trade-off：单 user 长 conversation 收益有限；需要 SGLang 或 vLLM Automatic Prefix Caching

**方向 5（最后手段）：tensor parallel 切到 2-4 卡**
- 收益：单卡 KV cache 显存占用 ÷ TP 度
- Trade-off：通信开销、卡多成本高、单卡 batch 反而下降
- 适用场景：模型本身权重单卡塞不下时必须；纯为 KV cache 加 TP 不划算

加分：
- 能说"先看 KV cache 占多少 / 算了 §2.4 公式再决策" —— 工程师素养
- 能区分"KV cache 量化"（cache 数据本身）与"权重量化 GPTQ / AWQ"（model weights）是两件事、可叠加
- 能提一句 chunked prefill / sliding window 应对超长 context

</details>

---

## 7. 延伸阅读

- [Hugging Face Blog — A guide to LLM inference and performance](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests) — prefill / decode 心智模型与性能优化的官方科普，本节 §1.3 的 mental model 可对照阅读
- [vLLM 官方文档 — PagedAttention](https://docs.vllm.ai/en/latest/dev/kernel/paged_attention.html) — 11.2 详讲前先读这篇建立直觉
- [Kwon et al. 2023 — vLLM PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — KV cache 内存管理的奠基论文，§3 内存分析与 §4 算法设计必读
- [Pope et al. 2022 — Efficiently Scaling Transformer Inference (arXiv:2211.05102)](https://arxiv.org/abs/2211.05102) — Google 推理 scaling 总论，KV cache 显存模型与 prefill / decode 分析的原始出处
- [Sebastian Raschka — Understanding and Coding KV Cache in LLMs](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) — 与本节 §3 风格一致的手撕 KV cache 教程
- [karpathy/llama2.c](https://github.com/karpathy/llama2.c) — 纯 C 实现的 LLaMA-2 推理含 KV cache，`run.c` 不到 1000 行，$O(T^2) \to O(T)$ 的优化在最简代码上看得最清楚
- 推荐继续读本教程的 **5.2 节《GQA / MQA / MLA：KV cache 压缩》**——本节算账后你会发现 KV cache 是 OOM 第一杀手，5.2 给出"在架构层一次性压 4-16×"的根本解
- 推荐继续读本教程的 **11.2 节《PagedAttention 与 Continuous Batching（vLLM）》**——本节预告的"按 block 管理 KV cache + iteration-level scheduling"在 11.2 系统讲透
- 推荐继续读本教程的 **11.3 节《RadixAttention 与 Prefix Cache（SGLang）》**——多 user 共享 system prompt 场景下 prefill 算力的根本性优化
- 推荐继续读本教程的 **11.4 节《量化：GPTQ / AWQ / GGUF / FP8 / SmoothQuant》**——KV cache 量化是 11.4 的子模块，与权重量化可叠加
