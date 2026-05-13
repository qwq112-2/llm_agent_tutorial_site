---
title: "11.5 投机解码：Speculative / Medusa / EAGLE"
description: "LLM 单 stream decode 是 memory-bound——70B model 在 H100 上单请求只跑 ~40 token/s，根本原因是每生成 1 个 token 都要把整个模型权重从 HBM 搬到 SRAM 一次（11.1 已讲）。投机解码（speculative decoding） 是当前唯一数学上无损且能把单 stream 速度提升 1.5-5× 的方案：用一个又快又略糙的"
---

> ⏱ 预计阅读 45 分钟 ｜ 难度 ★★★ 🔥 必考 ｜ 前置：4.6 完整 decoder-only 实现（必）；4.7 KV cache（强烈推荐）；11.1 推理性能指标

## 一句话本节讲什么

LLM 单 stream decode 是 **memory-bound**——70B model 在 H100 上单请求只跑 ~40 token/s，根本原因是每生成 1 个 token 都要把整个模型权重从 HBM 搬到 SRAM 一次（11.1 已讲）。**投机解码（speculative decoding）** 是当前唯一**数学上无损**且能把单 stream 速度提升 1.5-5× 的方案：用一个又快又略糙的 **draft model** 一次预测 K 个候选 token，再用 target model **一次 forward** 并行验证 K 个位置，配 rejection sampling 接受 / 拒绝，最终输出分布与原 model 完全一致。本节讲 Leviathan 2023 的原始 spec decoding 算法、Medusa 多头预测、EAGLE feature-level draft 三大方案的差异，给出 vLLM 实战配置与"什么场景用 / 不用"的判断准则——是 reasoning model（R1 类 long-CoT）部署的关键优化、是 2025-2026 推理引擎面试的标配考点。

---

## 1. Mental model（直觉）

### 1.1 为什么单 stream decode 慢：memory-bound 复盘

11.1 已经讲过：H100 的 FP16 算力约 1000 TFLOPS、HBM3 带宽约 3 TB/s——**算力 / 带宽比 ≈ 333 FLOP/byte**。LLaMA-3 70B 在 FP16 下权重 140 GB，单 stream 每 forward 一次（生成 1 个 token）必须把 140 GB 权重从 HBM 搬到 SM 上的 SRAM 一次，搬运耗时下界 $\frac{140 \text{ GB}}{3 \text{ TB/s}} \approx 47 \text{ ms}$，理论上限 1/0.047 ≈ 21 token/s；实测 H100 单 stream 大概 30-40 token/s（kernel 调度等还有其他开销）。

**关键洞察**：这 47 ms 里 GPU 算力几乎是闲置的。每生成 1 个 token 实际只需要 $2 \times 70 \times 10^9 = 1.4 \times 10^{11}$ FLOP，按 H100 算力只需 0.14 ms——**算力空了 99.7%**。这意味着你可以用同一次 forward 顺手算很多 token 而几乎不增加 latency，只要它们能"塞进同一次 weight load"。

**这就是 speculative decoding 的全部物理基础**：把"1 次 forward 算 1 个 token"改成"1 次 forward 验证 K 个候选 token"——weight load 还是一次，FLOP 多了 K 倍但远没用满算力，wall-clock 几乎不变。如果 K 个候选大部分都对，你就用 1 次 forward 的时间生成了 K 个 token——速度提升 K 倍。

### 1.2 投机解码的核心 idea：draft + verify

类比：**你在写 SQL**——一种慢方式是每写一行就让 DBA review 一次（每次 review 极慢）；快方式是**自己先写 5 行（你写得快但偶尔错）**，然后**让 DBA 一次性把 5 行都 review 一遍**——DBA 一次 review 5 行的成本与 review 1 行差不多，只要你写对的多就净赚。

LLM 投机解码完全同构：

- **draft model**（你）：又快又略糙——一个 1B 小 model 在同一硬件上跑 ~200 token/s，是 70B 大 model 单 stream 5×
- **target model**（DBA）：又慢又准——70B 跑 40 token/s，但**一次 forward 验证 K 个位置的 cost ≈ 1 个位置**（前面讲过 GPU 算力空闲）
- **流程**：draft 串行生成 K 个 token → target 并行 verify K 个位置 → rejection sampling 决定接受多少 → 接受 m 个、拒绝点重 sample 1 个 → 共得到 $m + 1$ 个 token / 1 次大 forward

如果接受率 60%、K = 5，每次大 forward 平均出 $0.6 \times 5 + 1 = 4$ 个 token——4× 加速（扣 draft 开销后净 2-3×）。

### 1.3 三大方案的演化路径（必背 ASCII 时间线）

```
2018 ─── Stern: Blockwise Parallel Decoding（早期类似思想，没火）
   │
2023 ─── Leviathan / Chen: Speculative Decoding ★ 经典版
   │       └── 用独立的小 LLM 当 draft；rejection sampling 严格保证 lossless
   │
2024 ─── Cai: Medusa
   │       └── 不要 draft model，给 target 加多个 prediction head
   │           一次 forward 同时拿 K 个候选；tree attention 并行 verify
   │
2024 ─── Li: EAGLE / EAGLE-2 / EAGLE-3 ★ 现 SOTA
           └── feature-level draft（在 hidden state 上做 auto-regressive
               预测），接受率 85%+，加速 3-5×
```

三者的核心矛盾都是：**draft 要快**（不然 draft 时间反而拖累）+ **draft 要准**（不然接受率低、白浪费一次大 forward）。三种方案在这两者间各有取舍：

| 方案 | "快" 怎么做 | "准" 怎么做 |
|---|---|---|
| Spec Decoding | 用小 LLM（1/30 大小）独立 forward | 同 family 同 tokenizer，分布天然接近 |
| Medusa | 1 次 target forward 顺便算 K 个 head | 多 head 训练 + tree attention 多候选 |
| EAGLE | feature 上轻量 draft head | feature 比 token 信息密度高，接受率天花板更高 |

### 1.4 lossless 的 mental model：rejection sampling 的"为什么"

很多人第一次听 spec decoding 会怀疑："draft model 都是猜的，怎么可能输出分布与不加 spec 一样？" 关键在 **rejection sampling**——不是无脑接受 draft 的 token，而是**用 target / draft 的概率比做加权接受**：

- target 同意 draft 的 token（target 概率 ≥ draft 概率）→ 一定接受
- target 觉得这 token 概率偏高（target < draft）→ **按比例 $p_\text{target} / p_\text{draft}$ 接受**，被拒就用 target 的"修正分布" $\max(0, p_\text{target} - p_\text{draft})$ 重新 sample

数学上严格证明（见 §2.3）：最终采到的 token 的分布**正好等于 target 的分布**——也就是说，spec decoding **不是近似**、**不是降质换速度**，而是**严格等价**于直接从 target sample，只是 wall-clock 更快。这一点是 spec decoding 区别于"用 draft 直接出"的根本：draft 直接出会损失质量，spec decoding 不会。

### 1.5 何时不会加速 / 反而变慢（关键 mental model）

spec decoding **不是万能加速**——它在以下场景会失效甚至变慢：

- **batch 大（≥ 8）**：当 batch 足够大时，target model 一次 forward 已经把 GPU 算力榨干（throughput-bound 而非 memory-bound），weight load 的成本被摊薄到每个 sample 上很小——这时 draft model 的额外开销反而成了纯负担
- **draft 太大**：draft 自己跑得慢，draft 时间本身就 > target 1 次 forward
- **draft 太小 / 太烂**：接受率 < 30%，平均每次大 forward 只多产 0.5 个 token，不偿
- **生成很短（< 10 token）**：spec decoding 的"启动开销"（draft 预热、tree 验证）摊不到几个 token 上

**结论**：spec decoding 是**单 stream / 小 batch / 长 generation / latency 敏感**场景的优化，**不**是高 throughput 服务的优化。这正好是 R1 类 reasoning model 部署的典型场景——long-CoT 单请求几千 token、用户开 1 个对话 1 个 stream，spec decoding 收益巨大。

---

## 2. 公式与原理

### 2.1 形式化定义

设 target model 的下一 token 分布为 $p(x \mid \text{context})$，draft model 的对应分布为 $q(x \mid \text{context})$，词表 $V$ 大小为 $|V|$。spec decoding 的目标是采样 $x \sim p$，但只能调用：

- **draft 串行**：用 $q$ 一次产生 $K$ 个候选 $\hat{x}_1, \hat{x}_2, \dots, \hat{x}_K$（成本 $K \cdot c_\text{draft}$，$c_\text{draft}$ 是 draft 1 次 forward 的 latency）
- **target 并行**：一次 forward 算出所有 $K$ 个位置的 $p_t(x)$（成本 $c_\text{target}$，与 1 次串行 forward 几乎一致）

### 2.2 算法流程（伪代码必背）

```
输入: prompt, target model p, draft model q, 每轮投机 K 个 token
输出: 与直接从 p sample 分布等价的 generation

while not done:
    # ===== 1. Draft 串行预测 K 个 token =====
    for i in 1..K:
        x_hat[i] ~ q(· | prompt + x_hat[1..i-1])
        q_prob[i] = q(x_hat[i] | ...)              # 记录 draft 概率

    # ===== 2. Target 并行 verify K 个位置（关键：1 次 forward）=====
    p_dist[1..K+1] = target_forward(prompt + x_hat[1..K])
    # p_dist[i] 是 "看到 prompt + x_hat[1..i-1] 后" target 的下一 token 分布
    # 多算的 p_dist[K+1] 是 "看到全部 K 个 draft 后" 的分布，用于 bonus token

    # ===== 3. Rejection sampling 接受 / 拒绝 =====
    accepted = []
    for i in 1..K:
        r = uniform(0, 1)
        if r < min(1, p_dist[i](x_hat[i]) / q_prob[i]):
            accepted.append(x_hat[i])              # 接受
        else:
            # 拒绝：从修正分布重新 sample 1 个
            p_resid = normalize(max(0, p_dist[i] - q))
            x_new ~ p_resid
            accepted.append(x_new)
            break                                  # 拒绝点之后的 draft 都丢弃

    # ===== 4. Bonus token（全部接受时）=====
    if all K accepted:
        # 多算的 p_dist[K+1] 直接 sample 1 个
        x_bonus ~ p_dist[K+1]
        accepted.append(x_bonus)

    prompt += accepted
```

每一轮 while 平均产出 token 数 = $\sum_{m=0}^{K-1} (\text{接受 m 个的概率}) \cdot (m+1) + (\text{全部接受概率}) \cdot 1$（bonus）；最少 1（第 1 个就拒绝），最多 $K + 1$（全接 + bonus）。

### 2.3 Rejection sampling 为什么 lossless（关键证明）

定理：上述算法每接受 / 重 sample 出的 token $x$，其边际分布精确等于 $p(x)$。

证明（对单个位置）：设接受 draft 的事件为 $A$，则被采到 $x$ 的概率：

$$\Pr(\text{output} = x) = \underbrace{\Pr(\text{draft} = x) \cdot \Pr(A \mid \text{draft} = x)}_{\text{path 1: 接受 draft}} + \underbrace{\Pr(\text{reject}) \cdot \Pr_\text{resid}(x)}_{\text{path 2: 拒绝后重 sample}}$$

第一项：

$$q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))$$

第二项的 $\Pr(\text{reject})$：

$$1 - \sum_y q(y) \cdot \min\left(1, \frac{p(y)}{q(y)}\right) = 1 - \sum_y \min(p(y), q(y))$$

第二项的 $\Pr_\text{resid}(x)$ 来自归一化的 $\max(0, p - q)$：

$$\Pr_\text{resid}(x) = \frac{\max(0, p(x) - q(x))}{\sum_y \max(0, p(y) - q(y))} = \frac{\max(0, p(x) - q(x))}{1 - \sum_y \min(p(y), q(y))}$$

代入：

$$\Pr(\text{output} = x) = \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)$$

最后一步因为：当 $p(x) \geq q(x)$ 时，$\min = q$、$\max = p - q$，加和 $= p$；当 $p(x) < q(x)$ 时，$\min = p$、$\max = 0$，加和 $= p$。**两种情况都等于 $p(x)$，即 lossless**。

> 数学上 lossless 的代价是：**采样种子的具体路径**与不加 spec 时不同（你接受的随机数序列变了），所以即使设了同一个 seed，spec on / off 的具体输出可能不同——但**分布相同**。这点在 §4 工程踩坑会展开。

### 2.4 加速比公式（必备）

设接受率为 $\alpha$（每个位置 draft 被接受的概率），每轮投机 $K$ 个 token，draft 1 次 forward 的 latency 是 target 的 $r$ 倍（典型 $r = 0.05 \sim 0.10$）。

每轮 while 的成本：$K \cdot r \cdot c_\text{target}$（draft K 次） + $c_\text{target}$（target 1 次 verify） $= (1 + K r) \cdot c_\text{target}$。

每轮 while 的产出：期望接受 token 数 $\bar{n}$。在简化假设下（每位置接受独立同分布），接受到第 $i$ 个就被拒的概率是 $\alpha^{i-1}(1 - \alpha)$，全接受的概率是 $\alpha^K$（再 + 1 个 bonus）：

$$\bar{n} = \sum_{i=1}^{K} \alpha^{i-1}(1-\alpha) \cdot i + \alpha^K \cdot (K+1) = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

**加速比**（vs 不加 spec 时每轮产 1 token、cost $c_\text{target}$）：

$$\boxed{\text{Speedup} = \frac{\bar{n}}{1 + K r} = \frac{1 - \alpha^{K+1}}{(1 - \alpha)(1 + K r)}}$$

代入典型值（$\alpha = 0.7$、$K = 5$、$r = 0.05$）：

- $\bar{n} = (1 - 0.7^6) / (1 - 0.7) = (1 - 0.117) / 0.3 \approx 2.94$
- Speedup $= 2.94 / (1 + 0.25) = 2.94 / 1.25 \approx 2.35×$

调参直觉：

- 接受率 $\alpha$ 是天花板；$\alpha \to 1$ 时 speedup $\to (K+1)/(1 + K r) \approx K$
- 接受率 $\alpha = 0.5$ 时 speedup ≈ 1.5×；$\alpha = 0.85$ 时 speedup ≈ 3.5×
- $K$ 不是越大越好——$K$ 增大但 $\alpha^{K+1}$ 衰减更快，存在最优 $K^*$（典型 4-7）

### 2.5 Medusa 与 EAGLE 的差别（一句话版）

- **Medusa**：在 target model 最后一层 hidden state 上接 $K$ 个 head $H_1, H_2, \dots, H_K$，$H_i$ 直接预测位置 $t + i$ 的 token logits。一次 target forward 同时输出位置 $t+1, t+2, \dots, t+K$ 的候选；用 **tree attention** 并行 verify 多个候选 path（不是单链）。**优点**：无独立 draft model、部署简单；**缺点**：需要单独 fine-tune Medusa head、接受率天花板比独立 draft model 低（约 65%）
- **EAGLE**：在 feature-level（hidden state）做 auto-regressive draft——一个 small head 拿"target 上一步的 hidden state + 当前 draft token"作为输入，预测下一步的 hidden state；这个 hidden state 经 target 自己的 lm_head 转成 token 分布。EAGLE-3 进一步用多层 feature 融合，接受率 $\alpha$ 达到 85%+，speedup 3-5×。**核心 insight**：feature 比 token 信息密度高得多（continuous vs discrete），用 feature 做 draft 信号更准

### 2.6 三大方案对比（必背表）

| 方案 | draft 类型 | lossless | 典型加速比 | 实现复杂度 | 训练成本 |
|---|---|---|---|---|---|
| **Speculative Decoding** | 独立小 LLM（1/10 - 1/30 大小） | ✓ 严格 | 1.5-3× | 中 | 0（即插即用） |
| **Medusa** | target 上加 K 个 head | ✓ 严格（tree attn） | 1.5-2.5× | 低 | 中（fine-tune K 个 head） |
| **EAGLE-3** | feature-level 小 AR head | ✓ 严格 | **3-5×** | 中 | 低（7B 数据 + 几小时） |
| **PLD** (Prompt Lookup) | n-gram lookup from prompt | ✓ 严格 | 1.2-2×（代码 task） | 低 | 0（无 model） |
| **Lookahead Decoding** | Jacobi iteration | ✓ 严格 | 1.5-2× | 中 | 0 |

---

## 3. 最小代码示例

### 3.1 手写 Spec Decoding 主循环（≤ 35 行）

self-contained 的 rejection sampling 核心循环。**target 与 draft 都用 nn.Module 接口**，便于直接套到任何 HF model 上。

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def speculative_decode_step(target, draft, prompt_ids, K=5, max_new=128):
    """单 stream spec decoding；返回生成的 token 序列（含 prompt）。
    target / draft: 任意 callable，签名 model(input_ids) -> logits (B, T, V)
    """
    seq = prompt_ids.clone()                                      # (1, T0)
    while seq.size(1) - prompt_ids.size(1) < max_new:
        # ===== 1. Draft 串行预测 K 个 token，记录 draft 概率 =====
        draft_seq = seq.clone()
        q_probs = []                                              # K 个分布
        for _ in range(K):
            q_logits = draft(draft_seq)[:, -1, :]                 # (1, V)
            q_dist = F.softmax(q_logits, dim=-1)
            x = torch.multinomial(q_dist, num_samples=1)          # (1, 1)
            q_probs.append(q_dist)
            draft_seq = torch.cat([draft_seq, x], dim=1)
        x_hat = draft_seq[:, -K:]                                 # (1, K) draft tokens

        # ===== 2. Target 一次 forward 验证 K 个位置 + 1 个 bonus =====
        p_logits = target(draft_seq)[:, -K-1:, :]                 # (1, K+1, V)
        p_dists = F.softmax(p_logits, dim=-1)                     # K+1 个分布

        # ===== 3. Rejection sampling =====
        accepted = []
        all_pass = True
        for i in range(K):
            tok = x_hat[0, i].item()
            r = torch.rand(1, device=seq.device).item()
            ratio = (p_dists[0, i, tok] / q_probs[i][0, tok]).clamp(max=1).item()
            if r < ratio:
                accepted.append(tok)                              # 接受
            else:
                # 拒绝：从 max(0, p - q) 归一化后 sample 新 token
                resid = (p_dists[0, i] - q_probs[i][0]).clamp(min=0)
                resid = resid / resid.sum()
                accepted.append(torch.multinomial(resid, 1).item())
                all_pass = False
                break
        # ===== 4. 全部接受时 sample bonus token =====
        if all_pass:
            accepted.append(torch.multinomial(p_dists[0, K], 1).item())
        seq = torch.cat([seq, torch.tensor([accepted], device=seq.device)], dim=1)
    return seq
```

**关键行解读**：

- **L13-19**（draft 循环）：串行 K 次 draft forward 是开销大头；production 用 KV cache（4.7）后 draft 1 步只跑 1 个 token，而非每次重算前缀
- **L22**（target 一次 forward）：spec decoding 的精华——`draft_seq` 长度是 prompt + K，target 只 forward 1 次拿到 K + 1 个位置的概率分布。生产环境下 target 也用 KV cache，只新算 K + 1 个新位置的 attention
- **L29-31**（接受判定）：`min(1, p/q)` 用 `clamp(max=1)`；接受概率比就是 §2.3 推导的核心
- **L34-36**（拒绝重 sample）：`max(0, p - q)` 归一化后 multinomial——这是 lossless 的数学保证
- **L40-41**（bonus）：全接受时多产 1 个 token——这是为什么 spec 最多能产 K+1 个 token / 1 次大 forward

### 3.2 vLLM Speculative Decoding 配置（≤ 20 行）

vLLM v0.6+ 把 spec decoding 做成一行配置；线上环境直接用，不需要手撕。

```python
# vllm 0.6+ 的 spec decoding 配置（Llama-3 70B + Llama-3 1B 作 draft）
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",        # draft model
    num_speculative_tokens=5,                                     # K，甜点 4-7
    use_v2_block_manager=True,                                    # 必开
    tensor_parallel_size=4,                                       # target TP
    gpu_memory_utilization=0.85,
)

prompts = ["请解释 speculative decoding 的工作原理。"]
out = llm.generate(prompts, SamplingParams(max_tokens=512, temperature=0.7))
print(out[0].outputs[0].text)

# 用 PLD（Prompt Lookup Decoding）替代 draft model（代码任务最优）：
llm_pld = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    speculative_model="[ngram]",                                  # 关键字
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,                                    # 最长 n-gram
)
```

**关键点**：

- `speculative_model` 必须与 target 同 family / 同 tokenizer——不同 tokenizer 算的 logits 没法逐位置对应（§4 第 2 条）
- `num_speculative_tokens` 通常 4-7；超过 10 加速反而下降（§4 第 3 条）
- `[ngram]` 是 vLLM 内置的 PLD draft，零成本——**代码生成 / JSON 输出场景几乎免费 1.5-2× 加速**（重复 token 多）
- 多卡部署时 draft 默认与 target 共享 TP；如果 draft 极小（1B）也可单卡放，看 vLLM 版本配置

### 3.3 Medusa Head 简化结构（≤ 25 行）

Medusa 不需要独立 draft model，给 target 接 $K$ 个并行 head——本质是 K 个独立的 "skip-i 位预测" lm_head。

```python
import torch
import torch.nn as nn

class MedusaHead(nn.Module):
    """单个 Medusa head: 预测 t+i 位置的 token（i 由位置决定）。
    每个 head = 1 个 ResBlock + 1 个 lm_head (共享 vocab)。
    """
    def __init__(self, d_model: int, vocab_size: int, lm_head: nn.Linear):
        super().__init__()
        self.skip = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
        )
        self.lm_head = lm_head                                    # 与 target 共享！

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: target 最后一层 hidden state (B, T, d)
        return self.lm_head(h + self.skip(h))                     # ResBlock + lm_head

class MedusaModel(nn.Module):
    def __init__(self, base_model, num_heads: int = 4):
        super().__init__()
        self.base = base_model                                    # target LLM (frozen)
        d, V = base_model.config.hidden_size, base_model.config.vocab_size
        self.heads = nn.ModuleList([
            MedusaHead(d, V, base_model.lm_head) for _ in range(num_heads)
        ])

    def forward(self, input_ids):
        h = self.base.model(input_ids).last_hidden_state          # (B, T, d)
        # 每个 head 预测一个 future 位置
        return [head(h) for head in self.heads]                   # list of (B, T, V)
```

**关键点**：

- **共享 lm_head**：所有 Medusa head 复用 target 的 lm_head，省 V·d × K 参数
- **训练时**：base 通常 frozen 或低 lr，只训 head；target 输出位置 t 的 head_i 监督是 $t + i$ 位置的 ground truth
- **推理时**：1 次 base forward 同时拿到 K 个 head 的输出，组成 candidate tree（每个 head top-2 产生 $2^K$ 条 path），用 **tree attention**（一次 forward 验证多 path）选出最长接受前缀

### 3.4 加速比 Benchmark（≤ 25 行）

测量 spec decoding 实际收益的最小 benchmark 脚本：

```python
import time
from vllm import LLM, SamplingParams

def bench(llm, prompts, max_tokens=256, n_runs=3):
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)   # greedy 便于对比
    # warmup
    llm.generate(prompts[:1], sp)
    # 计时
    t0 = time.perf_counter()
    total_tokens = 0
    for _ in range(n_runs):
        outs = llm.generate(prompts, sp)
        total_tokens += sum(len(o.outputs[0].token_ids) for o in outs)
    dt = time.perf_counter() - t0
    return total_tokens / dt                                      # token/s

prompts = ["写一段 100 行的 Python quicksort 实现。"] * 1            # 单 stream
base = LLM(model="meta-llama/Llama-3-70B-Instruct", tensor_parallel_size=4)
spec = LLM(model="meta-llama/Llama-3-70B-Instruct",
           speculative_model="meta-llama/Llama-3.2-1B-Instruct",
           num_speculative_tokens=5, tensor_parallel_size=4)

print(f"Base       : {bench(base, prompts):.1f} tok/s")
print(f"Spec (K=5) : {bench(spec, prompts):.1f} tok/s")
# 典型输出（H100 x4）:
# Base       : 38.2 tok/s
# Spec (K=5) : 92.4 tok/s   ← 2.4× 加速
```

**注意**：benchmark 必须用 `temperature=0.0`（greedy）或固定 seed 才能跨 run 复现 token 数；不同 batch size / 不同输入长度需要分别测，spec decoding 的收益在不同场景差异极大（§4 第 1 条）。

---

## 4. 工程踩坑与经验

- ❗ **Spec decoding 在 batch ≥ 8 时几乎无加速 / 甚至变慢**。这是最大的也是最被忽略的坑。spec decoding 的物理基础是"target memory-bound、算力空闲"——但 batch 大时 target 已经 throughput-bound（多个 sample 共用一次 weight load），算力被榨干，draft 的额外开销反而成纯负担。**结论**：spec decoding 是单 stream / 小 batch / latency 敏感场景的优化，**不要**在高 throughput 服务（batch 32+）上开。线上典型部署模式：**chat / agent 单用户长 stream → 单独的 spec decoding 实例**；**批量数据生成 → 不开 spec 的高 throughput 实例**。两类实例分开部署。

- ❗ **draft model 与 target model 必须 tokenizer 一致（同 family）**。spec decoding 要求 draft 输出的 token id 能直接放到 target 的输入序列里，并且 target 算的概率分布是在同一个词表上——tokenizer 不同时，draft 输出的 "id 5234" 在 target 那里可能对应完全不同的 subword。**正确选择**：Llama-3 70B 配 Llama-3.2 1B（同 family）；Qwen2 72B 配 Qwen2 1.5B（同 family）；**不要**混用 Llama 与 Qwen / Llama 与 Mistral——tokenizer / vocab 不同。如果非得用跨 family，需要 token-level alignment 或 retrain draft 的 lm_head 对齐 target 词表，工程成本极高。

- ❗ **`num_speculative_tokens` 太大（> 10）反而变慢**。直觉上 K 越大越快，实际不然——接受率 $\alpha < 1$ 时 $\alpha^K$ 衰减很快，K = 10 时即使 $\alpha = 0.7$ 也只有 $0.7^{10} \approx 2.8\%$ 概率全部接受；同时 draft 串行 K 次的成本线性增长。**实测甜点 K = 4-7**——业界论文几乎都用 5。**不要**为了"看起来更激进"开 K = 10+，用 §2.4 公式手算最优 K* 才靠谱。

- ❗ **Spec decoding 在 long generation（> 1000 token）收益巨大，short generation（< 50 token）几乎无收益**。短输出时 spec 的"启动开销"（draft 预热 KV cache、第一轮验证 latency）摊不到几个 token 上，net 加速可能 < 1.0。但 reasoning model（R1 类）输出几千 token 的 long-CoT 时，spec decoding 节省 **50-70% wall-clock**——是 R1 部署的关键优化。**业务决策**：chat 场景（响应 200 token）spec 收益中等；agent / reasoning 场景（响应 2000+ token）spec 收益巨大、必须开。

- ❗ **Medusa head 需要训练（fine-tune），不像 spec decoding 即插即用**。Medusa 的 K 个 head 是新参数，必须用 target model 的输出当 teacher、用 ground truth token 当 supervision 训练（论文典型 7B 数据 + 1-2 epoch）。**部署成本对比**：spec decoding = 0 训练成本（直接拿现有小 LLM 当 draft）；Medusa = 中训练成本但部署后无独立 draft model；**EAGLE = 低训练成本（小 head + 7B 数据 + 几小时）但加速比天花板最高**。如果你只想"今天就上线"，spec decoding；如果你能投入 1 周训练，EAGLE。

- ❗ **EAGLE 需要专门的 draft head，但训练成本远低于"训一个独立 draft model"**。EAGLE 的 draft head 是 1-2 层小 transformer 在 feature-level 做 auto-regressive，参数 < 1B，typically 在 7B token 数据上训几小时即可（vs 训一个 1B draft model 需要数百 B token）。**关键 insight**：因为 draft 是在 target 自己的 hidden state 空间上跑的，分布天然对齐，几 B 数据就够了。EAGLE-3 提供 Llama / Qwen / DeepSeek 的官方预训 head，直接下载用即可。

- ❗ **lossless 是分布意义上的，单次输出可能与不开 spec 不同**。spec decoding 的数学证明是"边际分布相同"——但**采样路径**（消耗的随机数序列）不同：开 spec 时一次接受多个 token、消耗多个随机数；不开时逐个 token 消耗。即使设了同一个 seed，spec on / off 的具体输出 sequence 可能不同。**坑场景**：你想做 A/B test 验证"开 spec 后质量是否退化"——直接 diff 输出会发现不一样，**这不一定是质量问题**，是采样路径差异。**正确做法**：跑 1000+ 个 prompt 看分布层面的指标（pass rate / human eval / benchmark score），不要逐 token diff。

- ❗ **vLLM 的 spec decoding 在某些 model / 配置上有 bug，建议先 benchmark 输出一致性**。vLLM v0.6 之前的 spec decoding 实现在 chunked prefill / TP 边界 / 某些 sampling 配置下有 known bug，可能输出 garbage 或抛异常。**Production 上线前 checklist**：(1) 跑同一 prompt 用 `temperature=0` 在 spec on / off 下出 token，看 logits 分布是否近似（不是严格相等，看分布距离）；(2) 跑标准 benchmark（MMLU / HumanEval）spec on / off，得分应该在噪声范围内（< 0.5%）；(3) 长 stream（2000+ token）压力测试 1 小时看是否崩溃。**强烈建议**用 vLLM v0.6.2+；老版本 spec decoding 不稳。

- ❗ **draft model 占显存**——70B 配 1B draft 要多吃 ~2 GB（fp16）；H100 80G 单卡装 70B 已经紧张，再装 draft 可能 OOM。**对策**：(1) draft 用 int8 量化（精度足够，draft 本来就糙）；(2) draft 放在独立卡上 vs target；(3) 用 PLD（n-gram）替代 draft model，零显存。代码生成 / 结构化输出场景 PLD 几乎与小 LLM draft 接近的接受率。

- ❗ **温度 / top_p 等 sampling 参数会影响接受率**。$\alpha$ 是 target 与 draft 概率分布的"重叠程度"——`temperature=0`（greedy）时分布退化为 one-hot，draft 与 target 的 argmax 不一致就直接拒绝，$\alpha$ 反而低；`temperature=1.0` + 中等 top_p 时分布平缓，$\alpha$ 反而高。**反直觉但实测如此**：spec decoding 在 chat 场景（温度 0.6-0.8）的加速比通常比 greedy decode 高。

---

## 5. 经典 paper

- **Leviathan, Kalman, Matias, 2023 — Fast Inference from Transformers via Speculative Decoding** — Google Research 提出 speculative decoding 的奠基 paper。读它的 §3 "Speculative Sampling" 一段就能看到本节 §2.3 那段 rejection sampling 的原始数学推导，以及加速比公式 $E[\#tokens] = (1 - \alpha^{K+1}) / (1 - \alpha)$。这一篇是所有 spec decoding 工作的源头，**必读**。
- **Chen, Borgeaud, Irving, et al., 2023 — Accelerating Large Language Model Decoding with Speculative Sampling** — DeepMind 同期独立提出 spec decoding。与 Leviathan 思路完全一致但工程实现细节不同（Chinchilla 70B + 4B draft 的 benchmark）；两篇并列被引为 spec decoding 的源 paper。读它的实验部分能看到"draft model 大小 vs 加速比"的经验曲线。
- **Cai, Li, Geng, et al., 2024 — Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** — Medusa 的提出。读它的 §3 能看到多 head 设计 + tree attention 的细节；§4.2 的 ablation 显示 head 数 K = 4-5 是甜点（与本节 §2.4 的最优 K* 分析一致）。Medusa-2 与 Medusa-1 的差别主要在训练方法（joint vs frozen base）。
- **Li, Wei, Zhang, Zhang, 2024 — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** + **EAGLE-2 / EAGLE-3** — 现 SOTA 的 spec decoding 方案。核心 insight 是"feature-level draft 比 token-level 好"——读 §3.1 能看到为什么 hidden state 信息比 token id 多得多、能让 draft head 接受率突破 80%。EAGLE-3 引入 multi-layer feature 融合 + 更激进的 tree expansion，加速 3-5×，工业界 2025 后的事实标准。
- **Stern, Shazeer, Uszkoreit, 2018 — Blockwise Parallel Decoding for Deep Autoregressive Models** — 早期类似思想（NeurIPS 2018），用并行预测多个位置加速 NMT。当时没引起广泛关注，5 年后被 Leviathan 重新发现。读它对理解"并行预测"思想的源头有帮助，**加分**。
- **Saxena, 2023 — Prompt Lookup Decoding (PLD)** — 用 prompt 自己的 n-gram 当 draft 的极简实现，**零训练成本、零额外显存**。在代码生成 / 结构化输出场景几乎免费 1.5-2× 加速。值得一读源码（< 200 行）理解为什么"什么 model 都不要"也能加速。

---

## 6. 自测与面试题

**Q1（数学）**：写出 spec decoding 的 rejection sampling 算法（接受概率公式 + 拒绝时如何重新采样）。证明这样做最终输出分布严格等于 target 分布（即 lossless）。

<details>
<summary>Answer sketch</summary>

**算法**：

- 设 target 分布 $p$、draft 分布 $q$、draft 采样到 $x$
- **接受概率**：$\Pr(A \mid x) = \min(1, p(x) / q(x))$
- **拒绝时重 sample**：从修正分布 $\Pr_\text{resid}(x) = \frac{\max(0, p(x) - q(x))}{Z}$ 采样，归一化常数 $Z = 1 - \sum_y \min(p(y), q(y))$

**证明 lossless**（输出 = $x$ 的总概率）：

$$\Pr(\text{out} = x) = \underbrace{q(x) \cdot \min(1, p(x)/q(x))}_{\text{接受 path}} + \underbrace{(1 - \sum_y \min(p, q)) \cdot \frac{\max(0, p(x) - q(x))}{1 - \sum_y \min(p, q)}}_{\text{拒绝重采 path}}$$

$$= \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x)$$

最后一步：

- 当 $p(x) \geq q(x)$：$\min = q$、$\max = p - q$，加和 $= p$ ✓
- 当 $p(x) < q(x)$：$\min = p$、$\max = 0$，加和 $= p$ ✓

**lossless 含义**（必须说到）：分布严格相等；但**采样路径不同**——同 seed 下 spec on / off 的具体输出可能不同。这不是质量退化，是数学上等价的不同采样路径。

加分：能说"$\min(1, p/q)$ 是 importance sampling 的标准接受率"；能解释为什么必须用 max(0, p - q) 而不能直接用 p（要保证拒绝点的修正分布抵消接受 path 的偏差）。

</details>

**Q2（trade-off）**：Spec Decoding / Medusa / EAGLE 三者的核心差异？为什么 EAGLE-3 能达到 3-5× 加速而经典 spec decoding 通常只有 1.5-3×？

<details>
<summary>Answer sketch</summary>

**三者核心差异**（必背表）：

| 维度 | Spec Decoding | Medusa | EAGLE-3 |
|---|---|---|---|
| draft 形式 | 独立小 LLM | target + K 个 head | feature-level 小 AR head |
| draft 在哪 | 完全独立 forward | target 1 次 forward 顺便算 | hidden state 上轻量 forward |
| 接受率 $\alpha$ | 60-75% | ~65% | **85%+** |
| 加速比 | 1.5-3× | 1.5-2.5× | **3-5×** |
| 训练成本 | 0（即插即用） | 中（fine-tune K head） | 低（small head + 7B 数据） |

**EAGLE-3 速度天花板更高的原因**（必须说到）：

1. **feature-level 信息密度高**：hidden state 是 continuous d 维向量（典型 4096-8192 维），比 token id（discrete 1 维）信息量大几个数量级；draft head 在 feature 上预测时能更精准捕捉 target 的下一步 hidden，自然接受率更高。
2. **draft 与 target 在同空间上跑**：EAGLE 的 draft head 输入 / 输出都是 target 自己的 hidden state 空间，**分布天然对齐**——经典 spec decoding 的小 LLM draft 是在不同 model 上训出来的，分布有偏。
3. **multi-layer feature 融合**（EAGLE-3 vs EAGLE-1）：EAGLE-3 用多层 hidden 而非只用最后一层，draft 信号更丰富。
4. **更激进的 tree expansion**：EAGLE-3 的 candidate tree 节点数比 spec decoding 的单链候选多得多，期望接受长度更长。

**接受率公式回顾**：$\bar{n} = (1 - \alpha^{K+1}) / (1 - \alpha)$；$\alpha$ 从 0.65 到 0.85 时 $\bar{n}$ 在 K=5 下从 2.6 升到 4.4——**接受率天花板决定加速比天花板**。

加分：能说 Medusa 的劣势是 head 之间独立预测、不能利用前面 head 的信息（vs EAGLE 是 AR），所以 Medusa 的接受率天花板低于 EAGLE。

</details>

**Q3（实战）**：你部署 LLaMA-3 70B 单 stream chat 服务（典型响应 500 token），列出完整的 spec decoding 配置，包括：draft model 选择、`num_speculative_tokens` 取值、何时不要开 spec、需要哪些监控指标。

<details>
<summary>Answer sketch</summary>

**配置选择**：

- **draft model**：Llama-3.2-1B-Instruct（同 family、同 tokenizer，必须）
  - 大小约 target 的 1/70，draft 1 次 forward latency 约 target 的 5-8%（满足 §2.4 公式中 $r$ 小的条件）
  - 备选：Llama-3.2-3B-Instruct（更准但更慢，draft latency 约 target 15-20%）
  - 不要用 Qwen / Mistral 当 draft——tokenizer 不同直接用不了
- **num_speculative_tokens (K)**：5（业界默认 / 论文最优）
  - 用 §2.4 公式估：$\alpha = 0.7, r = 0.05$ 时最优 K* ≈ 5；K = 7 也可，K > 10 不要试
- **vLLM 配置**：`speculative_model="meta-llama/Llama-3.2-1B-Instruct"`、`num_speculative_tokens=5`、`use_v2_block_manager=True`、`tensor_parallel_size=4`（H100 x4）
- **替代方案**：代码 / JSON 输出场景用 PLD（`speculative_model="[ngram]"`），零成本 1.5-2× 加速
- **温度建议**：保持 0.6-0.8，spec decoding 在中等温度下接受率反而更高

**何时不要开 spec**：

- ❌ 高 throughput 批处理服务（batch ≥ 8）——已 throughput-bound、spec 反而慢
- ❌ 短响应场景（< 50 token）——启动开销摊不开
- ❌ 跨 family 部署（target 与 draft tokenizer 不一致）
- ❌ 单卡显存紧张（draft 占额外 ~2GB；用 PLD 替代）

**必须监控的指标**：

1. **接受率 $\alpha$ / 平均接受 token 数 $\bar{n}$**：vLLM 暴露 `spec_decoding/draft_acceptance_rate` 指标；$\alpha < 0.5$ 说明 draft 选错了或场景不适合，关掉 spec
2. **end-to-end TBT（time between tokens）**：spec on 应比 spec off 快 2-3×；不到 1.5× 说明配置有问题
3. **输出质量 benchmark**：周期跑 MMLU / HumanEval（spec on 与 off），分数应该在噪声范围（< 0.5%）
4. **OOM / 异常率**：spec decoding 在 vLLM 老版本有 bug，监控异常 trace
5. **K 自适应**：高级配置可根据 $\alpha$ 动态调 K（接受率高升 K，低降 K）

**Reasoning model 特别说明**（加分）：R1 类 long-CoT model（响应 2000+ token）spec decoding 收益翻倍——节省 50-70% wall-clock，是 R1 部署的标配优化；甚至可以用 EAGLE-3 进一步榨到 4-5×。

</details>

---

## 7. 延伸阅读

- [Leviathan et al. 2023 — Fast Inference from Transformers via Speculative Decoding (arXiv)](https://arxiv.org/abs/2211.17192) — spec decoding 的奠基 paper，§3 必读
- [Chen et al. 2023 — Accelerating LLM Decoding with Speculative Sampling (arXiv)](https://arxiv.org/abs/2302.01318) — DeepMind 同期版本，对照 Leviathan 一起读
- [Cai et al. 2024 — Medusa (arXiv)](https://arxiv.org/abs/2401.10774) — Medusa-1 / Medusa-2 的提出，含 tree attention 设计
- [Li et al. 2024 — EAGLE / EAGLE-2 / EAGLE-3 (GitHub)](https://github.com/SafeAILab/EAGLE) — EAGLE 全系列代码 + 预训 head（Llama / Qwen / DeepSeek），现 SOTA 工业级方案
- [Saxena 2023 — Prompt Lookup Decoding (GitHub)](https://github.com/apoorvumang/prompt-lookup-decoding) — PLD 极简实现 < 200 行，代码 / JSON 场景免费加速
- [vLLM Spec Decoding 文档](https://docs.vllm.ai/en/latest/usage/spec_decode.html) — vLLM 官方配置指南，含已知限制清单
- [HuggingFace Assisted Generation Blog](https://huggingface.co/blog/assisted-generation) — HF 在 transformers 库中实现的 assisted generation（即 spec decoding），可以直接 `model.generate(..., assistant_model=draft)`
- [Andrej Karpathy 关于 spec decoding 的推文与解释](https://twitter.com/karpathy/status/1697318534555336961) — 一段直观解释，适合给非技术同事讲清楚 spec decoding 的物理直觉
- 推荐继续读本教程的 **11.4 节《量化：GPTQ / AWQ / GGUF / FP8 / SmoothQuant》** —— 量化是另一条单 stream 加速路线（降低 weight load 大小），可与 spec decoding 叠加使用（spec + INT8 = 5-8× 加速）
- 推荐继续读本教程的 **10.3 节《RLVR 与 DeepSeek-R1》** —— R1 类 reasoning model 输出 long-CoT、最受益于 spec decoding，理解 R1 的 inference 模式有助于设计 spec 部署方案
