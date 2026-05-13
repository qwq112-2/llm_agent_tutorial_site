---
title: "8.4 Adapter / Prefix-tuning / P-tuning v2"
description: "LoRA 之前与之外的 PEFT 谱系——Adapter 在 block 内插小 MLP、Prefix-tuning 在 KV cache 前拼可学习向量、P-tuning v2 把 prompt 推到所有 layer、IA³ 给中间值乘 scaling、GaLore 把 gradient 投影到低秩——它们各自踩中了 LoRA 没踩中的工程权衡，搞清楚它们才能理解\"为什么 2026 年的事实标准"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：8.3 LoRA / QLoRA / DoRA

## 一句话本节讲什么

LoRA 之前与之外的 PEFT 谱系——Adapter 在 block 内插小 MLP、Prefix-tuning 在 KV cache 前拼可学习向量、P-tuning v2 把 prompt 推到所有 layer、IA³ 给中间值乘 scaling、GaLore 把 gradient 投影到低秩——它们各自踩中了 LoRA 没踩中的工程权衡，搞清楚它们才能理解"**为什么 2026 年的事实标准是 LoRA**"。

---

## 1. Mental model（直觉）

8.3 把 LoRA 讲透了，但 LoRA 并不是 PEFT 的起点，更不是唯一答案。把 PEFT 家族放进一张谱系图：

```
                  PEFT（Parameter-Efficient Fine-Tuning）
                                │
        ┌───────────────────────┼─────────────────────────┐
        │                       │                         │
   ① 额外模块插入            ② prompt 类                ③ 重参数化
   (Additive)            (Prompt-based)             (Reparameterization)
        │                       │                         │
   Adapter (Houlsby)       Prompt-tuning              LoRA (8.3)
   Adapter (Pfeiffer)      Prefix-tuning              QLoRA (8.3)
   AdapterFusion           P-tuning v1 / v2           DoRA (8.3)
                                                      VeRA / rsLoRA
        │                       │                         │
        └─── ④ scaling-only ────┘──── ⑤ 梯度低秩 ────────┘
                  IA³                    GaLore
```

五条流派各自的核心 idea 一句话：

1. **额外模块（Adapter）**：在 Transformer block 内塞个小 MLP（bottleneck），base 冻死只训这个小 MLP。优点稳、缺点**推理时多一次 forward**
2. **prompt 类（Prefix / Prompt / P-tuning）**：base 全冻，只在 input（或每层 attention 的 K/V）前拼一段可学习向量，让模型"自己读到提示后再输出"
3. **重参数化（LoRA 家族）**：把 $\Delta W$ 假设成低秩 $BA$，**推理时可 merge 回 W 零开销**——这是 LoRA 在 LLM 时代胜出的根本原因
4. **scaling-only（IA³）**：连小矩阵都不训，只训"乘在中间值上的 scaling 向量"，参数极少
5. **梯度低秩（GaLore）**：训练参数仍是 100%（不是 PEFT），但**梯度**被投影到低秩子空间，optimizer state 显存断崖式下降——是给"想 full FT 但显存不够"准备的

时间线上：**Adapter (2019) → Prefix-tuning (2021) → Prompt-tuning (2021) → P-tuning v1/v2 (2021/2022) → LoRA (2021) → IA³ (2022) → DoRA (2024) → GaLore (2024)**。LoRA 不是最早的，但因为"推理 zero-overhead + 训练稳定 + 生态完善"，2023 年起就成了事实标准；其它流派则在特定场景（多任务、超低参数、full-FT 替代）有自己的位置。

本节聚焦"LoRA 之外"——把 Adapter / Prefix / P-tuning v2 / IA³ / GaLore 各讲透一节最少必要内容，最后用一张对比表回答"**为什么 LoRA 赢了**"这个面试高频问题。

---

## 2. 公式与原理

### 2.1 Adapter（Houlsby 2019）：bottleneck MLP

[Houlsby 2019] 是 PEFT 的开山之作。思路朴素：在每个 Transformer block 内插一个小 MLP，base 全冻只训这个小 MLP。形状用 **down → up bottleneck**：

$$
\text{Adapter}(h) = h + W_\text{up} \cdot \sigma(W_\text{down} \cdot h)
$$

其中 $h \in \mathbb{R}^d$ 是 block 中间隐藏状态、$W_\text{down} \in \mathbb{R}^{r \times d}$、$W_\text{up} \in \mathbb{R}^{d \times r}$、$r \ll d$ 是 bottleneck rank、$\sigma$ 是非线性激活（GELU / ReLU）。**残差**让 Adapter 起步等价于恒等变换（$W_\text{up}$ 通常零初始化），不会破坏 base 表征。

Houlsby 原版**每个 block 插两次**——attention 后一次、FFN 后一次。Pfeiffer (2020) 改进为**每个 block 只插一次**（attention 后），参数减半、效果几乎不变，是后来 AdapterHub 默认配置。

参数量（一层）：$2 \cdot r \cdot d$（Pfeiffer）或 $4 \cdot r \cdot d$（Houlsby），全模型典型占 base 的 **1-3%**——比 LoRA（0.1-1%）多一个数量级，但绝对值仍很小。

**Adapter 的致命缺点：推理时增加 latency**。LoRA 训完可以 merge 回 base.weight，部署时一行 nn.Linear 跑完；Adapter 是显式的额外模块，**每个 block 都多一次 down-up forward**，无法 merge。实测在 LLaMA-7B 上 Adapter 推理延迟比 base 高 5-10%，长 context 场景更明显。这是 LoRA 后来胜出的关键工程理由。

AdapterHub（adapterhub.ml）是 NLP 社区开源 adapter 仓库，现在仍有 BERT/RoBERTa 时代的多任务 adapter 库；LLM 时代基本被 LoRA 取代，但**多任务 / 模块化组合**的研究还在用（AdapterFusion / Mixture-of-Adapters）。

### 2.2 Prefix-tuning（Li & Liang 2021）：可训练的 KV cache 前缀

Prefix-tuning 完全换了一个套路：**base 一切都不动，只在每一层 attention 的 K、V 前拼一段可训练向量**。直觉上像是"给模型读到一段隐藏的提示"，但这个提示不是真 token、没有对应的 input embedding，而是直接以 KV cache 的形式注入：

$$
K_\text{eff} = [P_K; K_\text{base}], \quad V_\text{eff} = [P_V; V_\text{base}]
$$

其中 $P_K, P_V \in \mathbb{R}^{l_\text{prefix} \times d_\text{model}}$ 是每层独立的可训练向量、$l_\text{prefix}$ 是 prefix 长度（典型 10-200）。attention 计算时：

$$
\text{Attn}(Q, K_\text{eff}, V_\text{eff}) = \text{softmax}\left(\frac{Q K_\text{eff}^\top}{\sqrt{d_k}}\right) V_\text{eff}
$$

参数量：每层 $2 \cdot l_\text{prefix} \cdot d_\text{model}$，$L$ 层共 $2 \cdot L \cdot l_\text{prefix} \cdot d_\text{model}$。LLaMA-7B（$L=32, d=4096$）+ $l_\text{prefix}=20$：约 **5M 参数**（占比 0.07%）。

**为什么不直接训 prompt 的 input embedding？** 因为 input embedding 经过 32 层 attention 后影响会被稀释，控制力很弱。Prefix-tuning 在每层都注入，相当于"每层都贴一遍提示"，容量大得多。

**致命缺点：训练不稳定**。直接训 $P_K, P_V$ 经常炸，Li & Liang 原 paper 给出的 fix 是**reparametrization**：训一个小 MLP $f_\theta$，用一个低维 $P_\theta \in \mathbb{R}^{l_\text{prefix} \times d'}$ 经 MLP 得到 $P_K, P_V$：

$$
[P_K; P_V] = f_\theta(P_\theta)
$$

训完丢掉 MLP 只留 MLP 输出的 $P_K, P_V$ 部署。这个"训练时多一层 reparametrize、推理时丢掉"的 trick 是 Prefix-tuning 能 work 的关键，新人不知道这个 trick 直接训会发现 loss 一直炸。

历史地位：Prefix-tuning 是早期（2021-2022）GLM、T5 时代的主力 PEFT 方案，2023 后被 LoRA 全面取代。今天还会用到的场景：**研究 KV cache 注入机制**、需要每层独立提示的实验。

### 2.3 Prompt-tuning（Lester 2021）：极简版 Prefix

Prompt-tuning 是 Prefix-tuning 的极简版：**只在 input embedding 前拼可学习向量**，不动每层 KV：

$$
X_\text{eff} = [P; X_\text{embed}], \quad P \in \mathbb{R}^{l_\text{prompt} \times d_\text{model}}
$$

参数量：$l_\text{prompt} \cdot d_\text{model}$，比 Prefix 少 $L$ 倍。LLaMA-7B + $l_\text{prompt}=20$：约 0.08M 参数（占比 0.001%）——**PEFT 里最少的**。

但实证上有个魔咒：**只在 model size > 10B 时才能匹配 full fine-tune**。Lester 2021 的核心实验图：T5-Small (60M) / T5-Base (220M) 上 prompt-tuning 比 full FT 差 10-20 个点，到 T5-XXL (11B) 上才追平。原因直觉上是——只在 input 注入提示，要让信号不被 32+ 层 attention 稀释掉，需要 model 本身有足够强的"long-range 信号传递能力"，这只在大模型上才成立。

这就是 Prompt-tuning 在 LLM 时代几乎被弃用的根本原因：**小模型上不 work、大模型上又被 LoRA 完全 dominate**。今天提它主要是历史价值与"the power of scale"这个实证发现的命名意义。

### 2.4 P-tuning v1 / v2（Liu 2021/2022）

**P-tuning v1** 是 Prompt-tuning 的"重参数化"变体——跟 Prefix-tuning 用 MLP 的思路一样，但 P-tuning v1 用 **LSTM/MLP 编码器** 生成 prompt embedding：

$$
P = \text{LSTM}(P_\text{raw}), \quad P_\text{raw} \in \mathbb{R}^{l \times d'}
$$

主要解决了 Prompt-tuning 在中等 size 模型（100M-1B）上不稳的问题。但**架构有点 over-engineered**，LLM 时代基本不用。

**P-tuning v2** 是更重要的版本，把 Prompt-tuning 推回 Prefix-tuning 的方向——**在所有 layer 都加 prompt**（不只 input）。所以从公式上看 P-tuning v2 ≈ Prefix-tuning，差别更多在工程细节：

- P-tuning v2 用更通用的实现（兼容 BERT 的 encoder 与 GPT 的 decoder）
- 用更简单的训练流程（不强制需要 reparametrization）
- 在 NLU 任务（SuperGLUE 等分类 / 抽取）上能匹配 full fine-tune
- 在 NLG 任务（生成）上仍有差距

历史地位：P-tuning v2 是 BERT/RoBERTa 时代的强 PEFT 方案，今天在**纯 NLU 任务**（如 IE、分类、reranker fine-tune）仍有用武之地；LLM 生成任务全面被 LoRA 取代。HuggingFace PEFT 的 `PrefixTuningConfig` / `PromptEncoderConfig` 都还在维护。

### 2.5 IA³（Liu 2022）：scaling-only

IA³（Infused Adapter by Inhibiting and Amplifying Inner Activations）的核心 idea：**连小矩阵都不训，只训"乘在中间值上的 scaling 向量"**。给三个位置各配一个 learnable scaling vector：

$$
\begin{aligned}
K' &= K \odot l_K, & l_K \in \mathbb{R}^{d_k} \\
V' &= V \odot l_V, & l_V \in \mathbb{R}^{d_v} \\
\text{FFN}(x) &= W_2 \cdot \big( \sigma(W_1 x) \odot l_\text{ff} \big), & l_\text{ff} \in \mathbb{R}^{d_\text{ff}}
\end{aligned}
$$

其中 $\odot$ 是逐元素乘、$l_K, l_V$ 缩放每个 head 的 key/value、$l_\text{ff}$ 缩放 FFN 中间激活。每层只多 $d_k + d_v + d_\text{ff} \approx 3d$ 个参数——比 LoRA 还少 1-2 个数量级。

参数量：LLaMA-7B 全模型只有 **~50K 个可训练参数**（占比 0.0007%）。然而效果实测能接近 LoRA r=4，是 PEFT 谱系里"性价比之最"。

工程优势：scaling 向量推理时可以**直接吸收进 K / V / FFN 的权重矩阵**（与 LoRA merge 同理），**zero overhead**。HuggingFace PEFT 通过 `IA3Config` 支持。

适用场景：**多任务大规模 routing**（一个 base + 几百个 task-specific IA³ vector，每个 task 几十 KB）、**超低预算 PEFT 实验**。LLM 主流不用 IA³ 主要是因为 LoRA 的"r=16 表达力上限"在大多 SFT/RLHF 场景下都用得到，IA³ 的容量略显不足。

### 2.6 GaLore（Zhao 2024）：full-param 训练的省显存方案

GaLore 与上面所有方法**根本路线不同**——它不是 PEFT，而是 **full-parameter fine-tune 的显存优化**。

观察：训练时显存大头是 **optimizer state**（AdamW 一阶 + 二阶矩共 8 字节/参数），梯度本身只是瞬时量。GaLore 把每个权重的**梯度**投影到一个低秩子空间：

$$
G_\text{low} = P^\top G, \quad P \in \mathbb{R}^{d \times r}, \; G \in \mathbb{R}^{d \times d_\text{out}}
$$

optimizer 在低秩子空间维护 $m, v$（显存从 $O(d \cdot d_\text{out})$ 降到 $O(r \cdot d_\text{out})$），更新时再投影回去：

$$
W \leftarrow W - \eta \cdot P \cdot \text{AdamW}(G_\text{low})
$$

每隔几百步重新计算 $P$（用 SVD 提取梯度的主子空间），保证投影方向跟得上训练动力学。

关键差异（与 LoRA 对比）：

- **trainable params 仍是 100%**——所有权重都在更新，不是冻 base
- **省的是 optimizer state 显存**，权重 + 梯度本身仍是全量
- 训练动力学更接近 full FT，理论上效果上限比 LoRA 高
- 适合**预训练 + 长 fine-tune** 场景，PEFT 不能完全替代 full FT 的任务

显存账本（7B）：full FT 需要 ~120 GB（权重 + 梯度 + AdamW），GaLore 降到 ~30 GB；LoRA r=16 ~17 GB。GaLore 介于两者之间，但**训练效果更接近 full FT**。

工程库：`galore-torch`（pip 安装），与 HF Trainer 集成只需替换 optimizer。2024-2025 年新兴方案，作为"显存不够 full FT、但 LoRA 又怕掉点"时的 middle ground。

### 2.7 方案对比表（必背）

| 方案 | 参数量 | 推理 overhead | 训练稳定性 | LLM 时代地位 |
|---|---|---|---|---|
| LoRA / QLoRA | 0.1-1% | **0**（可 merge） | 稳 | **事实标准** |
| DoRA | 0.1-1% | 0（可 merge） | 稳 | 新兴主流 |
| Adapter (Houlsby/Pfeiffer) | 1-3% | 有（每 block 一次） | 稳 | 历史 / 多任务研究 |
| Prefix-tuning | < 0.1% | KV cache 占用 + dequant | 不稳，需 reparam | 历史 |
| Prompt-tuning | < 0.01% | 小（拼 input） | 不稳（< 10B 不 work） | 几乎弃用 |
| P-tuning v2 | < 0.5% | KV cache 占用 | 中 | NLU 仍用 |
| IA³ | < 0.01% | 0（可 merge） | 稳 | 极轻量场景 |
| GaLore | 100% trainable | 0 | 稳 | full-FT 替代 |

### 2.8 为什么 LoRA 赢了

把上表读懂就知道答案，但面试时要能一口气说清楚：

1. **推理时可 merge 回 base**——zero overhead，部署完全等价于普通 model；Adapter 做不到（额外 forward）、Prefix-tuning 也做不到（KV cache 永远占着 prefix 长度）
2. **训练稳定**——A-Kaiming + B-zeros + $\alpha/r$ scaling 一套下来基本不会炸；Prefix-tuning 必须 reparametrize 才能训
3. **效果好**——大 model 下接近 full FT；Prompt-tuning 在 < 10B model 上根本训不出
4. **生态完善**——HF PEFT、bitsandbytes、TRL、unsloth、DeepSpeed、Axolotl、vLLM multi-LoRA serving 全栈支持
5. **多任务可叠加**——vLLM multi-LoRA / S-LoRA 支持几百个 LoRA adapter 共享 base，工业 multi-tenant 标配

Adapter / Prefix / Prompt 各自踩中了上面 5 条里的一两条**反向**：Adapter 推理慢、Prefix 训练不稳、Prompt 小模型上不 work。LoRA 把这 5 条**全占了**——这就是它在 2023 年完成"对 PEFT 谱系的统一"的根本原因。

---

## 3. 最小代码示例

### 3.1 手撕 Houlsby Adapter（≤ 25 行）

```python
import torch
import torch.nn as nn

class HoulsbyAdapter(nn.Module):
    """插在 attention/FFN 后的 bottleneck MLP，base 冻结、只训这个 module。"""

    def __init__(self, d_model: int, r: int = 64, act: str = "gelu"):
        super().__init__()
        self.down = nn.Linear(d_model, r, bias=True)         # d -> r
        self.up = nn.Linear(r, d_model, bias=True)           # r -> d
        self.act = nn.GELU() if act == "gelu" else nn.ReLU()
        # 关键：up 用零初始化，让 Adapter 起步等价于恒等映射，不破坏 base 表征
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h 是 block 内某个隐藏状态 (B, T, d)
        return h + self.up(self.act(self.down(h)))           # 残差连接

# 用法：包在 Transformer block 的 attention/FFN 后
# block_out = self.attn(x); block_out = adapter(block_out)
# 训练时把 base 全部 freeze、只让 adapter.parameters() 走 optimizer
```

关键点：

- `up` 零初始化让 $\Delta = 0$ 起步，与 LoRA 的 B=0 同一个 trick
- `down` 用 PyTorch 默认 Kaiming uniform 即可
- 残差连接是必须的——没残差的话 Adapter 起步等价于把 $h$ 替换为 0，base 直接废
- 推理时 Adapter **无法 merge**，因为非线性 $\sigma$ 隔在中间——这是 Adapter 的根本缺点

### 3.2 手撕简化 Prefix-tuning（≤ 30 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrefixAttention(nn.Module):
    """演示 prefix KV 如何在 attention 时拼进去；base 的 q/k/v proj 已 freeze。"""

    def __init__(self, n_layer: int, n_head: int, d_head: int,
                 prefix_len: int = 20):
        super().__init__()
        # 每层一组 (P_K, P_V)，shape (n_layer, n_head, prefix_len, d_head)
        self.prefix_K = nn.Parameter(torch.zeros(n_layer, n_head, prefix_len, d_head))
        self.prefix_V = nn.Parameter(torch.zeros(n_layer, n_head, prefix_len, d_head))
        nn.init.normal_(self.prefix_K, std=0.02)
        nn.init.normal_(self.prefix_V, std=0.02)
        # 实务上还要叠一个 reparametrize MLP：P_K, P_V = MLP(P_raw)，训完丢掉

    def forward(self, q, k, v, layer_idx: int):
        # q, k, v: (B, n_head, T, d_head)
        B, H, T, D = q.shape
        # 取出当前层的 prefix，扩 batch
        pK = self.prefix_K[layer_idx].unsqueeze(0).expand(B, -1, -1, -1)  # (B,H,L,D)
        pV = self.prefix_V[layer_idx].unsqueeze(0).expand(B, -1, -1, -1)
        # 拼接到 K, V 前面
        k_eff = torch.cat([pK, k], dim=2)    # (B, H, L+T, D)
        v_eff = torch.cat([pV, v], dim=2)
        # 注意：Q 不动，attention mask 要扩展（prefix 部分对所有 query 可见）
        return F.scaled_dot_product_attention(q, k_eff, v_eff, is_causal=False)
```

关键点：

- prefix 是**每层独立**的可学习张量；shape `(n_layer, n_head, L, d_head)`
- 真正训练时强烈建议加 reparametrize MLP（注释里的 `P_raw → MLP → P_K,P_V`），否则 loss 容易炸
- attention mask 要重新构造：prefix 对所有真 token 可见，但 prefix 之间通常不需要 causal mask
- 推理时 prefix 永远占着 KV cache 的前 $l_\text{prefix}$ 槽位，**不是 zero overhead**

### 3.3 HuggingFace PEFT 多种方法切换（≤ 25 行）

```python
from peft import (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig,
                  PromptEncoderConfig, IA3Config, get_peft_model, TaskType)
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                            torch_dtype="bfloat16", device_map="auto")

# 切换 PEFT 方法只需替换 config，其余训练流程完全一致
configs = {
    "lora":   LoraConfig(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM,
                         target_modules=["q_proj", "v_proj"]),
    "prefix": PrefixTuningConfig(num_virtual_tokens=20, task_type=TaskType.CAUSAL_LM,
                                 prefix_projection=True),    # 启用 reparam MLP
    "ptuning_v2": PromptEncoderConfig(num_virtual_tokens=20,
                                      task_type=TaskType.CAUSAL_LM,
                                      encoder_hidden_size=128),
    "ia3":    IA3Config(target_modules=["k_proj", "v_proj", "down_proj"],
                        feedforward_modules=["down_proj"],
                        task_type=TaskType.CAUSAL_LM),
}
model = get_peft_model(base, configs["lora"])     # 改 key 即可换方法
model.print_trainable_parameters()
# LoRA r=16: ~4M / 8B = 0.05%  |  Prefix 20: ~5M  |  IA3: ~50K
```

关键点：

- 不同 PEFT method 的 config 字段差异很大（`r` / `num_virtual_tokens` / `target_modules` / `feedforward_modules`），**混合使用前要对照官方文档查字段**
- `PrefixTuningConfig(prefix_projection=True)` 启用 Li & Liang 论文里的 reparam MLP
- `IA3Config` 的 `feedforward_modules` 要单独指定（IA³ 在 FFN 中间值上的 scaling 与 K/V 的 scaling 实现不同）
- HF PEFT 还支持 `AdaLoraConfig`（自适应 rank）、`OFTConfig`（正交微调）、`BOFTConfig` 等更新方法

### 3.4 GaLore 用法（≤ 15 行）

```python
from galore_torch import GaLoreAdamW
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                             torch_dtype="bfloat16")
# 注意：所有参数都 trainable（不冻 base）
galore_params = [p for n, p in model.named_parameters() if "self_attn" in n or "mlp" in n]
optimizer = GaLoreAdamW(galore_params, lr=1e-5,
                        rank=128,                # 梯度低秩子空间维度
                        update_proj_gap=200,     # 每 200 步重计算投影矩阵
                        scale=0.25)              # 投影后梯度的 scaling
# 直接喂给 Trainer 即可
trainer = Trainer(model=model, optimizers=(optimizer, None), ...)
```

关键点：

- GaLore **不冻任何参数**，是 full-FT 的显存优化
- `rank=128 ~ 256` 是常用值；rank 越大越接近 full-FT 但显存增多
- `update_proj_gap` 是重计算投影矩阵的间隔，每次重算要做一次 SVD（开销不小，所以不能太频繁）
- 与 LoRA 是互补关系：LoRA 走 PEFT 路线、GaLore 走 full-FT 显存优化路线

---

## 4. 工程踩坑与经验

- ❗ **Adapter 推理 overhead 实测 5-10%**——每个 block 都多一次 down-up forward，无法 merge 进 base.weight。对延迟敏感场景（在线 chat、低 TBT 要求）慎用 Adapter；离线 batch 推理影响小。LoRA 的"merge 后 zero overhead"是工程上的杀招
- ❗ **Prefix-tuning 直接训会炸 loss，必须 reparametrize**——HuggingFace PEFT 的 `PrefixTuningConfig(prefix_projection=True)` 默认开启 MLP 重参数化，Li & Liang 论文 §3.2 也强调这一点。新人不知道这个 trick 直接训会发现 loss 在前几百步直接 NaN 或震荡
- ❗ **Prompt-tuning 在 < 10B 模型上几乎训不出效果**——这是 Lester 2021 paper 的核心实证，T5-Small 上 prompt-tuning 比 full FT 差 15-20 个点，到 11B 才追平。今天在 7B 模型上做 prompt-tuning 是错误选型，直接上 LoRA
- ❗ **HF PEFT 不同 method 的 config 字段差异很大**——`LoraConfig` 用 `r`、`PrefixTuningConfig` 用 `num_virtual_tokens`、`IA3Config` 又有 `feedforward_modules`，混用时容易拼错字段名。批量切方法做对比实验时建议封装成一个 dict（如 §3.3 代码），便于 ablation
- ❗ **Adapter / Prefix-tuning 的 ckpt 大小通常比 LoRA 大 5-10 倍**——Adapter 占 base 的 1-3%、Prefix 占 ~0.07% × 32 层 = ~2-5%，LoRA 默认 ~0.05-0.3%。在 multi-task / multi-tenant 场景（每个 task 一个 adapter），LoRA 的存储成本优势很显著
- ❗ **GaLore 不是 PEFT，思路与 LoRA 完全相反**——GaLore 训练全部参数、只压缩 optimizer state；LoRA 只训少量参数。面试时被问"你了解 GaLore 吗？"千万别说"是 LoRA 的改进"，会被纠错。两者是**互补**关系：显存极紧 → LoRA / QLoRA；显存够但想接近 full FT → GaLore
- ❗ **Multi-LoRA serving 在 vLLM 中支持很好（`enable_lora=True`），但 multi-Adapter / multi-Prefix 不支持**——vLLM 的 LoRA serving 利用了 LoRA 的"低秩矩阵可以合并 GEMM"特性，Adapter 的非线性、Prefix 的 KV cache 注入都打破了这个假设。工业 multi-tenant 部署是 LoRA 比其它 PEFT 多一个数量级生态优势的领域
- ❗ **Prefix / P-tuning v2 推理时 prefix 永远占 KV cache 槽位**——`prefix_len=20` 意味着每个请求的 KV cache 都比真实序列长 20 个 token，长 context 场景会放大显存占用。max_seq_len=4096 时影响小，但 max_seq_len=128k 时几乎可忽略

---

## 5. 经典 paper

- **Houlsby et al., 2019 — *Parameter-Efficient Transfer Learning for NLP*** — Adapter 起点，提出在 Transformer block 内插 bottleneck MLP 的 PEFT 范式。读它能搞清 PEFT 这条路线的最初动机（避免 BERT 时代每个 task 都要 fine-tune 一份完整模型），§3 的 bottleneck 设计与零初始化是后来所有 additive PEFT 的原型
- **Li & Liang, 2021 — *Prefix-Tuning: Optimizing Continuous Prompts for Generation*** — Prefix-tuning 原典，提出"在每层 KV 前拼可训练向量"的范式 + reparametrize MLP 的训练 trick。读它能理解"prompt 类 PEFT 为什么要每层注入"以及"为什么训练这么不稳"——§4.1 的 ablation 直接对比了"只 input prompt vs 每层 prefix"，差距巨大
- **Liu et al., 2022 — *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning*** — IA³ 提出的论文，主张 PEFT 在 few-shot 场景比 in-context learning 更有效。读它能搞清 IA³ 的"scaling-only"哲学——比 LoRA 还激进，把 PEFT 推到信息论意义上的"几乎不训任何参数"
- 选读：**Lester et al., 2021 — *The Power of Scale for Parameter-Efficient Prompt Tuning*** — Prompt-tuning 提出的论文，最大价值不在方法本身（已被弃用），而在 §4 的 scaling 实验——证明 PEFT 效果与 model size 强相关，"the power of scale" 这个洞见后来被 LoRA / DoRA 反复印证
- 选读：**Liu et al., 2022 — *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning*** — P-tuning v2 论文，主要价值是把 Prompt-tuning 推回 Prefix-tuning 的方向（每层加 prompt）并给出 NLU 任务的强基线
- 选读：**Zhao et al., 2024 — *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection*** — GaLore 原典，提出梯度低秩投影替代 LoRA 的"权重低秩"假设。读它能理解 PEFT 与 full-FT 之间还有第三条路

---

## 6. 自测与面试题

**Q1（对比）：** Adapter / Prefix-tuning / LoRA 三者的核心差异是什么？为什么 LoRA 在 LLM 时代成为事实标准？

<details>
<summary>Answer sketch</summary>

核心差异：

- **Adapter**：在 block 内插 bottleneck MLP（$h + W_\text{up}\sigma(W_\text{down} h)$），base 冻结、只训 MLP。属于 **additive** 流派，参数 1-3%
- **Prefix-tuning**：每层 attention 的 K、V 前拼可训练向量 $P_K, P_V$，base 全冻。属于 **prompt-based** 流派，参数 < 0.1%
- **LoRA**：把 $\Delta W$ 假设为低秩 $BA$，旁路加在原 linear 上。属于 **reparameterization** 流派，参数 0.1-1%

LoRA 胜出的 5 条理由（一定要全说出来）：

1. **推理时可 merge 回 base.weight**——zero overhead；Adapter 因为非线性隔在 down/up 之间无法 merge，每次推理多一次 forward；Prefix-tuning 永远占 KV cache 槽位
2. **训练稳定**——A-Kaiming + B-zeros + $\alpha/r$ scaling，几乎不需要调；Prefix-tuning 必须叠 reparametrize MLP 才不炸
3. **效果接近 full FT**——尤其 r=16/32 全加 7 个 linear 时；Prompt-tuning 在 < 10B 上根本训不出
4. **生态**——HF PEFT、bitsandbytes、TRL、unsloth、Axolotl、vLLM multi-LoRA serving 全栈
5. **多任务可叠加**——vLLM `enable_lora` 支持几百个 adapter 共享 base；Adapter / Prefix 都做不到 batched 多租户

加分：提一句 LoRA 也不是没缺点（容量上限低于 full FT，复杂任务上 DoRA / GaLore 表现更好）；指出 Adapter 在多任务 NLP（AdapterHub / AdapterFusion 研究）仍有用武之地。

</details>

**Q2（实现）：** 写出 Houlsby Adapter 的核心 forward 公式与 5 行 PyTorch 实现。

<details>
<summary>Answer sketch</summary>

公式：

$$
\text{Adapter}(h) = h + W_\text{up} \cdot \sigma(W_\text{down} \cdot h)
$$

其中 $W_\text{down} \in \mathbb{R}^{r \times d}$、$W_\text{up} \in \mathbb{R}^{d \times r}$、$r \ll d$、$\sigma$ 是 GELU/ReLU。

5 行 PyTorch（最精炼版本）：

```python
class Adapter(nn.Module):
    def __init__(self, d, r=64):
        super().__init__()
        self.down, self.up = nn.Linear(d, r), nn.Linear(r, d)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)  # 零初始化 up
    def forward(self, h):
        return h + self.up(F.gelu(self.down(h)))    # 残差 + bottleneck
```

关键点：

- `up` 必须零初始化——让 Adapter 起步等价于恒等映射，不破坏 base 表征
- 残差是必须的——没残差则起步把 h 替换成 0，base 直接废
- 推理时**无法 merge**（非线性隔在中间），这是 LoRA 后来胜出的关键
- 加分：提到 Pfeiffer adapter 把 Houlsby 的"每 block 插两次"减半为"只插 attention 后一次"，参数省一半效果几乎一样

</details>

**Q3（trade-off）：** 什么场景下你会选 Adapter / Prefix-tuning 而不是 LoRA？

<details>
<summary>Answer sketch</summary>

诚实答案：**绝大多数 LLM 场景都该选 LoRA**。少数几个场景值得用其它 PEFT：

**用 Adapter 的场景**：

- **多任务 NLP 研究**：AdapterHub / AdapterFusion 等组合 adapter 的研究路线还在用；NLU 任务的 task-specific adapter 仍有效
- **需要非线性表达**：LoRA 的低秩线性分解在某些任务上容量不足，Adapter 的 bottleneck MLP 有非线性，理论容量更大（实战很少触发）
- **历史遗留代码**：BERT/RoBERTa 时代留下的 codebase 改动成本高

**用 Prefix-tuning / P-tuning v2 的场景**：

- **NLU 任务（分类、抽取、reranker）**：P-tuning v2 在 SuperGLUE 等 NLU 上能匹配 full FT，是这类任务的合理 baseline
- **研究 KV cache 注入机制**：prefix 形式天然适合做 prompt encoding 类研究
- **极低参数预算**：prefix < 0.1% 比 LoRA 还少，但要承担训练不稳、推理 KV cache 占用的代价

**用 IA³ 的场景**：

- 超大规模 multi-tenant routing：每个 task 几十 KB 的 scaling vector，比 LoRA 还省存储
- 极端 few-shot 任务：参数极少反而不容易过拟合

**反过来——什么场景一定不该用 LoRA**：

- 需要 full-parameter 更新但显存不够 → **GaLore**
- 模型量化部署 → 直接 QLoRA（仍是 LoRA 家族）

加分：边界讨论——2026 年的现实是 **LoRA / QLoRA / DoRA 占 PEFT 实战 95%+**，其它方法主要是面试谈资 + 学术研究 + 历史 codebase 维护。

</details>

---

## 7. 延伸阅读

- [HuggingFace PEFT 文档](https://huggingface.co/docs/peft) — LoRA / Adapter / Prefix / P-tuning / IA³ / OFT 等全家桶官方实现，§3 代码的 source of truth
- [AdapterHub](https://adapterhub.ml/) — Adapter 家族的开源仓库，BERT/RoBERTa 时代多任务 adapter 的集大成者；查 PEFT 谱系的活历史
- [galore-torch GitHub](https://github.com/jiaweizzhao/GaLore) — GaLore 的官方实现 + HF Trainer 集成示例
- [Sebastian Raschka — Understanding Parameter-Efficient Finetuning](https://magazine.sebastianraschka.com/p/understanding-parameter-efficient) — PEFT 谱系的高质量综述长文，与本节互补
- [Lialin et al., 2023 — Scaling Down to Scale Up: A Guide to PEFT](https://arxiv.org/abs/2303.15647) — PEFT 综述论文，把谱系系统化梳理一遍，适合做完本节后查漏补缺
- 推荐继续读本教程的 **8.5 SFT 实战**——把 LoRA 配置端到端跑通；**Module 9 RLHF**——DPO / GRPO 几乎都基于 LoRA fine-tune；**11.2 vLLM**——multi-LoRA serving 的工程细节
