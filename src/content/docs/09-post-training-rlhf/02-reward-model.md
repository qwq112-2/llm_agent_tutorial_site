---
title: "9.2 Reward Model：Bradley-Terry + 训练实操"
description: "Reward Model（RM）是 RLHF 三段式（SFT → RM → PPO）的\"打分器\"——本节把 RM 的角色、Bradley-Terry pairwise loss 的推导、HF TRL 训练落地、Best-of-N 应用、以及 reward hacking 这条贯穿 9.x 的暗线讲清楚。"
---

> ⏱ 预计阅读 30 分钟 ｜ 难度 ★★ 🔥 ｜ 前置：1.4 损失函数

## 一句话本节讲什么

Reward Model（RM）是 RLHF 三段式（SFT → RM → PPO）的"打分器"——本节把 RM 的角色、Bradley-Terry pairwise loss 的推导、HF TRL 训练落地、Best-of-N 应用、以及 reward hacking 这条贯穿 9.x 的暗线讲清楚。

---

## 1. Mental model（直觉）

PPO / GRPO / Best-of-N 这些 RL / sampling 方法都需要一个东西：给一段 `(prompt, response)` 打一个**标量分数**，越高越好。这个"打分器"就是 Reward Model。

为什么不能直接用人？人工打分 1 万条 prompt 的实时打分成本是几千块美金、延迟以小时计——RL 训练要几百万次采样，根本不可能在线问人。所以 RLHF 的核心 trick 是：

> 先一次性收集人工标注的偏好数据 →  训一个 RM 当"人类口味的代理（proxy）" → 后面 PPO / GRPO 阶段全部用 RM 打分。

人工标注的形式不是"给这条 response 打 8 分"（绝对分数主观、跨标注员对不齐），而是"A 和 B 哪个更好"——这种 pairwise 比较的 inter-annotator agreement 显著高于 pointwise 评分。所以训练数据长这样：

```
{
  "prompt":   "讲个程序员笑话",
  "chosen":   "为什么程序员喜欢黑暗？因为他们怕 light themes。",
  "rejected": "我不知道。"
}
```

RM 要学的事是：把 `(prompt, chosen)` 的标量输出 $r_w$ 推得比 `(prompt, rejected)` 的 $r_l$ 高。**Bradley-Terry 模型**给出这件事的概率化形式：A 比 B 好的概率正比于 $e^{r_A} / (e^{r_A} + e^{r_B})$，等价于 $\sigma(r_A - r_B)$。负对数似然就是 RM 的训练 loss。

架构上 RM 不是另起炉灶——**RM = SFT 模型 + 一个 reward head**。把 SFT 模型最后一层 hidden state 的 last token（或 EOS token）那一维拿出来，过一个 `nn.Linear(d_model, 1)`，就得到标量 reward。base 的 LM head 在 RM 训练时通常被丢掉（不再做 next-token prediction），全部参数（或 LoRA adapter）+ reward head 一起 trainable。

一张图串起来：

```
              人类标注                       RM 训练                   下游使用
   prompt ─┬─→ chosen   ─┐                ┌──────────┐           ┌────────────────┐
           │             ├─ 偏好对 →      │  RM(SFT  │ → r_w ─┐  │ PPO / GRPO     │
           └─→ rejected ─┘                │  + head) │ → r_l ─┘  │ Best-of-N      │
                                          └──────────┘           │ Rejection Smpl │
                                          loss = -log σ(r_w-r_l) │ DPO 训练数据   │
                                                                 └────────────────┘
```

记一句话：**RM 是把"偏好数据"压缩成"可微分的奖励函数"的工具**。它的天花板是人类一致性（约 75-80%），它的死穴是 reward hacking——后面会反复回到这两点。

---

## 2. 公式与原理

### 2.1 Bradley-Terry 模型

Bradley-Terry（BT, 1952）是体育排名 / 偏好建模的经典工具。给每个候选 $i$ 赋一个 "实力" 标量 $r_i \in \mathbb{R}$，则 $i$ 战胜 $j$ 的概率定义为

$$P(i \succ j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \frac{1}{1 + e^{-(r_i - r_j)}} = \sigma(r_i - r_j)$$

其中 $\sigma(x) = 1 / (1 + e^{-x})$ 是 sigmoid。注意 $r$ 是 log-odds 尺度，绝对数值无意义、只有差值有意义（$r_i \to r_i + c$ 不改变任何 pair 的概率）。

把 $r$ 换成神经网络 $r_\theta(x, y) \in \mathbb{R}$（$x$ 是 prompt，$y$ 是 response），就得到 RM 的概率形式：

$$P(y_w \succ y_l \mid x) = \sigma\bigl(r_\theta(x, y_w) - r_\theta(x, y_l)\bigr)$$

设训练集 $\mathcal{D} = \{(x, y_w, y_l)\}$（$y_w$ = winning / chosen，$y_l$ = losing / rejected）。最大似然估计 = 最小化负对数似然：

$$\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Bigl[\log \sigma\bigl(r_\theta(x, y_w) - r_\theta(x, y_l)\bigr)\Bigr]$$

这就是 InstructGPT [Ouyang 2022] §3.5 / Christiano 2017 §2.2 给出的 RM loss——pairwise 二分类问题，"类"就是"chosen 还是 rejected"。

**为什么用 sigmoid 不用别的？** 因为 BT 模型本质是 logistic regression 在偏好空间的应用：你想用 $r_w - r_l$ 这一个标量预测一个二分类概率，sigmoid 是把 $\mathbb{R} \to (0, 1)$ 的标准选择，且对应的 NLL 等价于 logistic loss——梯度形式漂亮（详见 §2.3）、与 1.4 节的 CE 同根。

### 2.2 RM 架构

设 SFT 模型隐藏维度 $d$、词表大小 $V$。**典型 RM 架构**：

```
input_ids (B, T)
     ↓ embedding + transformer layers
hidden states  (B, T, d)
     ↓ 取每条样本的 last non-pad token: h_last = hidden[:, last_idx, :]   shape (B, d)
     ↓ reward_head: nn.Linear(d, 1)
reward         (B, 1)
```

几个工程要点：

1. **取哪个 token 的 hidden state？** 主流实现取**最后一个非 pad token**（通常是 EOS 或 response 最后一个 token），因为 decoder-only 的 causal attention 让最后一个 token 能 attend 到前面所有内容，信息最完整。HF `AutoModelForSequenceClassification` 默认就这么干。
2. **是否丢掉 LM head？** 通常丢——RM 不再做 next-token prediction，原 `lm_head` 占的内存浪费。HF `AutoModelForSequenceClassification` 自动替换 head。
3. **reward_head 的初始化**：常用 Xavier uniform 或直接复制 SFT 模型 lm_head 的某一行（实践差异不大），因为 RM 训练几千步后 head 完全主导，初始化影响很小。
4. **base 是谁**：**必须**从 SFT model init，不能从 base / pretrain model init——SFT 模型已经知道"什么叫 instruction following"，base 模型不知道，会让 RM 把"是否像 instruction-tuned 输出"也当成 reward 信号。

### 2.3 梯度与 margin 直觉

把 $\Delta = r_\theta(x, y_w) - r_\theta(x, y_l)$ 简记，单条样本的 loss 是 $\ell = -\log \sigma(\Delta)$。

$$\frac{\partial \ell}{\partial \Delta} = -\bigl(1 - \sigma(\Delta)\bigr) = \sigma(\Delta) - 1$$

- 当 $\Delta \gg 0$（chosen 已经远高于 rejected）：$\sigma(\Delta) \to 1$，梯度 $\to 0$——容易样本不再贡献梯度。
- 当 $\Delta \ll 0$（chosen 反而被低估）：$\sigma(\Delta) \to 0$，梯度 $\to -1$——hard sample 强信号。
- 当 $\Delta \approx 0$：梯度 $\approx -0.5$。

这与 SVM hinge loss 的"margin"思想一脉相承，但 RM 的"margin"是 soft 的——chosen 比 rejected 高出 5-10 logit 后基本就停止学这条样本，避免无谓的过拟合。

**Margin loss 变体**：Llama-2 [Touvron 2023] 用了带 margin 的扩展形式

$$\mathcal{L}_{\text{margin}} = -\log \sigma\bigl(r_\theta(x, y_w) - r_\theta(x, y_l) - m(r)\bigr)$$

$m(r)$ 是人工标的"chosen 比 rejected 好多少"档位（如 "明显更好" = 1.0，"差不多" = 0.0）。这能利用人工标注的"偏好强度"信息。Skywork-Reward 类似思路。

### 2.4 Pointwise / Pairwise / List-wise 三种范式

| 范式 | 输入形式 | loss | 优点 | 缺点 |
|------|---------|------|------|------|
| **Pointwise** | $(x, y) \to r$ | MSE / regression | 单样本可用 | 绝对分数难标，跨样本对不齐 |
| **Pairwise** | $(x, y_w, y_l)$ | BT (sigmoid) | 标注容易、loss 稳定 | 一次只能用 2 个 |
| **List-wise** | $(x, y_1, \dots, y_K)$ + ranking | Plackett-Luce / RankNet | 一次榨更多信息 | 标注成本高 |

工业主流是 **pairwise**——成本/效果比最优。少数场景（如 PairRM、ArmoRM）用 list-wise；pointwise 几乎只在"已经有人工 1-10 分标注"的场景才用（罕见）。

### 2.5 Multi-objective RM

Helpful / harmless / honest / coherent 这些维度往往冲突——一个直接拒绝（"我不能回答"）的 response 在 harmless 上满分但在 helpful 上零分。两种处理：

- **Single RM, weighted sum**：训一个 RM 同时拟合多目标的加权和，`reward = w_h * helpful + w_s * safe + ...`。Anthropic HH-RLHF [Bai 2022] 早期路线。问题：权重调参敏感，一次只能服务一个权重组合。
- **Multi-head RM**：一个 backbone + K 个 reward head，同时输出 K 个分数。下游 PPO 时再线性组合。**ArmoRM** [Wang 2024] 是这条路的代表，进一步训了一个 gating module 自动学权重。

---

## 3. 最小代码示例

### 3.1 手撕 RM training 核心 loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.backbone = base_model            # 已经 init 自 SFT model（去掉 lm_head）
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        # backbone 输出 last_hidden_state: (B, T, d)
        h = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        # 取每条序列的最后一个非 pad token
        last_idx = attention_mask.sum(dim=-1) - 1                         # (B,)
        last_h = h[torch.arange(h.size(0)), last_idx]                     # (B, d)
        return self.reward_head(last_h).squeeze(-1)                       # (B,)


def rm_loss(model, batch):
    # batch: chosen / rejected 已 tokenize 好且共享 prompt 部分
    r_chosen   = model(batch["chosen_input_ids"],   batch["chosen_attention_mask"])
    r_rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
    # Bradley-Terry pairwise NLL
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    # 同时记录 accuracy（chosen > rejected 的比例）作为 sanity check
    acc  = (r_chosen > r_rejected).float().mean()
    return loss, acc
```

关键点：

- `F.logsigmoid` 比 `torch.log(torch.sigmoid(x))` 数值稳定（避免 $\log 0$）——直接调它，不要分两步。
- `(r_chosen > r_rejected).float().mean()` 是 RM 的核心 metric——**pairwise accuracy**（不是 loss）才是看 RM 训得好不好的硬指标。健康 RM 训完通常在验证集 65-75%。
- 实战时 chosen / rejected 通常拼成一个 batch 一次 forward（避免两次 backbone 调用），后再切回两份算 loss。

### 3.2 用 HF TRL 的 RewardTrainer（生产推荐）

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from trl import RewardConfig, RewardTrainer
from peft import LoraConfig

base = "meta-llama/Llama-3.1-8B-Instruct"   # 必须用 Instruct / SFT 版本！
tok  = AutoTokenizer.from_pretrained(base)
tok.pad_token = tok.eos_token

# num_labels=1 → 单 scalar reward head
model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1)
model.config.pad_token_id = tok.pad_token_id

# UltraFeedback 的 binary 版本：每条含 chosen / rejected 文本
ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

cfg = RewardConfig(
    output_dir="./rm-llama3-8b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,         # RM lr 通常比 SFT 还小一个量级
    num_train_epochs=1,         # 1 epoch 通常够，多了过拟合
    bf16=True,
    max_length=2048,
    report_to="wandb",
)

# LoRA 是工业实战标配，全参 8B RM 显存吃不消
peft_cfg = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")

trainer = RewardTrainer(
    model=model, args=cfg, tokenizer=tok,
    train_dataset=ds, peft_config=peft_cfg,
)
trainer.train()
```

`RewardTrainer` 内部会处理：(a) chosen / rejected 分别 tokenize；(b) padding 到同一长度；(c) 上面 §3.1 的 BT loss；(d) log pairwise accuracy。读者只需关心数据格式（必须有 `chosen` / `rejected` 两列）和 LoRA / lr / epoch 这几个超参。

### 3.3 RM 推理给 reward

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

rm  = AutoModelForSequenceClassification.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
                                                          torch_dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

def score(prompt: str, response: str) -> float:
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    ids  = tok.apply_chat_template(msgs, return_tensors="pt").to(rm.device)
    with torch.no_grad():
        return rm(ids).logits[0, 0].item()       # (1, 1) → scalar

print(score("讲个笑话", "为什么程序员喜欢黑暗？因为他们怕 light themes。"))   # 例：5.3
print(score("讲个笑话", "我不知道。"))                                          # 例：-2.1
```

注意 chat template——RM 对输入格式非常敏感，**必须用 RM 训练时用的同一个 chat template**（每个开源 RM 的 README 会写）。换 template 直接掉 5-10 分 accuracy。

### 3.4 Best-of-N：RM 最简单、最强的应用

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. policy（SFT 模型）采样 N 个候选
policy_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
policy = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                               torch_dtype=torch.bfloat16, device_map="auto")

prompt = "解释什么是 Bradley-Terry 模型"
input_ids = policy_tok.apply_chat_template(
    [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True
).to(policy.device)

N = 16
out = policy.generate(input_ids, max_new_tokens=512, do_sample=True,
                      temperature=0.9, top_p=0.95, num_return_sequences=N)
candidates = policy_tok.batch_decode(out[:, input_ids.size(1):], skip_special_tokens=True)

# 2. 用 RM（上面 score 函数）给每个候选打分，选最高
rewards = [score(prompt, c) for c in candidates]
best = candidates[int(torch.tensor(rewards).argmax())]
print("Best response:", best)
```

Best-of-N 是 RM 最直接的"压榨方式"——不训 PPO、不需要 KL 约束、计算开销 $N \times$ 推理。Llama-2 [Touvron 2023] §3.2 论证了 BoN（N=8~64）经常和 PPO 不相上下、有时还更稳。**实战入门 RLHF 推荐先跑通 RM + BoN，再上 PPO**。

---

## 4. 工程踩坑与经验

- ❗ **RM 必须从 SFT model init，不能从 base / pretrain init**——base 模型不知道"什么叫 instruction following"，会把 "是不是像 instruction-tuned 输出" 也当成 reward 信号，下游 PPO 一上来 reward 就崩。如果你只有 base，先做最简单的几千条 SFT 再训 RM。
- ❗ **chosen / rejected 长度差异大 → RM 学到 length bias**，最后 reward 几乎只反映长度。三条对策：(1) 数据预处理时按长度分桶或丢掉极端样本；(2) loss 里加 length penalty（如 [Singhal 2023]）；(3) 推理用 length-controlled reward $r' = r - \alpha \cdot \text{len}(y)$。这个问题在 9.6 详谈。
- ❗ **RM accuracy 通常 65-75%，不要期望 95%+**——人类标注员之间的 inter-annotator agreement 也只有 75-80%（[Bai 2022] §2.3 报 70%+），RM accuracy > 人类一致性反而是过拟合信号。看到 95% acc 多半是数据泄漏 / chosen 全是 GPT-4 输出而 rejected 全是某 7B 输出这种作弊场景。
- ❗ **RM 不能跨域使用**——chat RM 不适合 code 任务，math RM 不适合 safety 评估。每个领域要么单独训，要么用 multi-objective + 显式 domain prompt。粗暴地把 chat RM 套到 code agent RL 上是 reward hacking 的捷径。
- ❗ **Pointwise RM 几乎不要训**——绝对打分跨样本对不齐（标注员 A 的 7 分 = 标注员 B 的 5 分），主流就是 pairwise BT loss。如果数据本身是 pointwise 评分，转换成 pairwise（每对样本比一比）再训。
- ❗ **lr 比 SFT 小一个量级**——SFT 常用 1e-5 ~ 5e-5，RM 常用 5e-7 ~ 5e-6（全参）或 1e-5 ~ 5e-5（LoRA）。RM 训练数据量小（10k-100k pair），过大 lr 直接过拟合。1 epoch 通常够，>2 epoch 几乎一定 overfit。
- ❗ **Best-of-N 是 RM 最划算的应用**，不是所有人都需要 PPO；Llama-2 paper 证明 BoN-64 经常打平甚至超过 PPO。生产链路上 RM + BoN 用作"在线 rerank" 是低风险方案，RM 漂移不会像 PPO 那样灾难性放大。
- ❗ **Multi-objective RM 的权重调参极度敏感**——helpful / harmless / honest 三维加权，权重从 (1,1,1) 变成 (1,2,1) 下游 PPO 风格可能完全变样。生产上常用 ArmoRM 那套 gating 自动学权重，或维持一个可在线调权重的 multi-head 设计。
- ❗ **DPO 时代 RM 仍然重要**：DPO 把 RM "隐含"进 loss 不需要显式 RM，但 PPO / GRPO / Best-of-N / Rejection Sampling / 在线评测全都需要。工业实战常见组合：先 DPO 拿稳定基线，再 RM + PPO 进一步推；或者 RM 训好后既用于 PPO 又用于离线评测（充当便宜的 LLM-as-Judge 替代品）。

---

## 5. 经典 paper

- **Christiano et al., 2017 — Deep Reinforcement Learning from Human Preferences** — RLHF 的奠基作。提出"用人类 pairwise preference 训 reward model 替代手工 reward function"的范式，本节 §2.1 的 BT loss 推导就出自这里 §2.2.2。读这一篇能看清楚为什么 RLHF 必须有 RM 这一段——LLM 时代继承了完全相同的范式，只是把 Atari 换成了 GPT。
- **Ouyang et al., 2022 — Training language models to follow instructions with human feedback (InstructGPT)** — 把 Christiano 的范式落到 LLM 上的奠基论文，确立 SFT → RM → PPO 三段式标准流程。读 §3.5（RM 训练细节）+ §3.6（PPO loss 形式），本节 §2.2 的架构选择和 §3.2 的 TRL 流水线全部基于此。
- **Bai et al., 2022 — Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback** — Anthropic 版 RLHF，HH-RLHF 数据集来源。比 InstructGPT 更详尽地讨论了 RM scaling、helpful vs harmless 多目标冲突、RM 的 calibration 问题。读 §3 RM training 与 §4.1 RM scaling laws，对训中等规模 RM 帮助很大。
- **Cui et al., 2024 — UltraFeedback: Boosting Language Models with Scaled AI Feedback** — 当代最常用的开源偏好数据集（含 64k prompt × 4 response 评分），现代 RM / DPO 训练几乎绕不开。看它的数据构造方法，理解 "AI 标注偏好" 在 9.7 RLAIF 里的角色。
- **Liu et al., 2024 — Skywork-Reward** + **Wang et al., 2024 — Interpretable Preferences via Multi-Objective Reward Modeling (ArmoRM)** — 2024 年开源 RM 的两个标杆。Skywork 关注数据策展 + margin loss，ArmoRM 关注 multi-objective + gating。要实际训 SOTA RM 必读它们的技术报告。

---

## 6. 自测与面试题

**Q1（公式）**：写出 Bradley-Terry RM 的 loss 公式，并解释为什么用 sigmoid。

<details>
<summary>Answer sketch</summary>

- 公式：$\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\bigl[\log \sigma\bigl(r_\theta(x, y_w) - r_\theta(x, y_l)\bigr)\bigr]$，其中 $\sigma$ 是 sigmoid，$r_\theta$ 是 RM 给出的标量 reward。
- BT 模型把"$y_w$ 比 $y_l$ 好"建模成概率 $P(y_w \succ y_l) = e^{r_w} / (e^{r_w} + e^{r_l}) = \sigma(r_w - r_l)$，这是 logistic regression 在偏好空间的应用。
- 为什么 sigmoid：(1) 想用一个标量差 $\Delta = r_w - r_l$ 预测一个二分类概率，sigmoid 是把 $\mathbb{R} \to (0, 1)$ 的标准选择；(2) 对应的 NLL 等价于 logistic loss，梯度形式漂亮（$\partial \ell / \partial \Delta = \sigma(\Delta) - 1$，自动给 hard sample 大梯度、易 sample 小梯度）；(3) 与 1.4 节的 CE 同根，是 K=2 二分类的特例。
- 直观结果：$\Delta$ 大（chosen 已远高于 rejected）时梯度趋零，$\Delta$ 小时梯度大——天然的 margin / curriculum 效果。

</details>

**Q2（实战）**：你训了一个 RM，验证集 accuracy 不低（72%），但下游 PPO 后发现 policy 输出越来越长，最后近乎全是冗余啰嗦。看了下 RM 评分发现 chosen 永远比 rejected 长。给出至少 3 个解决方向。

<details>
<summary>Answer sketch</summary>

这是经典的 **length bias** 导致的 reward hacking。诊断与对策：

- **数据层面**：检查训练集 chosen / rejected 长度分布——如果 chosen 平均比 rejected 长 30%+，RM 必然学到 length 信号。对策：(1) 按长度差分桶采样，丢弃长度差极端的 pair；(2) 主动构造"短的 chosen vs 长的 rejected" 反向 pair 平衡分布；(3) 用 length-controlled 数据集（如 length-controlled AlpacaEval 的训练集变体）。
- **Loss 层面**：训 RM 时加 length regularization。如 [Singhal 2023] 的做法是在 loss 里减去 length 与 reward 的相关项；或者直接用 length-disentangled loss（先回归出"长度贡献"再训一个 length-free reward head）。
- **RM 推理层面**：下游 PPO / BoN 用 length-normalized reward $r' = r - \alpha \cdot \text{len}(y)$，$\alpha$ sweep 出来；或者引入显式 length penalty 进 PPO 的 reward shaping。
- **PPO 层面**：(1) 加 length penalty 直接到 reward；(2) 提高 KL 约束 $\beta$ 限制 policy 漂移幅度；(3) 限制 max_new_tokens 强制截断。
- **架构层面**：训 multi-objective RM，把 length 作为 explicit head 输出，下游做 weighted sum 时把 length 项权重置 0 或负。
- **加分**：根本问题是人工标注员有 length bias（更长的回答看起来更"用心"），所以最干净的方案是修标注准则 + 重新收数据；但工程上来不及，就只能上面那些 patch。

</details>

**Q3（trade-off）**：DPO 不需要显式 RM 就能从偏好数据训出对齐模型，那为什么大公司还在花资源训 RM？至少给出 3 个 RM 仍然不可替代的应用场景。

<details>
<summary>Answer sketch</summary>

- **PPO / GRPO 训练**：在线 RL 必须有 reward function 给采样的 trajectory 打分，DPO 是离线 / 静态偏好数据，无法处理"policy 探索出新 response 后该给多少分"。所有 on-policy RL（PPO、GRPO、RLOO）都需要 RM。DeepSeek-R1 / GRPO 实操中 RM 仍是核心组件之一。
- **Best-of-N / Rejection Sampling**：推理时对每个 prompt 采 N 个候选，用 RM 选最优。这是工业上最低风险、性价比最高的"对齐手段"——不动模型权重、效果立等可见，Llama-2 实证 BoN-64 经常打平 PPO。**这一条 DPO 完全无法替代**。
- **数据筛选 / 合成**：RM 充当便宜的 quality scorer，从大规模 LM 生成的合成数据里筛掉低质条目（self-instruct / WizardLM / Magpie 类 pipeline 都需要）；也用于构造下一轮 DPO / SFT 的训练数据（rejection sampling fine-tuning, RFT）。
- **离线评测 / 在线 monitoring**：RM 比调用 GPT-4 当 LLM-as-Judge 便宜 100x 以上，用于 CI 回归测试、A/B 实验流量打分。Module 12.2 详谈。
- **多轮 / 长链路 reward 归因**：DPO 是 trajectory-level 的隐式 reward，无法对单步打分；多轮 agent / long-CoT 场景需要 RM 给中间步骤打分（更进一步的是 PRM，详见 10.2）。
- **加分讨论**：工业实战常见组合是 "DPO 拿到稳定基线 → RM + PPO 进一步推"，或 "DPO + RM 双管齐下，互为安全网"——单纯 DPO 容易在 OOD prompt 上失控，RM 能给在线 sanity check。
- **本质原因**：DPO 把 reward 隐含在 policy ratio 里，**回答了"怎么用偏好数据训 policy"，但没回答"怎么给任意 (x, y) 打分"**——而后者是 RM 的核心能力，凡是需要"分数"而不是"训练"的场景都仍然需要 RM。

</details>

---

## 7. 延伸阅读

- [HuggingFace TRL `RewardTrainer` 文档](https://huggingface.co/docs/trl/main/en/reward_trainer) — §3.2 代码的官方手册，含完整 API 与 LoRA / 多卡示例。
- [RewardBench Leaderboard](https://huggingface.co/spaces/allenai/reward-bench) — AllenAI 维护的 RM 评测榜单，看现代开源 RM 在 chat / safety / reasoning / code 各域的表现，找 RM 选型参考必读。
- [Skywork-Reward 技术报告](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2) — 2024 年 RewardBench 上长期榜单第一，关键创新在数据策展（不在 loss），值得整篇精读。
- [Lambert 2024 — RLHF Book Ch.6 "Reward Modeling"](https://rlhfbook.com/c/06-reward-modeling.html) — Nathan Lambert 在写的开源 RLHF 教材中 RM 章节，把 BT 推导、margin loss、ensemble、calibration 全讲到。
- 推荐继续读本教程的 **9.3 节《PPO 原理与在 LLM 上的形式》**——把训好的 RM 接到 PPO 主循环，理解为什么需要 4 个模型同时驻留显存；以及 **9.6 节《工程踩坑：reward hacking / RM 漂移 / KL 坍塌》**——本节末尾点到的 length bias / RM hacking 这条暗线在那里集中处理。
