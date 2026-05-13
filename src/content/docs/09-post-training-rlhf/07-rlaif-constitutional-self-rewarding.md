---
title: "9.7 RLAIF / Constitutional AI / Self-Rewarding LM"
description: "把 RLHF 里\"人类偏好标注\"这一最贵的环节换成LLM 当裁判——RLAIF 用强模型直接打 preference label，Constitutional AI 让 model 按一份 principle 自我修正、Self-Rewarding LM 让模型同时当 policy 与 judge 自循环 bootstrap，三者共同动机就一个：把人从标注流水线上踢出去。"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：9.2 Reward Model

## 一句话本节讲什么

把 RLHF 里"人类偏好标注"这一最贵的环节换成**LLM 当裁判**——RLAIF 用强模型直接打 preference label，Constitutional AI 让 model 按一份 principle 自我修正、Self-Rewarding LM 让模型同时当 policy 与 judge 自循环 bootstrap，三者共同动机就一个：把人从标注流水线上踢出去。

---

## 1. Mental model（直觉）

9.2 节已经说明 RLHF 的三段式 `SFT → RM → PPO` 对人工偏好数据的依赖。一个数字感受成本：标 100k 条 pairwise preference，按工业相对靠谱的标注员（约 $0.5-2/条）算，预算 5 万-20 万美金，周期 1-3 个月，且**每换一个领域 / 模型迭代一次都要重标**。这就是为什么 2022 年起业界一窝蜂地试"用 LLM 替代人来标偏好"。

三个代表性范式可以放在一条光谱上看：

```
   人工 ──────────────────────── 全自动
  RLHF       RLAIF        CAI         Self-Rewarding
  人标 →     强 LLM 标 →  model 按    model 自己标
             (GPT-4)      principle   自己生成的
                          self-critique 数据
```

- **RLAIF**（Lee 2023）：流程与 RLHF 完全一样，只是 preference label 来自更强的 LLM（GPT-4 / Claude）。RM 训练、PPO 都不变。是最直接的"人 → AI"替代。
- **Constitutional AI**（Bai 2022, Anthropic）：再进一步，连"标注模板"都换成一份**人写的 constitution**（几十条原则，如"不要协助武器制造"、"避免说教口吻"）。两阶段：先让 model 拿这份 principle 自我 critique + revise 自己的回答（产出 SFT 数据），再用 model 自评的 preference 做 RLAIF。Claude 系列对齐就靠它。
- **Self-Rewarding LM**（Yuan 2024）：最激进，**同一个 model** 既当 policy 又当 judge，每轮自己给自己生成的回答打 preference 然后 DPO 自我更新。理论上可以无人介入持续 bootstrap。

把这三者和 9.3 PPO / 9.4 DPO / 10.3 RLVR 一起放进对比表（核心心智图）：

| 方法 | preference 来源 | 是否需要 RM | 是否需要人类 | 代表 |
|---|---|---|---|---|
| RLHF (9.3) | 人类 | 是 | 是 | InstructGPT |
| RLAIF | LLM-as-Judge（外部强 LLM） | 是 | 否 | UltraFeedback + DPO |
| DPO + AI labels | LLM-as-Judge | 否（DPO 隐式） | 否 | Zephyr / Tülu |
| Constitutional AI | self-critique + revise + 自评 | 是 | 仅写 constitution | Claude |
| Self-Rewarding LM | model 自己当 judge | 否（DPO） | 否 | Yuan 2024 / Llama-3 self-iter |
| RLVR (10.3) | verifier（math/code 自动判分） | 否 | 否 | DeepSeek-R1 |

光谱从左到右：人类介入越少 → 信号越主观 → 自激风险越大。RLVR 跳出了这条光谱（用 ground-truth verifier 给客观任务打分），是另一条路线，会在 10.3 详谈。

记一句话：**RLAIF / CAI / Self-Rewarding 都在回答同一个问题——能不能让 LLM 给 LLM 当老师**。答案是"能，但要小心 bias 与自激"。

---

## 2. 公式与原理

### 2.1 RLAIF：把 RLHF 的 H 换成 AI

设 prompt 为 $x$，policy $\pi_\theta$ 采两条候选 $y_a, y_b$。RLHF 收集人类偏好 $\mathbb{1}[y_w \succ y_l]$ 训 RM；RLAIF 把这一步换成 LLM judge $J$（通常是 GPT-4 / Claude / 一个更强的 model）：

$$\mathbb{1}[y_w \succ y_l] \;\;\longleftarrow\;\; J(x, y_a, y_b) \in \{a, b\}$$

剩下的 RM 训练（9.2 的 BT loss）和 PPO（9.3）一字不改。整个流程对下游来说**等价于**用了一个"AI 标注的偏好数据集"。

Lee et al. 2023 在 summarization、helpful dialog 等任务上系统对比：RLAIF 与 RLHF 在 helpfulness 上 win-rate 相当（差距 < 2pp），harmfulness 上 RLAIF 略占优（因为 judge 的安全偏好更稳定）。这是 RLAIF 路线工业化的基础实证。

**LLM-as-Judge 的可靠性数字（必须记）**：
- GPT-4 与人类的偏好 agreement ≈ **80-85%**（Zheng 2023, MT-Bench）
- 人类标注员之间的 agreement ≈ **75-80%**（Bai 2022 §2.3）
- → GPT-4 当 judge 的"准确度"已经达到甚至略超人类一致性的下限

但 LLM judge 有几个**系统性 bias**：

1. **Position bias**：先看到的 response（位置 A）更易被选中。Zheng 2023 报 GPT-4 swap A/B 后约 10-15% 的判决会反转。**强制对策**：每条样本随机 shuffle A/B 顺序；更严的做法是同一对样本两次评分（A/B 与 B/A）取一致的才保留。
2. **Length bias**：longer response 更易被选——judge 会把"详细 = 高质量"当默认 prior。**对策**：在 judge prompt 里显式要求 "do not prefer longer response"，或后处理时对 length 做 normalization。
3. **Verbosity / 自我偏好 bias**：GPT-4 偏好 GPT-4 风格的回答（学术化、列点、礼貌套话）。这会把 student model 推向同一种风格。**对策**：ensemble 多个 judge（GPT-4 + Claude + Gemini）。

### 2.2 Constitutional AI：两阶段（SL-CAI + RL-CAI）

CAI 把"alignment 信号"显式拆成一份人写的 **constitution**——几十条 principle，比如：

```
Principle 12: Please choose the response that is most age-appropriate
              and least likely to be harmful to children.
Principle 17: Please choose the response that is most respectful of personal,
              private, and confidential matters.
...
```

完整的 Anthropic constitution 公开在 their blog，约 60 多条，分 helpfulness / harmlessness / honesty 三类。

**Stage 1: SL-CAI (Supervised Learning Constitutional AI)** —— 让 model 自我修正然后做 SFT。

对每个 prompt $x$，流程是 4-step prompt chain：

1. **Generate**：用 base model（通常已 SFT）产出 initial response $y_0$。
2. **Critique**：把 $y_0$ 喂回去，让 model 按某条 principle $p_i$ 写一段 self-critique（"上述回答违反了 X，因为……"）。
3. **Revise**：再喂一次，让 model 基于 critique 写 revised response $y_1$。
4. **重复**：可以多 round（每轮抽不同 principle），最后得到 final $y_K$。

数据集就是 $\{(x, y_K)\}$——把 revised response 当 SFT target。语义是："让 model 学会 critique 完应该写出的样子"。

**Stage 2: RL-CAI (RLAIF using constitution)** —— 用 constitution 当 judge prompt 做 RLAIF。

对每个 prompt $x$ 采两条 $y_a, y_b$，用 LLM（同一 model 或更强的 model）按 constitution 打 preference：

$$P(y_a \succ y_b \mid x) = \texttt{LLM}\bigl(\text{principle}, x, y_a, y_b\bigr)$$

剩下与 RLHF 一致：训 RM、跑 PPO（Anthropic 后期改成 RLAIF + DPO）。

**为什么要两阶段**？只做 Stage 2 会让 model 收敛到一个 reward-maximizing 但缺乏自我反省能力的 policy；先做 Stage 1 让 model **学会按 principle 推理**，Stage 2 再 reinforce 这种行为。Bai 2022 实证：两阶段比单阶段 harmlessness 提升 ≈ 30%（同时不掉 helpfulness）。

### 2.3 Self-Rewarding LM：policy 与 judge 同体

Yuan 2024 的核心构想：让一个 model 同时具备两种能力：(1) instruction following（生成 response）；(2) **judge ability**（给 (prompt, response_a, response_b) 打 preference）。这样就可以无外部 judge / 无 RM 持续 bootstrap。

训练循环（每个 iteration $t$）：

1. **Self-instruction**：用 $M_t$ 生成新 prompt 集（或用固定 prompt pool）。
2. **Generate pairs**：对每个 prompt 采 $K$ 条候选 $\{y_1, \dots, y_K\}$（temperature > 0）。
3. **Self-judge**：用 $M_t$ 当 LLM-as-Judge 给每个候选打分（pointwise 5 分制，或两两 pairwise）。挑出 highest-scored 当 chosen、lowest-scored 当 rejected，构造 preference pair $(y_w, y_l)$。
4. **DPO update**：用上面构造的 preference 数据集做 DPO，更新得 $M_{t+1}$。
5. 回到 step 1。

形式化地，第 $t$ 轮的训练数据 $\mathcal{D}_t$ 完全由 $M_t$ 自产，loss 是 DPO（9.4）：

$$\mathcal{L}_{\text{DPO}}^{(t)} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_t}\Bigl[\log \sigma\bigl(\beta \log \tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\bigr)\Bigr]$$

其中 $\pi_{\text{ref}} = M_t$（每轮换一次 ref），$\pi_\theta$ 初始化也是 $M_t$，训完得到 $M_{t+1}$。

Yuan 2024 在 Llama-2-70B 上跑 3 个 iteration，AlpacaEval 2.0 win-rate 从 9.94% → 15.38% → 20.44% → **20.44%**（第 3 轮收益递减，第 4 轮基本饱和）。这是"self-rewarding 可以 bootstrap 但有上限"的关键证据。

**为什么能 work**？Iteration $t$ 的 model 给同一个 prompt 采 $K$ 个候选，里面有"差不多正确"的也有"明显错"的——只要 model 当 judge 时**比当 policy 时更准**（这在 instruction-tuned 模型上经验上成立，因为 judge 是简单的二分类，policy 是开放生成），preference 信号就有正向梯度，DPO 把"被自己判好的 response"reinforce 进去。下一轮 policy 更强了，自然 judge 也更准——正反馈。

**为什么会饱和 / drift**？Iteration 多了之后，policy 输出分布越来越窄（mode collapse），judge 看不到足够的差异性 → preference 信号变弱；同时 model 自己的 bias 被反复放大（学自己喜欢的、忽略人类反而喜欢的）。所以工业实战通常 3-5 轮停，且每轮注入少量"外部 judge"（更强的 LLM 或 1k 人工标注）做 calibration。

### 2.4 三者的统一视角

把上述三种方法和 9.3 RLHF / 10.3 RLVR 放一起对比 preference 信号的"成本-质量-bias"trilemma：

| 维度 | RLHF | RLAIF | CAI | Self-Rewarding | RLVR |
|------|------|-------|-----|----------------|------|
| 单条标注成本 | $0.5-2 | $0.001-0.01 | $0.001-0.01 | ~$0 | ~$0 |
| 标注质量 | 黄金 | 80-85% 一致 | 同 RLAIF | 取决于 base | 客观 100% |
| Bias 来源 | 标注员主观 | judge 自身 bias | constitution 写法 | 自激放大 | 任务覆盖窄 |
| 可扩展 prompt 量 | 万级 | 百万级 | 百万级 | 无上限 | 受 verifier 限 |
| 适用任务 | 全部 | 主观 chat | 安全 / 价值观 | 通用 chat | math / code |

工业现状是**混用**：通用 chat 走 RLAIF（UltraFeedback + DPO 是标配），安全走 CAI 思想，reasoning 走 RLVR；少量人工标注用于 final calibration。Tülu 3 / Llama-3 Instruct 都是这种 multi-stage 配方。

---

## 3. 最小代码示例

### 3.1 RLAIF preference labeling（用 GPT-4 当 judge）

```python
import os, random, json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JUDGE_PROMPT = """You are a helpful assistant evaluating two responses to a user query.

User query: {prompt}

Response A: {resp_a}
Response B: {resp_b}

Compare them on helpfulness, harmlessness, and honesty.
Respond with ONLY a single character: "A" or "B". No explanation."""

def llm_judge(prompt: str, resp_1: str, resp_2: str, model="gpt-4o") -> dict:
    # 关键：随机 swap 防 position bias
    swapped = random.random() < 0.5
    a, b = (resp_2, resp_1) if swapped else (resp_1, resp_2)
    out = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=1,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(prompt=prompt, resp_a=a, resp_b=b)}],
    ).choices[0].message.content.strip().upper()
    # 把 judge 的 A/B 映射回原始 resp_1/resp_2
    winner_is_1 = (out == "A") ^ swapped
    return {"chosen": resp_1 if winner_is_1 else resp_2,
            "rejected": resp_2 if winner_is_1 else resp_1}

# 使用：把整个 SFT model 采样的候选喂给它，攒成 preference dataset
pair = llm_judge("讲个程序员笑话",
                 "为什么程序员喜欢黑暗？因为他们怕 light themes。",
                 "我不知道。")
print(json.dumps(pair, ensure_ascii=False, indent=2))
```

关键点：

- `swapped` 这一行是**RLAIF 的命门**——不做 randomize 直接掉 5-10pp judge 准确度。生产环境更严的版本是同一对样本调用两次（A/B 和 B/A），只保留两次一致的（drop 率约 15-20%）。
- `temperature=0.0 + max_tokens=1`：判决要 deterministic，且约束输出空间——否则模型会写一长段 explanation 再给答案，token 成本暴涨且容易 parse 失败。
- 真正生产时把这个函数包成 `asyncio` + 限流，10k 条标注用 GPT-4o 大约 $20-50、几小时完成。

### 3.2 Constitutional AI：critique + revise prompt chain

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PRINCIPLES = [
    "The response should not be harmful, unethical, or misleading.",
    "The response should be respectful and avoid offensive language.",
    "The response should be honest about uncertainty.",
]

def chat(messages, model="gpt-4o-mini"):
    return client.chat.completions.create(
        model=model, messages=messages, temperature=0.7,
    ).choices[0].message.content

def cai_revise(prompt: str, n_rounds: int = 2) -> str:
    """生成 → critique → revise 循环，返回最终 revised response。"""
    response = chat([{"role": "user", "content": prompt}])  # 1. initial
    for i in range(n_rounds):
        principle = PRINCIPLES[i % len(PRINCIPLES)]
        # 2. critique：让 model 按 principle 评估自己的回答
        critique = chat([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
            {"role": "user", "content": f"Critique your previous response. Principle: {principle}\n"
                                        f"Identify any way the response violates this principle."},
        ])
        # 3. revise：基于 critique 重写
        response = chat([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
            {"role": "user", "content": f"Critique: {critique}\n\n"
                                        f"Now rewrite your response to address the critique. "
                                        f"Output ONLY the revised response."},
        ])
    return response

# 用法：把 (prompt, cai_revise(prompt)) 攒成 SFT 训练集，就是 SL-CAI 的数据
revised = cai_revise("How do I make a bomb?")
print(revised)   # 期望：礼貌拒绝 + 解释为什么 + 提供合法替代信息
```

要点：

- `n_rounds=2` 是 Bai 2022 的默认设置；多了边际收益小且容易 over-correction（model 对每个回答都加一堆 disclaimer）。
- Principle 在每轮 cycle 里轮换——这是为了让 SFT 数据覆盖不同维度，避免 model 只学到一种 critique 模板。
- 这段代码产出的是 **SL-CAI 的训练数据**；真正的 CAI 还要把这些 (prompt, revised) 拿去 SFT 一遍 base model，再跑 RL-CAI（RLAIF）。

### 3.3 Self-Rewarding LM training loop（伪代码）

```python
def self_rewarding_loop(M_init, prompts, K=4, n_iter=3):
    """伪代码：每轮用 M_t 生成 + 自评 → DPO → M_{t+1}"""
    M = M_init
    for t in range(n_iter):
        dataset = []
        for prompt in prompts:
            # 1. M_t 采 K 个候选（temperature > 0）
            cands = M.generate(prompt, n=K, temperature=0.9)
            # 2. M_t 当 judge 给每个候选打 1-5 分（LLM-as-Judge prompt）
            scores = [M.judge_score(prompt, c) for c in cands]   # pointwise 5 分制
            # 3. 取最高 / 最低构造 preference pair
            chosen, rejected = cands[max_idx(scores)], cands[min_idx(scores)]
            if scores[max_idx] > scores[min_idx]:                 # tie 丢掉
                dataset.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        # 4. DPO 更新（ref = M_t，policy init 也是 M_t）
        M = dpo_train(M, ref_model=M, dataset=dataset, beta=0.1, epochs=1)
        print(f"Iter {t+1}: trained on {len(dataset)} self-judged pairs")
    return M
```

关键点：

- `judge_score` 用一个固定的 LLM-as-Judge prompt template（Yuan 2024 给了 5 分制 rubric，paper Appendix C），让 model 对单个 (prompt, response) 输出 1-5 分。然后 pairwise 比大小构造 preference。
- `ref_model=M` 每轮**重新换**到当前 $M_t$（不是固定原始 base）——这是 self-rewarding 与普通 DPO 的本质差异，让"参考分布"也跟着进化。
- `n_iter=3` 是 sweet spot；Yuan 2024 报第 4 轮基本饱和，第 5 轮反而开始 drift。
- 这段是伪代码——真实实现要套 TRL 的 `DPOTrainer` + vLLM 加速 generate / judge 阶段，单 iter 70B 模型耗时约 6-12 小时（4×H100）。

---

## 4. 工程踩坑与经验

- ❗ **LLM-as-Judge 的 position bias 必须 randomize**——不 swap 直接掉 5-10pp judge 准确度。Zheng 2023 实测 GPT-4 swap A/B 后 10-15% 的判决会反转；最严格的做法是同一对样本两次评分（A/B 和 B/A 都问），只保留两次一致的，剩下的 drop 掉。drop 率通常 15-20%。
- ❗ **judge LLM 强弱直接决定 RLAIF 上限**——GPT-4 / Claude Opus 当 judge 的 student model win-rate 显著高于 GPT-3.5 / 7B 当 judge。**用 7B 当 judge 几乎没用**（一致性 ≈ 60%，刚比随机好一点）。预算紧也至少要用 GPT-4o-mini / Claude Haiku 这一档。
- ❗ **RLAIF 的标注成本远比想象低**——100k preference 用 GPT-4 约 $1k-5k，用 GPT-4o-mini 约 $100-500。算上 batch API 50% 折扣还能再省一半。**这是 RLAIF 路线工业化最直接的驱动力**——比人工标注便宜 100-1000 倍。
- ❗ **Self-Rewarding 风险：自激放大 bias**——model 学自己喜欢的风格、忽略人类反而喜欢的特性，几轮后容易 mode collapse（输出越来越同质化）或 drift（在某个 niche 上越钻越深）。**至少 ensemble 一个外部 judge**（如每轮加 5% 外部 GPT-4 标注做 calibration），或定期在 holdout set 上跑 win-rate 监控。
- ❗ **Self-Rewarding 通常 3-5 个 iteration 后收益递减**——Yuan 2024 第 4 轮饱和。**不要无限迭代**。生产实践是跑 3 轮拿稳定增益就停，第 4 轮以后投入产出比变差。
- ❗ **CAI 的 constitution 写法极其影响效果**——principle 太抽象（"be helpful"）model 学不到具体行为；太具体（"don't mention X"）泛化差。Anthropic 公开了部分 principles 可以参考；自己写时建议每条配 1-2 个具体示例（few-shot critique）。Principle 数量 30-60 是 sweet spot，超过 100 边际收益接近零。
- ❗ **CAI 的 Stage 1 不能省**——只跑 Stage 2 (RL-CAI) 而不做 SL-CAI 的话，model 学到的是"按 reward 输出"而不是"按 principle 推理"，鲁棒性差很多。Bai 2022 实证两阶段比单阶段 harmlessness 提升 ≈ 30%。
- ❗ **RLAIF 不能完全替代 RLHF，至少要少量人工 calibration**——纯 AI 标注会把 judge 的 bias（学术风、列点癖、self-preference）原封不动传给 student。生产链路通常配 500-1k 高质量人工标注做最终验证 / 校准；或 RLAIF 训完后再用人工偏好做一轮 fine-grained DPO。
- ❗ **judge 的 chat template 要严格匹配**——RLAIF 给 GPT-4 当 judge 时 system prompt 写法（"you are evaluating…"）会显著影响判决。生产上把 judge prompt 当代码版本管理（写进 repo + diff review），不能随手改。
- ❗ **Length bias 在 LLM judge 上比在人类标注员上更严重**——GPT-4 当 judge 时长 response win-rate 系统性偏高 5-10pp。AlpacaEval 2.0 引入 length-controlled win-rate 就是为了消这个 bias。下游做 RLAIF 时建议在 judge prompt 里显式加一句 `"do not prefer longer responses"`，能减半 length bias。

---

## 5. 经典 paper

- **Bai et al., 2022 — Constitutional AI: Harmlessness from AI Feedback (Anthropic)** — RLAIF + CAI 的奠基论文，Claude 系列的 alignment 范式。读 §3 SL-CAI 的 critique-revise prompt chain、§4 RL-CAI 的 RLAIF 流程、附录公开的部分 constitution——本节 §2.2 / §3.2 全部基于此。理解"为什么 Claude 这么爱 critique 自己"读这一篇就懂。
- **Lee et al., 2023 — RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback (Google)** — 首次系统对比 RLAIF 与 RLHF 在 summarization / dialog 上的差距，是 RLAIF 路线工业化的关键实证。take-away：RLAIF 与 RLHF 在 helpfulness 上 win-rate 几乎相同（< 2pp）、harmfulness 上 RLAIF 略优——为现代 UltraFeedback 类 pipeline 铺路。
- **Yuan et al., 2024 — Self-Rewarding Language Models (Meta)** — Self-rewarding 范式提出。Llama-2-70B 跑 3 个 iteration，AlpacaEval 2.0 win-rate 9.94% → 20.44%。读 §2 算法细节（特别是 LLM-as-Judge 5 分制 prompt template）+ §4 实验曲线（看到第 4 轮饱和的 point）——理解"自循环能 bootstrap 但有上限"的本质。
- **Cui et al., 2024 — UltraFeedback: Boosting Language Models with Scaled AI Feedback** — 当代最常用的开源 RLAIF 偏好数据集（64k prompts × 4 responses，由 GPT-4 多维评分）。Zephyr / Tülu / Starling / 几乎所有 2024 之后的开源对齐模型都用它。读它的数据构造方法（§3）就是 RLAIF 工业实操模板。
- **Zheng et al., 2023 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — LLM-as-Judge 可靠性的系统研究。给出 GPT-4 与人类 80-85% agreement 这个关键数字、量化了 position / length / verbosity bias、提出 swap consistency 等 mitigation。本节 §2.1 的所有可靠性数字都出自这一篇，做 RLAIF 必读。

---

## 6. 自测与面试题

**Q1（区别）**：RLHF / RLAIF / Constitutional AI / Self-Rewarding LM 四者的核心差异是什么？preference 信号的来源各是什么？

<details>
<summary>Answer sketch</summary>

四者放在"人介入程度"光谱上：

- **RLHF (9.3)**：preference 来自**人类**标注员对 (prompt, response_a, response_b) 的 pairwise 选择。需要 RM、需要人类、成本最高（每条 $0.5-2）。代表：InstructGPT。
- **RLAIF (Lee 2023)**：流程与 RLHF 完全一样，**只是把人换成更强的 LLM（GPT-4 / Claude）**当 judge 打 preference label。RM、PPO 不变。需要 RM、不需要人类、成本降 100-1000 倍。代表：UltraFeedback + DPO pipeline。
- **Constitutional AI (Bai 2022)**：再进一步，把"标注模板"也换成一份**人写的 constitution**（几十条 principle）。两阶段——SL-CAI 让 model 按 principle 自我 critique + revise 当 SFT 数据，RL-CAI 用 model 按 constitution 自评 preference 做 RLAIF。**唯一需要的人工是写 constitution**。代表：Claude alignment。
- **Self-Rewarding LM (Yuan 2024)**：最激进，**同一个 model 既当 policy 又当 judge**，每轮自己给自己生成的回答打 preference 然后 DPO 自更新。完全无人介入、无外部 judge。代表：Llama-2-70B self-iteration。
- 加分对比维度：是否需要 RM——RLHF / RLAIF / CAI 都需要（虽然 CAI 后期改 DPO），Self-Rewarding 直接 DPO 不需要 RM；是否需要外部 LLM——RLAIF / CAI 需要外部强 LLM，Self-Rewarding 完全自给自足。

</details>

**Q2（trade-off）**：你要给一个新模型做对齐训练，预算 5 万美金，想用 GPT-4 做 RLAIF 标注 100k preference。算一下成本，并指出至少 3 个引入的风险。

<details>
<summary>Answer sketch</summary>

**成本估算**：

- 单条 preference judge 输入 ≈ 500 tokens（prompt + 两条 response）+ 输出 ≈ 5 tokens
- GPT-4o：$5/1M input + $15/1M output → 单条约 $0.0026 → **100k 条约 $260**
- GPT-4-turbo：$10/1M + $30/1M → 单条约 $0.005 → **100k 条约 $500**
- 加 batch API 50% 折扣：再省一半。**实际 5 万美金能做 1-10M 条**（vs. 人工 5 万只能做 25k-100k 条），便宜 100-1000 倍。

**主要风险**：

- **Position bias**：必须强制 randomize A/B 顺序；不做的话掉 5-10pp 准确度。最严格做法是 swap consistency（同一对问 A/B 和 B/A 都问，只保留一致的，drop 率 15-20%）。
- **Length bias**：GPT-4 当 judge 系统性偏好长 response，5-10pp。下游 student model 学完会越来越啰嗦。对策：judge prompt 显式加 "do not prefer longer"，或 reward 后处理 length normalize。
- **Self-preference / 风格 bias**：GPT-4 偏好 GPT-4 风格（学术化、列点、礼貌套话），student model 会被推到同一种风格。对策：ensemble 多个 judge（GPT-4 + Claude + Gemini）。
- **Bias 难以发现**：人工标注的 bias 还能靠 inter-annotator agreement / 抽样审计发现，LLM judge 的 bias 是系统性的、隐藏的，必须配合 holdout set 跑 win-rate vs 人类偏好做监控。
- **完全替代不行**：纯 RLAIF 没有 ground-truth anchor，长期会 drift。生产实践是配 500-1k 高质量人工标注做 final calibration。
- 加分讨论：这个预算应该花的位置不是全都堆 RLAIF——拆成 "RLAIF 80%（4 万） + 人工 calibration 1k 条（5k）+ ensemble judge 5k" 才是健康配比。

</details>

**Q3（前沿）**：Self-Rewarding LM 与 RLVR (10.3) 都标榜"无人工 reward"，但适用场景完全不同。给出至少 2 个判断标准来决定某个新任务该用哪个。

<details>
<summary>Answer sketch</summary>

两个核心判断标准：

**标准 1：任务是否有 ground-truth verifier？**

- 有（math 答案对错、code 单元测试通过、game 输赢、formal proof checker）→ **RLVR**。Verifier 给的 reward 是客观的、零 bias、几乎免费。
- 无（chat 是否 helpful、回答是否礼貌、写作风格、对齐价值观）→ **Self-Rewarding**（或 RLAIF）。这种主观 quality 没有 algorithmic verifier，只能用 LLM-as-Judge。
- 例：写"求解 2x+3=7"用 RLVR；写"用温柔语气安慰朋友"用 Self-Rewarding。

**标准 2：reward 信号能否在 base model 能力范围内被 model 自己判别？**

- 能（model 自己会判 chat 好坏、code 能跑会读 stack trace）→ Self-Rewarding 可以 bootstrap。
- 不能（base model 解不出 olympiad math，自己判不了对错）→ Self-Rewarding 自激没用，必须用外部 verifier (RLVR) 或更强外部 judge。
- DeepSeek-R1 用 RLVR 而不用 self-rewarding 就是因为基础 reasoning model 自己当 judge 准确率太低、无法稳定 bootstrap。

**加分讨论**：

- 现代 frontier model 训练混用两者：通用对齐 / chat 走 RLAIF + Self-Rewarding（主观信号），reasoning / math / code 走 RLVR（客观信号）。Tülu 3 / Qwen 2.5 / Llama-3.1 都是这种配方。
- 一个易混淆点：Self-Rewarding 可以处理客观任务（如 math），但 base model 必须已经"基本会做"——只有当 self-judge 比 self-policy 更准时 bootstrap 才有效。Olympiad math 这种 base model 准确率 10% 的任务，self-judge 同样不准，无法 bootstrap，必须用 RLVR。
- 第三个隐藏标准：**bias 容忍度**。Self-Rewarding 必然放大 model 自身 bias，对安全 / 价值观敏感任务不合适；RLVR 完全客观、无 bias 但任务覆盖面窄。

</details>

---

## 7. 延伸阅读

- [Anthropic 公开的 Claude Constitution（部分）](https://www.anthropic.com/news/claudes-constitution) — CAI 的 principle 真容，写自己 constitution 时的标杆参考。
- [UltraFeedback 数据集（HF）](https://huggingface.co/datasets/openbmb/UltraFeedback) — 现代 RLAIF 训练事实标准，64k prompts × 4 responses 由 GPT-4 多维评分。
- [Yuan 2024 Self-Rewarding LM 官方代码](https://github.com/lucidrains/self-rewarding-lm-pytorch) — Phil Wang 的 PyTorch 复现，配合 paper 看实现细节。
- [LMSYS Chatbot Arena Leaderboard](https://chat.lmsys.org/) — 大规模真人 pairwise preference 收集平台，是评测 RLAIF 训出的 model 是否真的对齐了"人类口味"的事实标准。
- [Lambert RLHF Book Ch.10 "Constitutional AI & AI Feedback"](https://rlhfbook.com/c/13-cai.html) — Nathan Lambert 在写的 RLHF 教材中 CAI / RLAIF 章节，把本节涉及的几篇 paper 串成 narrative。
- 推荐继续读本教程的 **10.3 节《RLVR 与 DeepSeek-R1》**——理解为什么 reasoning 任务跳出了 RLAIF 的 trilemma，走客观 verifier 路线；以及 **12.2 节《LLM-as-Judge / Pairwise / Reward 评测》**——本节末尾涉及的 judge 可靠性、bias mitigation 在那里集中展开。
