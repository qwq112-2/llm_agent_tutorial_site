---
title: "12.1 通用评测：MMLU / GSM8K / HumanEval / IFEval / Arena"
description: "现代 LLM 评测不是\"跑一个 benchmark 报一个数\"，而是沿着「通用知识 / 数学推理 / 代码 / 指令遵循 / 综合对话」5 大维度，用 MMLU、GSM8K/MATH/AIME、HumanEval/SWE-bench、IFEval、MT-Bench/Arena-Hard/Chatbot Arena 等 benchmark 组合给模型画像——同时要警惕 contamination（训"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：无（建议先了解 6.2 数据管线、8.x SFT 基础）

## 一句话本节讲什么

现代 LLM 评测不是"跑一个 benchmark 报一个数"，而是**沿着「通用知识 / 数学推理 / 代码 / 指令遵循 / 综合对话」5 大维度，用 MMLU(-Pro)、GSM8K/MATH/AIME、HumanEval/SWE-bench、IFEval、MT-Bench/Arena-Hard/Chatbot Arena 等 benchmark 组合给模型画像**——同时要警惕 contamination（训过 test set 导致虚高）、评测实现差异（同一个 MMLU 不同 harness 能差 5-10 分）、judge model bias（GPT-4 评 GPT-4 是循环裁判）这些"看似简单实则全是坑"的工程问题。本节把 5 大维度的代表 benchmark、评测原理、主流工具链（lm-eval-harness / OpenCompass / EvalPlus）、contamination 的检测与防御、以及 reasoning model 时代评测的新挑战讲清楚——读完应当能为一个新训出来的 7B chat 模型设计一份"必跑 benchmark 清单"，并能识别出"MMLU 89 分"这种声明背后的潜在水分。

LLM-as-Judge 的细节（裁判 prompt 设计、bias 校正、Pairwise 与 Reward 评测）是 12.2 的内容，安全 / 红队评测（HarmBench、GCG）是 12.3，本节只聚焦"模型能力到底有多强"这一面。

---

## 1. Mental model（直觉）

### 1.1 为什么不能"只看一个分数"

工业界一个常见的迷思是"我们有个 SOTA 数：MMLU 89"。这句话基本没有信息量，因为：

- **MMLU 高 ≠ chat 好**：MMLU 是 4 选 1 的多选题，模型只要 log p(A/B/C/D) 中选最大的就够了，**不需要会写一句通顺人话**；很多 base model MMLU 很高，但拿来直接 chat 一塌糊涂
- **MMLU 高 ≠ 数学好**：MMLU 里数学只占很小一部分，且都是高中以下；GSM8K / MATH / AIME 才是数学评测的本体
- **HumanEval 高 ≠ SWE-agent 强**：HumanEval 是 164 道独立的单文件 Python 函数题，**不涉及多文件 / 工程上下文 / git diff**；真的 SWE 能力要看 SWE-bench Verified
- **MMLU 高也可能是「训过题」**：benchmark contamination 是 2023 年以来开源 leaderboard 的常态

正确做法是**按维度组合多个 benchmark 形成画像**：

```
                 通用知识     数学推理     代码能力     指令遵循     综合对话
                 ────────     ────────     ────────     ────────     ────────
模型 A (base)    MMLU-Pro 65  MATH 28      HumanEval 0  IFEval 12    MT-Bench 2.1
                 (扎实)       (一般)       (没 SFT)     (没 SFT)     (不会聊)

模型 A (chat)    MMLU-Pro 64  MATH 32      HumanEval 78 IFEval 78    MT-Bench 8.2
                 (略掉)       (CoT 抬升)   (SFT 加成)   (chat 后会跟) (alignment 后)

→ 维度之间常见此消彼长，必须组合看，不能单维度断优劣
```

### 1.2 Auto benchmark vs 人工 / Judge 评测

按"打分主体"分两大流派：

- **Auto benchmark**（规则可验证）：MMLU 看 ABCD、GSM8K 看最终数字、HumanEval 跑 unit test、IFEval 用 regex 检查约束 → **客观、可复现、便宜**，但只能评测"答案有没有标准形式"的题
- **Judge / 人工评测**：MT-Bench / Arena-Hard 用 GPT-4 当 judge 给 1-10 分或 pairwise 选优；Chatbot Arena 直接让人类盲评 Elo → **更接近真实 chat 体验**，但慢、贵、有 bias

实战上**两者必须组合**：用 lm-eval-harness 跑一遍 MMLU/GSM8K/HumanEval/IFEval（一晚上能跑完），用 MT-Bench 或 Arena-Hard 跑一遍 chat 质量（GPT-4 当 judge，几小时能跑完），最后**关键节点**才上 Chatbot Arena 拿 Elo（要等 1-2 个月）。

### 1.3 2024-2026 的 benchmark 老化

2020 提的 MMLU 在 2024+ 顶级模型上已经接近饱和（GPT-4o / Claude-Sonnet-4 / Gemini-2 系列普遍 88-90+），区分度变差，加上严重 contamination → 2024 后**新标准**接棒：

- **MMLU-Pro**（2024）：10 选 1、推理题更多、去 contamination → 取代 MMLU 作为通用知识首选
- **GPQA**（"Google-proof"）：研究生级 STEM，**人类 PhD 也只 60-70 分**，给最强模型留区分空间
- **AIME / Olympiad**：reasoning model 时代的数学主战场（2024 R1 / o1 之后）
- **LiveCodeBench**：实时收题，对抗 contamination
- **SWE-bench (Verified)**：agent 时代的代码黄金标
- **Chatbot Arena**：chat 体验最权威，但慢

HuggingFace 在 2024 年废弃了"老 Open LLM Leaderboard"（MMLU/HellaSwag/ARC 那套），换成 **v2**：MMLU-Pro / GPQA / IFEval / BBH / MATH-LV5 / MUSR——本质就是承认老 benchmark 已经被打饱和或污染。

---

## 2. 5 大维度与代表 benchmark

### 2.1 总表（必背）

| 维度 | 评测什么 | 代表 benchmark | 评测形式 | 现代主流 |
|---|---|---|---|---|
| 通用知识 | 学科常识 | MMLU / **MMLU-Pro** / **GPQA** / C-Eval / CMMLU | 多选题（log-likelihood 或 generate） | MMLU-Pro、GPQA |
| 数学推理 | 应用题 / 竞赛 | GSM8K / MATH / **AIME** / Olympiad | generate + 答案匹配 | AIME、MATH |
| 代码能力 | 写代码 / 修 bug | HumanEval / MBPP / **HumanEval+** / **LiveCodeBench** / **SWE-bench** | unit test 通过率 | LiveCodeBench、SWE-bench Verified |
| 指令遵循 | 严格规则约束 | **IFEval** / FoFo / InfoBench | 规则 / regex 验证 | IFEval |
| 综合对话 | chat 体验 | MT-Bench / **Arena-Hard** / **Chatbot Arena** / AlpacaEval 2.0 | LLM-as-Judge / 人工 Elo | Arena-Hard、Chatbot Arena |

加粗的是 2024-2026 当下最常被引用的版本。

### 2.2 通用知识：MMLU 系列

**MMLU**（Hendrycks 2020）：57 学科（数学、物理、法律、医学、伦理学等），约 14k 道 4 选 1 多选题。评测时常见两种实现：

- **Log-likelihood scoring**（lm-eval-harness 默认）：把 prompt 拼上 4 个候选 "A. xxx / B. xxx / ..."，算模型在每个候选下的 $\log p$，选最大者；**不需要模型真的生成内容**，因此对 base model 也能跑
- **Generation scoring**：让模型自由生成第一个 token，看是不是 A/B/C/D 之一；更接近 chat 模型实际用法，但需要严格的 chat template

两种实现**同一个模型常差 5-10 分**，Llama / Qwen 各家公布的 MMLU 分数有时不可比就是这个原因——比较 MMLU 必须**指定 harness 与 scoring 方式**。

**MMLU-Pro**（Wang 2024, TIGER-Lab）：把 MMLU 中"凭背诵 / 凭直觉"能猜对的题大量删去，保留需要多步推理的题，并把选项从 4 个扩到 10 个 → **猜对的概率从 25% 降到 10%**，且做了 contamination 清洗。当下开源 leaderboard 的事实首选。

**GPQA**（Rein 2023, Google-proof QA）：448 道由 PhD 出题、PhD 也很难、Google 搜索几乎无解的研究生级 STEM 题。当 MMLU-Pro 也被强模型打饱和时，GPQA 是最强模型间区分度最高的通用知识 benchmark 之一。

**C-Eval / CMMLU**：中文版 MMLU，中国厂商必跑。C-Eval 偏学科，CMMLU 偏中国本土知识（包括法律、政治、历史等）。

### 2.3 数学推理

**GSM8K**（Cobbe 2021）：8.5k 道小学水平应用题（"小明买了 3 个苹果……一共多少钱？"）。最大的价值是**首次系统证明 CoT prompting 能让 LLM 数学能力跳一个台阶**——直出答案 acc 30%，加上 "Let's think step by step" 能到 60%+。当下顶级模型 GSM8K 已 95+，区分度低，但仍是 SFT 阶段必跑的 sanity check。

**MATH**（Hendrycks 2021）：12.5k 道高中数学竞赛题（AMC、AIME 等），答案是 LaTeX 表达式（如 $\frac{3}{2}$、$\sqrt{2}+1$）。评测难点在**答案解析**：模型可能输出 `3/2`、`1.5`、`\frac{3}{2}` → 现代用 **Math-Verify**（Hugging Face）这种符号化 verifier 做等价判定，比早期纯字符串匹配宽容很多。

**AIME**（American Invitational Math Exam）：每年 30 道竞赛题，**reasoning model 时代的主战场**。2024 R1 / o1 之后，AIME 24/25 几乎是新模型必报的指标，排行榜上 30-90 跨度极大，区分度最高。

**Olympiad / OlympiadBench**：进阶版数学 + 物理奥林匹克题，最强模型也才 30-50%，给前沿留空间。

评测流程统一：**generate 完整 CoT → 用正则提取 `\boxed{...}` 或 "the answer is X" 里的最终答案 → 用数值/符号 verifier 比对**。max_tokens 必须给足（reasoning model 一题动辄 1-10k token），否则 unfair。

### 2.4 代码

**HumanEval**（Chen 2021, OpenAI Codex 论文）：164 道 Python 函数实现题。每题给函数签名 + docstring，模型补全函数体，跑 3-10 个 unit test，全过算 pass。评测指标 **pass@1**（temperature=0，单次生成）/ **pass@k**（temperature=高，生成 k 次取一次过算过）。

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to
    each other than given threshold.
    """
    # 模型补全这里
```

**HumanEval+ / MBPP+**（EvalPlus, Liu 2023）：原 HumanEval 的 unit test 太弱，很多"错代码"也能 pass → EvalPlus 把每题 test case 扩到原来的 80×，大量"漏网之鱼"被卡掉，HumanEval+ 的分数普遍比 HumanEval 低 10-20。当下报代码能力**应当用 +版本**。

**LiveCodeBench**（Jain 2024）：每月从 LeetCode / AtCoder / Codeforces 实时收题，按时间窗口切片评测 → 模型只能用窗口前数据训，**结构性对抗 contamination**。当下抗污染代码 benchmark 首选。

**SWE-bench / SWE-bench Verified**（Jimenez 2023）：从 12 个真实 Python 开源项目（Django、sympy、scikit-learn 等）的 GitHub issue 抽出 2294 个 bug，给模型完整代码库，让模型产出 patch，跑项目自带的 test → 解决率（resolve rate）。**Verified** 子集是 OpenAI 人工审核过的 500 道高质量题。这是 **Agent 时代的代码黄金标**，2024-2025 各家发布 Coding Agent（Claude Code、Devin、SWE-Agent）都首报 SWE-bench Verified。

**Aider Polyglot**：Aider 工具自带的多语言 code editing 评测，覆盖 Python / JS / Go / Rust / C++ 等 6 种语言的 225 道题，更接近"做一个 PR"的工程语境。

### 2.5 指令遵循：IFEval

**IFEval**（Zhou 2023, Google）：核心思想是**用机器可验证的硬约束清单**直接判定，而不是让 judge 主观打分。题型例如：

- "用恰好 5 个 bullet point 回答" → 程序数 `^\s*[-*]` 行
- "回答必须包含 'sustainable' 这个词" → 程序 grep
- "整个回答全部用大写" → 程序检查 `s == s.upper()`
- "JSON 格式输出，包含 name / age 两个字段" → 程序 json.loads + assert key

每题可能有多个约束，记 strict（全过算过）和 loose（容忍部分大小写差异等）两套准确率。**IFEval 是 alignment 后的必测项**，因为 SFT 训出来的"看起来很顺"的模型经常不真的听话——叫它输出 5 行非要给 7 行，叫它用 JSON 非要包一层 markdown。GPT-4 / Claude 这类前沿模型 IFEval 也只 80-85%，远不饱和。

### 2.6 综合对话：MT-Bench / Arena-Hard / Chatbot Arena

**MT-Bench**（Zheng 2023）：80 道两轮对话题，覆盖 8 个类别（writing / roleplay / reasoning / math / coding / extraction / STEM / humanities）。**用 GPT-4 当 judge**给 1-10 分，最终报平均分。优点：80 道题快、一两小时能跑完；缺点：GPT-4 当 judge 有 length / position / verbosity bias，且**用 GPT-4 评 GPT-4 是循环裁判**。

**Arena-Hard**（Li 2024, LMSYS）：500 道难度筛选过的题，用 **pairwise 与 GPT-4-Turbo baseline 对比**，让 GPT-4 选谁更好，最终输出胜率。比 MT-Bench 更难、区分度更高，是当前自动 chat 评测的事实标准。

**Chatbot Arena**（LMSYS）：**用户在 chat.lmsys.org 盲评两个匿名模型 → 选优 → 用 Bradley-Terry 拟合 Elo**。这是 chat 质量最权威的指标，无法 hack（人类盲评），但慢——一个新模型上线后通常要 1-2 个月才能积累足够投票拿到稳定 Elo。

**AlpacaEval 2.0**（Dubois 2024）：805 道指令，让 LLM judge 与 GPT-4 baseline 比胜率，**带 length-controlled 校正**（修正长回答更容易赢的 bias）。比 MT-Bench 更轻量，开源社区也常用。

---

## 3. 最小代码示例

### 3.1 lm-eval-harness 跑 MMLU / GSM8K（事实标准）

```bash
pip install lm-eval[api]

# 评测本地 HuggingFace 模型在 MMLU 与 GSM8K 上的表现
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16 \
    --tasks mmlu,gsm8k,ifeval \
    --batch_size 8 \
    --device cuda:0 \
    --output_path ./results/

# 评测 vLLM 服务（更快，适合大模型）
lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-3.1-70B-Instruct,tensor_parallel_size=4,gpu_memory_utilization=0.9 \
    --tasks mmlu_pro,gpqa_main_zeroshot,ifeval \
    --batch_size auto \
    --output_path ./results/
```

```python
# 用 Python API 直接跑也行
from lm_eval import simple_evaluate
results = simple_evaluate(
    model="hf",
    model_args="pretrained=Qwen/Qwen2.5-7B-Instruct",
    tasks=["mmlu", "gsm8k_cot"],  # gsm8k_cot 是带 CoT 提示的版本，分数更高
    num_fewshot=5,                 # MMLU 标准是 5-shot
)
print(results["results"]["mmlu"]["acc,none"])
```

**关键点**：(1) `dtype=bfloat16` 与训练时一致，否则数值差异影响分数；(2) MMLU 标准是 5-shot，GSM8K 一般 5-shot CoT；(3) chat 模型必须配 chat template，base model 不需要，**两者 MMLU 实现不同**（base 用 log-likelihood，chat 多用 generation）。

### 3.2 HumanEval 自定义评测脚本

```python
# 简化版，真实工程用 EvalPlus 库
import json, signal
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset

def run_with_timeout(code, timeout=5):
    """在 sandbox 进程里跑代码，超时算失败"""
    def handler(signum, frame): raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        exec(code, {})  # 真实生产用 docker / nsjail
        return True
    except Exception:
        return False
    finally:
        signal.alarm(0)

def eval_humaneval(model_generate_fn):
    ds = load_dataset("openai_humaneval")["test"]
    correct = 0
    for ex in ds:
        # 让模型补全函数体（temperature=0 算 pass@1）
        completion = model_generate_fn(ex["prompt"], temperature=0.0, max_tokens=512)
        full_code = ex["prompt"] + completion + "\n" + ex["test"] + f"\ncheck({ex['entry_point']})"
        if run_with_timeout(full_code, timeout=5):
            correct += 1
    return correct / len(ds)  # 这就是 pass@1
```

**踩坑提示**：(1) 模型生成完代码后常带 markdown ```python``` 包裹，要 strip；(2) 必须在隔离环境跑（exec 不安全，生产用 docker）；(3) pass@k 要生成 k 次取一次过，不是 greedy 重复 k 次。

### 3.3 MT-Bench 风格 LLM-as-Judge

```python
import openai
client = openai.OpenAI()

JUDGE_PROMPT = """[Instruction]
You are an impartial judge. Rate the assistant's response 1-10.
Consider: helpfulness, relevance, accuracy, depth, creativity, level of detail.
Avoid position / length bias.

[Question]
{question}

[Assistant Response]
{response}

Output ONLY a number 1-10, nothing else."""

def judge(question, response, judge_model="gpt-4o-2024-11-20"):
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, response=response)}],
        temperature=0.0,
    )
    return float(resp.choices[0].message.content.strip())

# 实战：跑 MT-Bench 80 题
scores = [judge(q, my_model_answer(q)) for q in mt_bench_questions]
print(f"MT-Bench score: {sum(scores)/len(scores):.2f}")
```

注意这是简化版，真正 MT-Bench 是**两轮对话**且用 reference answer 做 anchor。LLM-as-Judge 的 bias 校正与 pairwise 形式细节是 12.2 节内容。

### 3.4 简短 contamination check（n-gram 匹配）

```python
from datasets import load_dataset

def ngrams(text, n=13):
    toks = text.split()
    return {tuple(toks[i:i+n]) for i in range(len(toks)-n+1)}

# 把 benchmark 题做成 n-gram 集合
mmlu = load_dataset("cais/mmlu", "all", split="test")
bench_ngrams = set()
for ex in mmlu:
    bench_ngrams |= ngrams(ex["question"])

# 流式扫训练 corpus，统计命中率
hit, total = 0, 0
for doc in load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True):
    total += 1
    if ngrams(doc["text"]) & bench_ngrams:  # 任一 13-gram 命中
        hit += 1
    if total >= 100_000: break
print(f"Contamination rate: {hit/total:.4%}")  # 工业界 13-gram 阈值常用
```

13-gram 是常见阈值：太短（5-gram）误报多，太长（30-gram）漏报多。这是 OpenAI / DeepMind contamination report 的标准做法之一。

---

## 4. 工程踩坑与经验

- ❗ **MMLU 分数不同实现差 5-10 分**：log-likelihood scoring（base model 友好）vs generation scoring（chat model 友好）vs 不同 prompt 格式（"Question: / Answer:" vs chat template）会让同一个模型分数差 5-10，跨模型比较**必须同一 harness 同一 setting**，最好直接用 lm-eval-harness 重跑 baseline 而不是抄各家技报数字
- ❗ **HumanEval pass@1 受 temperature 影响极大**：temperature=0 是常报的 pass@1，但 temperature=1.0 + pass@10 在某些模型上能比 pass@1 高 20+ 分；公平比较要么都 greedy（pass@1）要么都 pass@k 同 k，不要混报
- ❗ **benchmark contamination 普遍存在**：现代很多开源模型为冲榜在训练数据里混入 MMLU / GSM8K 改写题，score 虚高；遇到"小模型 MMLU 反超大模型"先怀疑污染。验证方法：用 LiveCodeBench、MMLU-Pro 这种较新或抗污染 benchmark 复测，或自己做 n-gram check
- ❗ **MT-Bench / GPT-4 judge 的多种 bias**：(a) position bias（同一回答放 A 位比放 B 位胜率高 5%+）；(b) length bias（长回答更容易被认为好）；(c) verbosity / 自我偏好（GPT-4 评 GPT-4 输出会偏高）；(d) 风格匹配 bias。Arena-Hard 用 swap 顺序两次取均值缓解 position bias，AlpacaEval 2.0 用 length-controlled 校正长度 bias，但**用 GPT-4 评 GPT-4 永远是循环裁判，需要交叉用 Claude / Gemini judge 取均值**
- ❗ **Chatbot Arena 慢**：新模型上线后 1-2 个月才能积累几千投票拿到 95% 置信区间的 Elo；急于发版的话 Arena-Hard 是更快的 proxy（与 Arena Elo 相关性 0.9+）
- ❗ **不要单看一个 benchmark**：MMLU 高不代表 chat 好（base model 也能 MMLU 高但不会聊天），HumanEval 高不代表 SWE-agent 强（HumanEval 是单文件 Python 函数，SWE-bench 是多文件真实 issue），IFEval 高也不代表 chat 自然（严格遵循约束的模型常显得机械）；**5 个维度的 benchmark 必须组合看**
- ❗ **评测时的 prompt 格式必须与生产一致**：如果生产用 chat template + system prompt，评测就也要用同一份 chat template；用错 template（比如把 chat 模型当 base 跑或者用错 special token）能让 MMLU 掉 10+ 分，IFEval 掉 30+ 分
- ❗ **Reasoning model 的 benchmark 必须放开 max_tokens**：R1 / o1 / DeepSeek-R1 类模型一题 thinking 动辄 5-50k token，max_tokens 设太小会"思考到一半被截断"导致没答案 → 0 分，**完全不公平**。AIME / Olympiad 评 reasoning 模型一般给 32k-64k max_tokens
- ❗ **chat template 的 BOS / EOS / system 顺序错一个，IFEval 就崩**：很多模型对 chat template 敏感，少加一个 `<|im_start|>` 或者 system 放错位置，模型就退化成 base 行为；评测前**必须用一两个手工 case 验证 chat template 渲染正确**
- ❗ **GSM8K / MATH 的答案解析容易漏**：模型输出 `\boxed{1.5}`、"The answer is 1.5"、"= 1.5\n\nDone." 等多种格式，正则要覆盖；MATH 用 Math-Verify 做符号等价判定（识别 $\frac{3}{2}$ 与 `1.5` 等价），比纯字符串匹配宽容得多

---

## 5. 经典 paper

- **Hendrycks et al., 2020 — Measuring Massive Multitask Language Understanding (MMLU)** — LLM 通用能力评测的奠基之作，57 学科 + 4 选 1 的格式定义了 2020-2024 几乎所有 leaderboard 的"通用知识"列。读它能理解为什么"多学科多选题"能成为事实标准，以及为什么 2024 年它需要被 MMLU-Pro 取代。
- **Cobbe et al., 2021 — Training Verifiers to Solve Math Word Problems (GSM8K)** — GSM8K 数据集首发，更重要的是首次系统证明 **CoT prompting + verifier reranking** 能让 LLM 在数学题上跨过一个台阶；本节的"为什么数学评测要 generate + 答案匹配"以及 Module 9-10 RLHF / PRM / RLVR 的源头都在这。
- **Chen et al., 2021 — Evaluating Large Language Models Trained on Code (HumanEval)** — Codex 论文，首次提出 **functional correctness（unit test 通过率）** 作为代码评测，定义了 pass@k 这个现代代码评测的标准指标。读它能理解为什么代码评测一定要跑 unit test 而不是看 BLEU / 字符串相似度。
- **Zheng et al., 2023 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — MT-Bench、Chatbot Arena 同篇论文，首次系统讨论 **LLM-as-Judge 的 position / verbosity bias 与 GPT-4 judge 与人类一致率（达 80%+）**。本节 12.2 的核心地基就在这里，强烈建议精读。
- **Zhou et al., 2023 — IFEval: Instruction-Following Eval** — 首次提出"用机器可验证的硬约束代替主观打分"评测指令遵循，alignment 时代必读。
- 加分：**Wang et al., 2024 — MMLU-Pro**（10 选 1 抗污染版）、**Jain et al., 2024 — LiveCodeBench**（实时收题抗污染）、**Jimenez et al., 2023 — SWE-bench**（真实 GitHub issue → patch 的 agent 黄金标）

---

## 6. 自测与面试题

**Q1（体系）：** 列出现代 LLM 评测的 5 大维度，每个维度举一个代表 benchmark，并说明该 benchmark 的评测形式。

<details>
<summary>Answer sketch</summary>

- **通用知识**：MMLU / MMLU-Pro / GPQA → 多选题，按 log-likelihood 选最大或 generate 出 A/B/C/D
- **数学推理**：GSM8K / MATH / AIME → generate 完整 CoT，正则提最终答案，数值/符号 verifier 比对
- **代码能力**：HumanEval / SWE-bench → 生成代码跑 unit test，pass@k
- **指令遵循**：IFEval → 程序化规则 / regex 验证硬约束（行数、关键词、JSON 格式等）
- **综合对话**：MT-Bench / Arena-Hard / Chatbot Arena → LLM-as-Judge 打分或 pairwise，或人类盲评 Elo
- 加分：能补充"自动 vs 人工 / Judge"的两大流派区别

</details>

**Q2（contamination）：** 你怀疑某开源模型公布的 MMLU 89 分有水分（同 size 的其他模型都 70 左右），列出 3 种 verify 方法。

<details>
<summary>Answer sketch</summary>

- **换抗污染 benchmark 复测**：用 MMLU-Pro / GPQA / LiveCodeBench 这种较新或动态收题的 benchmark 重跑，如果分数掉到与同 size 模型一致 → 高概率原 MMLU 训过题
- **n-gram 比对训练数据**：把 MMLU test 题切 13-gram，如果训练 corpus 公开（或部分公开），扫命中率；命中率显著高于 baseline 即污染
- **统一 harness 重跑 baseline + 该模型**：用 lm-eval-harness 同 setting（5-shot、log-likelihood、相同 prompt 格式）跑该模型与几个公认未污染的同 size 模型，看相对差距是否仍是 10-20 分；常见情况是该模型用了奇怪的 prompt / generation scoring 把分数刷上去
- **检查官方 report 的实现细节**：是否报了 contamination check（Llama 3 / DeepSeek 系列都会报），未报的可疑度高
- **加分**：用 perplexity / membership inference 等方法检测训练 leakage（如 min-K% prob 攻击）

</details>

**Q3（实战）：** 你训了一个 7B chat 模型用于通用助手场景，列出至少 4 个必跑 benchmark，并说明各自评测什么以及为什么必跑。

<details>
<summary>Answer sketch</summary>

参考"5 维度组合"思路：

- **MMLU-Pro**（或 MMLU + GPQA）→ 通用知识；必跑因为这是所有 LLM 通用能力的事实 baseline，与同 size 同代模型比较的入场券
- **GSM8K / MATH**（数学）→ 验证 CoT 与数学推理没掉，SFT 数据如果数学比例不够这里会显著低
- **HumanEval+ / MBPP+** 或 **LiveCodeBench**（代码）→ 即使非 code 模型，常见用法也涉及代码生成，不能掉太多
- **IFEval**（指令遵循）→ chat 模型必测，验证 SFT/RLHF 没把听话能力训坏
- **MT-Bench 或 Arena-Hard**（综合 chat）→ LLM-as-Judge 验证整体 chat 体验；7B 模型 Chatbot Arena 周期太长可以先跳
- **C-Eval / CMMLU**（如果做中文）→ 中文知识必跑
- 加分：能说明**为什么这套清单覆盖 5 维度**、**评测要与生产 chat template 一致**、**和上一版本模型回归对比**而不只看绝对值

</details>

---

## 7. 延伸阅读

- [lm-evaluation-harness 仓库](https://github.com/EleutherAI/lm-evaluation-harness) — EleutherAI 维护，事实上是**开源 LLM 评测的标准入口**，所有主流 benchmark（MMLU / GSM8K / IFEval / BBH / GPQA / MMLU-Pro / Math / TruthfulQA…）都已实现，HuggingFace Open LLM Leaderboard v2 也是基于它
- [OpenCompass 文档](https://opencompass.org.cn/) — 上海 AI Lab 维护，**中英文 benchmark 综合性最好**，跑 C-Eval / CMMLU / SuperCLUE 等中文榜常用
- [EvalPlus](https://github.com/evalplus/evalplus) — HumanEval+ / MBPP+ 与 LiveCodeBench 的工具仓库，代码评测必备
- [SWE-bench 官网](https://www.swebench.com/) — agent 时代代码评测的官方榜单与运行工具
- [Chatbot Arena Leaderboard](https://lmarena.ai/) — 人类盲评 Elo 排行榜，chat 质量最权威的 reference
- [HuggingFace Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — 当下开源模型综合 leaderboard
- [LiveCodeBench](https://livecodebench.github.io/) — 抗污染代码 benchmark 官网
- 推荐继续读 **12.2 LLM-as-Judge / Pairwise / Reward 评测**（深入 judge model 的 bias 校正与 pairwise 设计）、**12.3 安全评测与红队**（HarmBench / GCG / jailbreak 谱系）
