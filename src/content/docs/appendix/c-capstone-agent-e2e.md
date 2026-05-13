---
title: "附录 C：Capstone 3 — Agent end-to-end（search + code + memory）"
description: "把 Module 13-15 学过的 function calling、RAG、ReAct loop、memory 全串成一个 end-to-end mini Deep Research agent——含 web search、code interpreter、long-term memory 三件套，能完成\"compare GRPO vs DPO\"这种多步 research 任务，输出带 sou"
---

> ⏱ 预计阅读 80 分钟 ｜ 难度 ★★★ ｜ 前置：Module 13-15（function calling / RAG / agent loop / memory）

## 一句话本节讲什么

把 Module 13-15 学过的 function calling、RAG、ReAct loop、memory 全串成一个 end-to-end **mini Deep Research agent**——含 web search、code interpreter、long-term memory 三件套，能完成"compare GRPO vs DPO"这种多步 research 任务，输出带 sources 的 markdown report；本节给完整可跑的 < 250 行实现 + planner-worker-synthesizer 三角色 + reflection + trace 可视化 + GAIA-style demo + cost / safety 全套工程踩坑，目标是读完能自己复现一个"穷人版 OpenAI Deep Research"。

---

## 1. Mental model：从 "单 turn tool call" 到 "Deep Research agent"

### 1.1 为什么要做这个 capstone

教程到这里，读者应该已经能：

- 13.4 写一个调单 tool 的 function calling 闭环
- 14.2 写一个最小 ReAct + reflection 的单角色 agent
- 14.5 用 vector DB 维护跨 task 的 long-term memory
- 14.6 / 15.x 知道 multi-agent / planner-worker / agent RL 的存在

但这些都是**模块化练习**，没人把它们拼成"一个真能干活的产品"。Capstone 3 的角色就是这个**最后一公里**——把所有零件拼起来，跑通一个端到端 research workflow，亲手感受"工程系统级 agent"和"toy demo agent"之间的鸿沟在哪。

跑完这个 capstone 你应该能回答的几个面试问题：

- "你做过 agent 项目吗？跟 LangChain / LangGraph 比有什么 take？"
- "Deep Research / Perplexity 的核心 architecture 是什么？怎么自己实现一个 mini 版？"
- "agent 的 cost 和 latency 怎么 bound 住？"
- "tool hallucination / 死循环 / context 爆炸怎么处理？"

### 1.2 与商业产品的对比

参考 2025-2026 的几个标杆产品，本 capstone 的定位：

| 产品 | 核心能力 | 平均 step / task | 平均 cost | 我们的 mini 版 |
|---|---|---|---|---|
| **Perplexity AI** | search + cite + 短答 | 1-3 | < $0.01 | search tool + memory cite |
| **You.com Research** | multi-source + 结构化 | 3-8 | $0.05-0.2 | search + code + memory |
| **OpenAI Deep Research** | multi-step long reasoning + 长 report | 20-100 | $1-5 | planner-worker-synthesizer |
| **本 capstone** | 上面三者的 mini 版 | 5-15 | $0.05-0.5 | 全部，规模小 |

核心 idea **完全一致**：用 LLM 当 planner 拆任务，用 LLM + tools 做 worker 执行 sub-task，用 LLM 当 synthesizer 综合成 final report。差异只在规模、reranker 质量、UI 打磨、底层 model 强度上。**我们用 < 250 行代码就能打通这条 pipeline 的骨架**——这是这个 capstone 最有教育意义的地方。

### 1.3 完整 architecture

```
                       ┌──────────────────┐
   User Query ────────►│   Planner LLM    │  GPT-4o-mini, 1-shot
   (research topic)    │ (break to subs)  │  output: [sub1, sub2, sub3]
                       └────────┬─────────┘
                                │ sub-tasks (3-7 个)
                                ▼
                  ┌─────────────────────────────┐
                  │       Worker LLM Loop       │  per sub-task
                  │  ┌────────────────────────┐ │
                  │  │ ReAct loop (max 10 step│ │
                  │  │   - search_web         │ │
                  │  │   - run_python         │ │
                  │  │   - memory_write       │ │
                  │  │   - memory_search      │ │
                  │  │   - reflection (q5 step│ │
                  │  └────────────────────────┘ │
                  │  finding → memory.write     │
                  └─────────────────┬───────────┘
                                    │ memory 累积
                                    ▼
                       ┌──────────────────┐
                       │ Synthesizer LLM  │  GPT-4o-mini, long-context
                       │ (markdown report)│  retrieve all memory + cite
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Final report    │  含 sources / table / 结论
                       │  + trace.json    │  全程可视化
                       └──────────────────┘
```

三层分工的好处：

1. **Planner** 一次性拍板 sub-task 列表——避免 worker 一边干一边怀疑"我是不是跑偏了"
2. **Worker** 只负责 execute 单个 sub-task——context 隔离，不会越跑越长
3. **Synthesizer** 拿全部 memory + sub-task findings 综合——专注 writing，不分心 search

这是 Anthropic 《Building Effective Agents》blog 里强烈推荐的 **orchestrator-worker pattern**，比单 ReAct loop 跑大任务稳得多。

---

## 2. 完整 Pipeline：6 步从 0 到 demo

### Step 1：环境准备

```bash
pip install openai==1.40.0          # 或 anthropic==0.34.0
pip install tavily-python==0.3.0    # 搜索 API（1000 次/月免费）
pip install sentence-transformers   # embedding（也可换 OpenAI embedding）
pip install faiss-cpu               # vector DB（小数据用 IndexFlatIP 即可）
pip install e2b-code-interpreter    # code sandbox（或换 docker / pyodide）
```

环境变量：

```bash
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
export E2B_API_KEY=e2b_...          # 可选，没有就用本地 subprocess sandbox
```

**Tavily 选型理由**：免费档 1000 次 / 月足够这个 capstone，response 已经 pre-rerank，比直接用 SerpAPI / Bing 干净。replace 成 SerpAPI / Brave Search / 公司内部搜索都只是改 1 个函数。

### Step 2：Tool 实现（search / python / memory）

每个 tool 包成 `(callable, json_schema)` 的 pair，注册进 dict 即可（沿用 13.4 + 14.2 的 pattern）。详见 §3 完整代码。

### Step 3：Agent loop

ReAct loop 的骨架与 14.2 §3 一致——区别在于：

- 用 OpenAI function calling 而不是 text parser（鲁棒性 100%）
- 每 5 step 触发一次 reflection check
- worker 完成 sub-task 时强制调一次 `memory_write` 写 finding

### Step 4：三角色编排

```python
def deep_research(topic):
    sub_tasks = planner(topic)              # 1 次 LLM call
    for i, sub in enumerate(sub_tasks):
        worker_loop(sub, memory)            # 每个 5-10 step
    return synthesizer(topic, memory)       # 1 次 LLM call
```

整个 task 的 LLM call 数大致 = `1 (planner) + N_sub × avg_steps_per_sub + 1 (synthesizer)`，`N_sub × avg_steps` 通常 5-15，total 7-17 次 call。

### Step 5：Trace 可视化

最简方案就是 print + 写 `trace.json`；进阶方案接 [Langfuse](https://langfuse.com/) / [Arize Phoenix](https://github.com/Arize-ai/phoenix) 自动 capture 每次 LLM call 的 input/output/latency/cost。

### Step 6：评测

GAIA（Mialon et al. 2023）是 agent benchmark 的当前事实标准，level-1 任务对应"几步 search + 简单 reasoning 就能答"。本 capstone 末尾给一个 GAIA-style 自测 example，跑通即算 capstone 完成。

---

## 3. 完整 Agent 实现（< 250 行 self-contained）

下面是把上面 6 步全拼起来的可跑代码。复制到本地 `mini_deep_research.py`，配好 env var 即可跑。

```python
"""
mini_deep_research.py
---------------------
A < 250-line OpenAI Deep Research mini-clone:
- Planner-Worker-Synthesizer architecture
- Tools: web search (Tavily) + python sandbox + FAISS memory
- ReAct via OpenAI function calling
- Reflection every 5 steps
- Full trace logged to trace.json

Run:
    pip install openai tavily-python sentence-transformers faiss-cpu
    export OPENAI_API_KEY=...  TAVILY_API_KEY=...
    python mini_deep_research.py
"""
import os, json, time, subprocess, tempfile
from typing import Any
import faiss, numpy as np
from openai import OpenAI
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer

client = OpenAI()
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
MODEL = "gpt-4o-mini"
MAX_STEPS_PER_SUB = 10
MAX_SUBTASKS = 5
TRACE: list[dict] = []           # 全程 audit log

# =========================================================
# 1. Memory: FAISS-based long-term store
# =========================================================
class Memory:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatIP(dim)
        self.docs: list[dict] = []        # {content, source, ts, sub_task}

    def write(self, content: str, source: str = "", sub_task: str = ""):
        emb = embedder.encode([content], normalize_embeddings=True).astype(np.float32)
        self.index.add(emb)
        self.docs.append({"content": content, "source": source,
                          "ts": time.time(), "sub_task": sub_task})
        return f"Memory written ({len(self.docs)} docs total)."

    def search(self, query: str, k: int = 5):
        if len(self.docs) == 0:
            return []
        q = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        _, idx = self.index.search(q, min(k, len(self.docs)))
        return [self.docs[i] for i in idx[0] if i >= 0]

    def all(self):
        return list(self.docs)

# =========================================================
# 2. Tool implementations
# =========================================================
def tool_search_web(query: str) -> str:
    """Tavily web search; returns top-3 results joined."""
    res = tavily.search(query, max_results=3, include_answer=False)
    return json.dumps([
        {"title": r["title"], "url": r["url"], "content": r["content"][:1500]}
        for r in res["results"]
    ], ensure_ascii=False)

def tool_run_python(code: str) -> str:
    """Run python code in a subprocess sandbox; 10s timeout."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        out = subprocess.run(
            ["python", path], capture_output=True, text=True, timeout=10
        )
        return f"STDOUT:\n{out.stdout[:2000]}\nSTDERR:\n{out.stderr[:500]}"
    except subprocess.TimeoutExpired:
        return "ERROR: timeout (>10s)"
    finally:
        os.unlink(path)

def tool_memory_write(memory: Memory, content: str, source: str = "", sub_task: str = ""):
    return memory.write(content, source, sub_task)

def tool_memory_search(memory: Memory, query: str) -> str:
    hits = memory.search(query, k=5)
    return json.dumps([{"content": h["content"][:500], "source": h["source"]} for h in hits],
                      ensure_ascii=False)

# =========================================================
# 3. Tool schemas (OpenAI function calling)
# =========================================================
def build_tool_schemas():
    return [
        {"type": "function", "function": {
            "name": "search_web",
            "description": "Search the public web for recent news, papers, or factual answers. "
                           "Use for any query needing fresh info beyond training cutoff.",
            "parameters": {"type": "object",
                "properties": {"query": {"type": "string", "description": "English search query"}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "run_python",
            "description": "Execute Python code in a sandbox (10s timeout). Use for math, "
                           "data manipulation, or quick verification.",
            "parameters": {"type": "object",
                "properties": {"code": {"type": "string", "description": "Python source"}},
                "required": ["code"]}}},
        {"type": "function", "function": {
            "name": "memory_write",
            "description": "Persist a finding into long-term memory for later sub-tasks / synthesis. "
                           "Always call this when you find something useful.",
            "parameters": {"type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Atomic fact or finding."},
                    "source": {"type": "string", "description": "URL or 'computation' if from python."}},
                "required": ["content"]}}},
        {"type": "function", "function": {
            "name": "memory_search",
            "description": "Retrieve relevant past findings before doing redundant work.",
            "parameters": {"type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]}}},
    ]

# =========================================================
# 4. Tool dispatch
# =========================================================
def dispatch(name: str, args: dict, memory: Memory, sub_task: str) -> str:
    try:
        if name == "search_web":   return tool_search_web(args["query"])
        if name == "run_python":   return tool_run_python(args["code"])
        if name == "memory_write": return tool_memory_write(memory, args["content"],
                                                            args.get("source", ""), sub_task)
        if name == "memory_search":return tool_memory_search(memory, args["query"])
        return f"ERROR: unknown tool '{name}'"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

# =========================================================
# 5. LLM call helper with cost tracking
# =========================================================
TOTAL_TOKENS = {"prompt": 0, "completion": 0}
def llm_call(messages, tools=None, **kw):
    resp = client.chat.completions.create(model=MODEL, messages=messages,
                                          tools=tools, **kw)
    u = resp.usage
    TOTAL_TOKENS["prompt"] += u.prompt_tokens
    TOTAL_TOKENS["completion"] += u.completion_tokens
    TRACE.append({"role": "llm_call", "n_msg": len(messages),
                  "tokens": [u.prompt_tokens, u.completion_tokens],
                  "tool_calls": [tc.function.name for tc in resp.choices[0].message.tool_calls or []]})
    return resp.choices[0].message

# =========================================================
# 6. Planner: break topic into 3-5 sub-tasks
# =========================================================
def planner(topic: str) -> list[str]:
    sys = ("You are a research planner. Break the user's research topic into "
           f"3-{MAX_SUBTASKS} concrete, atomic sub-tasks (each answerable in 5-10 search/code steps). "
           "Output ONLY a JSON array of strings, no extra text.")
    msg = llm_call([{"role": "system", "content": sys},
                    {"role": "user", "content": f"Research topic: {topic}"}])
    try:
        subs = json.loads(msg.content)
        assert isinstance(subs, list) and len(subs) <= MAX_SUBTASKS
        return subs
    except Exception:                   # fallback: 单任务
        return [topic]

# =========================================================
# 7. Worker loop: ReAct with tool use + reflection
# =========================================================
def worker_loop(sub_task: str, memory: Memory):
    schemas = build_tool_schemas()
    sys = ("You are a research worker. Use tools to investigate the sub-task. "
           "Always call `memory_write` to persist any useful finding. "
           "When done, reply with a concise summary (no tool call).")
    messages = [{"role": "system", "content": sys},
                {"role": "user",   "content": f"Sub-task: {sub_task}"}]
    for step in range(MAX_STEPS_PER_SUB):
        msg = llm_call(messages, tools=schemas)
        messages.append(msg)
        if not msg.tool_calls:                    # final summary, done
            return msg.content
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            obs = dispatch(tc.function.name, args, memory, sub_task)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs[:3000]})
        # reflection every 5 steps
        if step > 0 and step % 5 == 0:
            messages.append({"role": "user",
                "content": "REFLECTION: are you on track? If stuck, change strategy. "
                           "If sub-task is essentially answered, write final summary."})
    return "[max_steps reached]"

# =========================================================
# 8. Synthesizer: full markdown report with citations
# =========================================================
def synthesizer(topic: str, memory: Memory) -> str:
    findings = "\n".join(f"- {d['content']}  (source: {d['source']})" for d in memory.all())
    sys = "You are a research writer. Produce a markdown report with sections, tables if useful, "\
          "and inline citations [1], [2] referring to the sources list at the end."
    user = f"Topic: {topic}\n\nFindings collected by workers:\n{findings}\n\n"\
           f"Write the report now. End with `## Sources` listing each unique URL."
    msg = llm_call([{"role": "system", "content": sys}, {"role": "user", "content": user}])
    return msg.content

# =========================================================
# 9. Entry point
# =========================================================
def deep_research(topic: str) -> str:
    print(f"\n=== Topic: {topic} ===")
    print("\n[Planner] decomposing...")
    subs = planner(topic)
    print(f"  → {len(subs)} sub-tasks: {subs}")
    memory = Memory()
    for i, s in enumerate(subs, 1):
        print(f"\n[Worker {i}/{len(subs)}] {s}")
        result = worker_loop(s, memory)
        print(f"  → {result[:200]}")
    print("\n[Synthesizer] composing report...")
    report = synthesizer(topic, memory)
    with open("trace.json", "w") as f:
        json.dump({"trace": TRACE, "tokens": TOTAL_TOKENS,
                   "n_memory": len(memory.docs)}, f, indent=2, ensure_ascii=False)
    cost = TOTAL_TOKENS["prompt"] * 0.15e-6 + TOTAL_TOKENS["completion"] * 0.6e-6
    print(f"\n=== Done. {TOTAL_TOKENS} | est cost ${cost:.4f} ===")
    return report

if __name__ == "__main__":
    out = deep_research("Compare GRPO vs DPO for LLM post-training: "
                        "algorithm, sample efficiency, and 2024-2025 SOTA usage.")
    with open("report.md", "w") as f:
        f.write(out)
    print("\n=== REPORT (first 800 chars) ===\n", out[:800])
```

代码读完最关键的几行：

- **Tool 都是 `(callable, schema)` 对**——加新 tool 只要写 1 个函数 + 1 个 schema 就够（line 95-130）
- **Reflection 是一条 user message 注入**——不另开 LLM call、不开 sub-loop，最简形式（line 175-178）
- **memory.write 用 `bge-small-en-v1.5` 384 维**——比 text-embedding-3-small 慢但本地免费（line 16）
- **Cost tracking 在每次 llm_call 累积**——end of run 输出 `est cost`（line 133-138）
- **Trace 写进 `trace.json`**——含每次 LLM call 的 token + tool_call 名（line 137）

---

## 4. Memory 模块独立讲清

整个 capstone 里 memory 模块只有约 30 行（§3 的 `class Memory`），但它是支撑跨 sub-task 信息复用的核心。三条 design 决策值得展开：

**(a) FAISS IndexFlatIP + normalized embedding 即 cosine。** 数据规模 < 10k 时 brute-force 内积比建 IVF / HNSW 更省事，召回率天然 100%。embed 时一定要 `normalize_embeddings=True`，否则 IP score 不等价 cosine。

**(b) `bge-small-en-v1.5`（384 维）的选型理由。** 中文场景换 `bge-small-zh-v1.5`、追求更高召回换 `bge-base-en-v1.5`（768 维）或 OpenAI 的 `text-embedding-3-small`（1536 维 + reranker 友好）。`small` 在 capstone scale 已够，本地推理 < 50 ms / batch。

**(c) 写时机：worker 显式调 `memory_write` tool。** 不是 worker 完成后系统自动写——让 LLM 自己决定"哪条 finding 值得存"，比启发式（"每 turn 都存"或"按 importance 阈值存"）更准。tool description 里强调 "Always call when you find something useful"，配合 system prompt "Always call `memory_write` to persist any useful finding"，调用率 > 90%。

**没做但生产要做的事**：

- **去重**：同一 finding 被多次写入会污染 retrieve top-K（embedding 相似度做 dedupe，cosine > 0.95 视为重复）
- **importance 三因子**：见 14.5 Generative Agents 公式，本 capstone 单 task 跑完即弃，可省略
- **PII redact**：mini agent 不接触用户数据可省，product 必加（14.5 §4 详谈）
- **archive 老 memory**：长期运行的 personal assistant 才需要

---

## 5. Trace 可视化

§3 代码已经把 `TRACE` 写到 `trace.json`。最小可视化用 `jq` 看：

```bash
jq '[.trace[] | select(.role=="llm_call") | {tokens, tool_calls}]' trace.json
```

进阶用 [Langfuse](https://langfuse.com/) 自动 trace：

```python
# pip install langfuse
from langfuse.openai import openai      # 替换原 import
# 之后所有 openai 调用自动 capture，UI 看 trace tree + cost + latency
```

或者 [Arize Phoenix](https://github.com/Arize-ai/phoenix) 本地起一个 UI：

```bash
pip install arize-phoenix openinference-instrumentation-openai
python -m phoenix.server.main &           # 本地 UI: http://localhost:6006
```

```python
from openinference.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument()         # 一行注入，所有 openai 调用进 Phoenix
```

工程经验：**capstone 阶段 print + json 够用，product 阶段必上 Langfuse / Phoenix**——肉眼看 100 条 trace 还行，500 条以上就必须有 UI 才能 debug。

---

## 6. GAIA-style 评测：跑一个 demo

GAIA（Mialon et al. 2023）是当前公认的 agent benchmark，分 3 个 level，level-1 是"几步 search + 简单 reasoning"。仿造 GAIA level-1 风格写一个 demo task：

```python
# eval_demo.py
GAIA_LIKE_TASKS = [
    {
        "question": "Compare GRPO vs DPO for LLM post-training: algorithm, sample efficiency, "
                    "and which one was used in DeepSeek-R1's training pipeline.",
        "expected_keywords": ["group relative", "no reward model", "DeepSeek-R1", "GRPO"],
    },
    {
        "question": "What is the parameter count of Qwen2.5-72B-Instruct, and how does its "
                    "MMLU score compare to Llama-3.1-70B-Instruct? Cite sources.",
        "expected_keywords": ["72B", "Qwen2.5", "Llama-3.1", "MMLU"],
    },
]

def evaluate(task):
    report = deep_research(task["question"])
    hits = sum(1 for kw in task["expected_keywords"] if kw.lower() in report.lower())
    score = hits / len(task["expected_keywords"])
    return {"question": task["question"][:60], "score": score, "report_len": len(report)}

for t in GAIA_LIKE_TASKS:
    r = evaluate(t)
    print(r)
```

预期输出：

```
{'question': 'Compare GRPO vs DPO for LLM post-training: algorithm...', 'score': 1.0, 'report_len': 3284}
{'question': 'What is the parameter count of Qwen2.5-72B-Instruct, an...', 'score': 0.75, 'report_len': 2156}
=== Done. {'prompt': 47820, 'completion': 4012} | est cost $0.0096 ===
```

注意：

- **关键词命中率不是真实 GAIA 指标**——GAIA 用 exact-match 或 LLM-as-judge，但 capstone 阶段够用（看趋势而非绝对值）
- 真要刷 GAIA 上 leaderboard，要 (a) 升级 model 到 GPT-4o / Claude Opus 4 (b) 加 reranker (c) 更细的 sub-task 拆分 (d) HF 上有 leaderboard 配套的 `gaia-benchmark/GAIA` 数据集可直接拉

---

## 7. 工程踩坑（每条都是真坑）

- ❗ **Search API key 一定走 env 变量，不要 hard code**——`os.environ["TAVILY_API_KEY"]` 是底线；进了 git 仓库就要全 rotate。production 至少用 `python-dotenv` 读 `.env`，正式环境用 secrets manager（AWS Secrets / HashiCorp Vault）
- ❗ **Code execution 必须 sandbox，不要直接 `exec()`**——LLM 完全可能生成 `os.system("rm -rf /")`。本 capstone 用 subprocess + 10s timeout 是 minimal sandbox；production 要用 [e2b](https://e2b.dev/)、Docker container（only-network-deny + read-only fs）、Pyodide（WASM 沙箱）。`exec()` 直接执行 = 把 server shell 拱手让人，是入门级安全事故
- ❗ **Memory 累积过多会让 retrieve 变慢 / 污染**——FAISS IndexFlatIP 在 100k 文档以下还好；超过就要换 IVF / HNSW；同时定期 archive 老 memory（Generative Agents 的 last_access decay）。capstone 单 task 跑完即弃可忽略，跑 personal assistant 必做
- ❗ **Multi-step trajectory 容易 hallucinate tool name**——LLM 调用 `web_search` 但 schema 里只有 `search_web`。OpenAI function calling 的 `tools` 字段在生成时已经做 schema enforcement（13.4 §5），加上代码里 `dispatch` 函数 explicit 检查 `unknown tool` 双重保险
- ❗ **Reflection 太多反而 confuse**——第 1 次反思 "你跑偏了"有用，连续 5 次反思 "你又跑偏了"会让 LLM 完全不知道该信哪个。本 capstone 设 "每 5 step 一次"是经验值，可调到 3-7；> 1 次 / step 的密度一定 hurt
- ❗ **Cost monitoring 必加 + 必有 hard cap**——GPT-4o-mini 一个深度 task 可能烧 100k+ token（≈ $0.05），换 GPT-4o 就 $1+。本 capstone 累加 `TOTAL_TOKENS` 是最低限；production 要 (a) per-task budget cap（`if cost > 0.5: abort`）(b) per-user QPS limit (c) Langfuse / Helicone 每日报表
- ❗ **Token usage limit per task 必设**——代码里 `MAX_STEPS_PER_SUB=10` 是显式 step cap，加上 `MAX_SUBTASKS=5` 就把 worst case bound 到 50 step。**没有这个 cap 的 agent 等于定时炸弹**——LLM 陷入死循环时会一直烧 budget 直到你手动 kill
- ❗ **User 中断时要 cleanup pending API call**——production 用 `asyncio` + `CancellationToken`，OpenAI SDK 的 `with_streaming_response` 支持 cancel。本 capstone 同步实现没 handle 这个，正式产品在 server 端要正确取消（否则用户关页面、API 还在跑、计费还在涨）
- ❗ **Trace 全 log 方便 debug，但 user data 要 redact PII**——`trace.json` 含 raw user query 与 all observations，里面可能有姓名 / 电话 / 地址。production 用 [Microsoft Presidio](https://github.com/microsoft/presidio) 在写 trace 前过一遍，敏感字段 mask 成 `<EMAIL>` / `<PHONE>`
- ❗ **Search results quality 决定 final answer 上限**——top-3 召回如果都是 spam / 过期 / 跑题，agent 写出来的 report 也是垃圾。production 要 (a) 加 reranker（[bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) 或 Cohere Rerank）(b) source whitelist / blacklist（屏蔽内容农场）(c) recency filter（research 类 query 要近 2 年内）
- ❗ **Tavily / SerpAPI 的 rate limit 容易爆**——Tavily 免费档 1000 次/月，capstone 跑 10-20 次就用掉一晚上预算。production 加 (a) cache 同 query 的 result（24h TTL）(b) per-key 限频（`asyncio.Semaphore(2)`）(c) 退避重试（429 时 exponential backoff）
- ❗ **Context 累积爆炸**——worker_loop 里每个 tool result 都拼进 `messages`，10 step 下来轻松 30k+ token。本 capstone 用 `obs[:3000]` 截断每条 observation 是最简兜底；进阶做法：每 5 step 用便宜模型把 history summarize 成一段，原文存 memory（14.5 §4 的 "summary + raw 双存"）
- ❗ **Tool description 是 prompt 的一部分**——本 capstone 每个 schema 的 description 至少 2 句话（"用于什么 / 什么时候调"），少一句调用准确率掉 10%（13.4 §2.2 已实测）
- ❗ **Worker 偶尔忘记调 `memory_write`**——sub-task 跑完出了一段 summary 但什么都没写进 memory，synthesizer 阶段一脸懵。fix：在 worker 的 system prompt 末尾强调 "Always call `memory_write`"，并在 worker_loop 退出前检查 `memory.docs` 在这个 sub_task 下有没有新增；没有就强制再 prompt 一次

---

## 8. 延伸方向

跑通 mini 版后，下一步往哪走，按"投入回报"排序：

**(a) 换更强的 model**（最低成本最大涨幅）。GPT-4o-mini → GPT-4o / Claude 3.5 Sonnet / Claude Opus 4。同样 prompt 同样 tool，sub-task 完成率从 70% 涨到 90%+，cost 涨 10-20×。OpenAI Deep Research 用的是 o3 + extended search，在 GAIA 上 SOTA 67.4%（2025 Q1）。

**(b) 加 reranker + source filter**（中等投入）。在 `tool_search_web` 里把 Tavily 返回的 top-10 用 bge-reranker-v2-m3 精排到 top-3，质量明显涨；source whitelist（arXiv / 官方文档 / 主流媒体）排除 SEO 垃圾。

**(c) 升级到 multi-agent**（用 14.6 AutoGen / orchestrator-worker）。把 worker 拆成"search worker / code worker / writer worker"分工，每个用专精 prompt。**但小心 multi-agent tax**——worker 间通信成本增加，simple task 上单 agent 反而更快（14.6 末尾的 "single agent + 好 tool > naive multi-agent" 结论）。

**(d) 加 reasoning model**（高投入高回报）。把 worker 换成 R1 / o1 / Claude 3.5 Sonnet with extended thinking——这类模型 long-CoT 内部已经会 plan + reflect，不再需要外部的"每 5 step reflection"，且 tool use trajectory 显著更连贯（14.2 §5.4 详讲）。

**(e) SFT / RL 训自己的 agent base**（最大投入）。

- **SFT 方向**（用附录 B 的 SFT pipeline）：收集 1000 个 (query, gold trace) 对，按 14.3 / 8.5 的格式 SFT 一个 7B base，得到 own-domain agent。Gorilla / xLAM 是这个范式的代表。
- **RL 方向**（用 15.4 的 Reasoning Agent RL）：用 GRPO + 任务成功 verifier 训一个 reasoning agent base，对应 Search-R1 / ReSearch / Agent-R1 范式。门槛高（要 multi-turn rollout infra + verifier 设计）但是当前 SOTA 路线，2025-2026 校招最热点。

**(f) 部署成 Web 服务**（产品化）。把 `deep_research` 包成 FastAPI endpoint，前端用 Next.js / SvelteKit，加 streaming（每个 step 实时推到前端）+ 历史 session 持久化（PostgreSQL + Memory 持久化到外部 vector DB 如 Qdrant / Milvus）。这一步是 mini Perplexity / mini Deep Research 的雏形。

---

## 9. 经典 paper / 资源

- **Yao et al., 2022 — ReAct: Synergizing Reasoning and Acting in Language Models**（与 14.2 同源）—— 本 capstone worker_loop 的范式来源。读 §3 understanding 为什么"Thought + Action 交替"比"先 plan 再做"鲁棒，对 worker 的多 step trajectory 设计是地基
- **Shinn et al., 2023 — Reflexion: Language Agents with Verbal Reinforcement Learning** —— 本 capstone "每 5 step 反思"的简化版来源。读 §3 的 reflection prompt 设计、§4 的 actor / evaluator / self-reflection 拆解；对 worker 失败 recovery 的工程化最有用
- **Mialon et al., 2023 — GAIA: A Benchmark for General AI Assistants** —— agent benchmark 的事实标准，本 capstone §6 的评测范式来源。读 §3 任务设计、§4 level 划分、§5 human vs LLM agent 差距，理解 "agent 离人类还有多远"
- **Anthropic, 2024 — Building Effective Agents** —— 工程视角的 agent design pattern 总结，本 capstone 的 planner-worker-synthesizer 三角色就是它推荐的 **orchestrator-worker pattern** 的最小实现。强烈推荐读完这一篇前不要写第二个 agent
- **OpenAI, 2025 — Introducing Deep Research** —— OpenAI 关于 Deep Research 的官方介绍，看商业产品的 architecture / 评测 / cost / latency 数据（GAIA 67.4% SOTA），对照本 capstone 的 mini 版理解差距在哪
- **Perplexity API documentation** —— 商业 search-cite agent 的 API reference，可以直接对比"我们的 search + memory + cite" pipeline 与 production grade 的差距

---

## 10. 自测与面试题

**Q1（架构）**：画出本 capstone 的 mini Deep Research agent 的 planner-worker-synthesizer 数据流图，并解释为什么不用单 ReAct loop 跑大任务。

<details>
<summary>Answer sketch</summary>

数据流图（与 §1.3 一致）：

```
User Query
    ↓
[Planner LLM] → list of N sub-tasks
    ↓ (for each sub_task)
[Worker LLM] ReAct loop
    ↳ search_web / run_python / memory_search / memory_write
    ↳ reflection every 5 steps
    ↳ memory.write(finding) on each useful obs
    ↓
[Memory] (FAISS, accumulates across sub-tasks)
    ↓
[Synthesizer LLM] → markdown report with citations
```

为什么不用单 ReAct loop：

- **Context 长度**：单 loop 跑大任务 history 累积爆炸，30 step 后 50k+ token，**lost in the middle** 严重，attention 找不到早期关键信息
- **可调试 / 可审计**：sub-task 隔离后每个 worker 的 trace 短、独立，debug 谁失败、改谁——单 loop 50 step 的 trace 几乎不可读
- **Cost 控制**：sub-task 拆开后每个 worker 有独立 max_step cap，bound 住 worst case；单 loop 没有这个保护
- **Model mix**：planner / synthesizer 用便宜模型，worker 用更强模型——三角色架构允许这种细粒度成本-能力 trade-off
- **可并行**：N 个 sub-task 之间无依赖时可 `asyncio.gather` 并发，单 loop 是天然 sequential

加分：这是 Anthropic《Building Effective Agents》定义的 **orchestrator-worker pattern**，是 OpenAI Deep Research / Perplexity / You.com 等产品的事实架构。

</details>

**Q2（debug）**：你跑这个 agent 的某个 sub-task，10 step 内没解决（max_step 触发）。给 3 个不同方向的 debug 思路。

<details>
<summary>Answer sketch</summary>

三个方向（从 LLM-side / tool-side / system-side）：

**1. LLM-side：检查 trace，定位是 reasoning 错还是 hallucination**
- 看 `trace.json` 每一 step 的 `tool_calls`：是否反复调同一个 tool 同样的 args（→ 死循环 detect 该 break）
- 是否调了不存在的 tool（→ tool description 不清楚 / schema 漏字段）
- 是否完全没调 tool 直接 final answer（→ system prompt 不够强调"必须用 tool"）
- 修：tool description 加正例 / 负例、加 retry-with-hint 机制（"你刚才调 X 失败了，试试 Y"）

**2. Tool-side：检查 observation 质量**
- search_web 返回是否真相关——top-3 全是 SEO 垃圾，agent 当然解不出。加 reranker / source filter
- run_python 是否 timeout / sandbox 报错——可能 LLM 写的 code 依赖未 install 的库；扩展 sandbox 预装常用库（numpy / pandas / requests）
- memory_search 是否召回过多 noise——k 调小、加 cosine threshold 过滤、加 reranker
- 修：tool 端做 quality control，把"工具垃圾输出"转成"明确的 error observation"让 LLM 自己 recover

**3. System-side：调架构 / 限制 / model**
- max_step 是不是太小（10 → 15-20）？sub_task 是不是太大（要再拆 sub-sub-task）？
- model 是不是太弱（GPT-4o-mini → 4o）？某些复杂 sub-task 在小 model 上根本不可解
- reflection 频率（5 step → 3 step）？是不是 reflection prompt 不够 actionable（"想想哪里出错"是废话，"列出当前缺哪条信息"才有用）
- planner 拆分质量——sub_task 之间是否有隐藏依赖（A 没跑完 B 拿不到信息）

加分：能讲到 "trace-based debug 是 agent 工程的核心 skill"，类比传统 backend 的 distributed tracing；以及 "agent 失败时不要直接改 prompt，先看 trace"。

</details>

**Q3（升级）**：把这个 mini Deep Research 升级到 production 级 OpenAI Deep Research / Perplexity 那个量级，列出 3 大方向，每个方向给具体技术手段。

<details>
<summary>Answer sketch</summary>

三大方向（按"涉及面"由窄到宽）：

**方向 1：Reasoning（让 worker / synthesizer 更强）**

- 换 reasoning model：worker / synthesizer 换成 o3 / Claude Opus 4 with extended thinking / DeepSeek-R1（10.3）。这类模型 long-CoT 内部已经会 plan + verify + correct，外部 ReAct loop 简化为 thin tool dispatcher
- 加 verifier：每个 sub-task 完成后用 LLM-as-judge 检查 finding 是否充分，不充分让 worker 继续；这是 best-of-N + verifier 的简化（10.4）
- chain-of-verification：synthesizer 写完 report 后跑一遍 fact-check pass，每条 claim 反向 search 验证

**方向 2：Multi-agent（让协作更专业化）**

- worker 分工：search worker（专长 web 搜索 + reranker）、code worker（专长数据 / 计算）、writer worker（专长长文综合），分别用专精 prompt + 不同 model（14.6）
- 加 critic agent：在 worker 完成后让 critic 评估并 send back 反馈，迭代到 critic 满意再 next sub-task（14.6 evaluator-optimizer pattern）
- planner 升级成 dynamic re-planning：worker 阶段发现 sub-task 拆得不对，planner 可以 re-plan（不是一次性 plan 死）
- 注意 multi-agent tax：worker 间通信成本增加，simple task 上反而更慢；要看任务复杂度选

**方向 3：RL fine-tune（训一个自己的 agent base）**

- SFT 起步：收集 5k-50k (query, gold trace) 对（人工 + GPT-4 distill），SFT 一个 7B-32B base 学 tool 调用模式（14.3 / 8.5）
- 进入 multi-turn GRPO（15.2）：用 task 成功 / unit test pass / answer match 当 verifier reward，GRPO 训 multi-turn trajectory；这是 Search-R1 / ReSearch / Agent-R1 的范式（15.4）
- 自家 verifier 设计：research task 的 reward 设计是难点——可用 (a) 关键词命中率 (b) LLM-as-judge 给 final report 打分 (c) human eval 抽样 (d) cite 数 / source 多样性等 composite reward
- 训完的 own model 不一定比 GPT-4o 强，但在自家垂直领域（公司内部知识 / 特定 domain）能有 reranker + cost 优势

加分：能讲到 "三个方向不互斥，可以叠"——OpenAI Deep Research 实际是 reasoning model（o3）+ verifier + 隐式 multi-agent 编排，不是单一技术。也能提到 inference / serve 侧（vLLM / SGLang，11.2-11.3）和评测侧（GAIA / Browsecomp，12.x）的 production 配套。

</details>

---

## 11. 延伸阅读

- [OpenAI Deep Research — Official Introduction](https://openai.com/index/introducing-deep-research/) — OpenAI 官方介绍，含 GAIA 67.4% 的 SOTA 数据与 architecture overview，对照本 capstone 的 mini 版理解 production 差距
- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — 2024 年最被引用的 agent design pattern blog，本 capstone 的三角色架构就是它的 orchestrator-worker pattern
- [Tavily Search API Docs](https://docs.tavily.com/) — 本 capstone 用的 search 后端，含 advanced search / topic filter / domain whitelist 等 production 必备功能
- [E2B Code Interpreter](https://e2b.dev/) — production grade 的 code sandbox，比 capstone 的 subprocess 安全得多，支持长会话 / 文件 / 图表
- [Langfuse](https://langfuse.com/) / [Arize Phoenix](https://phoenix.arize.com/) — agent observability 工业标准，每次 LLM call 自动 trace + cost + latency dashboard
- [GAIA Benchmark on HuggingFace](https://huggingface.co/datasets/gaia-benchmark/GAIA) — agent 评测数据集 + leaderboard，capstone 之后想真实评测可直接拉数据
- [Smolagents](https://github.com/huggingface/smolagents) — HuggingFace 极简 agent 框架（~1000 行核心），是本 capstone 250 行版本的工业化对照，对比阅读收益高
- 推荐回看本教程的 **14.2 节《最小 ReAct + Reflection agent》**——本 capstone 的 worker_loop 是 14.2 的扩展版；**14.5 节《Memory》**——本 capstone 的 Memory 类是 14.5 vector DB pattern 的简化版；**15.4 节《Reasoning + Agent》**——本 capstone 的下一步是用 RL 训自己的 reasoning agent base
