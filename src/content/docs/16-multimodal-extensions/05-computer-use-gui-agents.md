---
title: "16.5 Computer Use & GUI Agent：OSWorld / Anthropic / SeeClick / UI-TARS"
description: "GUI Agent 把 LLM/VLM 推到\"直接操作屏幕\"的最后一公里——输入是 screenshot、输出是 click / type / hotkey 这种像素级动作；本节梳理 Anthropic Computer Use / OpenAI Operator / SeeClick / UI-TARS 这条 2024-2025 主线，把 visual grounding、action repr"
---

> ⏱ 预计阅读 50 分钟 ｜ 难度 ★★★ ｜ 前置：14.3（Tool Use 训练）、15.3（OSWorld / WebArena 真实环境 RL）、16.1（VLM）

## 一句话本节讲什么

GUI Agent 把 LLM/VLM 推到"直接操作屏幕"的最后一公里——输入是 **screenshot**、输出是 **(x, y) click / type / hotkey** 这种像素级动作；本节梳理 Anthropic Computer Use / OpenAI Operator / SeeClick / UI-TARS 这条 2024-2025 主线，把 visual grounding、action representation、训练数据合成、OSWorld/AndroidWorld 评测、以及"为什么准确率仍然只有 ~40%"这几个核心问题讲清楚，作为 Module 16 与全书 Agent 章节的收官。

---

## 1. Mental model（直觉）

### 1.1 GUI Agent vs Web/Text Agent 的根本差异

14 章讲的 ReAct/Function-calling agent 拿到的是**结构化输入**（function schema、JSON 参数）；15.3 讲的 WebArena agent 拿到的是**半结构化 DOM**（accessibility tree）。GUI Agent 是这条 spectrum 的最末端：

```
                    输入抽象等级
   高 ──────────────────────────────────────────────► 低
   │                                                    │
Function call    DOM/a11y tree    Screenshot + a11y    Pure Screenshot
(JSON schema)    (WebArena)       (OSWorld)            (Anthropic CU)
   │                  │                  │                   │
   ▼                  ▼                  ▼                   ▼
"调 weather()"   "click [data-id=5]"  "click button '提交'"  "click (847, 320)"
```

差别一句话：**GUI Agent 必须自己看懂屏幕、自己算坐标**。这带来三个 web/text agent 完全没有的难题：

1. **Visual grounding**：从 screenshot 中识别"提交按钮在哪里"，输出像素坐标
2. **Precise click**：(x, y) 必须落在按钮命中区域内（通常几十像素见方），LLM 天生不擅长输出精确坐标
3. **OS heterogeneity**：Windows / macOS / Linux / Android / iOS 的 GUI 风格、控件排布、快捷键完全不同，跨 OS 几乎不能 zero-shot

### 1.2 核心范式

```
       ┌────────────────────────────┐
       │     Task instruction       │
       │  "帮我把桌面所有 PDF       │
       │   按月份归档到子文件夹"    │
       └────────────┬───────────────┘
                    │
        ┌───────────▼───────────┐
        │    Screenshot (t)     │  ← 1920×1080 RGB
        └───────────┬───────────┘
                    │
              ┌─────▼─────┐
              │   VLM     │  ← Claude/GPT/UI-TARS
              │ (Reasoning│
              │  +Action) │
              └─────┬─────┘
                    │
        ┌───────────▼───────────┐
        │ Action: click(847,320)│
        │   or type("PDF")      │
        │   or hotkey("ctrl+c") │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   OS executor         │  ← pyautogui / xdotool / ADB
        │ (真的去 click)        │
        └───────────┬───────────┘
                    │
              ┌─────▼─────┐
              │ 新的 Screenshot (t+1)
              └───────────┘   ← 进入下一轮
```

每一步是 **screenshot → reasoning → action → screenshot** 的闭环，与 ReAct 的 thought→action→observation 同构，但 observation 变成图像、action 变成像素坐标。

### 1.3 4 个时代的演化

```
2023 SeeClick      ──► 单步 visual grounding 模型
                       (screenshot, instruction) → (x, y)
                       预训练专门的 GUI grounding 能力

2024 Anthropic     ──► 商业化 Computer Use API
       Computer Use     Claude 3.5 Sonnet + computer/bash/edit tool
                       第一次让生产级 LLM 直接控制 desktop

2024 OSWorld       ──► 标准 OS-level benchmark
                       369 真实 task，VM-based eval

2025 UI-TARS       ──► 端到端 GUI 基础模型（字节）
       (字节)           7B-72B 全 size，开源
                       SFT + RL + reflection on OS/Android
                       OSWorld / AndroidWorld SOTA

2025 OpenAI        ──► 浏览器版 GUI agent (web-only)
       Operator       $200/月 Pro 用户专享
```

> 与 15.3 区分：15.3 关注**训练 setup**（env / reward / RL infra），本节关注**模型与产品形态**——同一个 OSWorld benchmark，15.3 讲怎么在上面跑 RL，本节讲在上面跑的 agent 长什么样、grounding 能力怎么训出来。
> 与 14.3 区分：14.3 是 function-calling tool use（结构化 tool），本节是 GUI tool use（像素级 tool）；可以理解为"把 GUI 整体当成一个超复杂、超不规则的 tool"。

---

## 2. Visual Grounding：GUI Agent 的核心难点

### 2.1 为什么 LLM/VLM 天生不擅长

在 16.1 介绍的 VLM 框架下，screenshot 经过 vision encoder + projector 变成 visual token 序列。这套 pipeline 在"描述图像内容"上效果好（CLIP-style 训练让它学会语义对齐），但**输出精确像素坐标**是另一回事：

- VLM 预训练目标几乎全是 caption / VQA / OCR——没有任何 (image, instruction) → (x, y) 的监督信号
- vision encoder（如 ViT）的 patch size 通常是 14×14 或 16×16，本身就丢掉了 sub-patch 精度
- LLM decoder 输出数字（"847"）是 token-by-token，对"接近 850 的数都行"这种近似没有 graceful 表达
- 高分辨率 screenshot（1920×1080）切成 patch 后 token 数巨大（与 16.1 高分辨率 patch 呼应），上下文压力大

### 2.2 SeeClick：第一个专门的 GUI grounding 预训练

**SeeClick** (Cheng et al. 2024) 给出第一个系统答案：**专门为 GUI grounding 做大规模预训练**。

数据合成 pipeline：

1. 抓取 mobile / desktop / web GUI screenshot 数百万张
2. 用 a11y tree / DOM / 自动 OCR 抽取每个可交互 element 的 (bbox, text/label, role)
3. 构造 (screenshot, "click 提交按钮") → (x, y) 的训练对

训练目标：把 base VLM（如 Qwen-VL）在这批数据上继续 SFT，让模型学会"看屏幕指元素"。

效果：在 ScreenSpot / Mind2Web 等 grounding benchmark 上 +20-30 个点，且能力**可迁移到下游 agent**——把 SeeClick checkpoint 当 vision backbone 给 OSWorld agent 用，整体成功率提升 5-10 个点。

**SeeClick 的核心 take-away**：**GUI grounding 是一种独立可训练的能力，必须显式预训练，不能指望通用 VLM 自动 emergence**。这条 insight 后续被 UI-TARS / CogAgent / SeeAct 全部继承。

### 2.3 输入表示的 4 种方案

不同方案在"通用性 vs 精度 vs OS-coupling"上做不同 trade-off：

| 方案 | 输入 | 输出 | 优点 | 缺点 |
|---|---|---|---|---|
| **Pure Screenshot** | 仅 screenshot | (x, y) | 最通用、跨 OS / app 无成本 | grounding 难、坐标易错 |
| **Screenshot + a11y tree** | screenshot + accessibility 节点列表 | element_id | 精度高、不需要算坐标 | OS-specific（macOS AX / Windows UIA / Linux AT-SPI 各不同） |
| **DOM (web only)** | HTML DOM | css selector | web 上最准 | 仅 web 可用 |
| **Set-of-Marks (SoM)** | screenshot 上叠加编号标签 | "click #5" | 把 grounding 转成 selection，难度大降 | 需先跑 element detector |

**Set-of-Marks (SoM)** (Yang et al. 2023) 是非常聪明的折中——先用一个 detector / a11y tree 找出所有可交互 element，在 screenshot 上叠加编号 "1, 2, 3..."，然后让 VLM 输出 "点击 #5"。这把"输出像素坐标"这种 LLM 不擅长的任务，转成"输出一个数字"这种 LLM 完全胜任的任务，端到端准确率显著提升。代价是需要一个上游 element detector。

### 2.4 Action space 的标准设计

主流 GUI agent 的 action space 大致收敛到：

```python
# 标准 action 集
click(x: int, y: int)                    # 单击
double_click(x: int, y: int)             # 双击
right_click(x: int, y: int)              # 右键
drag(x1, y1, x2, y2)                     # 拖拽
type(text: str)                          # 输入文本
hotkey(keys: list[str])                  # 组合键，如 ["ctrl", "c"]
scroll(x, y, direction, amount)          # 滚动
screenshot()                             # 主动截图
wait(seconds: float)                     # 等待加载
done(answer: str = None)                 # 任务完成
```

Anthropic Computer Use 在此之外还提供 `bash(cmd)` 与 `text_editor(view/create/str_replace)` 三个 tool，三者并列。这种设计让 agent 在"能用 shell 就别用 GUI"时走快路（如 git clone 用 bash 而不是去点 VS Code 菜单），是工程上的关键优化。

---

## 3. 主流方案巡礼

### 3.1 Anthropic Computer Use（2024.10）

**第一个商业化的 desktop computer use API**，与 Claude 3.5 Sonnet (new) 一起发布。技术细节：

- **Tool 三件套**：`computer`（screenshot / click / type / hotkey）+ `bash`（执行 shell）+ `text_editor`（结构化文件编辑）
- **Input**：默认 1024×768 截图（高分辨率会先 downscale，避免 token 爆炸）
- **Output**：直接输出 `tool_use` block，含坐标或命令
- **协议**：标准 Anthropic Messages API + 一个 special `betas=["computer-use-2024-10-22"]` header
- **使用方式**：用户自己起 sandbox / docker / VM，Claude 输出 action 后，**用户的 executor** 真的去 click（Anthropic 不直接执行）

公开数字：在 OSWorld 上 success rate ~22%（首发版本），到 Claude 3.7 Sonnet 提升到 ~30%+。这个数字看似低，但已是当时**唯一**可商用的 OS-level GUI agent，催生了 Browser-Use / E2B / Anthropic Quickstart 等一批 desktop agent 应用。

**安全设计**：Anthropic 反复强调 Computer Use 是 alpha 阶段、必须 sandbox + human-in-loop——agent 完全可能 `rm -rf /`、发邮件、转账。这与 15.3 §3.2 sandbox safety 那段直接呼应。

### 3.2 OpenAI Operator（2025.1）

**浏览器版 GUI agent**，与 Computer Use 的核心差异：

- **范围**：仅 browser，不直接控制 OS
- **形态**：消费产品（ChatGPT Pro $200/月专享），不是 API
- **实现**：基于内部 CUA（Computer-Using Agent）模型，screenshot + reasoning → click/type
- **目标场景**：购物、订餐、订票、表单填写等 web 操作

为什么 OpenAI 选 web-only 而不是全 OS？

1. Web 操作覆盖 80% 普通用户需求，足以撑起产品
2. Web 比 OS 更安全（沙箱由浏览器提供，不用担心 `rm -rf`）
3. Web 训练数据远比 OS 数据丰富（Mind2Web / WebArena trace 可直接用）

Operator 在 WebArena / VisualWebArena 上 SOTA，但具体训练细节没公开。

### 3.3 SeeClick（Cheng et al. 2024）

**作为 grounding pre-train 的代表**，已在 §2.2 详讲。补充一点：SeeClick 本身不是端到端 agent，而是给下游 agent 用的 visual grounding backbone。它启发了一整条 "GUI-pretraining → agent SFT" 的 pipeline。

### 3.4 UI-TARS（字节 2025）

**当前开源 GUI agent SOTA**，由字节 Doubao 团队开源。核心特点：

- **全 size 系列**：UI-TARS-7B / 72B，UI-TARS-1.5（推理增强版）
- **训练 recipe**：(1) GUI grounding pre-train（SeeClick 风格大规模数据）→ (2) agent SFT（合成 trajectory）→ (3) RL with reflection（错误恢复）
- **多平台统一**：Windows / macOS / Android 同一模型
- **OSWorld 上**：UI-TARS-72B 达到 ~40%+，与 Claude 3.7 Computer Use 同档
- **AndroidWorld 上**：~60%，移动端 SOTA
- **关键创新**：把 reasoning trace（System 2 thinking）显式写进训练数据，让模型在 click 之前先"想清楚下一步"——这是 R1 风格 reasoning 在 GUI agent 上的迁移

UI-TARS 的开源对中文圈意义重大——它是第一个公开权重 + 训练 recipe 的工业级 GUI agent，也是 2025-2026 国内 GUI agent 创业潮的技术底座。

### 3.5 Mobile-Agent / AppAgent（移动端代表）

**Mobile-Agent** (Wang et al. 2024, BAAI) 与 **AppAgent** (Yang et al. 2024, 阿里)：聚焦 Android。与 desktop agent 的关键差异：

- 大量利用 **accessibility tree**（Android 的 a11y info 比 desktop 完整且统一），不只靠 screenshot
- Action space 简单（手机交互方式有限：tap / swipe / type / back / home）
- 任务多为短序列（订外卖、发消息），但 app 间 context 切换难
- 屏幕分辨率统一（手机普遍 1080p / 1440p），训练数据更易合成

**AndroidWorld** (Rawles et al. 2024) 是该方向的标准 benchmark，116 个 task 覆盖 20 个真实 app（Gmail、Maps、WhatsApp 等）。

---

## 4. 训练 GUI Agent 的 5 阶段 pipeline

把上面 3 节的零散 insight 拼起来，2025 年训一个 GUI agent 的标准 pipeline：

```
Stage 1: VLM base
   │  Qwen2-VL-7B / InternVL-2.5 / Llama-3.2-Vision
   ▼
Stage 2: GUI Grounding Pre-train (SeeClick 风格)
   │  数百万 (screenshot, "click X") → (x, y) 数据
   │  让 VLM 学会"看屏幕指元素"
   ▼
Stage 3: Agent SFT (multi-step trajectory)
   │  Mind2Web / AndroidControl / 自合成 OS trace
   │  教模型 reasoning + action sequence
   │  典型数据格式：(history, screenshot_t) → (thought_t, action_t)
   ▼
Stage 4: Real env RL (15.3)
   │  WebArena / OSWorld / AndroidWorld
   │  reward = task completion (rule-based)
   │  GRPO + KL to ref
   │  从 SFT 30% → RL 40-50%
   ▼
Stage 5: Reflection / self-correction tuning
   │  在错误 trajectory 上加"我搞错了，让我重新看 screenshot"的修复样本
   │  显著提升 long-horizon 鲁棒性
```

**关键决策点**：

- **Stage 2 是否单独做**？数据量足够（百万级 grounding pair）+ 算力允许时单独做能让 grounding 能力专精；数据少时与 Stage 3 合并 SFT 也可行
- **Stage 4 是否值得做**？SFT 后准确率 < 30% 时优先补 SFT 数据；> 30% 且失败集中在"长序列错误传播"时上 RL
- **Stage 5 的 reflection 数据怎么来**？最常见做法是用 strong teacher（GPT-4V / Claude 3.5）跑 trajectory 时记录"出错→恢复"片段，作为修复样本

---

## 5. 最小代码示例

### 5.1 Anthropic Computer Use API demo

```python
# anthropic_computer_use.py — 调 computer tool 让 Claude 截图并点击
import anthropic, base64

client = anthropic.Anthropic()

# computer tool 的标准 schema（type=computer_20241022）
tools = [{
    "type": "computer_20241022",
    "name": "computer",
    "display_width_px": 1024, "display_height_px": 768, "display_number": 1,
}]

response = client.beta.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    betas=["computer-use-2024-10-22"],
    messages=[{
        "role": "user",
        "content": "请截一张当前屏幕的图，然后点击屏幕中央。",
    }],
)
# Claude 会输出 tool_use block，例如：
#   {"type":"tool_use","name":"computer","input":{"action":"screenshot"}}
#   {"type":"tool_use","name":"computer","input":{"action":"left_click","coordinate":[512,384]}}
for block in response.content:
    if block.type == "tool_use":
        print(block.input)  # 用户侧的 executor 真去执行 action
```

要点：Anthropic 只**输出** action，**用户必须自己起 executor**（pyautogui / xdotool 等）真去执行，再把新 screenshot 作为 `tool_result` 喂回去——这是一个手动 loop。Anthropic 提供 [quickstart docker image](https://github.com/anthropics/anthropic-quickstarts) 把 loop 封好。

### 5.2 SeeClick 风格 visual grounding 调用

```python
# seeclick_grounding.py — 用 SeeClick checkpoint 做 element 定位
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch, re

# SeeClick 是基于 Qwen-VL 继续 SFT 的 checkpoint
proc = AutoProcessor.from_pretrained("cckevinn/SeeClick")
model = AutoModelForCausalLM.from_pretrained(
    "cckevinn/SeeClick", torch_dtype=torch.bfloat16, device_map="auto",
)

img = Image.open("desktop_screenshot.png")  # e.g. 1280x800
prompt = (
    "In the UI, where should I click to "
    "'open the search bar'? Output the coordinate as (x, y) in 0-1 range."
)
inputs = proc(images=img, text=prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=20)
text = proc.decode(out[0], skip_special_tokens=True)
# 模型输出形如 "(0.85, 0.12)"，归一化坐标
m = re.search(r"\(([\d.]+),\s*([\d.]+)\)", text)
nx, ny = float(m.group(1)), float(m.group(2))
px, py = int(nx * img.width), int(ny * img.height)
print(f"click at pixel ({px}, {py})")
```

要点：SeeClick 输出的是 **0-1 归一化坐标**（与分辨率解耦），下游再乘以实际宽高得到像素坐标。这个 trick 让同一 model 能跨不同分辨率截图工作。

### 5.3 Set-of-Marks 实现

```python
# som_overlay.py — 在 screenshot 上叠加编号 mark
from PIL import Image, ImageDraw, ImageFont

def draw_som(image_path, elements, out_path):
    """
    elements: list of dict, each {"bbox": (x1, y1, x2, y2), "id": int}
    上游 element detector / a11y tree 提供 elements
    """
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    for el in elements:
        x1, y1, x2, y2 = el["bbox"]
        # 1. 画红框圈出元素
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=2)
        # 2. 在左上角放白底红字编号 tag
        tag = str(el["id"])
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 8, y1], fill=(255, 255, 255, 230))
        draw.text((x1 + 4, y1 - th - 2), tag, fill=(200, 0, 0, 255), font=font)
    Image.alpha_composite(img, overlay).convert("RGB").save(out_path)

# 用法：然后把 out_path 丢给 GPT-4V，prompt 说 "click element #5"
draw_som("raw.png",
         [{"bbox": (100, 80, 240, 130), "id": 1},
          {"bbox": (300, 200, 480, 250), "id": 2}],
         "som.png")
```

要点：SoM 把"输出 (x, y)"问题转成"输出一个 id"，难度骤降。代价是上游必须有 element bbox 来源——可以用 GroundingDINO / OmniParser 等 detector，也可直接用 a11y tree。OmniParser（微软 2024）是专门为 SoM 设计的 GUI element parser，与 GPT-4V 配合能在 OSWorld 上把 success rate 拉高 ~10%。

---

## 6. Benchmark 速查

| Benchmark | 范围 | 规模 | 当前 SOTA | 备注 |
|---|---|---|---|---|
| **OSWorld** (Xie 2024) | Ubuntu desktop | 369 task | UI-TARS-72B / Claude CU ~40% | OS-level 黄金标准 |
| **WebArena** (Zhou 2023) | 5 自托管 web app | 812 task | ~60% | web 黄金标准（15.3） |
| **VisualWebArena** (Koh 2024) | web + screenshot | 910 task | ~30% | 测 VLM agent |
| **Mind2Web** (Deng 2023) | 137 真实 site 静态 | 2350 task | - | SFT 数据为主，不是 interactive eval |
| **AndroidWorld** (Rawles 2024) | Android emulator | 116 task / 20 app | UI-TARS-72B ~60% | mobile 标杆 |
| **ScreenSpot** (SeeClick 2024) | grounding only | ~1.3k | UI-TARS / SeeClick ~80%+ | 单纯测 grounding |
| **AgentBench** GUI 子集 | 跨 5 类 env | - | - | 综合性，但 GUI 部分较少 |

人类基线参考：OSWorld 人类 ~70%+、WebArena 人类 ~78%、AndroidWorld 人类 ~80%。**当前 GUI agent 与人类还有 30-40 个点差距**，是真正的 wide-open frontier。

---

## 7. 工程踩坑与经验

- ❗ **Screenshot 分辨率高 → vision token 数巨大**（与 16.1 高分辨率呼应）。1920×1080 截图按 14×14 patch 切是 ~10k token，10 轮 trajectory 就是 100k+ token，单步推理延迟数秒、$$ 暴涨。**对策**：(1) 强制 downscale 到 1024×768 或更低（Anthropic CU 默认 1024×768）；(2) 用 dynamic resolution 让 VLM 自适应不同尺寸；(3) crop 关注区域（已知大致位置时只 crop 一块送 VLM）；(4) 历史 screenshot 只保留最后 1-3 张
- ❗ **Click coordinate 精度要求高（pixel-level），LLM 通常不擅长**——LLM 输出 "847" 和 "850" 在 token 空间相邻但在 button 命中区域可能差很多。**对策**：(1) 用归一化坐标 + 模型输出小数（0-1 范围），让数值误差被原图大小放大但训练分布稳定；(2) 用 SoM 把"坐标输出"转成"编号选择"；(3) grounding 阶段大规模预训练（SeeClick 路线）；(4) 给每个 click 加 ±5 像素的 jitter 容错（执行端做小范围 retry）
- ❗ **OS 之间 GUI 差异大，跨 OS 训出来的 model 一般要 fine-tune**。同一 "保存" 按钮在 macOS 是 ⌘+S、Windows 是 Ctrl+S；菜单栏在 macOS 顶部、Windows 在窗口顶部；文件管理器布局完全不同。在 macOS 数据上训的 model 直接放 Windows 上 success rate 砍半。**对策**：(1) 多 OS 数据混合训练（UI-TARS 这么做）；(2) 在目标 OS 上做小规模 fine-tune（千级 trajectory 即可显著恢复）；(3) action space 抽象出 OS-agnostic 中间层（如 "save current file" 而不是 "Cmd+S"），由 executor 翻译成具体快捷键
- ❗ **Computer Use 安全风险大（agent 可以 wipe disk / 发邮件 / 转账），必须 sandbox + human-in-loop**——这与 15.3 §3.2 sandbox safety 一脉相承，但 desktop agent 风险面更大（不像 docker container，desktop agent 可能直接控制本机邮箱、登录态社交账号、加密钱包）。**最低配**：(1) 隔离 VM 跑 agent（VirtualBox / VMware / Anthropic 提供的 docker quickstart），不要直接在主力机上跑；(2) `confirm()` 钩子拦截高风险 action（涉及钱、删除、网络外传时强制用户确认）；(3) tool whitelist / blacklist；(4) 全程 record video log，事后可追溯
- ❗ **训练数据稀缺，大量靠合成 + 人标**。GUI trajectory 数据天然稀缺——网上找不到大批量"用户操作 screenshot 序列"。**主流来源**：(1) 雇人标注（贵，每条几美元，UI-TARS / Anthropic 都重金标）；(2) 用 strong teacher（GPT-4V / Claude）跑现成 task pool 自动产数据，再过滤；(3) 录屏 + 自动 annotation（OCR 出 click target、用 a11y 反推 action）；(4) 程序化合成（在已知 GUI 上脚本化生成 trajectory，再让 VLM 描述）。**质量铁律**：合成数据的 action 必须 executable + 必须真去 replay 验证一次，否则训出"看起来合理但跑不通"的 model
- ❗ **Latency 高（screenshot → LLM → action → screenshot），影响交互**——单 step 通常 3-10 秒（截图 100ms + VLM 推理 2-5s + 执行 + 等页面响应 1-3s）。10 步 task 就是分钟级，用户等不及。**对策**：(1) screenshot 做 perceptual hash，新截图与上一帧太像时跳过 VLM 调用；(2) 把短期内可预见的 action 序列一次性 plan 好，减少 round-trip；(3) 用更小的 VLM（7B 而非 72B）做基础 step，遇困难再升级到大 model；(4) 截图分辨率自适应（简单页面用低分辨率快速决策）
- ❗ **OSWorld 评测 reproducibility 差（依赖外部状态），多次跑结果有波动 5-10%**。同一 model 同一 task 跑 3 次可能拿到 0/1/1 三个结果，因为 (1) 网页加载速度不同导致 element 出现时机不同；(2) 应用更新导致 UI 变化；(3) 系统时间相关 task（"今天日期"）不稳定；(4) reward evaluator 本身有边界 case bug。**对策**：(1) 至少跑 3 seed 取平均；(2) 锁定 OSWorld 镜像版本（`v0.x.x` tag）；(3) 自定义补 deterministic env layer（time freeze / network mock）；(4) 报数字时附带 95% 置信区间，单次 SOTA 数字不可全信
- ❗ **Long-horizon trajectory 错误传播严重**——5 步 task 准确率 60%，10 步骤降到 30%、20 步几乎 0。每步独立准确率 90% 时，20 步连乘只剩 12%。**对策**：(1) Stage 5 的 reflection 训练让模型识别"我刚搞错了"并回滚；(2) 中间设 checkpoint，错了从 checkpoint 重启而不是从头；(3) 用 planning agent 把长 task 拆成短 sub-task；(4) trajectory-level RL（15.3）显式优化 long-horizon 成功率而非单步
- ❗ **Headless / headful 模式 screenshot 可能不一致**——同一网页 headless Chromium 渲染与正常 Chrome 视觉上有差异（字体、缩放、widget），训练用 headless 数据但部署在真实桌面，model 准确率会掉。**对策**：训练数据生成时与部署环境对齐（同一 browser、同一 DPI、同一字体集），或者两个模式数据混训

---

## 8. 经典 paper

- **Anthropic 2024 — *Computer Use* blog & docs** — 必读。第一个商业化 desktop agent API，定义了 `computer/bash/text_editor` 三件套 tool 的工业标准。读官方 blog + cookbook + quickstart docker，理解 action loop 的工程实现以及 Anthropic 强调的 sandbox + human-in-loop 安全立场。Take-away：理解一个生产级 computer use 系统的 minimal viable architecture
- **Xie et al., 2024 — *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*** — 必读。OS-level GUI agent 的标准 benchmark。读 §3 的 VM-based env infrastructure、§4 的 369 task taxonomy（GUI / file / terminal / 跨应用）、§5 的 baseline 数字。Take-away：理解为什么 OSWorld 是当前最难的 agent benchmark 之一，以及人类 ~70% vs SOTA ~40% 的 gap 在哪
- **Cheng et al., 2024 — *SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents*** — 必读。第一个把 GUI grounding 作为独立可训练能力提出的工作。读 §3 的数据合成 pipeline（mobile/desktop/web 混合）、§4 的预训练目标、§5 的下游 agent 性能提升。Take-away：理解"GUI grounding 必须显式预训练，通用 VLM 不会自动 emergence"这条 insight
- **Qin et al., 2025 — *UI-TARS: Pioneering Automated GUI Interaction with Native Agents* (字节)** — 必读。当前开源 GUI agent SOTA，国内代表作。读 §3 全 size 训练 recipe（grounding pretrain → agent SFT → RL with reflection）、§4 在 OSWorld / AndroidWorld 上的 benchmark 数字、§5 reflection 训练设计。Take-away：理解 R1 风格 reasoning 与 GUI agent 结合的具体做法
- **Yang et al., 2023 — *Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V*** — 必读。把"输出像素坐标"转成"输出编号"的关键 trick。读 §2 的 SoM 构造、§3 的 GPT-4V grounding 实验。Take-away：理解 visual grounding 的"问题转换"思路，几乎所有现代 GUI agent 都在某层用了 SoM 思想
- 加分阅读：**Wang et al., 2024 — Mobile-Agent**（移动端代表）；**Rawles et al., 2024 — AndroidWorld**（Android benchmark）；**Lu et al., 2024 — OmniParser**（专门的 GUI element parser，与 SoM 配套）；**Hong et al., 2024 — CogAgent**（清华 GUI agent，早期重要工作）

---

## 9. 自测与面试题

**Q1（架构）**：GUI agent 的输入表示有哪 3-4 种？各自 trade-off 是什么？什么场景选哪个？

<details>
<summary>Answer sketch</summary>

四种主流输入表示：

1. **Pure Screenshot**：仅 screenshot
   - 优点：通用、跨 OS / app 零成本、实现简单
   - 缺点：grounding 难（输出像素坐标），LLM 不擅长精确数值
   - 场景：跨 OS 通用 agent（Anthropic Computer Use）、新 app / 没有 a11y 支持的 app

2. **Screenshot + Accessibility Tree**：screenshot 加 a11y 节点结构
   - 优点：grounding 用 element_id 不用算坐标、精度高、a11y 包含 role / label 语义
   - 缺点：OS-specific（macOS AX / Windows UIA / Linux AT-SPI 完全不同）；很多 app 的 a11y 实现质量差（特别是游戏、Electron）
   - 场景：固定平台（如只做 Windows 企业 agent）、a11y 完整的标准应用

3. **DOM (web only)**：HTML DOM 树 + CSS selector
   - 优点：web 上最精确、CSS selector 稳定、reward function 也容易写
   - 缺点：仅 web 可用、SPA 渲染异步带来时序问题、shadow DOM 复杂
   - 场景：纯 web agent（OpenAI Operator、WebArena）

4. **Set-of-Marks (SoM)**：screenshot 上叠加编号标签
   - 优点：把 grounding 转成 selection（"#5"）大幅降低难度，跨 OS 通用
   - 缺点：依赖上游 element detector（GroundingDINO / OmniParser / a11y），detector 漏掉的 element 模型看不见
   - 场景：与 GPT-4V / Claude 等通用 VLM 配合，无需重新训 grounding 能力

**多数 production 系统是混合**：a11y 能拿到就用 a11y，拿不到 fallback 到 SoM，最后兜底 pure screenshot。

加分：能指出 OmniParser 是当前 SoM 流派的标准工具；能指出 a11y tree 在 OSWorld 中是 baseline 而 pure screenshot 是更难的 setting。

</details>

**Q2（实战 pipeline）**：你要做一个 OS 助手 agent，列出从 base VLM 到 production 的完整 pipeline（visual grounding + action 训练 + RL）。

<details>
<summary>Answer sketch</summary>

完整 7 步 pipeline：

**数据准备**

1. **Base VLM 选型**：Qwen2-VL-7B / InternVL-2.5 / Llama-3.2-Vision——选支持高分辨率（dynamic resolution）、有 OCR 能力的 VLM
2. **Grounding 数据合成**：
   - Mobile：Rico / AMP screenshot
   - Desktop：录屏 + a11y 抽 element bbox
   - Web：scrape Common Crawl + DOM 反推
   - 目标量级：100 万 - 1000 万 (screenshot, instruction, bbox) 三元组
3. **Trajectory 数据合成**：
   - 用 strong teacher（GPT-4V / Claude）跑 OSWorld / Mind2Web task 录 trajectory
   - 人标修正错误 trajectory
   - 注意覆盖 multi-OS、长短 step、错误恢复

**训练阶段**

4. **Stage 2 grounding pretrain**：base VLM 在 grounding 数据上 SFT，损失只算坐标 token
   - 输出归一化坐标 (0-1)
   - lr=1e-5, bs=128, 1-2 epoch
5. **Stage 3 agent SFT**：在 multi-step trajectory 上 SFT
   - 数据格式：`(task, history, screenshot_t) → (thought_t, action_t)`
   - 每条 trajectory 最多 20 step，超长截断
   - mix 60% trajectory + 30% grounding + 10% chat 防能力退化
   - lr=2e-5, bs=64, 2-3 epoch
6. **Stage 4 real env RL（15.3）**：
   - Env：OSWorld dockerized VM
   - Reward：task completion (rule) + 可选 process reward（中间 sub-goal）
   - Algorithm：GRPO（group-relative advantage，省 critic）
   - 每 prompt 采 8-16 条 trajectory，KL to ref ≤ 0.05 防 reward hacking
   - 100+ 并行 container 维持 throughput

**评测 + 上线**

7. **多维评测 + 部署**：
   - OSWorld（主指标）+ ScreenSpot（grounding）+ 自定义业务任务
   - 严格 train/test split（参考 15.3 §6）
   - 部署：sandbox VM、human-in-loop confirm 高风险 action、record video log

加分要点：
- 提到 Stage 5 reflection 训练（错误恢复样本）
- 提到 Stage 4 必须 SFT 冷启把 success rate 推到 30%+ 才上 RL
- 提到 grounding pretrain 与 agent SFT 可合并（数据少时）也可分开（数据多时）
- 提到 action space 抽象出 OS-agnostic 中间层（safe save / open）让跨 OS 训练共享数据
- 提到 latency 优化：小 model 做基础 step、大 model 兜底困难 step
- 提到 sandbox safety / network isolation / 高危 action whitelist

</details>

**Q3（前沿）**：Reasoning model + GUI agent 的融合点在哪？2026 年这个方向有哪些可能突破？

<details>
<summary>Answer sketch</summary>

**融合点（已落地的）**：

1. **UI-TARS 的 System 2 thinking**：在 click 之前显式输出 reasoning trace（"我现在屏幕上看到 X，下一步应该 Y 因为 Z"），把 R1 风格 long-CoT 迁移到 GUI 决策。结果：long-horizon task 准确率提升 5-10 个点
2. **Reflection / self-correction**：错了之后能识别"我刚才搞错了"并回滚——这是 reasoning 能力的延伸，UI-TARS-1.5 的 reflection 训练就是显式做这个
3. **Plan-then-execute**：reasoning model 先规划完整 trajectory（"先打开浏览器、再搜索、再点第一个结果..."），再交给 GUI executor 执行——把"思考"和"行动"分离，每步行动更聚焦
4. **Verifier-guided GUI rollout**：reasoning model 在 candidate action 上打分（这个 click 是否合理），best-of-N 选择最好的——与 10.4 best-of-N 思路同构

**2026 年可能突破**：

1. **Real-env RL with reasoning reward**：OSWorld + GRPO + 让模型显式输出长 reasoning，reward 覆盖最终 + reasoning quality。需要 verifier 能判断 reasoning 是否合理（PRM 风格），是 10.2 PRM 在 GUI 上的直接迁移
2. **Multi-modal native reasoning**：当前 reasoning 是纯文本 CoT，未来可能 reasoning 内嵌 visual marker（"我看到这个区域 [crop_box] 应该是按钮"），让 reasoning 真正 grounded 到屏幕
3. **跨平台统一基础模型**：Windows / macOS / Linux / Android / iOS 一个 model，用 reasoning 抽象 OS-agnostic 中间动作（"save file" → executor 翻译成 Cmd+S 或 Ctrl+S），是 UI-TARS 已开始走的方向
4. **Long-context multi-screenshot 推理**：当前每步只看 1-3 张 screenshot，未来 1M context 下 agent 能"回忆"过去 50 步的视觉历史，做真正的长链规划——这要求 vision token 压缩 + memory 机制都跟上
5. **End-to-end RL 跨整个 desktop**：从"每个 task 单独训练 evaluator"到"通用 desktop reward model"——能判断任意 OS-level 操作是否完成。这是 GUI agent 通向 AGI 的核心 milestone
6. **Agent + Computer Use 在企业落地**：当前商业化主要是消费场景（Operator 订餐购物），2026 可能在企业 RPA 替代场景（财务报表、CRM 填写、客服后台）爆发——这些场景任务结构化、可标准化、reward 容易定义，是 GUI agent 真正的 product-market fit 落点

**反思**：当前 GUI agent OSWorld ~40% 的天花板瓶颈在哪？
- 一部分在 grounding（看不准）→ 继续做 SeeClick / OmniParser 风格 grounding pretrain
- 大部分在 long-horizon decision（多步累积错误）→ reasoning + reflection + RL 是主战场
- 一部分在环境理解（不懂应用本身的语义）→ 需要在更多 app 上预训练 trajectory

加分要点：
- 能联系本书 10.3 RLVR、15.4 Search-R1 / Agent-R1，指出"reasoning + agent + RL"三位一体是 2025-2026 大趋势
- 能指出 GUI agent 是 LLM 走向真实世界的"最后一公里"——一旦稳定可用，软件交互范式会被重写
- 能批判性指出 GUI agent 商业化的最大障碍不是技术而是安全与信任（用户不敢让 agent 真去操作支付）

</details>

---

## 10. 延伸阅读

- [Anthropic Computer Use docs](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) — 官方文档 + cookbook，含 quickstart docker
- [Anthropic Quickstart GitHub](https://github.com/anthropics/anthropic-quickstarts) — 把 computer use loop 封装好的参考实现
- [OSWorld GitHub](https://github.com/xlang-ai/OSWorld) — VM setup + 369 task evaluator + leaderboard
- [SeeClick GitHub](https://github.com/njucckevin/SeeClick) — SeeClick 模型 + grounding 数据合成代码
- [UI-TARS GitHub & paper](https://github.com/bytedance/UI-TARS) — 字节开源 7B/72B 全 size GUI agent，含训练 recipe
- [UI-TARS Desktop](https://github.com/bytedance/UI-TARS-desktop) — 基于 UI-TARS 的桌面 app，可直接试玩
- [Set-of-Mark prompting](https://github.com/microsoft/SoM) — 微软 SoM 的官方实现
- [OmniParser](https://github.com/microsoft/OmniParser) — 微软 GUI element parser，SoM 流派的标准工具
- [Mobile-Agent](https://github.com/X-PLUG/MobileAgent) — 阿里 mobile agent 开源实现
- [AndroidWorld](https://github.com/google-research/android_world) — Android 116 task benchmark
- 推荐继续读本教程的 **15.3 节《真实环境 RL》** —— 本节讲 GUI agent 模型与产品形态，15.3 讲怎么在 OSWorld 等 env 上跑 RL；以及 **15.4 节《Reasoning + Agent》** —— GUI agent 与 reasoning 融合的更多变体；以及 **16.1 节《VLM》** —— 理解 GUI screenshot 处理背后的 vision encoder 与高分辨率挑战
