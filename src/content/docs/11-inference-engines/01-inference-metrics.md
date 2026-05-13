---
title: "11.1 推理性能指标：TTFT / TBT / throughput / 容量"
description: "LLM 推理服务有两套截然不同的视角：单 stream 看延迟（TTFT、TBT、E2E latency），整个系统看吞吐（throughput、goodput、QPS）——而这两套视角的根本来源是 prefill compute-bound 与 decode memory-bound 两个阶段在硬件上完全不同的瓶颈。本节把指标定义、prefill / decode 的 roofline 直觉、b"
---

> ⏱ 预计阅读 35 分钟 ｜ 难度 ★★ ｜ 前置：4.7 KV Cache 原理与实现

## 一句话本节讲什么

LLM 推理服务有两套截然不同的视角：**单 stream 看延迟（TTFT、TBT、E2E latency），整个系统看吞吐（throughput、goodput、QPS）**——而这两套视角的根本来源是 **prefill compute-bound 与 decode memory-bound** 两个阶段在硬件上完全不同的瓶颈。本节把指标定义、prefill / decode 的 roofline 直觉、batch size 与 throughput 的关系、TTFT / TBT 的 trade-off、以及单 GPU 的容量估算公式讲清楚——读完应当能对 "我们的服务 P99 TTFT 800ms 怎么办"这种问题给出工程化的诊断路径，并能为 11.2-11.5 的具体优化技术建立"它们到底在优化哪个指标"的判断框架。

---

## 1. Mental model（直觉）

### 1.1 两套视角：单 sample vs 整个系统

LLM 推理性能讨论里最常见的混乱是**视角错位**：

- 用户视角（单 sample）：**我等了多久才看到第一个字？后续输出多流畅？整段答案多久出完？** → TTFT、TBT、E2E latency
- 运维视角（整个系统）：**这一台 GPU 一秒能为多少人产出多少 token？跑一批离线任务多久能跑完？** → throughput、goodput、QPS

两个视角不仅指标不同，**优化方向甚至矛盾**：把 batch 开大 → throughput 上升 → 单个 user 的 TBT 反而变差（同一份 KV cache 读上来要服务更多 user，每个 user 等的时间更长）。一个成熟的推理服务不是"追求某个指标极大化"，而是**在 SLO 约束下追求 goodput 最大化**——只有那些满足 SLO 的请求才算"有效产出"。

```
单 sample 视角 (latency-oriented)         整个系统视角 (throughput-oriented)
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  请求 ────────────► 第1个token │         │  N 个并发请求 同时入 batch    │
│         TTFT                 │         │  ────► continuous batching    │
│         ──────► 后续每个token │         │                               │
│              TBT (流式)       │         │   GPU ──► token/s ──► 用户   │
│  ──────────────► 完整回答完成 │         │                               │
│           E2E latency        │         │  关心 throughput / GPU 利用率 │
└──────────────────────────────┘         └──────────────────────────────┘
```

### 1.2 Prefill vs Decode：硬件瓶颈完全相反

这是 4.7 §1.3 已经建立、本节继续深化的核心心智模型。

**Prefill**（一次处理整段 prompt 的 N 个 token）：
- 输入 N 个 token、所有位置的 attention 并行算 → 与训练 forward 完全一样
- N 个 token × 模型参数 → **算术强度高**（每 byte 参数被复用 N 次）
- 瓶颈在 **GPU matmul 算力**（FLOPs）→ **compute-bound**
- A100 / H100 的 Tensor Core 利用率可以打到 60-95%
- 决定 **TTFT**

**Decode**（生成时一次产 1 个 token）：
- 输入 1 个 token、要从 HBM 把整个模型权重 + 整段 KV cache 读出来算一次 forward
- 算术强度极低（每 byte 参数只被 1 个 token 用一次）
- 瓶颈在 **HBM 带宽**（memory bandwidth）→ **memory-bound**
- Tensor Core 经常闲在 10-30% 利用率，等数据从 HBM 来
- 决定 **TBT（time between tokens）**

直觉对照表：

| 维度 | Prefill | Decode |
|---|---|---|
| 一次处理 token 数 | N (整段 prompt) | 1 |
| Attention 形状 | $N \times N$ (方阵) | $1 \times (T_{\text{cur}}+1)$ (一行) |
| 主要瓶颈 | 算力 (compute-bound) | 带宽 (memory-bound) |
| GPU 利用率 (FLOPs) | 高 (60-95%) | 低 (10-30%) |
| 决定指标 | TTFT | TBT |
| 优化方向 | FlashAttention / TP / chunked prefill | KV 量化 / GQA / batch 摊薄 |

**这个心智模型是 11.x 全章的地基**——后面 PagedAttention、Continuous Batching、Speculative Decoding 等技术，都可以归类成"这个技术主要优化哪个阶段、哪个指标"。

### 1.3 一句话直觉：单 stream LLM 推理为什么这么慢？

70B 模型 bf16 占 140 GB；单卡 H100 装不下，假设 8 卡 TP 后单卡 17.5 GB 权重。decode 一步要把这 17.5 GB 全部从 HBM 读出来过一遍 → H100 HBM3 带宽 3 TB/s → **理论上限 3000 / 17.5 ≈ 170 step/s 单卡** → 8 卡同步走，约 **170 token/s** 单 stream（实际 50-80 token/s，受调度、TP 通信、kernel overhead 影响）。

如果模型再大、context 再长（KV cache 也要读），单 stream 速度只会更慢——**这是单 stream LLM 推理慢的物理上限，FlashAttention / 算法优化都难突破**。要再快，只有：(1) 提升带宽（换更新硬件 / NVLink 互联）；(2) 减少 I/O（量化、KV 压缩）；(3) 用投机解码"一次蹦出多个 token"（11.5）。

但是！**多 user batch 起来一起 decode** 是另一回事——同一份模型权重读上来同时服务 N 个 user，每个 user 的 decode 计算几乎不增加（FLOPs 在 memory-bound 时几乎免费），整体 throughput 接近 N × 单 stream 的速度。这就是 vLLM Continuous Batching 收益的根本来源，也是为什么"throughput 大 ≠ latency 低"。

---

## 2. 公式与原理

### 2.1 核心指标速查表（必背）

| 指标 | 定义 | 数学 | 主要影响因素 |
|---|---|---|---|
| **TTFT** (Time To First Token) | 请求发出 → 第 1 个 output token 返回的时间 | $T_{\text{prefill}} + T_{\text{decode}}^{(1)}$ | prompt 长度 $N$、prefill batch、queue 等待 |
| **TBT** (Time Between Tokens) | 后续每个 output token 之间的间隔 | $\Delta t_i = t_i - t_{i-1}$ ($i \ge 2$) | KV cache 大小、batch size、HBM 带宽 |
| **TPOT** (Time Per Output Token) | 行业另一种叫法，等价于 TBT | 同 TBT | 同 TBT |
| **ITL** (Inter-Token Latency) | 同 TBT 的另一种叫法（vLLM 文档常用） | 同 TBT | 同 TBT |
| **E2E Latency** (端到端延迟) | 请求发出 → 最后一个 token 返回 | $\text{TTFT} + (M-1) \cdot \text{TBT}$ | $M$ 主导，TBT 主导 |
| **Throughput** (输出 token/s) | 整个系统每秒产出的 output token 数 | $\frac{\sum_i M_i}{T_{\text{wall}}}$ | batch size、GPU 利用率 |
| **TPS** (Tokens Per Second, 单 stream) | 单个用户的输出速度 | $1 / \text{TBT}$ | 单 stream 视角的"流式速度" |
| **QPS** (Queries Per Second) | 系统每秒处理的请求数 | $\frac{N_{\text{req}}}{T_{\text{wall}}}$ | latency 分布、并发数 |
| **Goodput** | 满足 SLO 的"有效" throughput | $\frac{\sum_i M_i \cdot \mathbf{1}[\text{SLO 满足}]}{T_{\text{wall}}}$ | SLO 越严，goodput 越低 |

**关于 percentile**：所有 latency 类指标必须看分布，不能只看 avg——production 通常用 **p50 / p95 / p99**：

$$\text{p99 TTFT} = \inf \{\, x : \Pr[\text{TTFT} \le x] \ge 0.99 \,\}$$

意思是"99% 的请求 TTFT 不超过这个值"——长尾才是 SLO 杀手（详见 §4 第 1 条）。

### 2.2 Prefill / Decode 的 roofline 模型

**算术强度** (Arithmetic Intensity, AI)：

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes loaded from HBM}} \quad (\text{FLOPs / byte})$$

GPU 的 roofline 决定了：

$$\text{Achievable throughput} = \min\Bigl(\text{Peak FLOPs},\ \text{HBM bandwidth} \times \text{AI}\Bigr)$$

当 $\text{AI} > \text{Peak FLOPs} / \text{HBM BW}$（即"屋脊点"）时是 compute-bound；反之 memory-bound。

H100 (SXM) 屋脊点：

$$\text{ridge point} = \frac{989 \text{ TFLOPS (bf16)}}{3.35 \text{ TB/s}} \approx 295 \text{ FLOPs/byte}$$

**Prefill 单层 attention（朴素 MHA、batch=1、seqlen=N）**：
- FLOPs $\approx 2 N^2 d$（$QK^\top$ + score·V）
- Bytes (KV) $\approx 2 N d \cdot \text{dtype}$
- AI $= N / \text{dtype}$ → bf16 下 $N / 2$；当 $N > 590$ token 时 compute-bound（一般 prompt 都到这个量级）

**Decode 单层 attention（batch=1、cache 长 T）**：
- FLOPs $\approx 2 T d$（1 行 q × T 行 K，再 score · V）
- Bytes $\approx 2 T d \cdot \text{dtype}$
- AI $= 1 / \text{dtype} = 0.5$（bf16）→ **永远 memory-bound**，与 T 无关

但 batch=B 的 decode（多 user 共享同一次模型权重读取）：
- FLOPs $\approx B \cdot 2 T d$
- Bytes (weights, 每 user 共用) + Bytes (KV, 每 user 一份) $\approx W + B \cdot 2 T d \cdot \text{dtype}$
- 当 $B$ 大到 weights 部分被摊薄、AI $\to 1/\text{dtype}$ 仍然是 memory-bound——但 **throughput 几乎随 B 线性增长**（FLOPs 几乎免费），直到 KV cache 把显存撑爆或 attention compute 自己变成瓶颈

**结论**：
- 单 stream decode 慢、多 user batch 摊薄读 weights 后 throughput 暴涨——**Continuous Batching 收益的物理基础**
- prefill 一开始就 compute-bound、加 batch 收益小（已经压满算力）——**chunked prefill 把 prefill 切片、给 decode 让出带宽**

### 2.3 单 stream decode 速度的物理上限

decode 一步要读 weights + 当前 KV cache。设模型 weight 大小 $W$（已分片到本卡）、KV cache $K$（本 sample），HBM 带宽 $B_{\text{HBM}}$：

$$T_{\text{step}}^{\text{lower}} \approx \frac{W + K}{B_{\text{HBM}}}$$

$$\text{TPS}^{\text{upper}} = \frac{1}{T_{\text{step}}} \approx \frac{B_{\text{HBM}}}{W + K}$$

例 1：**LLaMA-3 70B (140 GB bf16) 单卡 H100 假装能装下**（实际不行，仅作上限估算）：

$$\text{TPS} \le \frac{3000 \text{ GB/s}}{140 \text{ GB}} \approx 21 \text{ token/s}$$

例 2：**LLaMA-3 70B + TP=8 → 单卡 17.5 GB 权重**：

$$\text{TPS} \le \frac{3000}{17.5} \approx 170 \text{ token/s}$$

实测在 vLLM 上单 stream 大约 80-120 token/s（kernel overhead、TP all-reduce 通信、KV cache 读取占 10-30 GB/s 等吃掉差距）。

例 3：**LLaMA-3 8B (16 GB bf16) 单 H100**：

$$\text{TPS} \le \frac{3000}{16} \approx 187 \text{ token/s}$$

实测 100-150 token/s。

**这是诊断推理性能的"理论天花板尺"**——如果你的实测 TBT 离理论上限还差 5×，先怀疑工程实现（kernel、调度、量化没开），而不是去叫板硬件。

### 2.4 batch size 对 throughput 的影响

设单 user 的 decode TPS 为 $r$（受 §2.3 限制），weights 读一次后在 batch 内复用：

$$\text{Throughput}(B) \approx \min\bigl(B \cdot r,\ \text{compute roof}\bigr)$$

阶段性表现：

- $B = 1$：throughput = $r$（例：LLaMA-3 70B 80 token/s）
- $B$ 增加（仍 memory-bound）：throughput 近似 $B \cdot r$ 线性增长
- $B$ 超过某临界点（KV cache I/O 与 weight I/O 接近，AI 不再增长）：throughput 趋于饱和
- $B$ 再大：要么显存爆（KV cache 占满），要么 compute roof 切换为 compute-bound

**实战经验值**（vLLM benchmark 数据）：
- LLaMA-3 70B + 8×H100 + GQA-8 + 2k context：单 stream ~100 tok/s；batch=64 时总 throughput ~3000-4000 tok/s（每 user 平均掉到 ~50 tok/s）
- batch 越大，单 user TBT 越差但总 throughput 越高 → trade-off 在 SLO 这条线上

### 2.5 容量规划公式

单 GPU 能装的最大 batch size 取决于显存预算：

$$B_{\max} = \left\lfloor \frac{M_{\text{GPU}} - M_{\text{model}} - M_{\text{activation}} - M_{\text{overhead}}}{M_{\text{KV per sample}}} \right\rfloor$$

各项含义：
- $M_{\text{GPU}}$：单卡显存（H100 80 GB / A100 40 或 80 GB）
- $M_{\text{model}}$：本卡承载的模型权重（按 TP 分片后的份额）
- $M_{\text{activation}}$：前向激活、临时 buffer，约 1-4 GB
- $M_{\text{overhead}}$：CUDA context、framework 开销、KV cache 碎片，约 2-4 GB
- $M_{\text{KV per sample}}$：单 sample 的 KV cache（按 4.7 §2.4 公式算）

**例**：8×H100 跑 LLaMA-3 70B (TP=8)，目标 8k context：
- $M_{\text{GPU}} = 80$ GB
- $M_{\text{model}} = 140 / 8 = 17.5$ GB
- $M_{\text{activation}} \approx 2$ GB
- $M_{\text{overhead}} \approx 3$ GB
- $M_{\text{KV per sample}}$（GQA-8 @ 8k）≈ 2.5 GB（沿用 4.7 §2.4 表）
- $B_{\max} = (80 - 17.5 - 2 - 3) / 2.5 \approx 23$

**但** 实际配置不能"装满 23"——因为 batch 越满 latency 越飘，必须留余量给 SLO。production 配置通常取理论容量的 60-80%（如此例配 batch ~15）。

更"业界"的容量公式（综合 SLO）：

$$B_{\text{prod}} = \min\Bigl(B_{\max},\ \arg\max_B \{\text{p99 TBT}(B) \le \text{SLO}\}\Bigr)$$

---

## 3. 最小代码示例

### 3.1 测 TTFT 与 TBT（OpenAI streaming SDK）

```python
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")  # 指向 vLLM server

def measure(prompt: str, model: str, max_tokens: int = 200):
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    times, ttft = [], None
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0                              # 第 1 个 token 的时间
            times.append(now)
    # TBT 从第 2 个 token 开始算 token 间隔（第 1 个含 prefill）
    tbts = [times[i] - times[i - 1] for i in range(1, len(times))]
    e2e = times[-1] - t0
    return {
        "ttft_s": ttft,
        "tbt_avg_ms": 1000 * sum(tbts) / max(len(tbts), 1),
        "tbt_p99_ms": 1000 * sorted(tbts)[int(len(tbts) * 0.99)] if tbts else 0,
        "e2e_s": e2e,
        "n_tok": len(times),
        "tps": len(times) / e2e,                             # 1/平均TBT 的近似
    }

print(measure("用一段话解释 self-attention", model="meta-llama/Meta-Llama-3-8B-Instruct"))
```

**关键点**：
- `stream=True` 必须开，否则只有 E2E 没有 TTFT / TBT
- TTFT 是从请求发出到**第一个非空 delta** 返回的时间——它包含 prefill + 1 step decode（见 §4 踩坑 3）
- TBT 从第 2 个 token 开始统计 token 间隔，第 1 个间隔会被 prefill 拖大

### 3.2 简单并发 benchmark（locust 风格、≤ 35 行）

```python
import asyncio, time, statistics
import httpx

URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Authorization": "Bearer EMPTY"}

async def one_request(client, prompt, model="meta-llama/Meta-Llama-3-8B-Instruct"):
    t0 = time.perf_counter()
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 256, "stream": True}
    ttft, last = None, None
    n_tok = 0
    async with client.stream("POST", URL, headers=HEADERS, json=payload, timeout=120) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                now = time.perf_counter()
                if ttft is None: ttft = now - t0
                last = now
                n_tok += 1
    return ttft, (last - t0) if last else 0, n_tok

async def bench(concurrency=32, total=200, prompt="解释 transformer"):
    async with httpx.AsyncClient(http2=True) as client:
        sem = asyncio.Semaphore(concurrency)
        async def run():
            async with sem:
                return await one_request(client, prompt)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[run() for _ in range(total)])
        wall = time.perf_counter() - t0
    ttfts = sorted(r[0] for r in results)
    e2es  = sorted(r[1] for r in results)
    n_tok = sum(r[2] for r in results)
    p = lambda xs, q: xs[int(len(xs) * q)]
    print(f"concurrency={concurrency}  wall={wall:.1f}s")
    print(f"TTFT  p50={p(ttfts,.5)*1000:.0f}ms  p99={p(ttfts,.99)*1000:.0f}ms")
    print(f"E2E   p50={p(e2es,.5):.2f}s  p99={p(e2es,.99):.2f}s")
    print(f"throughput={n_tok / wall:.1f} tok/s   QPS={total/wall:.1f}")

asyncio.run(bench())
```

**关键点**：
- `Semaphore(concurrency)` 控制并发数——这才是 batch 真正大小的输入端
- 每条请求独立计时，最后**统计分布而非 avg**（必须看 p50 / p99）
- 真实 production benchmark 还要做：(1) prompt 长度采样自真实分布；(2) request 到达率服从 Poisson 而非 burst；(3) warmup（前 10s 不计入）。生产工具如 [vLLM benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py) 已含这些细节。

### 3.3 容量估算函数（≤ 15 行）

```python
def max_batch_size(gpu_mem_gb: float,
                    model_size_gb: float,
                    kv_per_sample_gb: float,
                    activation_overhead_gb: float = 5.0,
                    utilization: float = 0.8) -> int:
    """估算单 GPU 在显存约束下能装的最大 batch（不考虑 SLO）。
    utilization < 1：留余量，避免接近显存上限时 latency 飘。"""
    available = (gpu_mem_gb - model_size_gb - activation_overhead_gb) * utilization
    if available <= 0:
        raise ValueError("model + overhead > GPU memory; need TP / smaller model")
    return max(1, int(available // kv_per_sample_gb))


# 用法 / 校验：
# H100 80GB, LLaMA-3 70B TP=8（单卡 17.5GB weight），GQA-8 8k context KV=2.5GB
# >>> max_batch_size(80, 17.5, 2.5)
# 18  ← 含 80% 利用率裕度
# A100 40GB, LLaMA-3 8B 单卡（16GB weight），8k context MHA KV=4GB
# >>> max_batch_size(40, 16, 4)
# 3   ← 8B 在 A100 40G 上 batch 都很紧张
```

---

## 4. 工程踩坑与经验

- ❗ **永远看 p50 / p95 / p99，不看 avg**。Latency 分布是重尾的——一个 GC、一个 prefill 阻塞就能把单点拉到 5 秒，几百次正常请求也拉不动 avg，但对 p99 影响巨大。SLO 的本质是"99% 的用户体验"——优化 avg 而 p99 不动等于没优化。Production 监控板上 latency 默认 panel 应当是 p50 / p95 / p99 三条线，avg 仅供参考。

- ❗ **Benchmark 要用真实 prompt 长度分布，不要全用同长度**。一份测试 prompt 全是 256 token，跑出来 throughput 漂亮——上线后真实流量是 (短 prompt + 短回答 80%、长文档总结 15%、长对话 5%) 的混合，prefill / decode 比例完全不同，throughput 直接腰斩。**正确做法**：从你自己的 production log 采样 prompt 长度直方图，benchmark 用同分布合成（vLLM `benchmark_serving.py --dataset-name sharegpt` 提供了一个常用代理分布）。

- ❗ **TTFT 包含的不只是 prefill，还有 1 步 decode + 网络 / 序列化开销**。"用户视角的 first token"是 streaming response 的第一个 chunk 到达浏览器/客户端的时间——它 = queue 等待 + prefill + 1 step decode + 第 1 chunk 序列化与传输。论文里报告的"prefill latency"通常只测前 3 项中第 2 项；做 SLO 时一定问清"这个 TTFT 测的是哪一段"。

- ❗ **Throughput 在 batch 增大时不是线性 scale 的**。前面 §2.4 公式说"理想线性"，但实际 batch 大到一定程度后：(1) KV cache 把 HBM 用满 → 触发 swap / preempt；(2) attention 自己变 compute-bound（QK 矩阵随 batch 增大）；(3) GPU SM 调度饱和。实测 scaling 曲线一般在 batch ~64 之后就开始 sublinear。**Sweet spot 通常在显存允许的最大 batch 的 60-70%**——再大 throughput 几乎不涨而 latency 飘。

- ❗ **流式 (streaming) vs 非流式的 SLO 完全不同**。Chat / coding assistant 必须 streaming，关心 TTFT + TBT（用户能"边读边等"）；离线 batch infer (打分、数据合成) 非流式，只关心 throughput / total wall time，TBT 不重要。两类业务用同一个推理 cluster 时，**调度策略应当区分**——flag "实时" / "批量" 流量，前者 prefill 优先小 chunked、后者允许 prefill 排长队压满 GPU。

- ❗ **监控指标必须 per-model + per-context-bucket 分桶**。把所有请求的 TBT 混在一起看分布，结果是无意义的——长 context 的 TBT 自然比短 context 大（KV cache I/O 多）。**正确**：按 (model_name, prompt_length_bucket) 分桶，每个桶单独看 p50/p99；不同桶 SLO 也不同（chat 与 long-context QA 不能用同一个阈值）。

- ❗ **Sampling 参数对 throughput 影响大**。`max_tokens` 大、`temperature` 高（输出更"啰嗦"）、`top-p / top-k` 宽（接受率高）→ 平均 generation 长度变长 → 单请求占 GPU 时间长 → throughput 下降。Benchmark 时 `max_tokens` 必须和真实业务对齐；`temperature=0`（greedy）的 benchmark 数字与 `temperature=0.7` 差异可达 30%+。

- ❗ **A100 / H100 / L40S 不同硬件 benchmark 数据差异 2-5×，必须标硬件**。HBM 带宽：A100-40GB 1.6 TB/s、A100-80GB 2.0 TB/s、H100-SXM 3.35 TB/s、H100-NVL 3.9 TB/s——decode TPS 几乎正比于带宽。Compute：H100 bf16 990 TFLOPS vs A100 312 TFLOPS——prefill 速度直接 3×。**Benchmark 报告必须写"卡型 + TP/PP 配置 + framework + dtype"**，否则数字没有可比性。

- ❗ **Goodput > throughput 是更诚实的指标**。一台 GPU 跑出 5000 tok/s 的 throughput，但 30% 请求 p99 TTFT 超过 SLO（用户已经放弃) → 真正"有效产出"只有 3500 tok/s。容量规划应当对齐 goodput 而非 raw throughput；自动扩缩容触发条件也应当是"goodput 接近上限"而非"GPU 使用率高"。

- ❗ **Prefill 与 decode 不解耦时，长 prompt 用户能"霸占"整台 GPU**。一个用户发 32k token 的长文档，朴素调度下这次 prefill 要占 GPU 数百毫秒，期间所有其他用户的 decode step 全部 stall——他们 TBT 直接飙升 10×。**Chunked prefill**（11.2 详讲）把这次大 prefill 切成 (e.g.) 512-token 的 chunk，与其他人的 decode 混在一个 batch 跑，长 prompt 的 TTFT 略变长但全系统的 TBT p99 大幅改善——这是现代推理引擎的默认配置。

---

## 5. 经典 paper

- **Pope et al., 2022 — Efficiently Scaling Transformer Inference (arXiv:2211.05102)** — Google PaLM 团队系统化讨论 LLM 推理的开山之作。**为什么读**：本节 §1.2 prefill / decode 两阶段、§2.2 roofline 模型、§2.5 容量规划公式都源自这篇。Section 2 的 latency / throughput / cost 三角分析是 production 推理工程师的必读 framework。
- **Patel et al., 2023 — Splitwise: Efficient Generative LLM Inference Using Phase Splitting (arXiv:2311.18677)** — Microsoft Research 提出"prefill 与 decode 跑在不同硬件 / 不同 GPU pool 上"的彻底解耦方案。**为什么读**：把 §1.2 的两阶段分离思想推到极致——prefill 用算力强的卡（H100 / TP 多卡）、decode 用带宽强的卡（HBM 大），两者解耦调度。这是理解"为什么 prefill 与 decode 应当分开优化"的最佳论文，也启发了 NVIDIA 后续的 "disaggregated serving"。
- **Agrawal et al., 2024 — SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills (arXiv:2308.16369)** — Chunked Prefill 的提出。**为什么读**：本节 §1.2 末尾、§4 最后一条提的"chunked prefill 平衡 TTFT / TBT" 的原始论文。把长 prefill 切 chunk + 与 decode 混 batch 的调度算法详细讲了一遍——vLLM / SGLang 现代版的默认调度策略基本来自这里。

---

## 6. 自测与面试题

**Q1（概念）**：用两句话各解释为什么 LLM 推理 prefill 是 compute-bound、decode 是 memory-bound？

<details>
<summary>Answer sketch</summary>

**Prefill compute-bound**：一次性输入 N 个 token、所有位置 attention 与 FFN 并行算 → 同一份模型权重被 N 个 token 复用 → 算术强度（FLOPs / byte loaded）随 N 增长，N 不算太小（>590 on H100 bf16）就把 GPU 算力打满，瓶颈在 Tensor Core 的 FLOPs。

**Decode memory-bound**：一次只生成 1 个 token、要把整个模型权重 + 整段 KV cache 从 HBM 读出来过一遍 → 每 byte 权重只被 1 个 token 用一次 → 算术强度极低（约 1/dtype）→ 远低于 GPU 屋脊点 → GPU SM 大量闲置，瓶颈在 HBM 带宽。

加分：
- 能说"Continuous Batching 通过把多 user decode 拼成一个大 batch、共享一次权重读取，把 decode 的 AI 拉到接近 compute-bound" → throughput 收益的物理来源
- 能说"prefill 的 batching 收益小（已经 compute-bound）、decode 的 batching 收益大（memory-bound 时几乎免费加 user）"

</details>

**Q2（计算）**：H100 (HBM 带宽 3 TB/s) 跑 LLaMA-3 70B (140 GB bf16 权重)、TP=8 分片（单卡 17.5 GB 权重），不考虑 KV cache 与 kernel overhead，估算单 stream decode TPS 上限。如果换成 TP=4，单卡上限是多少？再如果用 INT8 weight 量化 + TP=8，单卡上限是多少？

<details>
<summary>Answer sketch</summary>

公式：$\text{TPS}_{\text{upper}} = B_{\text{HBM}} / W_{\text{shard}}$（不计 KV cache）

**TP=8（单卡 17.5 GB bf16）**：
$$\text{TPS} \le \frac{3000 \text{ GB/s}}{17.5 \text{ GB}} \approx 170 \text{ token/s}$$

**TP=4（单卡 35 GB bf16）**：
$$\text{TPS} \le \frac{3000}{35} \approx 86 \text{ token/s}$$

——TP 度增大单 stream 速度上升（因为单卡加载量 ÷ TP 度），代价是通信开销 + 卡数翻倍

**TP=8 + INT8 weight（单卡 8.75 GB）**：
$$\text{TPS} \le \frac{3000}{8.75} \approx 343 \text{ token/s}$$

——INT8 直接把 I/O 减半，速度翻倍

加分：
- 能指出实际 TPS 通常是理论上限的 50-70%（kernel overhead、TP all-reduce 通信、KV cache I/O 占带宽）
- 能说 KV cache 也要算进 I/O：8k context GQA-8 KV cache ≈ 2.5 GB → 每 step I/O 变成 (17.5 + 2.5) = 20 GB → TPS 上限降到 150 token/s
- 能说 batch=N 时 throughput ≈ N × 单 stream 上限（在 memory-bound 区间内）

</details>

**Q3（实战）**：你的 LLM 服务（70B + 8×H100 + GQA-8）p99 TTFT 已经飘到 1.5 秒（SLO 是 500ms），列出 3 个优化方向，按优先级排序，每个方向说明预期收益与会引入的 trade-off / 工程成本。

<details>
<summary>Answer sketch</summary>

**先看几个分桶诊断 (前置)**：是不是某些 prompt 极长把 p99 拉飞？是不是某段时间 QPS 暴涨触发 queue？是不是混用了流式 / 非流式请求互相阻塞？

**方向 1：开启 chunked prefill (11.2)**
- 收益：长 prompt 的 prefill 不再阻塞短 prompt 与 decode，p99 TTFT 通常降 2-5×
- Trade-off：超长 prompt 的 TTFT 略上升（被切成 chunk 排队）；需要 vLLM ≥ 0.4 / SGLang，配置 `enable_chunked_prefill=True`
- 工程成本：低（一行配置）→ **首选**

**方向 2：启用 Prefix Caching / RadixAttention (11.3)**
- 收益：如果业务有大量共享 system prompt（chat、agent、RAG），prefill 部分跳过缓存命中的 token，TTFT 降 50%+ 甚至更多
- Trade-off：纯单 turn / 无共享前缀的业务无收益；需要 SGLang 或 vLLM 的 Automatic Prefix Caching；新增 cache pool 占显存
- 工程成本：中（要监控命中率、cache 容量调参）

**方向 3：扩容 / 限流，从根本上降低 queue 等待**
- 收益：若 p99 TTFT 飘的根因是 burst 流量导致 prefill queue 堆积（而非单请求 prefill 慢），加 1-2 个 replica / 加限流即可
- Trade-off：扩容花钱；限流降总 throughput
- 工程成本：低（pod 扩容）→ **如果能确诊根因是排队，优先做**

**方向 4（次选）：投机解码 / Speculative Decoding (11.5)**
- 收益：主要降 TBT 不降 TTFT；只在 TBT 也飘时考虑
- 不直接对症 TTFT

**方向 5（深一层）：换 prefill / decode 分离架构 (Splitwise 风格)**
- 收益：prefill 与 decode 各跑各的硬件、互不干扰
- Trade-off：架构改动大、要新建 GPU pool；只有规模够大才合算

加分：
- 能说"先看 trace 与 metrics 确诊瓶颈再优化"——而不是上来就堆技术
- 能区分"prefill 算得慢（compute）" vs "排队等 prefill（queue）"——前者要 chunked prefill / TP 加大，后者要扩容
- 能提一句"sampling 参数与 max_tokens 也间接影响 TTFT 上下游"——大 batch 里 long generation 没释放、新 prefill 进不来

</details>

---

## 7. 延伸阅读

- [vLLM benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py) — 官方 production-grade benchmark 脚本，含 sharegpt 真实 prompt 分布、Poisson 到达、TTFT/TBT 分位数统计——本节 §3.2 的工业级版本
- [NVIDIA GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 官方 LLM 推理 benchmark 工具，标准化 OpenAI API 兼容接口下的 TTFT / TBT / throughput 测量
- [trtllm-bench (TensorRT-LLM)](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html) — TensorRT-LLM 自带的 benchmark CLI，与 vLLM 对比时常用
- [Anyscale Blog — Reproducible Performance Metrics for LLM Inference](https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference) — 对各家 framework benchmark 数据"水分"的诚实分析，必读避坑
- [Mosaic / Databricks — LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) — production 推理调优 best practice，TTFT / TBT / throughput 三角的工程案例
- [Pope et al. 2022 — Efficiently Scaling Transformer Inference (arXiv:2211.05102)](https://arxiv.org/abs/2211.05102) — Google 推理 scaling 总论，本节 §2 公式的源头
- [Patel et al. 2023 — Splitwise (arXiv:2311.18677)](https://arxiv.org/abs/2311.18677) — prefill / decode 物理解耦的开山论文
- [Agrawal et al. 2024 — SARATHI (arXiv:2308.16369)](https://arxiv.org/abs/2308.16369) — chunked prefill 算法的原始论文
- 推荐继续读本教程的 **11.2 节《PagedAttention 与 Continuous Batching（vLLM）》**——本节多次预告的"chunked prefill + iteration-level scheduling"在 11.2 系统讲透；KV cache 碎片管理也在那里
- 推荐继续读本教程的 **11.3 节《RadixAttention 与 Prefix Cache（SGLang）》**——多 user 共享 system prompt / multi-turn 场景下 TTFT 的根本性优化
- 推荐继续读本教程的 **11.4 节《量化：GPTQ / AWQ / GGUF / FP8 / SmoothQuant》**——weight / KV cache 量化对 throughput 的提升
- 推荐继续读本教程的 **11.5 节《投机解码：Speculative / Medusa / EAGLE》**——降 TBT 的核心技术
