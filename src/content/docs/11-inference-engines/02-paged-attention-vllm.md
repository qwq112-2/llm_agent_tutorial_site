---
title: "11.2 PagedAttention 与 Continuous Batching（vLLM）"
description: "4.7 已经讲清 KV cache 是 LLM 推理的 OOM 第一杀手、5.2 用 GQA/MLA 在架构层砍 KV cache；本节讲在内存管理层做的两件事——PagedAttention（Kwon 2023, vLLM）借鉴 OS 虚拟内存分页的思想，把 KV cache 切成固定大小的 block、逻辑连续物理离散，把\"为最长序列预留 max_seqlen buffer\"导致的 60-80"
---

> ⏱ 预计阅读 55 分钟 ｜ 难度 ★★★ 🔥 必考 ｜ 前置：4.7 KV Cache、5.2 GQA/MQA/MLA

## 一句话本节讲什么

4.7 已经讲清 KV cache 是 LLM 推理的 OOM 第一杀手、5.2 用 GQA/MLA 在**架构层**砍 KV cache；本节讲在**内存管理层**做的两件事——**PagedAttention**（Kwon 2023, vLLM）借鉴 OS 虚拟内存分页的思想，把 KV cache 切成固定大小的 block、逻辑连续物理离散，把"为最长序列预留 max_seqlen buffer"导致的 60-80% 碎片浪费一刀降到 < 5%；**Continuous Batching**（Yu 2022, Orca）把 batching 的颗粒度从"整个 request"细化到"每一步 iteration"，sample A 一旦生成完 EOS 就立刻让 queue 中的 sample B 接班——两者结合让 LLM 推理 throughput 较 HuggingFace TGI 提升 5-20×。本节会把朴素 KV cache 管理的两大问题（预留浪费 + 碎片化）讲透，给出 PagedAttention 的 block table 心智模型与简化的 Block Manager / 调度器伪代码，并用 vLLM 的 OpenAI Compatible API 跑一个完整 demo，最后给一张 vLLM / SGLang / TensorRT-LLM / LMDeploy / llama.cpp 的对比表与 6+ 条 production 踩坑——是从"能跑通推理"到"能上线服务"的关键一跳，也是 11.3 RadixAttention、11.4 量化、11.5 投机解码三节的共同前置。

---

## 1. Mental model（直觉）

### 1.1 朴素 KV cache 管理的三宗罪

回顾 4.7 §3.2：教科书写法是给每个 sample **预分配一块 max_seqlen 大小的 KV cache buffer**，每生成一个 token 就在 buffer 里 in-place 写下一格。HuggingFace `transformers.generate` 默认就是这样——简单、易实现、对单 sample 够用。但**到了 production 多 user 服务的场景，立刻撞到三个问题**：

**问题 1：内部碎片（internal fragmentation）— 预留浪费**

每个 sample 不知道自己最终会生成多少 token，所以保险起见按 `max_seqlen`（如 4096）预留。可实际请求的平均长度往往只有 200-500 token——**每个 sample 浪费 80-95% 的 KV cache 槽位**。

举例：LLaMA-3 70B GQA-8、bf16，每 token 每层 KV cache 2048 B，80 层 = 160 KB/token；4096 token 的 buffer 就是 640 MB/sample。如果实际只生成 256 token，有效占用 40 MB、**预留浪费 600 MB**——94%。要服务 32 个并发 user，预留浪费就是 19 GB——一张 H100 的 1/4 显存被白白浪费。

**问题 2：外部碎片（external fragmentation）— 不同长度凑不到一起**

不同 user 的 prompt + generation 长度天差地别——有人 100 token、有人 8k token。朴素实现要按最长 padding，前 31 个 sample 各浪费 97% 槽位；或者用 dynamic shape，但显存里就成了"长一块短一块"的奶酪状——新来的 sample 要找到一段连续 max_seqlen 的空闲显存往往找不到（即便总空闲量足够），等于**你有 20 GB 空闲显存却装不下一个 4 GB 的新 user**。

**问题 3：早结束的 sample 释放不掉空间**

朴素 batching 是 **request-level**——一个 batch 里所有 sample 必须**一起生成完**才能整体返回结果。如果 batch 里有 7 个 sample 在 50 步生成完 EOS、1 个 sample 要 500 步——剩下的 7 个槽位**空转 450 步**，KV cache 占着不释放，也不能让 queue 中等待的新 user 进来。GPU 利用率因此长期 < 30%。

**这三个问题的本质都是同一件事**：KV cache 的"分配粒度太粗"——按 sample 整体分配 max_seqlen buffer，无法应对动态长度与早结束。

### 1.2 PagedAttention 的核心 idea：从 OS 借虚拟内存分页

Kwon 等人（vLLM 团队）2023 年的关键 insight：**OS 几十年前就解决过完全同构的问题**——进程的虚拟地址空间是连续的，但 OS 不要求物理内存也连续；OS 把物理内存切成 4 KB 的 page，每个进程持有一张 page table 把"虚拟地址 → 物理 page"映射起来。新进程要内存 → 分配空闲 page、填进 page table；进程退出 → page 回收到 free list。**碎片几乎为 0、按需分配、动态扩缩容**。

PagedAttention 把这套机制原样搬到 KV cache 管理上：

- 把 KV cache 切成固定大小的 **block**（vLLM 默认每个 block 容纳 16 个 token 的 KV）
- 每个 sample 持有一张 **block table**：`[逻辑 token id 0..15 → 物理 block A; 逻辑 token id 16..31 → 物理 block B; ...]`
- **逻辑上连续，物理上离散**——sample 看自己的 KV cache 是 token 0, 1, 2, ... 一字排开；底下的物理 block 可以散落在显存任何角落
- 新 token 来了，当前 block 还没满 → 写进当前 block；满了 → 从全局空闲 block pool 拿一个新 block、追加到 block table
- sample 生成完 EOS → 把它持有的所有 block 还给空闲 pool

**收益是颠覆性的**：内部碎片只剩"最后一个 block 没填满"那点（最多浪费 16 个 token），外部碎片完全消失（block 大小固定、互相可替换），KV cache 显存利用率从朴素管理的 ~30% 提升到 ~95%。同样 80 GB H100 能服务的并发 user 数提升 2-4×。

### 1.3 PagedAttention 的 attention kernel：间接索引

OS 分页有一个代价——CPU 每次访问内存都要先查 page table。PagedAttention 同理：**标准 attention kernel 假设 K, V 是物理连续的 tensor `(B, h, T, d_k)`**，可以直接 `K[b, h, t, :]` 取数；PagedAttention 的 K, V 在物理内存里是离散的 block，要先查 block table 把 `逻辑 t → 物理 block_id + offset`、再去对应 block 里取数。

vLLM 团队为此**手写了专门的 CUDA kernel**——给 attention 一个额外的 `block_table` 参数，kernel 内部按表间接索引读 K, V。代价是引入一次 indirection：**性能损失 < 5%**（vs 物理连续 attention），但**显存收益 3-10×**——这笔账太划算。

近年的演进：vLLM 直接基于 FlashAttention v2/v3 做 paged 适配（FlashAttention 主线已合并 paged kernel，称作 `flash_attn_with_kvcache`）；性能与连续 attention 几乎打平。

### 1.4 Continuous Batching：从 request-level 到 iteration-level

PagedAttention 解决了"装得下"的问题，但还有"调度公平"的问题——朴素 batching 是 **request-level**：把 N 个 request 拼成一个 batch、所有 sample 一起 prefill、一起 decode 到所有 sample 都生成完 EOS、再一起返回。这有两个显著弊端：

- **早结束的 sample 拖累整体**——7 个 sample 50 步完成、1 个 sample 500 步，整个 batch 要陪跑 450 步无意义计算
- **新 request 进不来**——queue 中等待的 user 必须等当前 batch 全部结束才能加入下一个 batch

**Continuous Batching**（也叫 **iteration-level batching**，Yu et al. 2022 的 Orca 系统首次提出）的 idea 极简：**调度颗粒度从一整个 request 细化到每一步 decode iteration**。

```
传统 request-level batching：
  step 0:  [A][B][C][D]  ← 4 个 sample 一起开始
  step 50: [A][_][C][_]  ← B, D 已完成 EOS，槽位空转
  step 100:[_][_][C][_]
  ...
  step 500:[_][_][C][_]  ← 等 C 终于完
  return all 4

Continuous batching：
  step 0:  [A][B][C][D]
  step 50: [A][E][C][F]  ← B 完成 EOS → 立刻让 queue 中 E 顶上
                          D 完成 EOS → 立刻让 queue 中 F 顶上
  step 100:[A][E][C][G]  ← F 也完了 → G 顶上
  ...
  return A 200, B 50, C 500, D 50, E 80, F 30, G 100, ...
```

**关键**：每一步 iteration 后，scheduler 检查哪些 sample 完成了 EOS、立刻把它们的 block 释放、把 queue 中等待的新 request 拼进 batch。**没有空槽空转、没有 batch 整体等待**。

Continuous batching 与 PagedAttention 是天作之合：动态进出 batch 时 KV cache 必须能灵活扩缩容、不能要求"提前知道每个 sample 的最大长度"——这正是 PagedAttention 能做到的事。两者结合后，vLLM 在 LLaMA 7B 上 throughput 较 HuggingFace TGI 提升 14-24×（Kwon 2023 论文 §6 实测）；production 经验通常报告 5-20× 提升（取决于负载分布）。

### 1.5 vLLM 的工程定位

vLLM（Berkeley，2023.06 开源）是**第一个把 PagedAttention + Continuous Batching 系统化落地的开源推理引擎**——发布时性能直接碾压 HuggingFace TGI / FasterTransformer，迅速成为 LLM 推理的事实标准。今天 vLLM 已经覆盖：

- **核心调度**：Scheduler + Block Manager（本节主线）
- **kernel**：PagedAttention（自研）+ FlashAttention v2/v3 集成
- **并行**：tensor parallel（Megatron 风格）+ 实验性 pipeline parallel
- **优化**：chunked prefill / prefix caching / FP8 KV cache / multi-LoRA serving / speculative decoding（与 11.5 衔接）/ guided decoding
- **部署**：OpenAI Compatible API server，可直接替换 OpenAI SDK 后端

学习 LLM 推理时 vLLM 是绝对的"必装+必读"——本节会把它的使用、参数、踩坑全部讲透。

---

## 2. 公式与原理

### 2.1 朴素 KV cache 管理的浪费率算账

设服务面对的请求长度分布期望为 $\mathbb{E}[T] = T_{\text{avg}}$、推理引擎按 $T_{\max}$ 预留 buffer。朴素管理的**有效利用率**：

$$\eta_{\text{naive}} = \frac{T_{\text{avg}}}{T_{\max}}$$

典型 production 场景 $T_{\text{avg}} = 256, T_{\max} = 4096$ → $\eta_{\text{naive}} \approx 6\%$——94% 的预留 KV cache 是浪费。再叠加 batch 内 padding、早结束 sample 不释放等因素，整体显存利用率往往低至 **20-30%**——vLLM 论文 Table 1 的实测数字。

**PagedAttention 的浪费率**只剩"最后一个 block 没填满"那点：

$$\eta_{\text{paged}} = \frac{T_{\text{avg}}}{T_{\text{avg}} + (\text{block\_size} - 1)/2}$$

block_size = 16、$T_{\text{avg}} = 256$ → $\eta_{\text{paged}} \approx 97\%$——内部碎片几乎为 0。

### 2.2 PagedAttention 的形式化描述

设 KV cache 总容量 $C$ 个 block，每个 block 容纳 $B_s$ 个 token（vLLM 默认 $B_s = 16$）。**全局结构**：

- **Free Block Pool**：当前未分配的物理 block 索引集合 $\mathcal{F} \subseteq \{0, 1, \dots, C-1\}$
- **Block Manager**：负责 `allocate()` / `free()` 操作，维护 $\mathcal{F}$
- **每个 sample $s$ 的 Block Table** $\text{BT}_s = [b_0, b_1, \dots, b_{n_s-1}]$，其中 $b_i \in \{0, \dots, C-1\}$ 是该 sample 的第 $i$ 个逻辑 block 对应的物理 block id

**逻辑 token id $t$ 在物理 block 中的定位**：

$$\text{block\_idx} = \lfloor t / B_s \rfloor, \quad \text{offset} = t \mod B_s$$

$$K_t = K_{\text{phys}}[\,\text{BT}_s[\text{block\_idx}],\ \text{offset}\,]$$

**关键操作**：

- **生成新 token**：当前最后一个 block 还没满 → 直接写入；满了 → 从 $\mathcal{F}$ pop 一个新 block、append 到 $\text{BT}_s$、写入新 block 的 offset 0
- **sample 完成**：把 $\text{BT}_s$ 里所有 block id push 回 $\mathcal{F}$
- **prefix sharing**（高级特性，与 11.3 SGLang RadixAttention 衔接）：多个 sample 的 prompt 前缀相同 → 它们的 block table 前几个 entry 指向同一物理 block（**Copy-on-Write**：写入时若发现共享，先复制再写，避免污染）

### 2.3 PagedAttention kernel 的伪代码

```
# 标准（连续 K, V）：
for each Q row q in this batch:
    for t in 0..T_cur:
        score[t] = dot(q, K[t]) / sqrt(d_k)
    softmax(score)
    out = sum_t score[t] * V[t]

# Paged（K, V 离散在 block 里）：
for each Q row q in sample s:
    for t in 0..T_cur:
        block_idx  = t // B_s
        offset     = t %  B_s
        phys_block = block_table[s][block_idx]    # ← indirection
        K_t = K_phys[phys_block][offset]
        score[t] = dot(q, K_t) / sqrt(d_k)
    softmax(score)
    out = sum_t score[t] * V_phys[block_table[s][t // B_s]][t % B_s]
```

CUDA 实现里 block table 通常 prefetch 到 shared memory、按 block 循环展开取 K/V，性能损失在 < 5% 以内。FlashAttention v2.3+ 已直接支持 paged KV cache（API: `flash_attn_with_kvcache(q, k_cache, v_cache, block_table, cache_seqlens)`），vLLM 0.5+ 默认调它。

### 2.4 Continuous Batching 的调度逻辑

Scheduler 在每一步 iteration 维护两个集合：

- **Running set** $\mathcal{R}$：当前 batch 内的活跃 sample
- **Waiting queue** $\mathcal{W}$：等待加入 batch 的新 request

每步 iteration 的伪代码：

```
loop:
    # 1) 从 running 中剔除已完成的 sample
    for s in R:
        if s.last_token == EOS or s.length >= s.max_tokens:
            release_blocks(s)            # 还给 free pool
            R.remove(s)
            return_to_user(s)

    # 2) 从 waiting 中尽可能多塞新 sample 进 R
    while W and can_admit(W.front()):
        s_new = W.popleft()
        allocate_initial_blocks(s_new)
        R.add(s_new)

    # 3) 执行一步 forward
    #    - prefill 阶段的 sample 跑 prefill kernel
    #    - decode 阶段的 sample 跑 paged decode kernel
    #    - chunked prefill 把长 prefill 切成小块与 decode 混跑
    forward_one_iteration(R)
```

`can_admit()` 的判断核心是：**剩余空闲 block 数 $\ge$ 该 sample prefill 需要的 block 数 + 安全水位**。这个安全水位很重要——避免 admit 后没空 block 给 decode 继续 append。

### 2.5 显存预算与 swap-out（high-pressure 时）

vLLM 启动时根据 `gpu_memory_utilization`（默认 0.9）算出可用 KV cache 总显存：

$$\text{KV显存} = \text{GPU总显存} \times \text{util} - \text{model权重} - \text{activation预留}$$

总 block 数：

$$C = \frac{\text{KV显存}}{2 \cdot L \cdot h_{\text{kv}} \cdot d_k \cdot B_s \cdot \text{dtype\_bytes}}$$

例：H100 80 GB、`util = 0.9` → 可用 72 GB；LLaMA-3 8B（GQA-8）权重 16 GB、activation 预留 ~4 GB → KV cache 显存 ~52 GB；按 block_size = 16、bf16、80 layer × 8 KV head × 128 d_k 算，每 block 大小 32 KB → 总 block 数 $C \approx 1.7M$。一个 4k context 的 sample 占 $4096/16 = 256$ blocks → 单卡可同时服务 $\sim 6500$ 个 4k 上下文的 sample（理论上限，实际受其他因素影响）。

**preemption 与 swap**：当 running set 中的 sample 需要扩张但 free pool 已空，vLLM 有两种应对：
- **swap-out**：把某些低优先级 sample 的 block 拷贝到 CPU 内存、释放 GPU block（可恢复）
- **recompute**：直接丢掉某些 sample 的 KV cache、稍后重新 prefill（适合短 prompt）

### 2.6 Chunked Prefill：平衡 TTFT 与 TBT

朴素 vLLM 的调度有个问题：**长 prompt 的 prefill 是 compute-bound 大计算**（几秒级），它进 batch 的那一步会**把所有其他 user 的 decode step 卡住**——TBT（time between tokens）尖刺。

**Chunked Prefill**（vLLM 0.4+ 的关键特性）：把超长 prompt 的 prefill 切成固定大小的 chunk（如每 chunk 512 token），与其他 user 的 decode step **混在一个 batch 里跑**——长 prompt 的 prefill 分散到多步完成，期间 decode 流不被阻塞。

效果：
- TTFT 略微变慢（长 prompt 的 prefill 被拉长）
- TBT 大幅平滑（不再被长 prefill 尖刺）
- throughput 略降但**P99 latency 大幅改善**——production 通常更看重后者

vLLM 0.5+ 默认开启 chunked prefill。详细论文背景见 Patel et al. 2023 *Splitwise*（prefill / decode 分离思想）与 Agrawal et al. 2024 *Sarathi-Serve*（chunked prefill 系统化）。

### 2.7 vLLM 关键参数速查表（必背）

production 部署 vLLM 时这一张表的参数 90% 都要调一遍：

| 参数 | 含义 | 经验值 / 建议 |
|---|---|---|
| `model` | HF 仓库 id 或本地路径 | 必填 |
| `tensor_parallel_size` | TP 多卡切分度 | 1 / 2 / 4 / 8（必须 `n_head % tp == 0`） |
| `pipeline_parallel_size` | PP 流水并行（实验性） | 通常保持 1，跨机大模型才考虑 |
| `dtype` | 权重精度 | `bfloat16`（H100/A100）/ `float16`（V100/T4） |
| `max_model_len` | 最大 context 长度 | ≤ 模型 config，且必须 ≤ §3.5 算的 KV 上限 |
| `gpu_memory_utilization` | KV cache 显存预留比例 | 默认 0.9；OOM 时调到 0.85；负载平稳可到 0.92 |
| `kv_cache_dtype` | KV cache 精度 | `auto`（默认 bf16/fp16）/ `fp8` / `fp8_e5m2` 强烈推荐 |
| `enable_prefix_caching` | Automatic Prefix Cache | 多 user 共享 system prompt 时**必开** |
| `enable_chunked_prefill` | 长 prefill 切块与 decode 混跑 | vLLM 0.5+ 默认开 |
| `block_size` | KV block 大小 | 16（默认）；调大省管理 overhead 但增内部碎片 |
| `max_num_seqs` | 单步并发 sample 上限 | 默认 256；P99 高时降到 64-128 |
| `max_num_batched_tokens` | 单步混跑 token 上限 | 默认 2048-4096；调小降 TBT |
| `swap_space` | CPU swap 大小（GB） | 4-16 GB；high-pressure 时缓冲峰值 |
| `enforce_eager` | 关闭 CUDA graph | 调试 / 多卡 hang 时设 `True` |
| `enable_lora` + `max_loras` | Multi-LoRA serving | 见 §2.8 |
| `speculative_model` | 投机解码 draft model | vLLM 0.5+ 集成；与 11.5 衔接 |
| `quantization` | 权重量化格式 | `awq` / `gptq` / `fp8`（与 11.4 衔接） |

记住三句口诀：
- 启动失败 → 先看 `max_model_len` 与 `gpu_memory_utilization`
- TBT 尖刺 → 看 `enable_chunked_prefill` 与 `max_num_batched_tokens`
- throughput 不达预期 → 看 `enable_prefix_caching` 与 `kv_cache_dtype`

### 2.8 vLLM 与其他主流推理框架对比（必背）

| 框架 | 开发方 | 强项 | 弱项 | 适用场景 |
|---|---|---|---|---|
| **vLLM** | Berkeley | PagedAttention 鼻祖、生态最广、OpenAI API 兼容 | 部分新模型支持滞后、PP 实验性 | **GPU 服务化推理首选** |
| **SGLang** | Berkeley | RadixAttention prefix cache、结构化输出（grammar）、Python DSL | 生态较新、社区比 vLLM 小 | 多 user 共享前缀 + 结构化输出场景 |
| **TensorRT-LLM** | NVIDIA | NV 卡上极致性能（FP8/INT4 GEMM 优化深）、in-flight batching | 闭源 + 编译复杂、迭代慢 | 极致性能且只用 NV 卡的 production |
| **LMDeploy** | InternLM | 国内生态强、Qwen / InternLM 系优化好、TurboMind kernel 高效 | 海外社区小 | 国内 Qwen / InternLM 部署 |
| **llama.cpp** | Open（ggerganov） | CPU / Mac / 边缘、GGUF 量化生态、跨平台 | 不适合多 user 服务化、batch 弱 | 个人电脑 / Mac / 边缘设备 |
| **MLC-LLM** | CMU | 基于 TVM 跨平台编译（CUDA/Metal/Vulkan/WebGPU） | 性能略差于 vLLM/TRT-LLM | 跨平台 / 浏览器内推理 |
| **HuggingFace TGI** | HF | 与 HF 生态无缝 | 性能落后 vLLM 显著（被 vLLM 论文当 baseline 打） | demo / 小规模实验 |

**选型决策树**：
- GPU 服务化推理（90% 场景）→ **vLLM**
- 多 user 重前缀共享 / 需要 grammar-constrained output → SGLang
- 单一 NV 卡型 + 极致 throughput → TensorRT-LLM
- Qwen / InternLM 国内场景 → LMDeploy
- Mac / 个人电脑跑模型 → llama.cpp
- 浏览器 / 移动端 → MLC-LLM

### 2.9 vLLM 的 Multi-LoRA Serving

LoRA（8.3）让一个 base model 可以挂多个轻量 adapter 服务不同任务/客户。**朴素部署**：每个 LoRA 一个 vLLM instance、N 个垂类 = N 张 H100——成本爆炸。

**Multi-LoRA Serving**（vLLM 0.4+ 支持）：单 base model 同时挂多个 LoRA、每个 incoming request 通过 `--lora-modules` 路由到对应 adapter；attention forward 时 base 部分共用、LoRA 增量部分按 sample 路由——基于 Punica / S-LoRA paper 的 batched LoRA kernel 实现。

**收益**：
- N 个 LoRA = 1 个 base 显存 + N × LoRA 增量显存（每个 LoRA 只 MB 级）
- N 张 H100 → 1 张 H100，**per-LoRA 成本 ÷ N**

**约束**（务必记住）：
- 所有挂载 LoRA 的 `r` 必须一致（Punica kernel 限制，详见 §4）
- `--max-loras 4` 控制同时载入数；超出会按 LRU 换出
- `--max-lora-rank 32` 设上限 rank
- LoRA target modules（如 `q_proj`, `v_proj`）也最好在所有 adapter 里统一

**启动示例**：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B \
    --enable-lora \
    --lora-modules sql=/path/sql_lora medical=/path/med_lora chat=/path/chat_lora \
    --max-loras 4 --max-lora-rank 16
```

调用时 `model="sql"` 即路由到 `sql_lora`，`model="chat"` 路由到 `chat_lora`。这是 to-B 业务"一个 base 服务 N 个客户私有 finetune"的标准方案。

---

## 3. 最小代码示例

### 3.1 vLLM 基础用法（≤ 30 行）

```python
# pip install vllm
from vllm import LLM, SamplingParams

# 1) 启动一个 vLLM engine（首次运行会下载模型权重）
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    tensor_parallel_size=1,            # 单卡；多卡填 2/4/8
    dtype="bfloat16",
    max_model_len=8192,                # context 上限，按显存调
    gpu_memory_utilization=0.9,        # 默认 0.9，OOM 时调到 0.85
    enable_prefix_caching=True,        # 多 user 共享 prompt 时大幅加速
)

# 2) 准备 prompts 与采样参数
prompts = [
    "解释一下 PagedAttention 的核心 idea。",
    "什么是 RLHF？",
]
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

# 3) 一次性 batch 推理（vLLM 内部已自动 continuous batching）
outputs = llm.generate(prompts, sampling_params)

for out in outputs:
    print("=" * 40)
    print("PROMPT:", out.prompt[:50])
    print("OUTPUT:", out.outputs[0].text)
```

**关键点**：
- 用户调用层只看到 `LLM.generate(list_of_prompts)` 这种"批量函数式"接口；底下 vLLM 自动做 PagedAttention + continuous batching
- `enable_prefix_caching=True` 几乎免费的优化——重复 system prompt 场景 prefill 计算可省 50%+
- `tensor_parallel_size > 1` 自动启动多 worker 并把模型切到多卡（NCCL 通信）

### 3.2 vLLM OpenAI Compatible API 部署 + 调用（≤ 25 行）

```bash
# 启动 server（这一行就是 production 部署的最小命令）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --port 8000
```

```python
# 客户端：直接用 OpenAI SDK 调（无需改业务代码）
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",                       # vLLM 默认不校验 key
)

resp = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个专业的 AI 助手。"},
        {"role": "user",   "content": "用一句话解释 KV cache。"},
    ],
    temperature=0.7,
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

**关键点**：
- 启动 server 的命令把 vLLM 暴露成与 OpenAI API **完全兼容的 HTTP 服务**——业务代码只换 `base_url`，无需改一行
- 这正是 vLLM 在 production 普及的核心原因之一：**zero migration cost**
- 高级路由（如多模型 / 多 LoRA）可以再叠 LiteLLM、Anthropic API gateway 等

### 3.3 手写简化 Block Manager（≤ 35 行 Python）

下面这段代码不能跑真 model，但完整演示 vLLM Block Manager 的核心逻辑——block 分配 / 释放 / block table 索引：

```python
from collections import deque
from typing import Dict, List


class SimpleBlockManager:
    """演示 vLLM Block Manager 的最小可读实现。"""

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.free_blocks: deque = deque(range(num_blocks))     # 全局空闲池
        self.block_tables: Dict[int, List[int]] = {}            # sample_id → [phys_block_id, ...]

    def allocate(self, sample_id: int, num_tokens: int) -> bool:
        n_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if n_blocks_needed > len(self.free_blocks):
            return False                                        # 显存不够，admit 失败
        self.block_tables[sample_id] = [self.free_blocks.popleft() for _ in range(n_blocks_needed)]
        return True

    def append_token(self, sample_id: int, cur_len: int) -> bool:
        """生成一个新 token 后扩展 block table（仅当当前 block 已满时分配新 block）。"""
        if cur_len % self.block_size == 0:                      # 当前 block 满了，需要新 block
            if not self.free_blocks:
                return False                                    # OOM
            self.block_tables[sample_id].append(self.free_blocks.popleft())
        return True

    def free(self, sample_id: int) -> None:
        for b in self.block_tables.pop(sample_id):
            self.free_blocks.append(b)                          # 还回 pool

    def locate(self, sample_id: int, token_id: int):
        """逻辑 token id → (物理 block id, block 内 offset)——PagedAttention kernel 的查表逻辑。"""
        return self.block_tables[sample_id][token_id // self.block_size], token_id % self.block_size


# === 用法演示 ===
mgr = SimpleBlockManager(num_blocks=10, block_size=16)
mgr.allocate(sample_id=0, num_tokens=20)            # prompt 20 token → 占 2 block
print(mgr.block_tables[0])                           # 例如 [0, 1]
print(mgr.locate(0, token_id=17))                    # → (block 1, offset 1)
mgr.append_token(0, cur_len=32)                      # 第 32 个 token 触发新 block
mgr.free(0)                                          # sample 完成 → 全部还回
```

**可以从这 35 行代码里看到 vLLM 整个 Block Manager 的骨架**——真实实现额外加了 swap、prefix sharing、Copy-on-Write、reference count 等机制（见 vLLM `csrc/cache_kernels.cu` 与 `vllm/core/block_manager_v2.py`）。

### 3.4 Continuous Batching 调度逻辑伪代码（≤ 25 行）

```python
def continuous_batching_loop(scheduler, block_manager, model):
    """vLLM Scheduler 的核心调度循环（极简版）。"""
    while True:
        # 1) 剔除已完成的 sample，释放它们的 block
        finished = [s for s in scheduler.running if s.is_done()]
        for s in finished:
            block_manager.free(s.id)
            scheduler.running.remove(s)
            scheduler.return_to_user(s)

        # 2) 从 waiting queue 尽可能多塞新 sample（前提：能分到 block）
        while scheduler.waiting and block_manager.can_allocate(scheduler.waiting[0]):
            s_new = scheduler.waiting.popleft()
            block_manager.allocate(s_new.id, num_tokens=len(s_new.prompt))
            scheduler.running.append(s_new)

        if not scheduler.running:                       # 没活干，等下一秒
            scheduler.wait_for_new_request()
            continue

        # 3) 一步 forward——prefill / decode 在同一 batch 里混跑（chunked prefill）
        next_tokens = model.forward_one_step(scheduler.running, block_manager)

        # 4) append 新 token、按需分配新 block
        for s, tok in zip(scheduler.running, next_tokens):
            s.tokens.append(tok)
            block_manager.append_token(s.id, cur_len=len(s.tokens))
```

**几个核心点**：
- 第 1 步是 continuous batching 的精髓——**早结束的 sample 立刻让位**，不空转
- 第 2 步控制 batch 大小——只要有空 block 就 admit，最大化 GPU 利用率
- 第 3 步把 prefill / decode sample 混在一个 forward 里跑——chunked prefill 的工程基础
- 真实 vLLM 还有 priority、preemption、swap-out、speculative decoding 集成等机制——参见 `vllm/core/scheduler.py`

### 3.5 KV cache 容量上限的应试速算

```python
def vllm_kv_cache_budget(gpu_mem_gb: float, model_size_gb: float,
                          n_layer: int, n_kv_head: int, d_k: int,
                          dtype_bytes: int = 2,
                          util: float = 0.9, activation_reserve_gb: float = 4.0) -> int:
    """估算 vLLM 在给定显存预算下能装多少 token 的 KV cache（单 sample 累计）。
    返回值是「总 token 数」——可用来校验 max_model_len × 期望 batch 是否可行。"""
    available_gb = gpu_mem_gb * util - model_size_gb - activation_reserve_gb
    bytes_per_token = 2 * n_layer * n_kv_head * d_k * dtype_bytes  # K + V 两份
    return int(available_gb * (1024 ** 3) / bytes_per_token)


# 例：H100 80GB、LLaMA-3 8B GQA-8 (n_kv_head=8, d_k=128, n_layer=32)
total = vllm_kv_cache_budget(80, 16, 32, 8, 128)
print(f"total tokens: {total:,}")                # ~14M token
print(f"max_model_len=8192 时可服务 batch ≈ {total // 8192}")   # ~1700
```

启动 vLLM 时如果设的 `max_model_len × 预期 batch` 超过这个上限，会报"No available memory for the cache blocks"——本节 §4 第 2 条就是这个坑。

---

## 4. 工程踩坑与经验

- ❗ **`gpu_memory_utilization` 默认 0.9，OOM 时往下调到 0.85 / 0.8；activation 与碎片要给余量**。vLLM 启动时按这个比例预分配 KV cache pool；剩下 10% 给 forward 时的临时 activation、CUDA workspace、NCCL buffer 等。负载不稳定（长 prompt 偶发）时 0.9 容易触发 OOM——降到 0.85 或 0.8 通常稳。**反过来**：如果你的负载 prompt 短而平均，0.92-0.95 也 OK，能多服务 5-10% batch。`--swap-space 4`（GB）开启 CPU swap 也能缓解短期峰值。

- ❗ **`max_model_len` 设过大但实际 KV cache 装不下 → vLLM 启动直接失败（"No available memory for the cache blocks"）**。新手常以为 `max_model_len = 32768` 是"上限可以这么大、实际按需用"，错——vLLM 启动时**会按 `max_model_len × 至少 1 个 sample` 预校验 KV cache 容量**，不够就拒启。**正确算法**：先用 §3.5 的公式算 `total_kv_tokens`，确保 `max_model_len ≤ total_kv_tokens`（留余量给 batch）；如果模型 config 的 `max_position_embeddings` 是 128k 但你显存只够 16k，**显式传 `max_model_len=16384`** 把上限压下来。

- ❗ **Multi-LoRA serving 时所有 LoRA 的 `r` 必须一致**。vLLM 0.4+ 支持 `--enable-lora` 同时挂多个 LoRA adapter（不同 user 的请求路由到不同 LoRA），但所有挂载的 LoRA 的 rank `r` 必须一致——否则 Punica kernel 不工作。**实战**：训 LoRA 时项目内统一定 `r=16` 或 `r=32`，不要每个任务一个 rank。`--max-loras` 控制同时载入的 LoRA 数（默认 4），`--max-lora-rank` 设上限 rank。这是降低 per-LoRA 部署成本的杀手锏——一个 base model 同时服务 N 个垂类，原本要 N 张 H100 现在 1 张就行。

- ❗ **`tensor_parallel_size > 1` 在某些 model 上需要单独配 `dtype` 与 `enforce_eager`**。多卡 TP 启动时若模型 config 的 `head_dim` 与 TP 切分不齐（例如 head_dim=80 切 4 卡）、或自定义 attention 与 vLLM 默认 path 不兼容、或 NCCL 死锁——会出现 hang 或 error。**调试 checklist**：(1) `--enforce-eager` 关掉 CUDA graph 排除编译问题；(2) 确认 `n_head % tp_size == 0`；(3) `--dtype float16` vs `bfloat16` 试切换（部分老 model 只支持 fp16）；(4) NCCL 卡死时设 `NCCL_P2P_DISABLE=1` 或 `NCCL_IB_DISABLE=1`。

- ❗ **Chunked prefill 在 vLLM 0.4+ 默认开启，老版本不支持**。vLLM ≤ 0.3 的 prefill 是整 prompt 一次性算——长 prompt（如 32k）的 prefill 会把所有 user 的 decode 卡住几秒，TBT 严重尖刺。**遇到长 prompt 服务卡顿时第一反应：升级到 vLLM ≥ 0.5**。新版本默认 `enable_chunked_prefill=True`、`max_num_batched_tokens=512`（每步混跑的最大 token 数），可按负载调。如果你的负载是"短 prompt + 长 generation"为主，关掉 chunked prefill 可以略提 throughput；但绝大多数 production 场景应当开。

- ❗ **`kv-cache-dtype=fp8` 省 50% 显存、质量几乎无损，强烈推荐 production 必开**。vLLM 0.4+ 支持 KV cache 量化到 FP8（`fp8_e5m2` 或 `fp8_e4m3`），显存直接减半（与 5.2 的 GQA / MLA **正交可叠加**）；MMLU / GSM8K 等 benchmark 通常掉点 < 0.5 分，对话场景几乎察觉不到。Hopper（H100）原生支持 fp8 GEMM，比 bf16 还略快。**注意 dtype 不要与 weight 量化混淆**——`kv-cache-dtype=fp8` 是量化 KV cache 数据本身、与 GPTQ/AWQ（11.4，权重量化）是两件事，**可以叠加**。

- ❗ **vLLM 与 HuggingFace `transformers` 的 chat template 不一致 → output 结果偏差**。vLLM `LLM.chat()` / OpenAI API 的 messages → prompt 转换走的是模型的 `tokenizer.apply_chat_template`；但有些 finetune 模型的 chat template 与 base 不同、或 system prompt 默认值不一样——同样的对话历史在 vLLM 与 HF 里 tokenize 出的 prompt 可能差几个 token，导致输出微妙不同。**Production 校验方法**：用 `tokenizer.apply_chat_template(messages, tokenize=False)` 在 vLLM 启动前手算一遍 prompt、比对 HF 直跑结果；不一致时显式传 `--chat-template my_template.jinja`。

- ❗ **vLLM 不直接支持 prefix-LM mask 与某些 custom attention pattern**。vLLM 的 PagedAttention kernel 是为标准 causal decoder-only 设计的——遇到 prefix-LM（prompt 部分双向、generation 部分 causal，T5 / GLM 用）、或 sliding-window attention 的某些变体、或 Mamba/SSM 模型——要么需要专门 patch、要么走不通。**踩坑经验**：上线前先去 vLLM `Models supported` 列表确认你的模型在；不在列表里的话先用小 model 跑通再扩大。Mamba、ChatGLM、Qwen-VL 等都经历过"vLLM 支持滞后几个月"的阶段。

- ❗ **`enable_prefix_caching=True` 在多 user 共享 system prompt 场景下吞吐翻倍，但偶发会增大单 sample latency**。vLLM 的 Automatic Prefix Caching 把 prompt 的 KV cache hash 后缓存到全局 pool，命中时直接复用 prefill 结果——多 user 共享 system prompt 场景下 throughput 提升 50-200%。但 cache 命中表的查询本身有 overhead，**单 user 长 prompt 不重复**的场景下反而慢一点点。判断方法：你的 P50 prompt 重复率 > 30% 时一定开；< 10% 时可关。RadixAttention（11.3）是这一思路的延伸版本。

- ❗ **`speculative_model` / 投机解码集成有版本兼容陷阱**。vLLM 0.5+ 集成了 Medusa / EAGLE / draft-model speculative decoding（与 11.5 衔接），但不同 vLLM 版本的 API 不同、speculator 与 target 模型架构必须匹配。**踩坑经验**：先用纯 vLLM 跑稳再加投机解码；投机解码引入的 throughput 提升通常 1.5-3×、但调参（draft length、acceptance threshold）有讲究——production 上线前必做 A/B。

- ❗ **多机 TP / PP 启动比单机 TP 复杂得多，pipeline parallel 仍是实验性**。vLLM 的 multi-node TP 走 Ray 集群（`--engine-use-ray`）；pipeline parallel（`--pipeline-parallel-size`）截至本节写作时仍在 active 开发中，部分模型不稳。**优先级建议**：能 TP 切单机 8 卡解决就不上 PP；必须跨机时（如 405B 模型）务必用 Ray 起、关掉 CUDA graph 排查问题、跑端到端 smoke test。

---

## 5. 经典 paper

- **Kwon et al., 2023 — Efficient Memory Management for Large Language Model Serving with PagedAttention** — vLLM 的奠基论文，**本节核心必引**。§3 用一张图把朴素 KV cache 的内部 / 外部碎片讲透；§4 给 PagedAttention 的 block table + Copy-on-Write 设计；§5 给 swap-out 与 recompute 应对 high pressure 的策略；§6 实测 throughput 较 HuggingFace TGI 提升 2-24×。读 §3-4 即可建立完整心智模型。
- **Yu et al., 2022 — Orca: A Distributed Serving System for Transformer-Based Generative Models（OSDI'22）** — Continuous Batching（iteration-level scheduling）的鼻祖，比 vLLM 早一年。§3 给出"selective batching"——每步 iteration 后剔除已完成的 sample、admit 新 request——这一 idea 直接被 vLLM 采纳。读 §3-4 理解 continuous batching 的"为什么"。
- **Patel et al., 2023 — Splitwise: Efficient Generative LLM Inference Using Phase Splitting** — prefill / decode 分离思想的代表 paper。提出"prefill cluster + decode cluster"分别部署、用专门硬件配比的设计——这一思路在 vLLM 的 chunked prefill、Sarathi-Serve、TensorRT-LLM 的 in-flight batching 里都有体现。读它能理解"为什么 prefill / decode 性能 profile 不同 → 调度上要区别对待"。
- **vLLM GitHub — [vllm-project/vllm](https://github.com/vllm-project/vllm)** — 直接读源码，重点 `vllm/core/scheduler.py`（调度器）+ `vllm/core/block_manager_v2.py`（block 管理）+ `csrc/attention/paged_attention*.cu`（kernel）。本节代码示例的"完整版"都在这里。

---

## 6. 自测与面试题

**Q1（原理）**：PagedAttention 借鉴了 OS 的哪个机制？解决了朴素 KV cache 管理的什么问题？为什么这套机制能让显存利用率从 ~30% 提升到 ~95%？

<details>
<summary>Answer sketch</summary>

**借鉴的机制**：OS 的**虚拟内存分页（virtual memory paging）**——把物理内存切成固定大小的 page、每个进程持有一张 page table 把虚拟地址映射到物理 page。

**解决的问题**（必须答全 3 个）：
1. **内部碎片（预留浪费）**：朴素管理给每个 sample 预分配 `max_seqlen` buffer（如 4096 token），实际平均只用 200-500 → 浪费 80-95%
2. **外部碎片**：不同 sample 长度差异大、batch 内 padding 浪费严重；显存里出现"长一块短一块"的奶酪状空洞，新 sample 找不到连续大块
3. **早结束 sample 释放不掉**：朴素 batching 是 request-level，batch 内有 sample 早完 EOS、它的 KV cache 槽位空转直到所有 sample 都结束

**为什么 30% → 95%**：PagedAttention 把 KV cache 按 16-token block 切碎、按需分配（不预留）、随用随还（早结束立即释放）：
- 内部碎片只剩"最后一个 block 没填满"那点（最多 16 token，可忽略）
- 外部碎片几乎为 0（block 大小固定、互相可替换）
- 配合 continuous batching，sample 一完成 EOS 就立刻把 block 还给 free pool 给新 user 用

加分：
- 能说出代价是 attention kernel 多一次 indirection（block table 查表），性能损失 < 5%
- 能提 Copy-on-Write 用于 prefix sharing
- 能说 vLLM 论文 Table 1 的 30% / 95% 数字来源
- 能区分这是"内存管理层"优化、与架构层（GQA/MLA）和数值精度层（FP8 KV cache）正交可叠加

</details>

**Q2（计算）**：算 LLaMA-3 70B 在 80 GB H100 上、开 PagedAttention + bf16 KV cache、`max_model_len = 4096`，能同时服务多少 sample？与朴素管理对比（朴素按 max_seqlen=4096 预留 buffer，且 sample 完成前 buffer 不能复用）能多服务几倍？

<details>
<summary>Answer sketch</summary>

**前置数据**：LLaMA-3 70B 用 GQA-8（$h = 64, g = 8, d_k = 128, L = 80$）、bf16 KV cache。

**单 sample 4k context KV cache**（用 4.7 / 5.2 的公式）：
$$M_{\text{KV}} = 2 \times L \times g \times d_k \times T \times \text{bytes} = 2 \times 80 \times 8 \times 128 \times 4096 \times 2 \approx 1.3 \text{ GB}$$

**显存预算**：
- H100 总显存 80 GB
- 70B 权重 bf16 ≈ 140 GB → 必须 TP 切 2 卡，每卡装 70 GB；或 fp8 量化后 70 GB 单卡装下
- 假设 TP=2、单卡装 70 GB 权重 → 单卡剩 ~10 GB（远不够）；改为 TP=4，单卡 35 GB 权重 + activation reserve ~4 GB → 剩 ~41 GB 给 KV cache（util=0.95 略激进，常用 0.85 → ~30 GB）

**PagedAttention（按实际 token 用量分配）**：
- 假设平均每 sample 实际用 1024 token（典型对话场景，prompt + generation 总和 1k 远小于 max_seqlen 4k）
- 单 sample 实际占用 $\approx 1.3 \times 1024/4096 = 0.33 \text{ GB}$
- 30 GB / 0.33 GB ≈ **~90 sample** 同时服务

**朴素管理（按 max_seqlen=4096 预留）**：
- 每 sample 必须预留 1.3 GB（不论实际多少）
- 30 GB / 1.3 GB ≈ **~23 sample**
- 而且早结束的 sample 不释放、batch 内还要 padding → 实际更少（~15 sample）

**对比**：PagedAttention 比朴素管理多服务 **~4-6×** sample——这是 vLLM 论文报告的典型 throughput 提升量级（5-20×，再考虑 continuous batching 的额外收益）。

加分：
- 能说出 GQA-8 已经把 MHA 的 KV cache 砍 8×（与本计算 base）
- 能说 fp8 KV cache 再 ÷ 2，可同时服务 ~180 sample
- 能指出 4k 平均占用 1k 是典型对话负载——长 context 服务时差距更大（如 max_model_len=32k、平均 4k 时差距可达 10×）

</details>

**Q3（实战）**：你的 vLLM 服务 P99 latency 高（用户偶发等待 5+ 秒），从 (chunked prefill / 量化 / TP / batch size) 四个方向各给一个具体优化动作。

<details>
<summary>Answer sketch</summary>

**先诊断 P99 高的根因**——是 TTFT（首 token 慢）高还是 TBT（token 间）高？vLLM `--metrics` + Prometheus 看分布。两者优化方向差异很大。

**方向 1：Chunked Prefill（缓解 TBT 尖刺）**
- 动作：升级 vLLM ≥ 0.5、确认 `--enable-chunked-prefill`、调 `--max-num-batched-tokens 512`（甚至 256）
- 原理：长 prompt 的 prefill 切成多个 chunk 与 decode 混跑，避免长 prefill 把所有 user 的 decode 卡住
- 适用：负载里偶发出现长 prompt（如 16k 文档 QA）、P99 是 TBT 尖刺导致

**方向 2：量化（降单 token 计算与显存）**
- 动作：开 `--kv-cache-dtype fp8` + 模型权重换成 AWQ / GPTQ / FP8 量化版（需 11.4 配合）
- 原理：KV cache fp8 减半显存压力 → 能装更大 batch → throughput 升、单 user 等待短；权重量化让 forward 更快 → 直接降 TBT
- Trade-off：质量略掉（< 0.5 分 benchmark）；要确认业务可接受

**方向 3：TP（多卡降单步 latency）**
- 动作：单卡 TP=1 → 多卡 TP=4 / TP=8（同时调 `--enforce-eager` 排除编译问题）
- 原理：weights 切 N 卡、单步 forward 时间 ÷ N → TTFT 与 TBT 同时下降
- Trade-off：通信开销、卡多成本高、不能解决 batch 调度问题（throughput 不一定升）；适合"想极致降单 user latency"场景

**方向 4：Batch size（控并发避免过载）**
- 动作：调 `--max-num-seqs 64`（默认 256）、`--max-num-batched-tokens 4096`，限制单步 batch 上限
- 原理：超大 batch 让单步 forward 时间过长，反而拖累 P99——找到 batch / latency 的甜点
- 适用：throughput 已饱和、单 user 等待时间被堵住的情况

**额外加分**：
- 开 `--enable-prefix-caching`（多 user 共享 system prompt 场景）
- 上 11.5 投机解码（`--speculative-model`）2-3× decode 加速
- prefill / decode 分离部署（Splitwise 思路，需多 instance）

**优先级排序**（性价比）：先 1（chunked prefill，免费）→ 6（prefix cache，免费）→ 2（量化，稍有质量代价）→ 4（batch size 调参）→ 3（多卡 TP，硬件成本高）→ 5（投机解码，工程复杂）

</details>

---

## 7. 延伸阅读

- [Kwon et al. 2023 — Efficient Memory Management for LLM Serving with PagedAttention (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — vLLM 奠基论文，本节核心引用
- [Yu et al. 2022 — Orca (OSDI'22)](https://www.usenix.org/conference/osdi22/presentation/yu) — Continuous Batching 鼻祖
- [Patel et al. 2023 — Splitwise (arXiv:2311.18677)](https://arxiv.org/abs/2311.18677) — prefill / decode 分离思想
- [Agrawal et al. 2024 — Sarathi-Serve (arXiv:2403.02310)](https://arxiv.org/abs/2403.02310) — Chunked Prefill 系统化
- [vLLM GitHub](https://github.com/vllm-project/vllm) — 源码必读：`vllm/core/scheduler.py` + `block_manager_v2.py` + `csrc/attention/`
- [vLLM 官方文档](https://docs.vllm.ai/) — 参数手册与最佳实践
- [vLLM Blog — How vLLM uses PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 官方科普版本，配合 §1 食用
- 推荐继续读本教程的 **11.3 节《RadixAttention 与 Prefix Cache（SGLang）》**——本节 prefix caching 的延伸：用 radix tree 把多 user 的共享前缀做到 token 粒度复用
- 推荐继续读本教程的 **11.4 节《量化：GPTQ / AWQ / GGUF / FP8 / SmoothQuant》**——本节多次预告的"权重量化与 KV cache 量化叠加"在 11.4 系统讲透
- 推荐继续读本教程的 **11.5 节《投机解码：Speculative / Medusa / EAGLE》**——decode 阶段的另一类加速思路，与 PagedAttention + Continuous Batching 正交可叠加
