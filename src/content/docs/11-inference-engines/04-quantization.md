---
title: "11.4 量化：GPTQ / AWQ / GGUF / FP8 / SmoothQuant"
description: "推理量化 = 用更少 bits 存 weight / activation / KV cache，省显存 + 蹭低精度算力；GPU 服务高 throughput 选 GPTQ / AWQ（W4A16）或 FP8，CPU / 边缘选 GGUF（llama.cpp），TensorRT 服务选 SmoothQuant（W8A8），KV cache 单独 fp8 / int4 才能扛长上下文。"
---

> ⏱ 预计阅读 40 分钟 ｜ 难度 ★★★ 🔥 必考 ｜ 前置：7.4 混合精度

## 一句话本节讲什么

推理量化 = **用更少 bits 存 weight / activation / KV cache，省显存 + 蹭低精度算力**；GPU 服务高 throughput 选 GPTQ / AWQ（W4A16）或 FP8，CPU / 边缘选 GGUF（llama.cpp），TensorRT 服务选 SmoothQuant（W8A8），KV cache 单独 fp8 / int4 才能扛长上下文。

---

## 1. Mental model（直觉）

7.4 节讲的是**训练**侧的混合精度：bf16 forward / backward + fp32 master weight。本节讲**推理**侧的低精度——训练完了，模型已经定型，唯一目标是把它跑得更快、占显存更少。这两件事的约束完全不同：

- **训练**对小梯度敏感（master weight 必须 fp32，否则更新被吃掉），但 forward / backward 的精度损失可以靠"梯度累积 + 多步优化"自我修正。
- **推理**没有梯度、没有 step，weight 是死的。只要量化误差不显著影响 token 分布，就能一路压到 4 bit 甚至更低。换言之，**推理量化的精度容忍度比训练大得多**。

推理量化的两条主流路线，要先在脑子里分清：

```
        ┌─────────── PTQ (Post-Training Quantization) ───────────┐
        │   训完直接量化，用一小批 calibration data 校准         │
        │   GPTQ / AWQ / SmoothQuant / GGUF / FP8 都属于此类     │
        │   工程主流，落地几乎全是 PTQ                            │
        └─────────────────────────────────────────────────────────┘

        ┌─────────── QAT (Quantization-Aware Training) ──────────┐
        │   训练时插入 fake-quant 算子，让 model 学会"忍受量化"   │
        │   质量上限高，但要重训，工程成本巨大                    │
        │   只在 W4A4 / W2A4 这种极端低 bit 才必须用              │
        └─────────────────────────────────────────────────────────┘
```

第二个 mental model：**weight 量化和 activation 量化是两件事**。

- **Weight quantization**（W4 / W8）：weight 是静态的、训完不变，离线一次量化好就行。难点是用 4 bit 表达 fp16 范围而不掉点。
- **Activation quantization**（A8 / A4）：activation 是动态的、每个 input 都不同，且**带 outlier**（少数 channel 数值比其他大几十倍），是真正难啃的骨头。

所以你看到的命名格式是 **W?A?**：W4A16 表示 weight 4-bit、activation 还是 fp16，相对简单；W8A8 表示两端都量化，要解决 outlier 问题；W4A4 是地狱难度，PTQ 基本搞不定。

第三个 mental model：**KV cache 是另一个独立战场**。Long-context 推理时 KV cache 显存早就超过 weight（128k context 的 70B 模型 KV cache > 100 GB），所以**KV cache 量化（fp8 / int8 / int4）正在变成和 weight 量化同等重要**的事情。vLLM 的 `kv_cache_dtype="fp8_e5m2"` 就是这条路。

---

## 2. 公式与原理

### 2.1 均匀量化：scale 与 zero-point

把一个 fp16 张量 $X$ 量化到 $b$ 位整数，本质是把连续区间 $[\min, \max]$ 映射到 $[0, 2^b - 1]$（非对称）或 $[-2^{b-1}, 2^{b-1} - 1]$（对称）。

**非对称量化**（asymmetric, 适合 activation）：

$$X_q = \text{round}\left(\frac{X}{s}\right) + z, \quad s = \frac{\max(X) - \min(X)}{2^b - 1}, \quad z = -\text{round}\left(\frac{\min(X)}{s}\right)$$

其中 $s \in \mathbb{R}$ 是 **scale**，$z \in \mathbb{Z}$ 是 **zero point**（"原 0 在量化空间的位置"）。反量化：$X \approx s \cdot (X_q - z)$。

**对称量化**（symmetric, 适合 weight）：直接令 $z = 0$，量化区间以 0 为中心：

$$X_q = \text{round}\left(\frac{X}{s}\right), \quad s = \frac{\max(|X|)}{2^{b-1} - 1}$$

Weight 通常分布以 0 为中心（接近高斯），用对称量化几乎不丢精度还省一次加法；activation 经过 ReLU / GELU 后是正偏的，对称量化会浪费一半表示空间，所以用非对称。

### 2.2 量化粒度：per-tensor / per-channel / per-group

同一个 weight 矩阵里，不同列（output channel）的数值分布可能差很多。**用一个 scale 量化整个 tensor**，会让分布最大的那一列把 scale 撑大，其他列被压成几个台阶。粒度越细，scale 越多，量化误差越小，开销也越大：

| 粒度 | scale 数量（对一个 $[O, I]$ 的 weight 矩阵）| 精度 | 存储开销 | 谁在用 |
|---|---|---|---|---|
| **per-tensor** | 1 | 最差 | 几乎为 0 | 老 toy demo |
| **per-channel** | $O$（每个 output channel 一个） | 中 | $O$ × 4 bytes | 经典 INT8 weight |
| **per-group** | $O \times \lceil I / g \rceil$（每 $g$ 个元素一个）| 好 | 显著但可接受 | **GPTQ / AWQ 主流** |
| **per-token**（仅 activation）| 每个 token 一个 scale | 好 | 动态计算 | SmoothQuant 的 activation 侧 |

**Group size $g = 128$** 是 W4 量化的工业甜点：精度接近 per-channel，开销可控。$g = 64$ 精度略好但收益边际；$g = 32$ 几乎无额外收益但 scale 存储翻倍。

### 2.3 量化方案分类速查（必背）

| 类别 | 含义 | 代表方案 | 主战场 |
|---|---|---|---|
| **W8A16 / W4A16** | weight 8/4-bit, activation fp16 | GPTQ / AWQ | 推理服务 GPU |
| **W8A8** | weight + activation 都 8-bit | SmoothQuant / LLM.int8() | TensorRT / 极致 throughput |
| **W4A8** | weight 4-bit, activation 8-bit | 较新方案（如 QoQ） | 2024+ 趋势 |
| **W4A4** | 极端：两端都 4-bit | QuaRot / Atom（实验性） | 学术 SOTA |
| **FP8** | 浮点 8-bit (E4M3 / E5M2) | TE / vLLM | H100+ 推理 |
| **KV cache 量化** | 单独压 K/V cache | FP8 KV / INT4 KV | long-context 必备 |

### 2.4 GPTQ：基于二阶信息的列量化

**Frantar et al., 2022** 提出 GPTQ，核心是把"weight 量化"看成一个**最小化输出误差**的优化问题：

$$\arg\min_{\hat W} \| W X - \hat W X \|_F^2$$

其中 $W \in \mathbb{R}^{O \times I}$ 是 fp16 weight，$\hat W$ 是量化后的 weight，$X \in \mathbb{R}^{I \times N}$ 是 calibration data 的 activation（用 128-512 条真实样本前向一次得到），$N$ 是总 token 数。

GPTQ 的解法继承自 OBS（Optimal Brain Surgeon）/ OBQ（Optimal Brain Quantizer）：**按列量化，每量化一列就用剩余列补偿这一列的量化误差**。具体来说，记 Hessian $H = 2 X X^T \in \mathbb{R}^{I \times I}$，量化第 $j$ 列引入误差 $\delta w_j = w_j - \text{quant}(w_j)$，则其他列要做的更新是：

$$\Delta w_k = -\frac{\delta w_j}{[H^{-1}]_{jj}} \cdot [H^{-1}]_{kj}, \quad k > j$$

直觉：用 $H^{-1}$ 估计的"二阶曲率"告诉我们，把误差分摊到哪些列代价最小。

**GPTQ 工程实现的关键点**：

- **Cholesky 分解 + lazy update**：原始 OBQ 复杂度 $O(I^3)$ 不可行；GPTQ 通过 Cholesky 把复杂度降到 $O(I^2)$ 量级，128 个 sample 上量化 13B 模型只要几十分钟。
- **Per-channel + group=128**：典型配方是 group-wise 4-bit weight，对 70B 模型量化后掉点 < 1%。
- **Activation 不量化**：W4A16，activation 仍是 fp16，所以 GPTQ 加速主要来自 **memory bandwidth** 节省（weight 体积砍 4×），decode 阶段 throughput 大约 2× 提升。

### 2.5 AWQ：activation-aware 的 weight 量化

**Lin et al., 2023** 的关键观察：**weight 不是平等的，少数 channel 对 activation 影响极大**——把这些 "salient channel" 识别出来并保护好，剩下的随便量化都不会掉点。

具体地，对每个 output channel $j$，定义"重要性"为 calibration activation 的平均幅度 $|x_j|$（不是 weight 大小，而是 activation 大小，这就是 "activation-aware" 的含义）。Top 1% 的 channel 是"显著通道"，对它们做特殊处理。

AWQ 的妙处不是"把这 1% channel 留 fp16"（混合精度部署麻烦），而是**等价数学变换**：对显著 channel $j$ 引入 scale $s_j > 1$：

$$y_j = \sum_i w_{ij} x_i = \sum_i (w_{ij} \cdot s_j) (x_i / s_j) = \sum_i \tilde w_{ij} \tilde x_i$$

其中 $\tilde w_{ij} = w_{ij} \cdot s_j$ 仍然量化到 4 bit（但因为放大了 $s_j$ 倍，量化误差相对值变小），$\tilde x_i = x_i / s_j$ 仍然是 fp16（不存在精度损失）。**显著通道的 weight 在量化前被放大 → 量化误差被稀释 → 反量化时 scale 抵消**——完全等价但精度提升。

$s_j$ 的取值通过 grid search 在 calibration data 上最小化输出 MSE。AWQ 实测**和 GPTQ 同等质量但量化更快、推理更稳**，autoawq 是当前 vLLM 服务的主推方案。

### 2.6 SmoothQuant：W8A8 的 outlier 平滑

**Xiao et al., 2022** 解决的问题更难：**activation 也要量化到 INT8**。但 LLM 的 activation 有个臭名昭著的现象——少数 channel 的数值比其他大 50-100 倍（Dettmers 在 LLM.int8() 里首次系统性指出）。直接 per-tensor INT8 量化，scale 被 outlier 撑爆，正常元素全压成几个台阶 → 精度崩。

SmoothQuant 的核心 idea 又是一次**等价数学变换**：把 activation 的 outlier"平移"到 weight 上，让两边都好量化。对每个 channel $i$ 引入 smoothing factor $s_i$：

$$Y = X W = (X \, \text{diag}(s)^{-1}) (\text{diag}(s) \, W) = \tilde X \tilde W$$

选 $s_i$ 让 $\tilde X = X / s$ 的 outlier 被压平、$\tilde W = s \cdot W$ 的数值膨胀在可接受范围。论文里 $s_i = \max(|X_i|)^\alpha / \max(|W_i|)^{1-\alpha}$，超参 $\alpha = 0.5$ 是默认值。

效果：**activation 平滑后两端都能 INT8 量化**，Tensor Core INT8 算力是 fp16 的 2×（A100 上），实测 W8A8 比 fp16 throughput 提升约 2×，且掉点 < 1%。SmoothQuant 是 TensorRT-LLM W8A8 路径的标配。

### 2.7 GGUF / GGML：llama.cpp 的 K-quants

**GGUF** 是 llama.cpp 项目的 model 文件格式（前身 GGML），同时定义了一套独有的量化方案。和 GPTQ / AWQ 不同，GGUF 量化**不需要 calibration data**，是纯 weight 统计 + 启发式分块，主战场是**CPU / Mac / 边缘设备推理**。

K-quants 的命名规则：

- **Q4_K_M** = 4-bit, K-quant family, Medium variant
- 字母后缀 S / M / L 表示 block 内 scale 精度递增

K-quants 的关键技巧：**block 内 mix bits**。一个 block（典型 32 或 256 个元素）里，绝大多数元素用 4 bit，少数 outlier 用 6 bit；scale 自身也用 6 bit fp 储存。这样平均位数在 4.5 bit 左右，但精度接近纯 6-bit 量化。常用档位：

| 量化档 | 平均 bits | 质量 | 用途 |
|---|---|---|---|
| Q2_K | ~2.6 | 明显掉点 | 极限部署 |
| Q3_K_M | ~3.7 | 略掉点 | 显存紧张 |
| **Q4_K_M** | ~4.8 | **甜点** | 主流推荐 |
| Q5_K_M | ~5.7 | 接近 fp16 | 高质量 |
| Q6_K | ~6.6 | 几乎无损 | 不在乎显存 |
| Q8_0 | 8 | 无损 | 校验用 |

**Q4_K_M 是 llama.cpp 社区的事实标准**——质量与显存的最佳折中，70B 模型量化到 ~40 GB，能在 Mac M2 Ultra 上跑。

### 2.8 FP8 推理（与 7.4 fp8 训练对比）

7.4 节讲过 fp8 训练用 E4M3 + E5M2。**推理侧 fp8 的玩法不太一样**：

- **Weight FP8**：vLLM 支持加载 fp8 weight（通常是 fp16 weight 离线量化得到），单卡显存砍半，吞吐 1.5-2×。
- **KV cache FP8**：`kv_cache_dtype="fp8_e5m2"`，把 KV cache 从 fp16/bf16 压到 fp8，long-context 显存砍半。**这是 fp8 在推理侧最有价值的应用**——因为 KV cache 通常比 weight 还大。
- **Activation FP8**：H100 上有 fp8 Tensor Core，但纯 fp8 activation 推理对模型质量挑战大，目前 vLLM 主要支持 weight + KV cache 的 fp8，activation 仍是 bf16。

**FP8 vs INT8 的工程取舍**：
- INT8 算力老 GPU（A100、V100）就有，FP8 必须 H100/H200/MI300X。
- INT8 是定点，需要 scale 管理（per-tensor / per-channel）；FP8 是浮点，**动态范围天然更宽**，精度损失小一截。
- 推理服务里如果硬件允许（H100），**FP8 优先于 INT8**——同样 8 bit，FP8 几乎无损而 INT8 偶尔掉点。

DeepSeek-V3 走的是 fp8 训练 + fp8 推理一致路线，2024 年成为 fp8 inference 的标志性参考。

### 2.9 方案对比总表（必背）

| 方案 | bits | 实现库 | 推理框架 | 质量损失 | 加速比 | 适用硬件 |
|---|---|---|---|---|---|---|
| **GPTQ** | W4A16 | `auto-gptq` | vLLM / TGI | < 1% | ~2× | A100 / H100 |
| **AWQ** | W4A16 | `autoawq` | vLLM / TGI | < 1% | ~2× | A100 / H100 |
| **GGUF Q4_K_M** | ~4.8 mix | `llama.cpp` | llama.cpp | < 2% | 4-8×（CPU 对比 fp16）| CPU / Mac / 边缘 |
| **SmoothQuant** | W8A8 | `smoothquant` | TensorRT-LLM | < 1% | ~2× | A100 / H100 |
| **FP8** | W8A8 / KV | `torch` / `vLLM` | vLLM | < 0.5% | 1.5-2× | H100+ |
| **bnb 4-bit (NF4)** | W4A16 | `bitsandbytes` | HuggingFace | 1-2% | ~1.5× | 任意（QLoRA） |

**记忆口诀**：服务端首选 AWQ / GPTQ（W4A16，质量稳），TensorRT 走 SmoothQuant（W8A8），llama.cpp 走 GGUF（CPU），有 H100 就 FP8（KV cache 必上），QLoRA 训练阶段才用 bnb。

### 2.10 什么场景选什么（必背决策树）

- **GPU 服务高 throughput 推理**：AWQ 或 GPTQ（W4A16）+ vLLM；H100 上叠加 FP8 KV cache。
- **CPU / Mac / 边缘部署**：GGUF Q4_K_M + llama.cpp。
- **训练与推理一致 / 极致吞吐**：FP8（H100 必备）→ DeepSeek-V3 路线。
- **超低显存 fine-tune**：QLoRA NF4（bitsandbytes）→ 见 8.3 节。
- **TensorRT 服务**：SmoothQuant W8A8。
- **Long-context 推理**：weight 量化 + **KV cache 量化（FP8 / INT4）必上**，否则 KV 撑爆显存。

### 2.11 现代趋势（一段话扫盲）

- **FP8 推理普及**：H100 大规模部署后，FP8 渐成默认（vLLM、TensorRT-LLM 都内建支持）。
- **KV cache 量化升级链**：fp16 → fp8 → int4，long-context 推理的显存账靠这个续命。
- **混合量化**：weight 用 W4，attention KV 用 FP8，FFN activation 用 W8——一个模型不同部位用不同精度，nuQmm / QuaRot 等学术方案在推。
- **量化感知 SFT**：在 SFT 阶段加 quantization noise（fake-quant），让 model 提前适应量化，PTQ 量化后掉点更小。

---

## 3. 最小代码示例

### 3.1 手写 per-tensor 对称量化（理解公式）

```python
import torch

def quantize_symmetric(x: torch.Tensor, n_bits: int = 8):
    """Per-tensor 对称量化（weight 用法）"""
    qmax = 2 ** (n_bits - 1) - 1          # int8: 127
    s = x.abs().max() / qmax              # scale = max(|x|) / qmax
    x_q = torch.round(x / s).clamp(-qmax - 1, qmax)  # quantize → int
    return x_q.to(torch.int8), s          # 返回 int 张量 + scale

def dequantize_symmetric(x_q: torch.Tensor, s: float):
    """反量化：x ≈ s * x_q"""
    return x_q.to(torch.float32) * s

# 演示：随机 weight 量化 → 反量化 → 看误差
W = torch.randn(1024, 1024) * 0.1
W_q, s = quantize_symmetric(W, n_bits=8)
W_hat = dequantize_symmetric(W_q, s)
print(f"max abs error: {(W - W_hat).abs().max():.6f}, scale: {s:.6f}")
```

**要点**：14 行代码就把 §2.1 的对称量化公式写完了。`clamp` 处理 round 后超出 int 范围的情况。生产实现把这套换成 per-channel / per-group 即可（多算几个 scale）。

### 3.2 GPTQ 量化与加载（auto-gptq）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_id = "meta-llama/Llama-2-7b-hf"
tok = AutoTokenizer.from_pretrained(model_id)

# Calibration data：必须用真实 prompt，不能随机 token
calib_texts = [open("c4_sample.txt").read() for _ in range(128)]
examples = [tok(t, return_tensors="pt", max_length=512, truncation=True) for t in calib_texts]

quant_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
model = AutoGPTQForCausalLM.from_pretrained(model_id, quant_config)
model.quantize(examples)                                   # GPTQ 量化（耗时 30 min ~ 数小时）
model.save_quantized("./llama2-7b-gptq-4bit")

# 加载量化后的模型推理
q_model = AutoGPTQForCausalLM.from_quantized("./llama2-7b-gptq-4bit", device="cuda:0")
out = q_model.generate(**tok("Hello, ", return_tensors="pt").to("cuda:0"), max_new_tokens=50)
print(tok.decode(out[0]))
```

**要点**：`group_size=128` 是甜点；`desc_act=True` 会按 activation 大小重排列量化（精度更好但推理慢些）；calibration 必须用真实文本不能用随机 token，否则 calibration 出来的 Hessian 完全失真。

### 3.3 AWQ 量化与加载（autoawq）

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-2-7b-hf"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoAWQForCausalLM.from_pretrained(model_id)

quant_config = {
    "zero_point": True, "q_group_size": 128,
    "w_bit": 4, "version": "GEMM",          # GEMM = 通用，GEMV = batch=1 优化
}
# AWQ 自带 calibration data（pile 子集），也可传 calib_data 自定义
model.quantize(tokenizer=tok, quant_config=quant_config)
model.save_quantized("./llama2-7b-awq")

# 加载推理
q_model = AutoAWQForCausalLM.from_quantized("./llama2-7b-awq", device_map="cuda:0")
print(tok.decode(q_model.generate(**tok("Hi", return_tensors="pt").to("cuda:0"), max_new_tokens=30)[0]))
```

**要点**：AWQ 比 GPTQ 量化快得多（10-30 分钟搞定 7B），版本选 GEMM（服务批量推理）或 GEMV（单条流式）。

### 3.4 vLLM 加载量化模型（生产推理一行）

```python
from vllm import LLM, SamplingParams

# 关键：必须显式传 quantization 参数，否则会按 fp16 加载导致报错
llm = LLM(model="./llama2-7b-awq", quantization="awq", dtype="float16")
# llm = LLM(model="./llama2-7b-gptq-4bit", quantization="gptq")  # GPTQ 同理
# H100 上加 KV cache fp8：kv_cache_dtype="fp8_e5m2"
# llm = LLM(model="meta-llama/Llama-3-70B", kv_cache_dtype="fp8_e5m2")

out = llm.generate(["Hello, world."], SamplingParams(max_tokens=50, temperature=0.7))
print(out[0].outputs[0].text)
```

**要点**：vLLM 的 `quantization` 参数支持 `awq` / `gptq` / `marlin` / `fp8` 等，**漏传会按 fp16 解析 weight，立刻报 dtype mismatch**。

### 3.5 GGUF 转换 + llama.cpp 推理（CPU / Mac 部署）

```bash
# 1) 拉 llama.cpp
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
make -j

# 2) HF 模型 → GGUF（先 fp16 全精度 GGUF）
python convert_hf_to_gguf.py /path/to/llama-2-7b-hf \
    --outfile llama-2-7b-f16.gguf --outtype f16

# 3) Q4_K_M 量化（甜点档位）
./llama-quantize llama-2-7b-f16.gguf llama-2-7b-Q4_K_M.gguf Q4_K_M

# 4) 推理（CPU / Mac，-ngl 指定 offload 到 GPU 的层数）
./llama-cli -m llama-2-7b-Q4_K_M.gguf -p "Hello, " -n 100 -ngl 0
```

**要点**：`Q4_K_M` 是社区主流；想极致小用 `Q3_K_M`，想接近 fp16 用 `Q5_K_M`。`-ngl 0` 全 CPU，`-ngl 999` 全 GPU。

---

## 4. 工程踩坑与经验

- ❗ **GPTQ / AWQ 量化必须用真实 prompt 做 calibration，不能用随机 token**。Calibration data 决定 Hessian / activation 统计的准确性，随机 token 的分布与真实推理分布完全脱节，量化后的 model 在真实 prompt 上掉点严重（典型表现：MMLU 掉 5-10 分而不是 < 1）。建议从 C4 / WikiText / 项目自己的真实 prompt 抽 128-512 条，长度 512-2048。

- ❗ **AWQ 的 `group_size=128` 是甜点**，`group_size=64` 精度略好但收益边际递减；`group_size=32` 几乎无收益但 scale 存储翻倍。除非追求论文级 SOTA，工程默认 128 不要乱调。

- ❗ **vLLM 加载 GPTQ / AWQ model 必须显式传 `quantization="gptq"` / `"awq"`**。漏传时 vLLM 按 fp16 解析 weight，立刻 dtype mismatch 报错；更隐蔽的情况是某些 fork 不报错但悄悄按 fp16 加载，吞吐和原始 fp16 一样，你以为量化生效了其实没生效——**部署后必须 benchmark 显存占用确认**。

- ❗ **KV cache 量化在 long-context 推理上是必选项，不是优化**。70B model 在 128k context 上 KV cache 可达 100+ GB（远超 weight 的 ~140 GB fp16），不量化 KV 直接 OOM。fp8 KV cache 是当前 vLLM / TensorRT-LLM 的成熟方案，掉点 < 0.5%；INT4 KV 在 256k+ 才考虑，质量风险更高。

- ❗ **INT4 / INT2 极端量化通常需要 QAT（quantization-aware training）**，PTQ 直接量化掉点严重。Q2_K 在 70B 上能勉强工作（GGUF 的 K-quant 设计有 outlier 保护），但 W2A16 纯 PTQ 几乎全军覆没。学术 W4A4（QuaRot / Atom）需要 Hadamard 旋转 + 仔细 calibration，工程化早期。

- ❗ **量化后的 model 不能再 fine-tune**。Weight 已经离散化，gradient 没有意义。要 fine-tune 必须走 **QLoRA 路线**（8.3 节）：weight 保持 NF4 量化但反量化到 fp16 做 forward，LoRA adapter 在 fp16 下训练。直接 PEFT 一个 GPTQ / AWQ model 通常报错或训不收敛。

- ❗ **不同硬件最优量化方案不同**：
  - **H100**：FP8（weight + KV cache）首选，AWQ/GPTQ 也可用。
  - **A100**：W4A16（AWQ / GPTQ）+ INT8 KV cache，没有 fp8 tensor core。
  - **V100 / 老 GPU**：只能 W4A16，没有 INT8 算力优势，量化主要为省显存。
  - **CPU / Mac**：GGUF（llama.cpp），别的都没意义。
  跨硬件复用同一个量化模型通常不可行，要按目标硬件单独量化。

- ❗ **benchmark 量化加速比必须用真实 batch / context，单 token decode 看不出加速**。量化的核心收益是 **memory bandwidth 节省**——decode 阶段每个 token 都要把整个 weight 从 HBM 搬到 SRAM，weight 砍 4× 直接 4× 带宽。但单 token / 小 batch / 短 context 下 latency 主要被 kernel launch 和 attention 计算拖累，量化收益不明显。**正确 benchmark**：固定 batch=32 / 64，input 1024、output 512 token，跑 10 轮取中位 throughput。

- ❗ **AWQ / GPTQ 模型在 vLLM 上的 throughput 不会 4× 提升，通常 1.5-2×**。理论上 weight 4× 但 activation 还是 fp16，KV cache 还是 fp16，attention compute 没省，PCIe 传输没省——所以端到端 throughput 在 W4A16 下大约 1.5-2×。想要更高加速比要叠加 KV cache 量化、FP8 等。

- ❗ **GGUF / llama.cpp 的"加速比"很容易误读**：社区常说 Q4_K_M 比 fp16 快 4-8×，那是在 CPU 上比；GPU 上 GGUF 远不如 vLLM + AWQ。**llama.cpp 不是 GPU 推理引擎**，是 CPU / 边缘引擎，不要拿它和 vLLM 直接比 GPU throughput。

- ❗ **混合精度量化（W4 weight + FP8 KV + W8 activation）的工具链尚未统一**。vLLM / TensorRT-LLM / SGLang 各家配置项不同，组合时要查每个框架的支持矩阵，常出现"理论上能开但实际报 unsupported"的尴尬。生产环境优先选成熟单一方案（如 AWQ + FP8 KV cache）。

---

## 5. 经典 paper

- **Frantar et al., 2022 — GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** — PTQ 路线的开山之作，把 OBS / OBQ 的二阶量化思想搬到 LLM scale，4-bit 量化 175B 模型只要单卡几小时。本节 §2.4 的 Hessian + 列量化补偿全部出自这篇。读完能彻底理解"为什么 PTQ 不只是 round 一下"。
- **Lin et al., 2023 — AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** — 提出 "salient weight + scaling 保护" 的极简思想，质量与 GPTQ 持平但量化更快、推理更稳。本节 §2.5 的等价数学变换、salient channel 识别、grid search scale 全来自这篇。算是工业最爱的量化 paper。
- **Xiao et al., 2022 — SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** — W8A8 量化的关键工程突破，揭示了 LLM activation outlier 现象并给出"把 outlier 平移到 weight 上"的优雅解法。本节 §2.6 数学变换出自这篇 §3-4。理解 W8A8 路径必读。
- *延伸*：**Dettmers et al., 2022 — LLM.int8()** 是 outlier 现象最早系统性描述，**Micikevicius et al., 2022 — FP8 Formats for Deep Learning** 是 fp8 标准（与 7.4 节呼应），**Tseng et al., 2024 — QuaRot** 是 W4A4 学术 SOTA（用 Hadamard 旋转消 outlier），有兴趣可补。

---

## 6. 自测与面试题

**Q1（概念）**：W4A16 / W8A8 / FP8 三种量化方案分别是什么？各自适合什么硬件 / 场景？

<details>
<summary>Answer sketch</summary>

- **W4A16**：weight 4-bit、activation 仍 fp16。代表方案 GPTQ / AWQ。**只压 weight，不动 activation**——所以工程门槛低，质量稳，掉点 < 1%。适合 **GPU 服务通用推理**（A100 / H100 / vLLM），主要收益是 memory bandwidth 砍 4×，throughput 提 1.5-2×。
- **W8A8**：weight + activation 都 INT8。代表方案 SmoothQuant。**两端都量化**，需要解决 activation outlier 问题（SmoothQuant 用数学变换平移 outlier 到 weight）。适合 **TensorRT-LLM 部署 / 极致 throughput** 场景，A100 上能利用 INT8 Tensor Core（算力 2× fp16），实测 throughput ~2×。
- **FP8**：浮点 8-bit（E4M3 forward / E5M2 backward）。**浮点保留动态范围**，比同 bit 的 INT8 精度更好（掉点 < 0.5%）。**只在 H100 / H200 / MI300X 等 Hopper+ GPU 上有意义**（A100 没有 fp8 tensor core）。推理上最有价值的应用是 **KV cache fp8**（long-context 必备），weight fp8 + KV fp8 的组合是 DeepSeek-V3 路线的延续。

加分：
- 指出 W4A16 的"加速主要靠带宽节省"vs W8A8 的"加速靠 INT8 算力"——两条不同的物理机制。
- 指出 KV cache 量化是与 weight 量化平行的独立战场，long-context 必上。

</details>

**Q2（实战）**：你要部署 LLaMA-3 70B 到 4× A100 80GB，列出量化 + 推理框架的完整选型与理由。

<details>
<summary>Answer sketch</summary>

**先算账**：
- LLaMA-3 70B fp16 weight = 140 GB，4× A100 80GB 共 320 GB，weight 用 TP=4 分到每卡 35 GB，剩 ~45 GB / 卡给 KV cache + activation。
- 假设要支持 8k context、batch=32：fp16 KV cache 一条 ≈ 1.3 GB，batch 32 = ~42 GB，每卡分摊 ~10 GB，刚好能塞但裕度低。

**选型**：

1. **量化方案：AWQ W4A16**
   - 理由：A100 上首选 W4A16（无 fp8 tensor core），AWQ 在 LLaMA 系列实测优于 GPTQ（量化更快、推理稍稳），70B 模型 AWQ W4 后 weight ~35 GB，TP=4 后每卡 ~9 GB，省出大量空间给 KV cache 和 activation。
2. **推理框架：vLLM**
   - 理由：vLLM 原生支持 AWQ（`quantization="awq"`）、PagedAttention、continuous batching，开箱即用 throughput 在 70B 上是 SOTA 之一；TensorRT-LLM 性能更优但工程门槛高（要写 model definition），原型阶段不优先。
3. **KV cache：INT8 KV cache（A100 没 fp8）**
   - 理由：A100 没 fp8，但 vLLM 0.5+ 支持 INT8 KV cache，能再砍 KV 显存 50%，把 8k context 的 batch 提到 64+。
4. **并行策略**：TP=4，不开 PP（70B 在 4 卡上 TP 足够，PP 只有更大 model 才有意义）。
5. **batching**：continuous batching + chunked prefill（vLLM 默认）。

**端到端预期**：throughput 比 fp16 baseline 提升 1.5-2×（来自 AWQ + INT8 KV cache + continuous batching 的组合收益），首 token TTFT 略有上升（量化反算开销），decode TBT 显著下降。

加分：
- 提到先做 quality benchmark（MMLU / GSM8K）确认 AWQ 量化掉点 < 1%，再上线。
- 提到部署后必须 benchmark 显存占用，确认 quantization 真的生效（vLLM 漏传参数会按 fp16 加载）。
- 提到如果换到 H100，方案应改为 FP8 weight + FP8 KV cache。

</details>

**Q3（trade-off）**：AWQ 和 GPTQ 都是 W4A16，AWQ 的"activation-aware"具体在哪？两者实战差异？

<details>
<summary>Answer sketch</summary>

要点：

- **GPTQ 是 weight-aware**：基于 Hessian $H = X X^T$ 做二阶优化，用 calibration activation **统计 Hessian** 但量化目标是 weight 误差 $\| W X - \hat W X \|^2$。activation 的角色是"提供 Hessian 信息"，不进入量化决策。
- **AWQ 是 activation-aware**：观察到"少数 weight channel 对应的 activation 幅度极大"，用 calibration activation 的 **per-channel 平均幅度** $|x_j|$ 直接定义 weight 的"重要性"。重要 channel 通过 scale $s_j > 1$ 在量化前放大（量化误差被稀释），反量化时 scale 抵消——是一次**等价数学变换**。
- **关键差异**：
  - GPTQ 用 activation 算 Hessian → **解一个二次优化**；AWQ 用 activation 算 channel importance → **grid search 一个 scaling factor**。AWQ 数学更简单、计算更轻。
  - **量化速度**：AWQ 通常 10-30 分钟搞定 7B；GPTQ 几小时（要算 Cholesky + 列更新）。
  - **质量**：在 LLaMA / Mistral / Qwen 上两者打平（AWQ 论文报告略好）；在某些少见架构上 GPTQ 偶尔更稳。
  - **推理性能**：AWQ 的 GEMM kernel（autoawq、Marlin）通常比 GPTQ 的 ExLlamaV2 / GPTQ kernel 更快，vLLM 默认偏好 AWQ。

加分：
- 指出两者其实可以**结合**：用 AWQ 的 scaling 保护重要 channel，再用 GPTQ 的二阶补偿剩余误差，是少数 SOTA 方案的玩法。
- 指出工业界趋势是 AWQ > GPTQ：autoawq 维护活跃、vLLM 集成深、量化耗时短，新项目优先选 AWQ。

</details>

---

## 7. 延伸阅读

- [auto-gptq GitHub](https://github.com/AutoGPTQ/AutoGPTQ) — GPTQ 量化的事实标准实现，README 就是速通教程
- [autoawq GitHub](https://github.com/casper-hansen/AutoAWQ) — AWQ 量化实现，含与 vLLM / TGI 集成的最小示例
- [llama.cpp 量化文档](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md) — GGUF K-quants 各档位详细说明
- [vLLM Quantization 文档](https://docs.vllm.ai/en/latest/quantization/supported_hardware.html) — 各量化方案 × 硬件 × kernel 的支持矩阵，部署前必查
- [NVIDIA TensorRT-LLM Quantization 指南](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization) — SmoothQuant / AWQ / FP8 在 TensorRT-LLM 的官方实现
- [DeepSeek-V3 Tech Report §3.3](https://arxiv.org/abs/2412.19437) — fp8 训练与推理的工业级实证，本节多处呼应
- 推荐继续读本教程的 **11.5 投机解码：Speculative / Medusa / EAGLE**——量化解决"weight 占空间"，投机解码解决"decode 一个 token 一次 forward 太慢"，是推理加速的另一条正交路线
