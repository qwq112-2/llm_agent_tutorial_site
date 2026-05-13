---
title: "16.3 语音：Whisper / VALL-E / CosyVoice / 全双工 Moshi"
description: "语音任务全栈——ASR（speech → text）/ TTS（text → speech）/ Speech LLM（speech ↔ speech）/ Voice Conversion——的现代解法已经全部\"LLM 化\"：Whisper 用 encoder-decoder Transformer + 680k 小时弱监督数据成为 ASR 事实标准；VALL-E 把 audio 用 Encodec"
---

> ⏱ 预计阅读 45 分钟 ｜ 难度 ★★★ ｜ 前置：Module 4（理解 encoder-decoder / decoder-only Transformer + token-level autoregressive）

## 一句话本节讲什么

语音任务全栈——**ASR（speech → text）/ TTS（text → speech）/ Speech LLM（speech ↔ speech）/ Voice Conversion**——的现代解法已经全部"LLM 化"：**Whisper** 用 encoder-decoder Transformer + 680k 小时弱监督数据成为 ASR 事实标准；**VALL-E** 把 audio 用 **Encodec** quantize 成 discrete token、然后用 LLM 范式预测下一个 audio token，3 秒参考音即可 zero-shot 克隆音色，启发了所有现代 TTS（CosyVoice / F5-TTS）；**Moshi** 把 speech-in / speech-out 全部走 audio token + 双 stream 并行预测，做出第一个工业级**全双工**（双方可同时说话）speech LLM——把传统 "ASR + LLM + TTS" 三阶段 pipeline 压成单模型 200ms 端到端延迟。本节讲清这四类模型的范式、audio tokenization 的关键作用、以及"如何用 Whisper encoder + LLM decoder 整合 multimodal speech LLM"，并给出 Whisper transcribe / CosyVoice TTS / Encodec tokenize 三段最小可跑代码。

---

## 1. Mental model（直觉）

### 1.1 语音 vs 文本：核心矛盾

文本是**离散 token 序列**，长度 $\sim 10^3$；语音是**连续波形**（16kHz 采样意味着 1 秒 = 16000 个 float）。直接给 LLM 喂原始波形，序列长度爆炸 100×。所以语音 NLP 的所有进展都围绕一个核心问题：**怎么把语音变成 LLM 友好的"短序列 token"**？答案有两条路：

- **continuous embedding 路线（ASR 主流）**：用 CNN + Transformer encoder 把 16kHz 波形压成 50Hz 的 hidden vector 序列（约 320× 下采样），再 cross-attention 给 text decoder。Whisper 走这条路。
- **discrete token 路线（TTS / Speech LLM 主流）**：用 neural codec（Encodec / SoundStream / SNAC）把波形量化成离散 token（如 75Hz × 8 codebook），audio 与 text 在 token 级别完全等价，可以直接走 GPT 范式。VALL-E / CosyVoice / Moshi 走这条路。

第二条路是过去三年最大的范式跃迁——它让"audio 像 text 一样"，所有 LLM 的训练 / 推理 / 工程经验直接复用。

### 1.2 语音任务全栈一张图

```
                       ┌────── ASR ───────┐
                       │                  │
   speech (16kHz wav) ─┤                  ├─→ text
                       │                  │
                       └─ Whisper-style ──┘
                          encoder + text decoder

                       ┌────── TTS ───────┐
                       │                  │
   text + ref voice  ──┤                  ├─→ speech (24kHz wav)
                       │                  │
                       └ VALL-E / CosyVoice ┘
                          LLM + Encodec token decoder

                       ┌── Speech LLM ────┐
   speech in  ─────────┤                  ├─→ speech out + text out
                       │                  │
                       └─ Moshi / GPT-4o ─┘
                          双 stream audio token + text head

                       ┌─── VC (转音色) ───┐
   speech A + spk B ───┤                  ├─→ speech with B's timbre
                       └ CosyVoice / SoVITS ┘
```

**记住这张图**：四类任务对应不同的 input / output 模态组合，但**底层骨架都是 Transformer + (continuous embedding | discrete token)**。区别只在于 audio 怎么进、怎么出。

### 1.3 Audio tokenization：第二条路的"BPE"

文本 LLM 之所以能跑，是因为先有 BPE（Module 3）把字符串变成 vocab 5 万的 token 序列。语音要走 LLM 化，必须有等价物——这就是 **neural audio codec**（Encodec / SoundStream / SNAC）。它们的工作流程：

```
24kHz waveform (24000 floats / sec)
   │
   ▼  Encoder CNN (跨步卷积下采样 320×)
75Hz continuous embedding (75 × 128-dim / sec)
   │
   ▼  Residual Vector Quantization (RVQ, K codebook)
75Hz × K discrete tokens (e.g., K=8, vocab=1024)
   │
   ▼  Decoder CNN (跨步反卷积上采样 320×)
24kHz waveform (重建)
```

关键 insight：**1 秒 24kHz 音频 ≈ 75 × 8 = 600 个 token**——已经足够 LLM-friendly。**RVQ（Residual Vector Quantization）** 的"residual"含义：第一层 codebook 量化主信号，后续每层量化前一层的残差，K 层叠加得到高保真重建。这是与传统 VQ-VAE 的关键差异——单个 codebook 容量不足以覆盖音频的高频细节，多层 RVQ 用更小的 codebook 总和达到同等质量。

### 1.4 Whisper：ASR 的"BERT 时刻"

2022 年之前的 ASR 江湖：HMM-GMM → DNN-HMM → CTC（DeepSpeech）→ RNN-T → Conformer，每代都靠 architecture innovation 挤出几个 % WER。**Whisper 一夜之间把战场结束**，靠的不是新 architecture（仍是标准 encoder-decoder Transformer），而是**数据**：从 web 抓 680k 小时 multilingual speech + 字幕，弱监督训练，没有任何 finetune 就在 LibriSpeech / Common Voice / FLEURS 上达到或超越 SOTA。这是 ASR 领域的 "BERT 时刻"——证明**大数据 + 大模型 + 标准架构 > 精巧的小模型**。

Whisper 的另一个工程贡献是**多任务统一**：同一个 model 在 prompt 里加不同的 special token（`<|transcribe|>` / `<|translate|>` / `<|en|>` / `<|zh|>` ...），就能切换任务（transcription / translation）和语言（99 种）。这是把 T5 的 "task as text" 范式搬到语音的成功案例。

### 1.5 VALL-E：TTS 的"GPT 时刻"

VALL-E（Microsoft 2023）的核心 idea 简单到让人惊叹：**既然 audio 可以 token 化，那 TTS 不就是 conditional language modeling 吗？**

```
input: [text tokens] + [3秒参考语音的 Encodec tokens]
       └─────────────── LLM (decoder-only) ───────────────┘
output: [continued Encodec tokens] → Encodec decoder → wav
```

3 秒参考语音相当于 GPT 的 prompt——LLM "续写"出与参考音色一致的 audio token，再用 Encodec decoder 还原成波形。这种 **in-context voice cloning** 是 TTS 史上的范式跃迁，之前 zero-shot voice cloning 需要专门的 speaker embedding + 修改架构，VALL-E 直接靠 LLM 的 in-context learning 能力做到。

VALL-E 还引入了 "AR + NAR" 双阶段：第一阶段 AR 模型预测第 1 个 codebook（粗音色），第二阶段 NAR 模型并行预测第 2-8 个 codebook（细节）——这是后续 CosyVoice / SoundStorm 等的设计原型。

### 1.6 Moshi：speech LLM 的"全双工"突破

传统 speech 对话（Siri / Alexa）：你说完 → ASR → LLM → TTS → 它说。**半双工**——任意时刻只有一方在说，端到端延迟 800ms+。

Moshi（Kyutai 2024）的突破：把 user audio 与 model audio **作为两个并行 stream**，每一帧（80ms）模型同时观察 user 这一帧的 token + 决定自己这一帧要发什么 token（包括"沉默" token）。两路并行预测意味着模型可以**边听边说**——你说话时它可以立刻打断 / 附和 / 确认。延迟降到 200ms，接近人类对话延迟（160ms）。

这是 speech LLM 的"全双工"（full-duplex）突破，OpenAI 的 GPT-4o realtime 是同期商业产品。这两者背后的核心思想都是：**audio token 化 + 多 stream 并行预测**，把对话的物理同步性建模进 LLM。

---

## 2. 公式与原理

### 2.1 Whisper 架构：encoder-decoder Transformer

设输入 1 段 30 秒、16kHz 的 audio waveform $w \in \mathbb{R}^{16000 \times 30 = 480000}$。

**Step 1：log-Mel spectrogram 提取**。先做 STFT（短时 Fourier）+ Mel filterbank，得到 $X_{\text{mel}} \in \mathbb{R}^{T_a \times 80}$，其中 $T_a = 3000$（每 10ms 一帧、80 个 Mel 频带）。

**Step 2：encoder**。两层 stride=2 的 CNN 把 $T_a$ 从 3000 下采样到 1500，再加 sinusoidal positional embedding，进 24 层（large 是 32 层）standard Transformer encoder，输出 $H_{\text{enc}} \in \mathbb{R}^{1500 \times d}$，$d = 1280$（large）。

**Step 3：decoder**。decoder 是 standard text Transformer，prompt 形式：

$$[\text{SOT}] \, [\text{lang}] \, [\text{task}] \, [\text{notimestamps}] \, [\text{text tokens}\dots] \, [\text{EOT}]$$

其中 `[lang]` 是 99 个语言 token 之一（`<|en|>` / `<|zh|>` / ...）、`[task]` 是 `<|transcribe|>` 或 `<|translate|>`。Decoder 的每一层有：

- **self-attention**（causal mask，prompt 内部）
- **cross-attention**（query 来自 decoder，key/value 来自 $H_{\text{enc}}$）
- **FFN**

训练目标是标准 next-token prediction：$\mathcal{L} = -\sum_t \log p(y_t \mid y_{<t}, X_{\text{mel}})$。

**Whisper-large-v3** 的 size：encoder 32 层 + decoder 32 层、$d = 1280$、$h = 20$，约 1.55B 参数。在 LibriSpeech test-clean 上 WER ≈ 3.0%，是开源 SOTA。

### 2.2 Encodec：audio token 化的标准做法

Encodec（Défossez 2022）的目标：把 24kHz waveform 压成 75Hz × 8 个 codebook 的离散 token，bitrate 6 kbps，可逆重建。

**架构**：

$$\text{Encoder}: w \in \mathbb{R}^{T} \to z \in \mathbb{R}^{T/320 \times d}$$

(stride 2/4/5/8 的四层 1D conv，总共下采样 320×；$T = 24000$ → 75 帧 / 秒)

$$\text{RVQ}: z \to (q_1, q_2, \dots, q_K), \quad q_k \in \{0, 1, \dots, V-1\}^{T/320}$$

(K 层 vector quantization，每层 codebook size $V = 1024$；第 $k$ 层量化前 $k-1$ 层的累计残差)

$$\text{Decoder}: \sum_{k=1}^{K} \text{Embed}_k(q_k) \to \hat{w} \in \mathbb{R}^{T}$$

(对称的转置卷积上采样回 24kHz)

**RVQ 的递归量化**：

$$\text{res}_0 = z, \quad q_k = \arg\min_{c \in C_k} \|\text{res}_{k-1} - c\|^2, \quad \text{res}_k = \text{res}_{k-1} - C_k[q_k]$$

每个 codebook 1024 entries → 10 bits / token，K=8 时 80 bits / 75Hz 帧 = 6 kbps，能保留 24kHz speech 的近无损质量。

**为什么 RVQ 而不是单个大 codebook**：要达到 80 bits 信息量，单个 codebook 需要 $2^{80}$ 个 entries，根本无法训练；而 8 个 1024-entry codebook 总参数只有 $8 \times 1024 \times d$，且每层只学"前一层没建模到的残差"，训练稳定。

### 2.3 VALL-E：discrete audio token 上的 LLM

设 target text 的 BPE token 为 $T = (t_1, \dots, t_N)$、3 秒参考语音的 Encodec token 为 $A^{\text{prompt}} = (a_{1:K, 1:M^{\text{prompt}}})$（$K$ codebook × $M^{\text{prompt}}$ 帧），目标 audio token 为 $A^{\text{target}} = (a_{1:K, 1:M^{\text{target}}})$。

**AR stage（第 1 个 codebook）**：

$$p(a_{1, 1:M^{\text{target}}} \mid T, A^{\text{prompt}}) = \prod_{m=1}^{M^{\text{target}}} p(a_{1, m} \mid T, A^{\text{prompt}}, a_{1, <m})$$

decoder-only Transformer，输入序列是 $[T; a^{\text{prompt}}_{1, :}; a^{\text{target}}_{1, :}]$，标准 causal LM 训练。

**NAR stage（第 2-K 个 codebook）**：给定第 1 层的全部 token + 前 $k-1$ 层的全部 token，**并行**预测第 $k$ 层的所有位置：

$$p(a_{k, 1:M^{\text{target}}} \mid T, A^{\text{prompt}}, a_{1:k-1, 1:M^{\text{target}}})$$

NAR 用 non-causal attention（没有 mask），所有位置一次出。这样 8 层 codebook 只需 1 次 AR + 7 次 NAR forward，比纯 AR（8 × $M$ 步）快 K 倍。

**为什么 AR + NAR 拆开**：第 1 层 codebook 编码"主导音色 + 韵律"，强依赖时序，必须 AR；第 2-8 层是"细节残差"，给定前面层后位置间相对独立，可以 NAR 并行。

**Zero-shot voice cloning**：训练时只见过"text + 同一段 audio"配对；推理时把 3 秒参考音放在 prompt 位置，target text 后接续生成的 token 自动继承参考音色——这是 LLM in-context learning 在 audio 上的体现。

### 2.4 Moshi：双 stream 并行预测的全双工

Moshi 的核心是一个 7B 的 decoder-only Transformer（基于自家 Helium 7B），但 token 序列长这样：

```
time:     t=0          t=1          t=2          t=3       ...
user:    [u_0]        [u_1]        [u_2]        [u_3]      ...   ← user audio token (Mimi codec)
moshi:   [m_0]        [m_1]        [m_2]        [m_3]      ...   ← Moshi audio token (Mimi codec)
text:    [w_0]        [w_1]        [w_2]        [w_3]      ...   ← text token (内部独白 inner monologue)
```

每一时间步 $t$ 有 3 类 token 并行：user audio、moshi audio、text。模型预测：

$$p(m_t, w_t \mid u_{\le t}, m_{<t}, w_{<t})$$

注意 user token $u_t$ 是**观察输入**（不预测），model 只预测自己的 $m_t$ 与 text $w_t$。这就是"边听（$u_t$ 流入）边说（$m_t$ 流出）"的全双工建模。

**几个关键工程点**：

- **Mimi codec**：Moshi 自研的 streaming neural codec，12.5Hz × 8 codebook（比 Encodec 75Hz 更稀疏），用 distillation 注入 semantic 信息（兼具 codec 的低 bitrate + semantic token 的语义对齐）
- **inner monologue**：text $w_t$ 与 audio $m_t$ 同步生成，相当于 model "默念" 自己要说的内容。empirical 上加 text stream 显著提升 audio 生成质量与可控性
- **delay pattern**：text 提前 audio 几帧（如 2 帧），让 audio 总能"看着 text"生成
- **Depth Transformer**：每个 time step 内 8 个 codebook 也用一个小 Transformer 顺序预测（类似 RQ-Transformer），保留 RVQ 的层间依赖

端到端延迟：80ms × 帧延迟 + 模型 forward ≈ 160-200ms，逼近人类对话延迟。

### 2.5 ASR / TTS 评测指标

**WER (Word Error Rate)**——ASR 黄金标准：

$$\text{WER} = \frac{S + D + I}{N}$$

其中 $S$ / $D$ / $I$ 分别是替换 / 删除 / 插入错误数，$N$ 是 reference 总词数。Whisper-v3 在 LibriSpeech test-clean WER 约 3%、test-other 约 5.4%；中文 AISHELL-1 约 7-9%（中文 Whisper 较 OpenAI 内部数据弱）。

**MOS (Mean Opinion Score)**——TTS 自然度的人评，1-5 分：

- 5 = 与真人无差
- 4 = 自然但能听出是合成
- 3 = 听得懂但明显机械
- 2 = 勉强听得懂
- 1 = 完全不可懂

现代 TTS（CosyVoice 2 / F5-TTS）在英文 / 中文都已经 MOS > 4.0，逼近真人 4.5。

**Speaker Similarity (SECS / SIM)**——voice cloning 时合成音与参考音的音色相似度，用 ECAPA-TDNN 等 speaker encoder 算 cosine similarity。> 0.7 算优秀。

**SECS 的本质问题**：speaker encoder 自身有偏，且对韵律 / 情感不敏感——它只是个"音色相似度"近似，不能完全替代人评。

---

## 3. 最小代码示例

### 3.1 Whisper transcribe demo（< 25 行）

```python
# pip install openai-whisper soundfile   或  pip install faster-whisper
# Apple Silicon / CPU 推理友好
import whisper

# 加载模型：tiny / base / small / medium / large-v3
# large-v3 是 1.5B 参数，CPU 上 30 秒音频要 1-2 分钟；GPU 几秒
model = whisper.load_model("base")     # base ≈ 74M，CPU 友好

# 一句话推理：自动检测语言、自动断句、自动加时间戳
result = model.transcribe(
    "audio.wav",
    language="zh",                     # 强制中文，比自动检测稳
    task="transcribe",                 # 或 "translate"（任意语言 → 英文）
    initial_prompt="以下是普通话内容。",  # 风格 prompt，影响 punctuation / 用词
    word_timestamps=True,              # 返回每个词的时间戳
    fp16=False,                        # CPU 必须关
)

print("text:", result["text"])
print("language:", result["language"])
for seg in result["segments"][:3]:
    print(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")
# segments 里还有 'words'（如果 word_timestamps=True）
```

**生产环境推荐 `faster-whisper`**——基于 CTranslate2 的 INT8/FP16 量化推理，比 openai-whisper 快 4-10×、显存 / 内存少 60%：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.wav", language="zh", beam_size=5)
for seg in segments:
    print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}")
```

### 3.2 Encodec audio tokenize demo（< 20 行）

演示 audio → discrete token → audio 的完整往返，理解 RVQ 的 K 个 codebook：

```python
# pip install encodec
import torch, torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

model = EncodecModel.encodec_model_24khz()    # 24kHz 模型；另有 48kHz stereo
model.set_target_bandwidth(6.0)               # 6 kbps → 8 个 codebook 激活

wav, sr = torchaudio.load("speech.wav")
wav = convert_audio(wav, sr, model.sample_rate, model.channels)  # → 24kHz mono
wav = wav.unsqueeze(0)                         # (1, 1, T)

with torch.no_grad():
    encoded_frames = model.encode(wav)         # list of (codes, scale) per chunk
    codes = torch.cat([f[0] for f in encoded_frames], dim=-1)  # (1, K=8, T_frames)
    print("audio shape:", wav.shape, "→ codes shape:", codes.shape)
    # 1 秒 24kHz audio (24000) → ~75 帧 × 8 codebook → 600 个 int 表示 1 秒
    print("token range:", codes.min().item(), codes.max().item())  # 0..1023

    # 反向解码
    wav_recon = model.decode(encoded_frames)   # (1, 1, T)
    torchaudio.save("recon.wav", wav_recon[0].cpu(), model.sample_rate)
```

**关键观察**：原始 1 秒 = 24000 个 float（96 KB fp32），编码后 = 600 个 10-bit int（750 字节），**压缩 128×**。这就是为什么 audio LLM 能跑——600 token / 秒已经是 LLM-friendly 量级（与 text 的 3-5 token / 秒不算同量级，但已经可承受）。

### 3.3 CosyVoice TTS demo（< 25 行）

```python
# pip install cosyvoice  (或从 github clone https://github.com/FunAudioLLM/CosyVoice)
# 模型从 modelscope / huggingface 下载（CosyVoice2-0.5B）
from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio

cosyvoice = CosyVoice2("FunAudioLLM/CosyVoice2-0.5B")

# 模式 1：zero-shot voice cloning（最常用）
prompt_speech_16k, _ = torchaudio.load("reference_3s.wav")    # 3 秒参考音
for i, out in enumerate(cosyvoice.inference_zero_shot(
        tts_text="你好，欢迎使用 CosyVoice 进行语音合成。",
        prompt_text="参考音频对应的文本",                       # 参考音的 transcript
        prompt_speech_16k=prompt_speech_16k,
        stream=True,                                            # 流式输出
)):
    torchaudio.save(f"out_{i}.wav", out["tts_speech"], 24000)

# 模式 2：跨语言（中文参考音 → 英文输出）
for out in cosyvoice.inference_cross_lingual(
        tts_text="<|en|>Hello, this is a test of cross-lingual voice cloning.",
        prompt_speech_16k=prompt_speech_16k,
):
    torchaudio.save("en_out.wav", out["tts_speech"], 24000)

# 模式 3：指令式（控制情绪 / 风格）
for out in cosyvoice.inference_instruct2(
        tts_text="这条新闻太令人震惊了！",
        instruct_text="用激动的语气朗读。",
        prompt_speech_16k=prompt_speech_16k,
):
    torchaudio.save("emotion.wav", out["tts_speech"], 24000)
```

**工程要点**：
- `stream=True` 让 first-byte latency 降到 ~300ms（chunked output），实时对话必开
- 参考音 3-10 秒最佳；过短音色不稳，过长 prompt 拼接耗时
- 输出固定 24kHz；接 ASR / 语音通话需要 resample 到 16kHz

---

## 4. 工程踩坑与经验

- ❗ **Whisper-large 在 streaming / 实时场景慢，必须用 faster-whisper 或 distil-whisper**。openai-whisper 的官方实现 PyTorch 版本，large-v3 在 RTX 4090 上 30 秒 audio 推理约 1.5 秒——离实时还差不少，且不支持流式分块。**生产推荐**：(1) `faster-whisper` 基于 CTranslate2，INT8/FP16 量化 + 流式 chunk，30s audio 0.3s 内完成；(2) `distil-whisper`（HuggingFace 蒸馏版）保留 99% WER 但 6× 加速；(3) 实时场景用 `whisper-streaming` 这种带 VAD + chunk overlap 的流式封装。
- ❗ **TTS 的 latency 比 ASR 重要得多——实时对话 TTFB（first-byte）要 < 500ms，CosyVoice / Moshi 等必须支持 streaming generation**。ASR 是"用户说完后等 1 秒可接受"，TTS 是"模型说话前等 1 秒就尬"。Streaming TTS 的实现要点：(1) audio token 一边生成一边送 codec decoder（chunk size 通常 25-50 帧 ≈ 333-666ms）；(2) **不能等全部 token 出完再 decode**——那 latency = 完整时长；(3) streaming 模式 quality 略低于 offline 模式（chunk boundary artifacts），生产要在两者间取舍。
- ❗ **Voice cloning 引发 deepfake 安全问题，commercial 部署必须加 watermark**。VALL-E / CosyVoice 的 zero-shot voice cloning 只需 3-10 秒参考音，技术门槛极低——已经被用于诈骗（"绑架家人"、"老板转账"）。负责任的部署：(1) **音频水印**（如 Meta AudioSeal / Microsoft Audio Watermark），在生成的 wav 中嵌入 18-bit ID，肉眼听不出但能检测；(2) **明确标注 AI-generated**；(3) 限制可克隆的 speaker（白名单 / 一次性授权 token）。OpenAI 至今未开放 voice cloning API 部分原因就是这个。
- ❗ **中文 ASR 单靠 Whisper 不够——Whisper-large-v3 在 AISHELL-1 上 CER 约 5-7%，远高于专门为中文训的 Paraformer / SenseVoice（CER 2-3%）**。Whisper 训练数据 680k 小时，但中文占比有限，且字幕质量参差。生产中文 ASR 选型：(1) 阿里 **SenseVoice** / **Paraformer-large** 是中文 SOTA 开源方案；(2) **FunASR** 是 SenseVoice 的工程化封装，支持 streaming + VAD + 标点；(3) 如果一定用 Whisper，做 **Whisper + LoRA finetune on 中文数据**（如 AISHELL-3 + WenetSpeech）能把 CER 降到 4% 内。
- ❗ **Audio tokenization 是 lossy——high-quality TTS 必须用 high-bitrate codec**。Encodec 6 kbps 的重建质量已经可懂，但有"金属感"；TTS 直接以 6 kbps 为目标会让生成 audio 听着不自然。**对策**：(1) 用更高 bitrate（12-24 kbps，对应 16-32 codebook）；(2) 用专门为 TTS 设计的 codec（CosyVoice 的 SenseVoice token、Moshi 的 Mimi codec）——这些 codec 在 token 化时注入 semantic 信息，重建质量在低 bitrate 下也优于通用 Encodec；(3) **后处理 vocoder**（HifiGAN / BigVGAN）对 codec decoder 输出做超分修饰。
- ❗ **Speech LLM 的 latency 是核心 KPI——不像 text LLM 可以 stream 容忍，audio 是"听到才算"**。Text LLM stream 模式下 TTFT 200ms 用户感觉不到延迟（因为还要看着 token 一个个出）；speech LLM 即使 stream 模式，第一个 audio chunk 必须在 ~200-400ms 内出来，否则用户会感觉"AI 还没反应过来"。这把整个工程难度提升一个量级：(1) Speech LLM 必须 PagedAttention + KV cache + 投机解码（Module 11）；(2) audio codec 必须 streaming decoding；(3) 端到端 ASR + LLM + TTS pipeline 的延迟拆账：ASR 100ms + LLM TTFT 200ms + TTS first-byte 200ms = 500ms 已经是用户体验的上限。Moshi / GPT-4o realtime 把这个数字压到 200ms 是工程上的暴击。
- ❗ **语音 model 的 sampling rate 必须严格一致——16kHz / 22.05kHz / 24kHz / 48kHz 接错就塌**。Whisper 喂 24kHz 进去？输出全是噪声。CosyVoice 输出 24kHz 直接送 16kHz 通话系统？听起来变调或卡顿。**纪律**：(1) ASR 业界标准 16kHz（Whisper / SenseVoice / Wav2Vec2 全是）；(2) TTS 输出常见 22.05kHz（VITS）/ 24kHz（CosyVoice / VALL-E）/ 48kHz（高保真 codec）；(3) 任何 pipeline 必须在 input / output 处用 `torchaudio.transforms.Resample` 统一；(4) 通话系统上下游一般 8kHz / 16kHz——TTS 输出要 downsample，丢失高频但不卡顿。
- ❗ **Whisper 的 hallucination 在长沉默 / 噪声段非常严重**。Whisper 训练数据是字幕，几乎不见纯噪声片段；推理时遇到 30 秒沉默会"幻觉"出"thank you for watching"、"字幕由 XX 提供" 这种从训练数据学来的口头禅。**对策**：(1) 推理前用 **Silero VAD** 切除静音段，只把有 voice 的 chunk 喂 Whisper；(2) 设 `no_speech_threshold=0.6` + `logprob_threshold=-1.0`——超出阈值的段标记为静音不输出；(3) 真正鲁棒的 ASR pipeline 必须 VAD + Whisper + 后处理过滤幻觉短语。
- ❗ **TTS 的 voice cloning 数据偏见——参考音如果是"标准普通话女声"，生成的音几乎都是同一种声音**。模型在训练时如果见到的某些口音 / 性别 / 情感分布稀疏，对应的 cloning 质量就差。例：CosyVoice 对粤语 / 川普的 clone 质量明显低于普通话；对老人 / 儿童音色的 clone 不稳定。**生产对策**：(1) 评测必须覆盖多 demographic 的 reference voice；(2) 做 finetune 时刻意补稀疏维度的数据；(3) 给 reference voice 一个 "quality score"，质量低的拒绝克隆，避免输出音色失真带来的客户投诉。
- ❗ **ASR + LLM + TTS pipeline 与端到端 speech LLM 的取舍**。pipeline 优势：每个组件可独立替换、ASR 输出的文本可作为 LLM 的 RAG context、可控性强；劣势：3 段延迟累加 + 信息损失（ASR 丢韵律 / 情感、TTS 加机械感）。端到端（Moshi / GPT-4o）：低延迟、保留韵律 / 情感的端到端学习；劣势：模型权重大且训练数据稀缺、可控性差（TTS 内容审计困难）、debug 难。**实战建议**：早期 MVP / 内部工具用 pipeline；2C 实时对话产品如果延迟 / 自然度是核心 KPI，端到端是未来方向；折中方案是 "Whisper encoder + LLM decoder"（如 Qwen2-Audio）——ASR + LLM 一体但 TTS 还是外部组件。

---

## 5. 经典 paper

- **Radford et al., 2022 — Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)** — ASR 的"BERT 时刻"。读它的 §2 "Approach" 看"680k 小时数据 + 标准 encoder-decoder Transformer + multi-task prompt"如何不靠任何架构创新就 SOTA；§4 "Analysis" 的 "robust to noise / accent" 实验是 ASR 工业部署的最佳论据。理解这篇你才理解为什么 2022 之后所有 ASR 工作都基于 Whisper finetune。
- **Wang et al., 2023 — Neural Codec Language Models are Zero-Shot Text-to-Speech Synthesizers (VALL-E)** — TTS 的"GPT 时刻"。读它的 §3 "Background and Method"——"audio token + LLM + 3 秒 prompt → zero-shot voice cloning"的整个范式跃迁就是这一段。AR + NAR 双阶段也来自这篇（§3.2.3 / §3.2.4）。读完 VALL-E 你能理解 CosyVoice / SoundStorm / NaturalSpeech 3 等所有现代 TTS 都在 VALL-E 范式上演化。
- **Défossez et al., 2024 — Moshi: a speech-text foundation model for real-time dialogue** — 全双工 speech LLM 的奠基作。读 §3 的 inner monologue（text + audio dual stream） 和 §4 的 Mimi codec 如何把 streaming + semantic 一起塞进 codec。这篇的工程细节密度极高，是了解"怎么做出 GPT-4o realtime 同等产品"的最详细公开材料（Kyutai 是少数把 speech LLM 完全开源的团队）。
- **Défossez et al., 2022 — High Fidelity Neural Audio Compression (Encodec)** — audio token 化的 de facto 标准。读 §3 "Method" 的 RVQ 设计（§3.4）就够了——理解为什么是"K 个 1024-entry codebook"而非"1 个大 codebook"。这是后续所有 audio LLM 的 vocabulary 来源，是这一节绕不开的底座。

---

## 6. 自测与面试题

**Q1（架构）**：VALL-E 用 LLM 范式做 TTS，关键 idea 是什么？为什么必须有 Encodec 这种 neural audio codec？AR + NAR 双阶段的 motivation 是什么？

<details>
<summary>Answer sketch</summary>

**关键 idea**：把 audio 用 codec quantize 成 discrete token 后，TTS 就变成 conditional language modeling——给定 [text token + 3 秒参考音的 audio token] 作为 prompt，让 LLM "续写"目标 audio token，再用 codec decoder 还原成波形。这把 TTS 从"专门设计的 acoustic model + vocoder"变成"标准 GPT 范式 + audio vocab"，享受 LLM 的所有红利（in-context learning、scaling、prompt engineering）。

**为什么需要 Encodec**：

- 原始 24kHz 波形 = 24000 floats / 秒，序列长度爆炸，LLM 处理不了
- 必须把 audio 变成短的 discrete token 序列才能套 LLM——Encodec 把它压到 75Hz × 8 codebook = 600 token / 秒
- discrete 是关键——continuous embedding 无法用 cross-entropy 学，必须量化成 vocab
- RVQ 比单个大 codebook 实用——8 个 1024-entry 的总信息量等于 $2^{80}$ entries 的单 codebook，但参数和训练难度差几个数量级

**AR + NAR 双阶段的 motivation**：

- 8 层 codebook 如果都 AR 串行预测，时间步 = $8 \times M$，慢 8 倍
- 第 1 层 codebook 编码"主导音色 + 韵律"，强时序依赖，必须 AR
- 第 2-8 层是"细节残差"——给定第 1 层后位置间相对独立，可以 NAR 并行（一次出全部位置）
- 总 forward 次数 = M（AR）+ 7（NAR）≈ M + 7，比纯 AR 的 8M 快 8×

加分：能说现代 TTS（CosyVoice 2 / SoundStorm / NaturalSpeech 3）大都基于 VALL-E 范式演化；能说 inner monologue / semantic codec（Mimi）是后续改进方向。

</details>

**Q2（trade-off）**：现代 speech LLM（Moshi / GPT-4o realtime）vs 传统 ASR + LLM + TTS pipeline，在延迟 / 质量 / 工程复杂度三个维度做对比。在哪些场景应该选哪种？

<details>
<summary>Answer sketch</summary>

| 维度 | ASR + LLM + TTS pipeline | 端到端 Speech LLM (Moshi) |
|---|---|---|
| 端到端延迟 | 500-1500ms（3 段累加 + 等待对方说完）| 200-300ms（边听边说，全双工）|
| 韵律 / 情感保留 | 差（ASR 输出纯文本，丢失） | 好（audio token 直接预测，保留语调）|
| 内容质量 | 高（LLM 是 native text，可用最大模型） | 中（speech LLM 通常较小，且训练数据少）|
| 可控性 | 高（中间是文本，可加 RAG / 工具调用 / 审计）| 低（audio token 内部，难审计）|
| 工程复杂度 | 中（每个组件独立可替换、debug 友好）| 高（端到端训练数据稀缺、整体重训成本极大）|
| 模型大小 | 总和大（Whisper + LLM + TTS = 多个模型）| 单模型大（Moshi 7B 但只一个）|
| 多语言 / 方言 | 强（每个组件可独立选型） | 弱（端到端模型语种覆盖受限于训练数据）|
| 成本（推理）| 高（3 段都要算）| 中（只 1 段，但模型大）|

**场景选型**：

- **早期 MVP / 内部工具 / 客服外呼 / IVR**：pipeline。延迟可接受、可控性 / 审计友好
- **2C 实时对话 / 教育陪练 / 同传**：端到端。延迟与情感是核心 KPI，pipeline 不及格
- **音频内容生产（podcast / 配音）**：pipeline。完全 offline，质量 / 可控性 > 延迟
- **多模态 agent**（speech-in 但 text-out）：折中——Whisper encoder + LLM decoder（如 Qwen2-Audio）。低延迟 ASR + 强 LLM 推理，但仍由文本输出

加分：能说"全双工"是端到端最大的不可替代优势——pipeline 必须等用户说完才能 ASR，而端到端可以边听边形成回应、必要时打断；能说 GPT-4o realtime / Moshi 也保留 text stream 用于审计与工具调用（hybrid 设计）。

</details>

**Q3（实战）**：你要做一个**中文实时语音助手**（手机 App，对话场景）。给出 ASR / TTS / LLM 三段的选型理由 + 整体 latency 优化方案。说出你预期的 TTFB 与端到端延迟。

<details>
<summary>Answer sketch</summary>

**选型方案 A（pipeline 路线，2025 落地友好）**：

- **ASR**：阿里 SenseVoice-Small / Paraformer-large + Silero VAD。中文 CER 4-6%，比 Whisper-large 在中文上更准；INT8 量化在 RTX 4090 / Apple M3 上单句 200ms 内
- **LLM**：Qwen3-32B（云端）或 Qwen3-7B-INT4（端侧）；vLLM / SGLang 部署 + KV cache prefix sharing；流式 first-token 200ms
- **TTS**：CosyVoice 2-0.5B 或 GPT-SoVITS。streaming generation、first-byte 300ms，24kHz 输出
- **端到端拼接**：

```
用户说话 → VAD 检测 endpoint → SenseVoice ASR (200ms) → Qwen3 LLM TTFT (200ms)
       → CosyVoice TTS first-chunk (300ms) → 播放
TTFB ≈ 700ms（ASR + LLM TTFT + TTS first-chunk）
```

**latency 优化技巧**（必须答到 3-5 条）：

- **VAD endpoint detection** 提前——用户停顿 300ms 即触发 ASR，不等用户说完
- **ASR streaming**：边听边转，不等说完。SenseVoice / Whisper-streaming 都支持
- **LLM 投机解码**（Module 11）：用 0.5B draft model + Qwen3-32B 验证，TTFT 不变但 throughput 翻倍
- **TTS streaming**：text 一边出 token，TTS 一边合成；不等 LLM 写完整句
- **句末标点驱动 TTS chunk**：以 "，。！？" 为分块边界送 TTS，每个 chunk 独立合成，user 听到第一句话时 LLM 还在出第二句
- **PagedAttention + Prefix Cache**：多用户共享 system prompt 的 KV cache（Module 11.3 SGLang）
- **端侧 / 云端混合**：ASR + TTS 端侧（隐私 + 0 网络延迟），LLM 云端（大模型质量）

**预期延迟**：
- TTFB（用户说完到第一个字播放）：800-1200ms（pipeline 极限）
- Streaming 优化后：500-700ms（用户感知"接近自然对话"，但仍有"AI 在想"的感觉）
- 完整一段回复：取决于 LLM 输出长度，每秒 8-15 字

**方案 B（端到端 Speech LLM，2025 前沿）**：

- 用 Moshi / GPT-4o realtime 替代整个 pipeline，TTFB 200-300ms
- 缺点：中文支持目前不如 pipeline 成熟、可控性 / 审计差、训练 / finetune 成本高
- 适合"未来 6-12 个月迭代到位"的产品路线，不适合今天就上线

加分：能说出"中文音色克隆 / 方言支持"是 voice cloning 的 differentiator；能说出 endpoint detection 的精度直接决定用户感知（误检会让 AI 抢话、漏检会让 AI 不响应）；能说出 watermark + AI 标识是合规底线。

</details>

---

## 7. 延伸阅读

- [Whisper GitHub](https://github.com/openai/whisper) — 官方仓库，含模型 ckpt 与 inference code
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper) — 生产级 Whisper 推理优化（CTranslate2，4-10× 加速）
- [Distil-Whisper GitHub](https://github.com/huggingface/distil-whisper) — HuggingFace 蒸馏版，6× 加速保留 99% WER
- [VALL-E demo page](https://www.microsoft.com/en-us/research/project/vall-e-x/) — Microsoft 官方 demo，听 3 秒克隆效果（VALL-E 本身未开源，社区有复现 [VALL-E-X](https://github.com/Plachtaa/VALL-E-X)）
- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice) — 阿里开源 SOTA TTS，中英双语强、2024 年最值得用的 TTS
- [F5-TTS GitHub](https://github.com/SWivid/F5-TTS) — diffusion-based TTS，无需 phoneme alignment，质量高
- [Moshi GitHub](https://github.com/kyutai-labs/moshi) — 全双工 Speech LLM 完整开源（含 Mimi codec / model weights / inference）
- [Encodec GitHub](https://github.com/facebookresearch/encodec) — Meta 官方 audio codec，理解 audio tokenization 必读
- [SNAC GitHub](https://github.com/hubertsiuzdak/snac) — 2024 年 multi-scale neural codec，TTS 友好
- [SenseVoice / FunASR (modelscope)](https://github.com/FunAudioLLM/SenseVoice) — 阿里开源中文 ASR SOTA，CER 比 Whisper-large 在中文上低 2-3%
- [Silero VAD GitHub](https://github.com/snakers4/silero-vad) — 工业级 voice activity detection，ASR pipeline 的标配前置
- [HuggingFace Audio Course](https://huggingface.co/learn/audio-course) — 系统化的 audio NLP 教程
- 推荐继续读本教程的 **16.4 节《Embedding：bge / E5 / Instructor / NV-Embed》**——retrieval / RAG 的底座，与多模态向量化呼应
- 推荐回顾 **Module 4** 的 encoder-decoder（4.6 §1.2 ASCII 图）——Whisper 是它的标准实例化；同时回顾 Module 11 的推理优化——speech LLM 的 latency 工程是 Module 11 所有技巧的极限场景
