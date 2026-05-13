---
title: "附录 B：Capstone 2 — SFT + LoRA + RAG demo"
description: "把 Module 8 的 QLoRA 训练管线和 Module 13 的 RAG 检索管线串成一个完整工程：在一张 24 GB RTX 4090 上 QLoRA fine-tune LLaMA-3-8B-Instruct 成医学领域 chat 模型，再外挂 bge-m3 + FAISS 的 RAG，最后用 vLLM 多 LoRA 部署——本节给出的是一份能复制就跑的端到端 recipe，所有踩过的"
---

> ⏱ 预计阅读 60 分钟 ｜ 难度 ★★★ ｜ 前置：Module 8（SFT/PEFT）、Module 13（RAG）

## 一句话本节讲什么

把 Module 8 的 QLoRA 训练管线和 Module 13 的 RAG 检索管线串成一个完整工程：在一张 24 GB RTX 4090 上 QLoRA fine-tune LLaMA-3-8B-Instruct 成医学领域 chat 模型，再外挂 bge-m3 + FAISS 的 RAG，最后用 vLLM 多 LoRA 部署——本节给出的是一份能复制就跑的端到端 recipe，所有踩过的坑都标在踩坑章节里。

---

## 1. Mental model（直觉）

Capstone 1（附录 A）让你**手撸**了一个 GPT-2 124M 从数据到训完到评测的完整 pipeline——目标是理解"一个 LLM 是怎么从随机权重变成会续写文本的"。Capstone 2 切换到**应用工程师**视角：base model 已经存在（LLaMA-3-8B-Instruct，Meta 出钱训的），数据也现成（PubMedQA / MedQA），你的活是把它**变成一个会看医学资料的领域 chatbot**。

整套工程对应两条独立又互补的"知识注入"路径：

```
        参数化知识（写到模型权重里）
   ┌──────────────────────────────────┐
   │  base LLM  ──QLoRA SFT──►  专家 │  ← 教它「医学问答的格式 / 风格 / 常识」
   └──────────────────────────────────┘
                    +
        非参数化知识（外挂检索）
   ┌──────────────────────────────────┐
   │  Vector DB ──retrieve──►  context│  ← 接最新文献 / 内网手册 / 长尾事实
   └──────────────────────────────────┘
                    ↓
              用户 query → 答案 + 引用
```

把这两条路径分开想，能解决一个新人最常问的问题——"既然 RAG 能塞知识，我还 SFT 干嘛？反过来既然 SFT 能学知识，RAG 还要吗？" 答案是它们解决的根本不是同一类问题：

- **SFT 教的是**"怎么说话"——回答风格、医学缩写、安全免责声明、思维链格式。这些是**模式**，反复出现在训练数据里被参数学会，**不依赖检索**。
- **RAG 注的是**"具体说什么"——某 paper 的具体结论、某药 2025 年新出的副作用警告、医院内部诊疗指南。这些是**事实**，**不应该塞进权重**（每次更新都要重训），而应该外挂检索。

所以 production-grade 医学问答系统几乎必然是 **SFT + RAG 两件套**，不是单选题。本 capstone 走完整两段，然后用 vLLM 把 LoRA + base 一起 serve 起来，提供一个支持 multi-LoRA 切换的 OpenAI-compatible API。

---

## 2. Pipeline 总览：8 步 end-to-end

把整个工程拆成 8 步，每步有明确的输入 / 输出 / 估时 / 显存——这是面试 / 工程交付时被问到"这个项目你怎么做的"的标准回答骨架：

| 步骤 | 内容 | 输入 | 输出 | 单 4090 估时 |
|---|---|---|---|---|
| **1** | 环境准备 | — | conda env + CUDA 12.1 | 30 min |
| **2** | 数据准备 | PubMedQA / MedQA raw | `messages` 格式 SFT 数据集 | 20 min |
| **3** | QLoRA SFT 训练 | base + dataset | `lora_adapter/` 目录 (~80 MB) | 4-6 h |
| **4** | merge LoRA + 测试 chat | base + adapter | `merged_model/` 目录 (~15 GB) | 20 min |
| **5** | 构建 RAG 知识库 | 医学 PDF / FAQ | FAISS 索引 + chunk metadata | 30 min |
| **6** | 组装 RAG + LLM pipeline | merged model + index | 一个 `chat(query)` 函数 | 30 min |
| **7** | vLLM 多 LoRA 部署 | base + adapter(s) | OpenAI-compatible API server | 20 min |
| **8** | 端到端 eval | API + eval set | MedQA acc / RAGAS faithfulness | 1 h |

显存总览：**Step 3 训练阶段 ~18 GB**（QLoRA + ckpt + bs=2 + seq=2048）、**Step 4 merge 阶段 ~16 GB**（base bf16 全量 load）、**Step 7 vLLM serve 阶段 ~16 GB**（bf16 base + 1-4 个 LoRA adapter，gpu-mem-util 0.85）。RTX 4090 24 GB 全程留 6-8 GB buffer。

下面逐步给完整代码 + 调试经验。

---

## 3. Step 1：环境准备

### 3.1 关键依赖与版本

```bash
# 推荐 Python 3.11 + CUDA 12.1（bitsandbytes 在 12+ 兼容性最好）
conda create -n med-rag python=3.11 -y && conda activate med-rag

# 训练栈
pip install "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.45" "peft>=0.13" "trl>=0.11" \
            "datasets>=3.0" "accelerate>=1.0" \
            "bitsandbytes>=0.43" "flash-attn>=2.6" --no-build-isolation

# RAG 栈
pip install "sentence-transformers>=3.1" "faiss-cpu>=1.8" \
            "pypdf>=4.0" "rank_bm25>=0.2.2"

# 部署 + eval
pip install "vllm>=0.6" "ragas>=0.2" "openai>=1.50"
```

`flash-attn` 需要源码编译且要求 CUDA toolkit 与 nvcc 在 PATH 上；如果 build 报错，可以先去掉它，模型加载时把 `attn_implementation="flash_attention_2"` 改成 `"sdpa"`，性能掉 30% 但能跑。

### 3.2 GPU 与磁盘检查

```bash
# 必须 ≥ Ampere（RTX 30/40/50 系、A10/A100、H100）才支持 bf16 + 4-bit
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
# RTX 4090 显示 compute_cap=8.9（Ada Lovelace）→ 全套支持

# base + adapter + index 大约要 50 GB 磁盘
df -h ~  # 确认 home 目录空间
```

如果是 V100 / T4（compute_cap < 8.0）就别试 QLoRA 了，跑普通 LoRA + bf16/fp16 base。

---

## 4. Step 2：数据准备

### 4.1 选数据：PubMedQA + MedQA

- **PubMedQA**（[bigbio/pubmed_qa](https://huggingface.co/datasets/bigbio/pubmed_qa)）：1k 条专家标注 + 200k 弱标注，"问题 + abstract + yes/no/maybe"格式，适合 grounding 风格 SFT
- **MedQA**（[bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa)）：USMLE 多选题，3.7k 训练 + 1.3k 测试，适合 reasoning + 医学常识

为了 demo 速度，下面只取 PubMedQA 的 expert-labeled 部分（1k 条）+ MedQA train（3.7k 条），合计约 4.7k 样本——足够 SFT 出明显效果，又能在 4090 上 4-6 小时训完。

### 4.2 转成 chat messages 格式

```python
# data_prep.py
from datasets import load_dataset, Dataset

def pubmed_to_messages(ex):
    abstract = " ".join(ex["context"]["contexts"])
    return {"messages": [
        {"role": "system", "content": "You are a clinical expert. Answer concisely with evidence."},
        {"role": "user", "content": f"Question: {ex['question']}\n\nReference abstract:\n{abstract}"},
        {"role": "assistant", "content": f"{ex['final_decision']}. {ex['long_answer']}"},
    ]}

def medqa_to_messages(ex):
    options = "\n".join(f"{k}. {v}" for k, v in ex["options"].items())
    return {"messages": [
        {"role": "system", "content": "You are a medical expert. Choose the best option and explain."},
        {"role": "user", "content": f"{ex['question']}\n\n{options}"},
        {"role": "assistant", "content": f"Answer: {ex['answer_idx']}. {ex['answer']}"},
    ]}

pmq = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", split="train")
mdq = load_dataset("bigbio/med_qa", "med_qa_en_source", split="train")

ds = Dataset.from_list(
    [pubmed_to_messages(x) for x in pmq] +
    [medqa_to_messages(x) for x in mdq]
).shuffle(seed=42)
ds.save_to_disk("./data/med_sft")
print(ds[0]["messages"])  # 肉眼校验一条
```

注意几件事：

- **system prompt 简短但定调**——SFT 训完模型会"记住"这个 system 的语气；如果训练时是英文 system 推理时换中文，模型表现会跌
- **assistant 内容必须是模型应该输出的最终格式**——包含答案 + 解释，不要只放选项字母（推理时模型会模仿）
- **多模板混合 OK，但要保持 messages schema 一致**——TRL 的 `SFTTrainer` 期望每条样本都有 `messages` 字段

### 4.3 为 RAG 准备文档语料

RAG 知识库的语料和 SFT 数据**必须不同**——SFT 已经"看过"的东西再放 RAG 里没意义；RAG 要补的是 SFT 数据没覆盖的长尾事实。这个 demo 用 PubMedQA 的 unlabeled 子集（21 万 abstract）当做"医学文献库"：

```python
unlabeled = load_dataset("bigbio/pubmed_qa",
                         "pubmed_qa_unlabeled_source", split="train[:5000]")
docs = [{
    "doc_id": ex["pubid"],
    "text": " ".join(ex["context"]["contexts"]),
    "title": ex["question"],
} for ex in unlabeled]

import json
with open("./data/rag_corpus.jsonl", "w") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
```

5000 条 abstract 平均 200-300 token，chunk 后约 1-2 万段，FAISS HNSW 索引几十 MB——4090 单卡完全 hold 得住。

---

## 5. Step 3：QLoRA SFT 训练（核心，≤ 100 行）

```python
# sft_train.py
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
OUT   = "./out/llama3-med-lora"

# 1. tokenizer：Llama-3 默认无 pad token，用 <|finetune_right_pad_id|>
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

# 2. base model：4-bit NF4 + double quant + bf16 compute（QLoRA 三件套）
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # ❗ 必须 bf16，fp16 会 NaN
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb,
    device_map={"": 0},
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)   # 提 LN 到 fp32 + 关 cache + 开 ckpt

# 3. LoRA：r=64、加全部 7 个 linear（attn 4 + ffn 3）
lora_cfg = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_dora=False,                  # 想再涨 1-3 点开 use_dora=True，慢 10%
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
# trainable params: ~85M || all params: 8B || trainable%: 1.06

# 4. SFT 配置
ds = load_from_disk("./data/med_sft")
sft_cfg = SFTConfig(
    output_dir=OUT,
    num_train_epochs=3,
    learning_rate=2e-4,                      # LoRA 经验值，比 full FT 大 ~10x
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,                               # 全程 bf16
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,           # effective bs = 16
    max_seq_length=2048,
    packing=True,                            # 吞吐 +60%~200%
    completion_only_loss=True,               # 自动只在 assistant 段算 loss
    gradient_checkpointing=True,             # 用计算换显存
    optim="paged_adamw_8bit",                # bnb 8-bit Adam + paged
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="wandb",                       # 没装就改成 "none"
)

# 5. 训
trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=ds,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(OUT)                      # 只保存 ~80 MB 的 LoRA adapter
```

启动：

```bash
CUDA_VISIBLE_DEVICES=0 python sft_train.py 2>&1 | tee train.log
```

**显存账本**（实测 4090 24 GB）：

| 组件 | 占用 |
|---|---|
| NF4 base (8B → 4-bit) | ~4 GB |
| LoRA weight + grad（r=64，~85M） | ~0.7 GB |
| 8-bit AdamW state | ~0.3 GB |
| Activation（gradient checkpointing 后） | ~10 GB |
| FlashAttention scratch + overhead | ~3 GB |
| **合计** | **~18 GB** |

留 6 GB buffer 给 batch 内的 long sample spike，4090 够用。如果 OOM，按下面优先级降：(1) `per_device_train_batch_size=4`、(2) `max_seq_length=1024`、(3) `gradient_accumulation_steps` 加倍保持 effective bs 不变。

**训练时长**估算：4.7k 样本 × 3 epoch / effective bs 16 ≈ 880 steps，单 step ~20s（packing 后），总计 ~5 小时。

---

## 6. Step 4：merge LoRA + 测试 chat

### 6.1 merge 出一个普通 model

```python
# merge.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA = "./out/llama3-med-lora"
MERGED = "./out/llama3-med-merged"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cpu",   # ❗ merge 在 CPU 做
)
peft = PeftModel.from_pretrained(base, LORA)
merged = peft.merge_and_unload()                           # W' = W + (α/r)·BA
merged.save_pretrained(MERGED, safe_serialization=True)
tok.save_pretrained(MERGED)
print(f"merged model saved to {MERGED}")
```

为什么 merge 要在 **CPU 上做**？因为 NF4 量化的 base model 不能直接 merge——`merge_and_unload` 要求 base 是 bf16/fp16 的普通 `nn.Linear`，4-bit 包装的 `Linear4bit` 不支持。所以这步重新 load 一份 bf16 base 到 CPU（~16 GB 内存够），merge 完直接 dump。如果 CPU 内存 < 32 GB，就开 swap 或者用 `device_map="auto"` 走 disk offload，会慢但能跑通。

### 6.2 quick smoke test

```python
# chat_test.py
from transformers import pipeline
import torch

pipe = pipeline("text-generation",
                model="./out/llama3-med-merged",
                torch_dtype=torch.bfloat16, device_map="auto")

msgs = [
    {"role": "system", "content": "You are a clinical expert."},
    {"role": "user", "content": "What is the first-line treatment for type 2 diabetes?"},
]
out = pipe(msgs, max_new_tokens=256, do_sample=False)
print(out[0]["generated_text"][-1]["content"])
```

期望对比 base：

- base LLaMA-3-8B-Instruct：会答得相对泛化，可能漏关键药品名（如 metformin）
- 训完的 med-merged：答案更专业（直接说 metformin、提到 contraindication / monitoring），格式接近训练数据风格

如果 merged model 输出乱码 / 重复 token / 完全没变化，按 §10 踩坑表逐项排查。

---

## 7. Step 5：构建 RAG 知识库

```python
# build_index.py（≤ 60 行）
import json, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMB_MODEL = "BAAI/bge-m3"               # 中英双语 dense + sparse + multi-vector
INDEX_PATH = "./out/rag_index.faiss"
元文档_PATH  = "./out/rag_meta.jsonl"

# 1. 读语料 + chunking
docs = [json.loads(l) for l in open("./data/rag_corpus.jsonl")]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100,
    separators=["\n\n", "\n", "。", ". ", " ", ""],
)

all_chunks, all_meta = [], []
for d in docs:
    for i, c in enumerate(splitter.split_text(d["text"])):
        all_chunks.append(c)
        all_meta.append({"doc_id": d["doc_id"], "chunk_id": i,
                         "title": d["title"]})
print(f"total chunks: {len(all_chunks)}")

# 2. embed（bge-m3 自动加 query/passage prefix）
emb_model = SentenceTransformer(EMB_MODEL, device="cuda")
emb = emb_model.encode(all_chunks,
                       batch_size=64,
                       normalize_embeddings=True,
                       show_progress_bar=True).astype(np.float32)
print(f"embedding shape: {emb.shape}")    # (N, 1024)

# 3. 建 HNSW 索引（M=32, ef_c=200 是稳定甜点）
dim = emb.shape[1]
index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.add(emb)
faiss.write_index(index, INDEX_PATH)

with open(元文档_PATH, "w") as f:
    for c, m in zip(all_chunks, all_meta):
        m["text"] = c
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
print(f"index saved to {INDEX_PATH}, meta to {元文档_PATH}")
```

5000 abstract → 约 1.5 万 chunk → 1.5 万 × 1024 × 4B ≈ **60 MB embedding**，HNSW 索引另加约 30 MB。bge-m3 在 4090 上 batch=64 编码速度约 200 chunks/s，全量 75 秒搞定。

注意 `IndexHNSWFlat` 的距离类型——`bge-m3` 输出已 normalize（`normalize_embeddings=True`），用 `METRIC_INNER_PRODUCT` 等价于 cosine similarity；如果用默认的 L2 距离会颠倒排序。

---

## 8. Step 6：组装 RAG + LLM pipeline

```python
# rag_chat.py（≤ 40 行）
import json, faiss, numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

INDEX_PATH = "./out/rag_index.faiss"
元文档 = [json.loads(l) for l in open("./out/rag_meta.jsonl")]
emb_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
index = faiss.read_index(INDEX_PATH)

LLM = "./out/llama3-med-merged"
tok = AutoTokenizer.from_pretrained(LLM)
llm = AutoModelForCausalLM.from_pretrained(
    LLM, torch_dtype=torch.bfloat16, device_map="auto")

PROMPT = """You are a clinical expert. Answer the question using ONLY the references
below. If the answer is not in the references, reply "I don't know based on the
provided context." Cite sources as [doc_id].

References:
{ctx}

Question: {q}
Answer:"""

def retrieve(q, k=5):
    q_emb = emb_model.encode([q], normalize_embeddings=True).astype(np.float32)
    scores, idx = index.search(q_emb, k)
    return [元文档[i] for i in idx[0]]

def chat(q, k=5):
    refs = retrieve(q, k)
    ctx = "\n\n".join(f"[{r['doc_id']}] {r['text']}" for r in refs)
    msgs = [{"role": "user", "content": PROMPT.format(ctx=ctx, q=q)}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt",
                                  add_generation_prompt=True).to("cuda")
    out = llm.generate(ids, max_new_tokens=512, do_sample=False,
                       eos_token_id=[tok.eos_token_id,
                                     tok.convert_tokens_to_ids("<|eot_id|>")])
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), refs

if __name__ == "__main__":
    ans, refs = chat("What are the cardiovascular benefits of GLP-1 agonists?")
    print(ans, "\n\nReferences:", [r["doc_id"] for r in refs])
```

要点：

- **prompt 模板必须写 grounding 指令**——"ONLY use references"、"reply I don't know"、"cite [doc_id]"，三件套缺一不可，否则 SFT 后的模型会把参数化知识和检索 context 混着用，幻觉照旧
- **`eos_token_id` 同时包括 `<|eot_id|>`**——LLaMA-3 的 turn 结束是 `<|eot_id|>` 不是 `<|end_of_text|>`，漏了模型会一路吐到 max_new_tokens
- **k=5** 是 demo 起点；正式产品建议先 retrieve 50-100、再用 `bge-reranker-v2-m3` 精排到 5-10（参考 13.2 §3.3 reranker 章节）

---

## 9. Step 7：vLLM 多 LoRA 部署

把 base 一次 load、多个 LoRA adapter 动态切换——这是工业 SaaS 多租户的标配。vLLM 0.5+ 原生支持。

```bash
# serve.sh
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 64 \
    --lora-modules med-expert=./out/llama3-med-lora \
                   safety-shield=./out/llama3-safety-lora \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --port 8000
```

调用（OpenAI-compatible）：

```python
# call_vllm.py
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="med-expert",                 # 选哪个 LoRA = 选 model 名
    messages=[{"role": "user", "content": "What is metformin's MoA?"}],
    max_tokens=256, temperature=0.0,
)
print(resp.choices[0].message.content)
```

注意几个关键 flag：

- **`--enable-lora`**：必须开，否则 vLLM 完全不加载 adapter
- **`--max-lora-rank 64`**：必须 ≥ 训练时的 r，**所有同 batch 的 LoRA 必须 rank 一致**否则不能合并 GEMM
- **`--max-loras 4`**：同时常驻 GPU 的 adapter 数；超过会做 LRU 换出（要重新 load 几百 MB，有 latency spike）
- **`--gpu-memory-utilization 0.85`**：vLLM 默认 0.9，4090 上偏激进易 OOM，0.85 更稳

base 显存（LLaMA-3-8B bf16）~16 GB + 4 个 LoRA adapter ~320 MB + KV cache buffer ~5 GB ≈ 21 GB，刚好 fit 24 GB 4090。

**为什么不直接 serve `merged_model`？** 两个理由：(1) 多租户场景一份 base + N 个 adapter 比 N 份 merged model 省 N×16 GB 显存；(2) 业务需要快速热切 adapter（AB test、rollback）时 merged model 要 N×15 GB 上下传。但**单租户单模型** demo 直接 serve merged model 更简单：

```bash
vllm serve ./out/llama3-med-merged --dtype bfloat16 --port 8000
```

---

## 10. Step 8：端到端 eval

两层评测：retrieval 端 + 生成端。

### 10.1 SFT 模型独立 eval：MedQA accuracy

```python
# eval_medqa.py（≤ 40 行）
import json, re
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
test = load_dataset("bigbio/med_qa", "med_qa_en_source", split="test[:500]")

def ask(model, q, options):
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    msgs = [{"role": "user",
             "content": f"{q}\n\n{opts}\n\nReply with only the letter."}]
    r = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=8, temperature=0.0)
    return r.choices[0].message.content.strip()

def acc(model):
    n_correct = 0
    for ex in test:
        pred = ask(model, ex["question"], ex["options"])
        m = re.match(r"\s*([A-E])", pred)
        if m and m.group(1) == ex["answer_idx"]:
            n_correct += 1
    return n_correct / len(test)

print("base   acc:", acc("meta-llama/Meta-Llama-3-8B-Instruct"))
print("med-FT acc:", acc("med-expert"))
# 期望：base ~50%，SFT 后 ~60-65%（+10-15 pp）
```

### 10.2 RAG 端到端：RAGAS faithfulness

```python
# eval_rag.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from rag_chat import chat   # 复用 §8 的 chat 函数

questions = ["What are the side effects of metformin?",
             "How does CRISPR-Cas9 work?",
             # ... 50 条人工或 GPT 生成的医学 query
            ]
records = []
for q in questions:
    ans, refs = chat(q, k=5)
    records.append({"question": q, "answer": ans,
                    "contexts": [r["text"] for r in refs]})

result = evaluate(Dataset.from_list(records),
                  metrics=[faithfulness, answer_relevancy, context_precision])
print(result)   # 三个指标都在 0-1，> 0.8 算可上线水平
```

**期望成果**：

- **MedQA accuracy**：base ~50% → SFT 后 ~60-65%（绝对 +10-15 pp，相对 +20-30%），与公开 medAlpaca / Meditron 同量级
- **RAG faithfulness**：纯 SFT 模型 ~0.55（幻觉重）→ 加 RAG 后 ~0.85（"only use references" prompt 起效）
- **latency**：纯 SFT 模型 vLLM serve 约 200 ms / token、TTFT 250 ms；加 RAG 多约 300-500 ms（embedding query 30 ms + FAISS 检索 5 ms + context 多塞 4-5k token 让 prefill 多 200-400 ms）

---

## 11. 工程踩坑与经验（必背 12 条）

- ❗ **bitsandbytes 安装地狱**：CUDA 12+ + Linux + Python 3.10/3.11 是最稳组合。Windows 原生 bitsandbytes 长期残废，请走 WSL2；Mac MPS / AMD ROCm 完全不支持 NF4，QLoRA 直接放弃，只能用普通 LoRA + bf16 base
- ❗ **Llama-3 chat template 必须用 `tokenizer.apply_chat_template`**——不要手拼 `<|start_header_id|>`！Llama-3 / 3.1 / 3.2 之间 template 微调过（`<|python_tag|>` 等新 token），手拼版本一旦与 tokenizer 不一致就是潜伏 bug，模型推理输出"重复用户问题"或"不会停"
- ❗ **LoRA `target_modules` 必须含全 7 个 linear**：`q/k/v/o + gate/up/down`。LoRA 原 paper 只加 `q,v`，但近 2 年实证全加才接近 full FT。在 r=64 这个量级下，全加 vs 只加 q,v 的 MedQA 差距可达 5-8 个百分点
- ❗ **QLoRA 必须用 bf16，fp16 会 NaN**：`bnb_4bit_compute_dtype=torch.bfloat16`，dequant 后矩阵乘的中间量动态范围大，fp16 上下溢概率高。如果 GPU 不支持 bf16（V100 / GTX 系列），不要用 QLoRA
- ❗ **`merge_and_unload` 必须在 bf16 base 上做、不能在量化 base 上做**：QLoRA 训完保存的是 LoRA adapter（与 base 解耦），merge 时需要重新 load 一份 bf16 base 到 CPU/GPU，然后调用 merge——直接对 4-bit base 调 merge 会报"Linear4bit doesn't support merge"
- ❗ **merge 完保存路径要包含 tokenizer**：`tok.save_pretrained(MERGED)` 千万别忘——只保存了 model weights、没保 tokenizer，下次 load 会 fall back 到默认 tokenizer，chat template 完全错位
- ❗ **vLLM 加载 LoRA 必须开 `--enable-lora` + `--max-lora-rank ≥ 训练 r`**：默认这两个 flag 都关着，直接传 `--lora-modules` 会被 silently 忽略；max_lora_rank 设小了会在 batch 内不同 LoRA 时 fall back 到串行执行，吞吐暴跌
- ❗ **vLLM batch 内不同 LoRA 必须同 rank**：r=16 和 r=64 的 adapter 不能在同 batch 调用——cuBLAS GEMM 不能合并。如果业务有混合 r 的 adapter，要么统一 r、要么按 r 分 batch 调度
- ❗ **RAG embedding model 与 LLM tokenizer 不一致没事**：bge-m3 用 XLM-Roberta tokenizer，LLaMA-3 用自己的 BPE，两者完全独立工作——embedding 只负责把文本映射到向量空间，LLM 看到的是 retrieve 后的纯文本 chunk，不需要 tokenizer 对齐
- ❗ **FAISS HNSW 默认 L2 距离与 normalize embedding 不兼容**：bge-m3 的 `normalize_embeddings=True` 输出的是单位向量，应该用 `METRIC_INNER_PRODUCT`（等价 cosine）；用默认 L2 会让排序变成"距离最近"而不是"相似度最高"，结果颠倒。索引大时（> 1M 向量）再叠 IVF 或换 Milvus，sub-millisecond 级检索没问题
- ❗ **chunk size 500-1000 token 是经验甜点**：太小（< 200）→ 单 chunk 信息不全；太大（> 1500）→ 与 query 相似度被稀释。医学 abstract 这种结构化文本可往 1000 走；FAQ / 对话 log 往 300-500 走。**overlap 设 chunk size 的 10-15%**，再多就是浪费
- ❗ **retrieve top-k 是 3-5 vs 10-20 的 trade-off**：多了上下文长 → prefill 慢 + lost in the middle；少了关键信息漏。Demo 用 k=5；上线版本用 retrieve top-100 + reranker top-5（参考 13.2 §3.3）
- ❗ **prompt 必须显式写 "ONLY use references"**——不写的话，SFT 后的医学专家模型会优先用参数化知识（训练时背下来的），把检索 context 当装饰，幻觉照旧。这是 RAG 项目第一个"看着接好了但 faithfulness 低"的 bug

---

## 12. 进阶方向

把这套 capstone 打通后，自然延伸到 4 条进阶路径——也是面试时被问"你这个项目还能怎么继续做"的标准回答：

- **加 reranker**：在 §8 的 `retrieve` 后插 `bge-reranker-v2-m3`，把 retrieve top-100 精排到 top-5——MTEB 上 reranker 平均能涨 5-10 个点 NDCG@10，是 ROI 最高的一招
- **升级到 DPO**：SFT 完之后用 `trl.DPOTrainer` 跑一轮偏好优化（参考 9.4），可以把"诚实 / 不幻觉 / 短输出"等难以用 SFT 表达的偏好教进模型；只动 LoRA、显存预算几乎不变
- **进阶 RAG**：用 HyDE（先让 LLM 生成一个假答案、再用它做 retrieve）或 GraphRAG（构建知识图谱、按实体跳转），解决 multi-hop 问题（参考 13.3）
- **Multi-LoRA serving 上规模**：把 vLLM 换成 [LoRAX](https://github.com/predibase/lorax)（专做 multi-LoRA 的 fork）或 NVIDIA NIM，可以做到几十个 adapter 同 GPU 共享 base、按 traffic 自动 LRU 换入换出

---

## 13. 经典 paper

- **Hu et al., 2021 — *LoRA: Low-Rank Adaptation of Large Language Models*** — LoRA 原典，Step 3 训练用到的 $\Delta W = BA$ 假设、$\alpha/r$ scaling、target_modules 选择都直接来自这篇 §4。本 capstone 的 r=64 / α=128 配置正是 §6 推荐的"接近 full FT 的甜点"
- **Dettmers et al., 2023 — *QLoRA: Efficient Finetuning of Quantized LLMs*** — Step 3 的核心配置（NF4 + double quant + paged optimizer）一一对应这篇 §3，§5 实测 4-bit QLoRA 在 24 GB 单卡上 SFT 7B-65B 模型与 full FT 几乎无差距——本 capstone 能在 4090 跑通的根本原因
- **Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*** — RAG 概念原典，Step 5-6 的 retrieve→prompt augmentation→generate 这条管线就是这篇定义的；现代工业 RAG 多数不再 joint train（retriever 与 LLM 解耦），但心智模型完全沿用
- **Xiao et al., 2024 — *BGE M3-Embedding*** — Step 5 用的 embedding model 的原 paper，§3 解释为什么一个模型能同时输出 dense + sparse + multi-vector 三种表示——做中英 / 多语 RAG 几乎绕不开
- 选读：**Touvron et al., 2023 — *Llama 2***、**Meta, 2024 — *Llama 3***：理解 Llama-3 的 chat template 设计与 SFT 超参（lr=2e-5、epoch=2、cosine warmup）的来源
- 工程必读：**HuggingFace TRL / PEFT 文档**、**vLLM Multi-LoRA Serving 文档**、**RAGAS 文档**——本 capstone 所有代码的实现 reference

---

## 14. 自测与面试题

**Q1（pipeline）**：写出 24 GB GPU 上 SFT + LoRA + RAG 的完整 8-step pipeline，每步说明输入 / 输出 / 显存。

<details>
<summary>Answer sketch</summary>

8 步骨架（每点都要点到，缺步面试扣分）：

1. **环境准备**：CUDA 12 + Linux + Python 3.11 + transformers/peft/trl/bitsandbytes/sentence-transformers/faiss/vllm；GPU compute_cap ≥ 8.0
2. **数据准备**：raw dataset → 转成 `messages` 格式 → save_to_disk；同时为 RAG 单独准备 corpus（与 SFT 数据不重叠的长尾）
3. **QLoRA SFT 训练**：base 用 `BitsAndBytesConfig(load_in_4bit, nf4, double_quant, bf16)`、`prepare_model_for_kbit_training`、LoRA r=64/α=128/全 7 个 linear、`SFTTrainer + packing + completion_only_loss + paged_adamw_8bit + grad ckpt`，输出 ~80 MB adapter；显存 ~18 GB
4. **merge LoRA**：CPU 上重 load bf16 base + `PeftModel.from_pretrained` + `merge_and_unload` + `save_pretrained`，输出 ~15 GB merged model；CPU 内存 ~32 GB
5. **构建 RAG 知识库**：document → `RecursiveCharacterTextSplitter` (chunk 800/overlap 100) → `bge-m3.encode(normalize=True)` → `faiss.IndexHNSWFlat(dim, M=32, INNER_PRODUCT)`，输出 index + meta
6. **RAG + LLM 集成**：`retrieve(q, k=5)` → 把 top-k chunk 拼成 grounded prompt（**必须显式 ONLY use references**）→ LLM generate（`eos=[eos, <|eot_id|>]`），返回答案 + 引用
7. **vLLM 部署**：`vllm serve base --enable-lora --max-loras 4 --max-lora-rank 64 --lora-modules name=path`，提供 OpenAI-compatible API；base + 4 个 LoRA + KV cache ≈ 21 GB
8. **端到端 eval**：MedQA accuracy（vs base 涨 10-15 pp）+ RAGAS faithfulness/answer_relevance/context_precision（> 0.8 可上线）

加分点：能给出每步显存账本、能解释 step 4 为什么必须 CPU 做 merge、能区分 step 7 的 multi-LoRA serving 与 step 4 的 merge 的使用场景

</details>

**Q2（trade-off）**：QLoRA r=64 vs r=128 vs full SFT 三种方案的 trade-off？给定 24 GB / 80 GB / 8×80 GB 三种预算，分别选哪个？

<details>
<summary>Answer sketch</summary>

三个维度对比：

| 维度 | QLoRA r=64 | QLoRA r=128 | Full SFT |
|---|---|---|---|
| **显存（8B）** | ~18 GB | ~20 GB（多 ~85M LoRA） | ~120 GB（base + grad + AdamW state） |
| **trainable params** | 0.7%（85M） | 1.4%（170M） | 100%（8B） |
| **效果（MedQA-like）** | full FT 的 95-98% | full FT 的 97-99% | 100%（baseline） |
| **训练速度** | 慢 ~30%（dequant 开销） | 慢 ~35% | 最快（无 dequant） |
| **保存大小** | ~170 MB | ~340 MB | ~16 GB |
| **multi-task 灵活度** | 高（多 adapter 共享 base） | 高 | 低（每任务独立 model）|

预算选择：

- **24 GB（4090）**：只能 QLoRA r=64——r=128 边际收益小、显存又紧；想再涨点可以叠 DoRA（`use_dora=True`）涨 1-3 pp
- **80 GB（H100）**：LoRA r=64/128 + bf16 base（不量化）——速度更快、效果更稳；如果是垂域要塞大量知识，可以试 QLoRA r=256 或 full SFT
- **8×80 GB**：full SFT 走起——FSDP/DeepSpeed ZeRO-3、bf16 + grad checkpoint，~3-4 小时训完 8B；效果上限最高，但要权衡 multi-task serving 的便利性

加分：能指出"r 不是越大越好"——超过 128 后边际收益急剧下降，且参数量已经接近 full FT 一部分，性价比降低；能说出 LoRA + QLoRA 不是竞争而是补全显存预算

</details>

**Q3（实战）**：你做客服 agent，base 用 Qwen-7B-Chat，要让它"会回答公司产品手册问题"。三选一：(a) 把手册塞 SFT 数据 fine-tune、(b) 用 long-context（128k）每次塞整本手册、(c) 用 RAG。说理由。

<details>
<summary>Answer sketch</summary>

正确答案：**(c) RAG**，但要解释为什么 (a) (b) 不合适：

**(a) SFT 注入手册知识为什么不行**：

- **可更新性差**：手册一周一更新，每次都要重训 + 重部署；模型权重一旦冻住就过时
- **容量不足**：7B 模型参数化知识容量有限，长尾事实（具体型号、specs、价格）记不住；测试时随便问个边角细节就幻觉
- **可溯源差**：SFT 后模型输出无法明确说"这条信息来自手册第 X 节"，合规性 0
- **训练成本高**：每次手册改动都重 SFT，时间和显存预算都吃不消
- **唯一适用场景**：教模型"客服话术风格"——固定话术、礼貌用语、call to action 这类"模式"——这部分应该 SFT，但**事实知识不应该**

**(b) Long-context 塞整本手册为什么不行**：

- **成本爆炸**：Qwen 128k context 每次 query prefill 都要算 128k token，几美元/query，QPS 一上去成本不可承受
- **延迟不可接受**：128k prefill 即使有 prefix cache 也要数秒级 TTFT，客服场景用户等不起
- **lost in the middle**：手册中间段的关键信息被 LLM 忽略，召回率反而低于 RAG 的精准 retrieve
- **唯一适用场景**：手册总长 < 32k token、单 user 长会话需要持续 grounding 时，long-context 直接装入更省事

**(c) RAG 为什么是正解**：

- **可更新性**：手册新版上线 → 重新 chunk + embed + upsert vector DB（分钟级），模型不动
- **可扩展**：手册体量大到 GB 级也 OK，向量 DB 水平扩展
- **可溯源**：每个答案带 [doc_id]，前端反查直接给手册原文链接
- **延迟 / 成本可控**：retrieve 几十 ms + 短 prefill（5-10k context）几百 ms，单 query 成本几分钱
- **配合 SFT 起协同效应**：SFT 教"客服风格 + 引用格式 + 拒答策略"（写到权重里反复用），RAG 注入"具体事实"（每次现查）——这才是 production 做法

**最终方案**：SFT 的对话风格（用 Qwen-7B-Chat 在 ~3000 条标注对话上 LoRA SFT）+ RAG 注入手册知识（bge-m3 embed + Milvus + bge-reranker），prompt 模板必须写 "only use references"。

加分：能指出 (a) + (c) 是 production 标配、不是单选；能区分"知识"（事实，应 RAG）vs"模式"（风格 / 格式 / 工具调用，应 SFT）；能给出大致预算分配（embedding GPU 1 卡 + LLM GPU 2 卡 + Milvus 集群 3 副本）

</details>

---

## 15. 延伸阅读

- [HuggingFace TRL — SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer) — Step 3 的所有 SFTConfig 字段官方说明
- [HuggingFace PEFT — LoRA 文档](https://huggingface.co/docs/peft/main/en/developer_guides/lora) — Step 3 的 LoraConfig 字段、merge_and_unload 行为
- [QLoRA 官方 repo (artidoro/qlora)](https://github.com/artidoro/qlora) — Dettmers 团队的端到端训练脚本，本 capstone Step 3 的灵感来源
- [bitsandbytes 文档](https://huggingface.co/docs/bitsandbytes/main/en/index) — NF4 / 8-bit Adam / paged optimizer 工程细节与 GPU 兼容性表
- [vLLM Multi-LoRA Serving 文档](https://docs.vllm.ai/en/latest/features/lora.html) — Step 7 的 `--enable-lora` / `--lora-modules` 完整参数表
- [BAAI bge-m3 模型卡](https://huggingface.co/BAAI/bge-m3) — Step 5 用的 embedding model，含使用示例
- [Unsloth (unslothai/unsloth)](https://github.com/unslothai/unsloth) — 把 QLoRA 训练速度做到 HF 的 2-5 倍、显存再省 30%；本 capstone 替换 Step 3 的训练后端可显著加速
- [RAGAS 文档](https://docs.ragas.io/) — Step 8 端到端评测的官方框架
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 在 chunk 前 prepend context summary 把 retrieval failure 降 49%，本 capstone Step 5 升级路径
- 推荐继续读本教程的 **附录 C：Capstone 3 — Agent end-to-end** —— 把本 capstone 的"chat + 检索"升级成"reasoning + tool use + memory"的完整 agent 系统
- 推荐继续读本教程的 **9.4 节《DPO》** —— SFT 之后下一阶段偏好优化，把"幻觉率"压得更低
