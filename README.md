# End-to-End Model Hosting: From Fine-Tuning to Production

> A complete learning project covering every layer of modern ML infrastructure:
> fine-tuning → quantization → inference servers → Docker → Kubernetes → monitoring.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Step 1 — Data Download](#3-step-1--data-download)
4. [Step 2 — Fine-Tuning with LoRA and MoRA](#4-step-2--fine-tuning-with-lora-and-mora)
   - [PyTorch Dataset and DataLoader](#41-pytorch-dataset-and-dataloader)
   - [nn.Module Model Wrapper](#42-nnmodule-model-wrapper)
   - [LoRA — Low-Rank Adaptation](#43-lora--low-rank-adaptation)
   - [MoRA — High-Rank Updating](#44-mora--high-rank-updating)
   - [Training Loop](#45-training-loop)
   - [Evaluation Metrics — ROUGE](#46-evaluation-metrics--rouge)
5. [Step 3 — Quantization](#5-step-3--quantization)
   - [Why Quantize?](#51-why-quantize)
   - [BitsAndBytes (INT8 and NF4)](#52-bitsandbytes-int8-and-nf4)
   - [GGUF (llama.cpp)](#53-gguf-llamacpp)
   - [AWQ — Activation-Aware Weight Quantization](#54-awq--activation-aware-weight-quantization)
   - [ONNX Export](#55-onnx-export)
   - [Quantization Comparison](#56-quantization-comparison)
6. [Step 4 — vLLM and SGLang](#6-step-4--vllm-and-sglang)
   - [Why Naive HuggingFace Inference Is Slow](#61-why-naive-huggingface-inference-is-slow)
   - [vLLM — PagedAttention and Continuous Batching](#62-vllm--pagedattention-and-continuous-batching)
   - [SGLang — Structured Generation Language](#63-sglang--structured-generation-language)
7. [Step 5 — FastAPI and Docker](#7-step-5--fastapi-and-docker)
   - [FastAPI Endpoint Design](#71-fastapi-endpoint-design)
   - [Prometheus Metrics in the API](#72-prometheus-metrics-in-the-api)
   - [Dockerfile — Multi-Stage Build](#73-dockerfile--multi-stage-build)
   - [Docker Compose — Full Local Stack](#74-docker-compose--full-local-stack)
   - [NGINX — Reverse Proxy and Load Balancer](#75-nginx--reverse-proxy-and-load-balancer)
8. [Step 6 — Kubernetes](#8-step-6--kubernetes)
   - [Core Concepts and Architecture](#81-core-concepts-and-architecture)
   - [Namespace](#82-namespace)
   - [ConfigMap and Secret](#83-configmap-and-secret)
   - [Deployment, ReplicaSet, and Pod](#84-deployment-replicaset-and-pod)
   - [Service — Virtual IP and Load Balancing](#85-service--virtual-ip-and-load-balancing)
   - [Ingress and Ingress Controller](#86-ingress-and-ingress-controller)
   - [HorizontalPodAutoscaler](#87-horizontalpodautoscaler)
   - [Persistent Volumes](#88-persistent-volumes)
   - [RBAC — Role-Based Access Control](#89-rbac--role-based-access-control)
9. [Step 7 — Prometheus and Grafana](#9-step-7--prometheus-and-grafana)
   - [How Prometheus Works](#91-how-prometheus-works)
   - [Metric Types Explained](#92-metric-types-explained)
   - [PromQL — Computing p90 and p95](#93-promql--computing-p90-and-p95)
   - [Grafana Dashboards](#94-grafana-dashboards)
10. [High-Level Architecture Diagram](#10-high-level-architecture-diagram)
11. [Inference Request Flow — End to End](#11-inference-request-flow--end-to-end)
12. [Running the Project](#12-running-the-project)

---

## 1. Project Overview

This project is a **learning-first** implementation of the full stack required
to take a language model from raw weights to a monitored, auto-scaling
production service. Every layer is explained from first principles.

**Model**: `google/roberta2roberta_L-24_cnn_daily_mail`
- Architecture: encoder-decoder (seq2seq), two tied RoBERTa-Large (24-layer) checkpoints
- Task: abstractive news summarisation
- Dataset: CNN / DailyMail (abisee/cnn_dailymail on HuggingFace)

**What you will learn**:

| Layer | Technology | Concept |
|-------|-----------|---------|
| Data | HuggingFace Datasets | Dataset versioning, splits |
| Training | PyTorch nn.Module | Autograd, gradient flow |
| PEFT | LoRA / MoRA | Adapter-based fine-tuning |
| Evaluation | ROUGE | Summarisation quality metrics |
| Quantization | BNB, GGUF, AWQ, ONNX | Model compression trade-offs |
| Serving | vLLM, SGLang | PagedAttention, continuous batching |
| API | FastAPI + uvicorn | Async ASGI web server |
| Containerisation | Docker multi-stage | Image layers, security |
| Reverse proxy | NGINX | Load balancing algorithms |
| Orchestration | Kubernetes | Scheduling, self-healing |
| Auto-scaling | HPA | Metric-driven scaling |
| Monitoring | Prometheus + Grafana | Pull-based metrics, PromQL |

---

## 2. Repository Structure

```
model_hosting_e2e_decoder/
│
├── data/
│   ├── download_data.py            ← Step 1: download CSV chunk from HuggingFace
│   ├── cnn_dailymail_train.csv     ← 5,000 articles + highlights
│   ├── cnn_dailymail_validation.csv← 500 rows
│   └── cnn_dailymail_test.csv      ← 500 rows
│
├── notebooks/
│   ├── 01_training_lora_mora.ipynb ← Step 2: full training pipeline
│   ├── 02_quantization.ipynb       ← Step 3: BNB, GGUF, AWQ, ONNX
│   └── 03_vllm_sglang_inference.ipynb ← Step 4: high-throughput inference
│
├── api/
│   ├── main.py                     ← Step 5: FastAPI app + Prometheus metrics
│   └── requirements.txt
│
├── docker/
│   ├── Dockerfile                  ← Multi-stage container build
│   ├── docker-compose.yml          ← Local stack: API + NGINX + Prometheus + Grafana
│   └── nginx.conf                  ← Reverse proxy configuration
│
├── k8s/
│   └── manifests/
│       ├── 01-namespace.yaml       ← Step 6: resource isolation
│       ├── 02-configmap.yaml       ← non-secret config injection
│       ├── 03-deployment.yaml      ← Pod spec, replicas, probes
│       ├── 04-service.yaml         ← virtual IP, load balancing
│       ├── 05-ingress.yaml         ← HTTP routing rules
│       ├── 06-hpa.yaml             ← auto-scaling
│       └── 07-monitoring.yaml      ← Prometheus + Grafana in k8s
│
├── monitoring/
│   ├── prometheus.yml              ← scrape targets config
│   └── grafana/
│       ├── datasources.yml         ← auto-wire Grafana → Prometheus
│       └── dashboards/
│           └── summarisation_dashboard.json ← pre-built p95/p90 dashboard
│
├── checkpoints/                    ← .pth checkpoint files (git-ignored)
├── models/                         ← merged HF model + quantized variants
├── requirements.txt                ← full dependency list
└── RUNBOOK.md                      ← copy-paste commands for every step
```

---

## 3. Step 1 — Data Download

**File**: [data/download_data.py](data/download_data.py)

### What is CNN / DailyMail?

The CNN/DailyMail dataset is the standard benchmark for abstractive
summarisation. Each sample has:
- `article` — full news article text (avg ~3,600 characters)
- `highlights` — bullet-point summary written by journalists (avg ~260 chars)
- `id` — unique hash identifier

We download only **5,000 training samples** (out of 287,000) to keep
training feasible on a local machine or free Colab.

### How HuggingFace Datasets works

```
load_dataset("abisee/cnn_dailymail", "3.0.0")
         │
         ▼
   HuggingFace Hub
   (dataset card + parquet shards cached in ~/.cache/huggingface/datasets/)
         │
         ▼
   DatasetDict {
     train:      Dataset(287,113 rows)
     validation: Dataset(13,368 rows)
     test:       Dataset(11,490 rows)
   }
```

The `.select(range(5000))` call slices the first 5,000 rows without
loading the rest into memory — it uses Arrow memory-mapped files
(zero-copy reads from disk).

### Running it

```bash
python data/download_data.py
```

Output:
```
✓ Saved 5,000 rows → data/cnn_dailymail_train.csv
✓ Saved 500 rows   → data/cnn_dailymail_validation.csv
✓ Saved 500 rows   → data/cnn_dailymail_test.csv
article length  (mean chars): 3664
highlight length(mean chars): 260
```

---

## 4. Step 2 — Fine-Tuning with LoRA and MoRA

**File**: [notebooks/01_training_lora_mora.ipynb](notebooks/01_training_lora_mora.ipynb)

### 4.1 PyTorch Dataset and DataLoader

#### `torch.utils.data.Dataset`

A Dataset is a Python class that knows how to load and transform one sample.
You subclass it and implement two methods:

```python
class CNNDailyMailDataset(Dataset):
    def __len__(self):
        return len(self.df)          # total number of samples

    def __getitem__(self, idx):
        # Load sample at index idx and return a dict of tensors
        ...
```

The `__getitem__` method does **tokenisation** — converting raw text into
integer IDs that the model understands:

```
"Scientists found water on Mars."
         │
    tokenizer(...)
         │
         ▼
{
  input_ids:      [0, 34389, 303, 514, 15, 14364, 4, 2]  ← token IDs
  attention_mask: [1, 1,     1,   1,   1,  1,     1, 1]  ← 1=real, 0=padding
}
```

**Why label masking with -100?**

The target sequence (highlights) is padded to `max_length`. Padding
positions are meaningless — we don't want the loss function to penalise
the model for not predicting `[PAD]`. By setting those positions to `-100`,
PyTorch's `CrossEntropyLoss` automatically ignores them:

```python
labels[labels == tokenizer.pad_token_id] = -100
```

#### `torch.utils.data.DataLoader`

DataLoader wraps a Dataset and provides:
- **Batching**: groups `batch_size` samples, stacks tensors into `(B, seq_len)`
- **Shuffling**: random order each epoch (prevents memorising order)
- **Multi-process loading**: `num_workers` worker processes pre-load batches in parallel
- **Pin memory**: `pin_memory=True` on CUDA uses page-locked RAM → faster GPU transfer

```
Dataset[0]  Dataset[1]  Dataset[2]  Dataset[3]
    │           │           │           │
    └───────────┴───────────┴───────────┘
                     │
                DataLoader (batch_size=4)
                     │
                     ▼
         {input_ids: (4, 512), labels: (4, 128), ...}
```

### 4.2 nn.Module Model Wrapper

Everything trainable in PyTorch is an `nn.Module`. Our wrapper:

```
SummarisationModel (nn.Module)
└── self.backbone  (EncoderDecoderModel)
    ├── encoder   (RobertaModel, 24 transformer layers)
    │   └── embeddings + 24 × RobertaLayer
    │       └── attention (self-attention)
    │           ├── query  (Linear: 1024→1024)  ← LoRA/MoRA injected here
    │           ├── key    (Linear: 1024→1024)
    │           └── value  (Linear: 1024→1024)  ← LoRA/MoRA injected here
    └── decoder   (RobertaForCausalLM, 24 transformer layers)
        └── 24 × RobertaLayer
            ├── self-attention     ← LoRA/MoRA here
            └── cross-attention    ← LoRA/MoRA here
```

The `forward()` method wires inputs → backbone → loss:

```
input_ids (article tokens)  ─────────────────────► Encoder
                                                       │ context vectors
decoder_input_ids (shifted highlights) ──────────► Decoder ◄─── cross-attention
                                                       │
                                                   logits (vocab distribution)
                                                       │
                                           labels ─► CrossEntropyLoss
                                                       │
                                                     loss (scalar)
```

**Teacher forcing**: during training, the decoder is fed the ground-truth
previous tokens (not its own predictions). This stabilises training but
creates a train/inference gap — at inference time the decoder must use
its own (imperfect) previous outputs.

### 4.3 LoRA — Low-Rank Adaptation

**Paper**: *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)

#### The core problem

A pre-trained weight matrix `W ∈ ℝ^(d×k)` (e.g. d=k=1024 for query projection)
has **1,048,576 parameters**. Fine-tuning all of them:
- Is expensive (full gradient computation)
- Often causes catastrophic forgetting
- Requires storing a full copy per task

#### LoRA's solution

Freeze `W` (original weights, never updated). Learn two small matrices
`A ∈ ℝ^(d×r)` and `B ∈ ℝ^(r×k)` where rank `r ≪ min(d,k)`:

```
Forward pass:
  output = x · W^T  +  (x · A^T) · B^T · (α/r)
            ───────     ────────────────────────
            frozen         LoRA adapter (learned)
```

With `r=16`, `d=k=1024`, the adapter has `2 × 1024 × 16 = 32,768` parameters
instead of `1,048,576` — a **32× reduction**.

**Why does rank-r work?**

Research (Aghajanyan et al., 2020) shows that during fine-tuning, weight
updates have a very **low intrinsic rank** — the gradient matrix ΔW naturally
lives on a low-dimensional subspace. LoRA directly parameterises this subspace.

**Hyperparameters**:

| Parameter | What it controls | Typical value |
|-----------|-----------------|---------------|
| `r` | rank — higher = more capacity | 4, 8, 16, 32, 64 |
| `lora_alpha` | scaling: output scaled by α/r | 2×r (so scale=2) |
| `lora_dropout` | regularisation on adapter | 0.05–0.1 |
| `target_modules` | which weight matrices to adapt | query, value |
| `bias` | whether to adapt bias terms | "none" |

**What happens at inference?**

You can either:
1. Keep adapters separate and add them at runtime (enables swapping adapters)
2. **Merge**: `W_merged = W + B·A·(α/r)` — fold back into original weights,
   zero overhead at inference

```python
merged_model = peft_model.merge_and_unload()  # standard HF model, no PEFT overhead
```

### 4.4 MoRA — High-Rank Updating

**Paper**: *MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning* (Jiang et al., 2024)

#### The limitation of LoRA

LoRA uses rank-`r` updates, which means `ΔW = BA` has at most `r` linearly
independent rows. For tasks requiring **memorisation** (e.g. learning new
facts, instructions with many details), a low-rank update is insufficient
— the model cannot represent the required full-rank weight change.

#### MoRA's solution

With the **same parameter budget** as LoRA (same `r²` parameters), use a
**square matrix** `M ∈ ℝ^(r×r)` instead of two rectangular matrices:

```
LoRA:  A ∈ ℝ^(d×r)   +   B ∈ ℝ^(r×k)   =  d·r + r·k  parameters
                                              (rank-r update)

MoRA:            M ∈ ℝ^(r×r)             =  r²  parameters
       (same budget when r² ≈ d·r + r·k)     (potentially full-rank update)
```

To go from `r×r` back to the full `d×k` shape, MoRA uses fixed (non-learned)
**compression and decompression** functions:

```
x ∈ ℝ^(in)
   │
   ▼  compress via fixed random projection A ∈ ℝ^(r × in)
z ∈ ℝ^r
   │
   ▼  learned square core M ∈ ℝ^(r × r)   ← THE only learned matrix
z' ∈ ℝ^r
   │
   ▼  decompress via fixed random projection B ∈ ℝ^(out × r)
output ∈ ℝ^(out)
   │
   + (added to frozen original output)
```

Because `M` is square and unconstrained, a single `M` matrix can represent
any mapping in ℝ^r — the update is **full rank within the compressed space**.

**When to prefer MoRA over LoRA**:
- Knowledge injection tasks (learning new facts)
- Instruction following with many detailed rules
- Domain adaptation requiring broad weight changes

**When LoRA is fine**:
- Style transfer, light adaptation
- Tasks where the pre-trained distribution is close to the target

### 4.5 Training Loop

The standard PyTorch training loop:

```
for epoch in range(N_EPOCHS):
    model.train()                    ← enables dropout, BatchNorm update

    for batch in train_loader:
        batch → GPU                  ← move tensors to device

        optimizer.zero_grad()        ← clear gradients from previous step

        output = model(**batch)      ← forward pass
        loss   = output.loss         ← cross-entropy over non-(-100) positions

        loss.backward()              ← backward pass: compute ∂loss/∂params
                                       only adapter params have requires_grad=True
                                       so only they get gradients

        clip_grad_norm_(params, 1.0) ← prevent exploding gradients

        optimizer.step()             ← update: param -= lr × gradient
        scheduler.step()             ← adjust learning rate

    evaluate(val_loader)             ← compute val loss + ROUGE
    checkpoint if val_loss improved  ← torch.save({model_state, optim_state})
```

**Gradient flow with adapters**:

```
loss.backward() computes:
  ∂loss/∂B   (LoRA matrix B — updated)
  ∂loss/∂A   (LoRA matrix A — updated)
  ∂loss/∂W   (base weight  — computed but NOT applied, requires_grad=False)
```

**Learning rate schedule — linear warmup + decay**:

```
lr
│     /‾‾‾‾‾‾‾‾‾‾‾\
│    /             \
│   /               \
│  /                 \
│ /                   \────────────
│/                              step
│← warmup →│←── linear decay ──────►
```

Warmup prevents large early updates that can derail training.
Decay ensures fine-grained convergence at the end.

### 4.6 Evaluation Metrics — ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
text overlap between generated and reference summaries.

**ROUGE-1** — unigram (single word) overlap:
```
Reference: "the cat sat on the mat"
Generated: "a cat sat"

Precision = matching unigrams / generated length = 3/3 = 1.00
Recall    = matching unigrams / reference length = 3/6 = 0.50
F1        = 2 × P × R / (P + R)             = 0.67
```

**ROUGE-2** — bigram (two consecutive words) overlap:
```
Reference bigrams: {the cat, cat sat, sat on, on the, the mat}
Generated bigrams: {a cat, cat sat}
Overlap: {cat sat} → Recall = 1/5 = 0.20
```
ROUGE-2 better captures local fluency and phrase-level accuracy.

**ROUGE-L** — Longest Common Subsequence:
```
Reference: "the cat sat on the mat"
Generated: "a cat sat on a rug"
LCS:        "cat sat on"  (length 3)
Recall    = 3/6 = 0.50
```
ROUGE-L captures global sentence structure, even with word insertions.

**Typical CNN/DailyMail scores** (higher is better):
- ROUGE-1: ~43 (SOTA), ~30–35 (fine-tuned small model)
- ROUGE-2: ~21 (SOTA), ~12–15 (fine-tuned small model)
- ROUGE-L: ~40 (SOTA), ~28–32 (fine-tuned small model)

---

## 5. Step 3 — Quantization

**File**: [notebooks/02_quantization.ipynb](notebooks/02_quantization.ipynb)

### 5.1 Why Quantize?

Our model uses FP32 (4 bytes per parameter). With ~700M parameters:

```
Storage: 700M × 4 bytes = 2.8 GB
```

Quantization reduces precision:

| Format | Bytes/param | 700M params | Accuracy loss |
|--------|------------|-------------|---------------|
| FP32   | 4.0        | 2.8 GB      | 0 (baseline)  |
| FP16   | 2.0        | 1.4 GB      | Minimal       |
| INT8   | 1.0        | 0.7 GB      | Very small    |
| NF4    | 0.5        | 0.35 GB     | Small         |
| INT4   | 0.5        | 0.35 GB     | Small-medium  |
| INT2   | 0.25       | 0.175 GB    | Large         |

Beyond size, quantized models run faster because:
- Smaller tensors fit in GPU L2 cache
- INT8 GEMM (matrix multiply) is 2-4× faster than FP32 on modern GPUs
- Bandwidth is the main bottleneck for LLMs — smaller = more throughput

### 5.2 BitsAndBytes (INT8 and NF4)

**INT8 quantization** — absmax per-row scaling:

```
W_fp32 = [1.2,  -0.8,  0.3,  -2.1,  0.9]   (one weight row)

scale   = max(|W_fp32|) / 127  =  2.1 / 127  =  0.01654

W_int8  = round(W_fp32 / scale) = [73, -48, 18, -127, 54]

At runtime:  output = (x @ W_int8) × scale  (dequantize on the fly)
                               ↑
                      INT8 GEMM (fast!)
```

**NF4 (Normal Float 4) — QLoRA quantization**:

Observation: pre-trained weights follow a **normal distribution** N(0, σ).
Instead of uniform INT4 buckets, NF4 places bucket boundaries at the
quantiles of a standard normal — this minimises quantization error for
normally-distributed weights.

```
INT4 uniform:  [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
               ← equal spacing, wastes precision in sparse tails

NF4:           bucket edges at N(0,1) quantiles
               ← most buckets near 0 where most weights cluster
```

`bnb_4bit_use_double_quant=True` quantizes the quantization constants
themselves (from FP32 to INT8), saving an extra ~0.4 bits per parameter.

**How to use**:
```python
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModel.from_pretrained("...", quantization_config=config)
```

### 5.3 GGUF (llama.cpp)

**GGUF** (GGML Unified Format) is the file format used by **llama.cpp** —
a pure C++ inference library that runs LLMs on CPUs (and Apple Silicon via Metal).

**Why llama.cpp exists**:
PyTorch is Python-first and requires a Python runtime. llama.cpp is a single
C++ binary — it can run on anything: a Raspberry Pi, an old MacBook, a server
with no CUDA. It loads GGUF files and runs quantized inference.

**Quantization levels in GGUF**:

| Format   | Bits/weight | Quality      | Use case |
|----------|------------|--------------|----------|
| Q2_K     | 2.6        | Poor         | Extreme compression |
| Q4_0     | 4.0        | Good         | Balance of size/quality |
| Q4_K_M   | 4.5        | Better       | Most popular for local LLMs |
| Q5_K_M   | 5.7        | Very good    | When quality matters |
| Q8_0     | 8.0        | Near-lossless| Quality close to FP16 |
| F16      | 16.0       | Lossless     | Full precision |

**Conversion pipeline**:
```
1. HuggingFace model (PyTorch .bin / .safetensors)
        │
        ▼   python convert_hf_to_gguf.py model_dir/ --outtype f16
2. model.gguf  (lossless F16 GGUF)
        │
        ▼   ./llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M
3. model-Q4_K_M.gguf  (4-bit quantized, ~25% of original size)
        │
        ▼   ./llama-cli -m model-Q4_K_M.gguf -p "Summarize: ..."
4. Inference output
```

**Architecture note**: llama.cpp primarily supports **decoder-only** models
(LLaMA, Mistral, Falcon, GPT-2, etc.). For encoder-decoder models like
roberta2roberta, you would need to use ONNX or BitsAndBytes instead.

### 5.4 AWQ — Activation-Aware Weight Quantization

**Paper**: *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (Lin et al., MIT, 2023)

**Key insight**: Standard INT4 quantization treats all weight channels equally.
But some channels are **much more important** than others — specifically,
channels that correspond to **large activation values** cause disproportionate
quantization error because a small rounding error gets amplified by a large activation.

**How AWQ protects salient channels**:

```
Step 1: Run calibration data through the model to find which input
        channels produce the largest activations.

        Activation magnitudes: [0.1, 0.8, 0.2, 3.5, 0.1, 0.9]
                                              ↑
                                        salient channel (large activation)

Step 2: Before quantizing weight W, scale UP the salient columns:
        W_scaled[:,j] = W[:,j] × s[j]     where s[j] is large for salient j

Step 3: To keep output correct, scale DOWN the activations:
        x_scaled[:,j] = x[:,j] / s[j]     (absorbed into previous layer norm)

Step 4: Quantize W_scaled to INT4.
        The salient columns, now larger, have less relative rounding error.

At runtime: x_scaled @ Q(W_scaled) ≈ x @ W   (with much less error)
```

**Result**: 4-bit AWQ often matches 8-bit BitsAndBytes quality,
at half the memory cost.

### 5.5 ONNX Export

**ONNX** (Open Neural Network Exchange) is a file format that represents
a computation graph in a framework-agnostic way. The same ONNX file can
be executed by:
- `onnxruntime` (CPU/CUDA, Python/C++/Java)
- TensorRT (NVIDIA — maximum GPU performance)
- OpenVINO (Intel CPUs/NPUs)
- DirectML (Windows GPU)
- CoreML (Apple, via conversion)

**Why HuggingFace Optimum for seq2seq**:

A seq2seq model has three inference phases:
1. **Encoder**: run once per request to encode the input
2. **Decoder init**: run once, no past key-values
3. **Decoder step**: run for each output token, with growing KV cache

Each phase has a different input/output shape, so Optimum exports three
separate ONNX graphs:

```
encoder_model.onnx              ← input_ids → encoder_hidden_states
decoder_model.onnx              ← decoder_input_ids + encoder_hidden_states → logits
decoder_with_past_model.onnx    ← same + past_key_values → logits + new_past_kv
```

**Graph-level optimizations** (`--optimize O2`):
- Layer fusion: merge LayerNorm + Attention into a single op
- Constant folding: pre-compute static sub-expressions
- Redundant node elimination

**INT8 ONNX** (dynamic quantization):
```
Q(W) for all Linear layers   ← weights quantized offline
At runtime: dequantize(Q(W)) → FP32 for GEMM
```
No calibration data needed for dynamic quantization (vs static which requires it).

### 5.6 Quantization Comparison

| Method    | Size reduction | Latency improvement | ROUGE drop | GPU required |
|-----------|---------------|---------------------|------------|--------------|
| FP32      | 1×            | 1×                  | 0          | Optional     |
| BNB INT8  | 2×            | 1.2–1.5×            | ~0.5       | Yes (CUDA)   |
| BNB NF4   | 4×            | 1.5–2×              | ~1–2       | Yes (CUDA)   |
| GGUF Q4   | 4×            | 2–4× (CPU)          | ~1–2       | No (CPU)     |
| AWQ INT4  | 4×            | 2–3× (GPU)          | ~0.5–1     | Yes (CUDA)   |
| ONNX FP32 | 1×            | 1.2–1.8×            | ~0         | Optional     |
| ONNX INT8 | 2×            | 2–3×                | ~0.5       | Optional     |

---

## 6. Step 4 — vLLM and SGLang

**File**: [notebooks/03_vllm_sglang_inference.ipynb](notebooks/03_vllm_sglang_inference.ipynb)

### 6.1 Why Naive HuggingFace Inference Is Slow

When you call `model.generate()` naively, several inefficiencies occur:

**Problem 1 — Static batching**:
```
Batch of 4 requests with lengths: [10, 50, 100, 200] tokens
Static batch: process all together, pad to max length (200)
→ GPU computes attention over 200 tokens for ALL requests
→ Request 1 (10 tokens) wastes 95% of compute on padding
```

**Problem 2 — KV cache fragmentation**:
```
KV cache (key-value pairs from attention) must be pre-allocated.
If max_length=512, each request reserves 512 slots even if it only uses 20.
→ With 8GB VRAM, you can only serve ~10 concurrent requests
```

**Problem 3 — No prefix reuse**:
```
System prompt: "You are an expert summariser. Always be concise."  (50 tokens)
Request 1: [system prompt] + article 1   ← computes system prompt KV
Request 2: [system prompt] + article 2   ← recomputes system prompt KV again
→ Wasted compute proportional to shared prefix length
```

### 6.2 vLLM — PagedAttention and Continuous Batching

**PagedAttention** (Kwon et al., UC Berkeley, 2023):

Inspired by OS virtual memory. Instead of one contiguous KV cache block
per sequence, vLLM manages a **pool of fixed-size physical blocks** and
maintains a per-sequence **page table** mapping logical → physical blocks.

```
Physical KV block pool (say 100 blocks of 16 tokens each):
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│B1│B2│B3│B4│B5│B6│B7│B8│...         │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Request A (generates 35 tokens → needs 3 blocks):
  Page table: [B1, B3, B7]   ← non-contiguous, no fragmentation waste

Request B (generates 12 tokens → needs 1 block):
  Page table: [B2]

Request C (generates 55 tokens → needs 4 blocks):
  Page table: [B4, B5, B6, B8]

→ Zero internal fragmentation (last block may be partially used)
→ Copy-on-write enables prefix sharing between beams/requests
```

**Result**: 2–4× more concurrent sequences in the same VRAM vs naive allocation.

**Continuous batching**:

```
Static batching (naive):
Time ──────────────────────────────────────►
GPU:  [=====Req A=====][=====Req B=====][=====Req C=====]
       GPU sits idle while waiting for all in batch to finish

Continuous batching (vLLM):
Time ──────────────────────────────────────►
GPU:  [Req A][Req B][Req C][Req D][Req E][Req F][Req G]...
       As each sequence finishes a token, new sequences inserted immediately
       GPU utilisation ~90%+ vs ~35% static
```

At every forward pass, vLLM's **scheduler**:
1. Selects which sequences to run (based on priority, memory)
2. Assigns physical blocks from the pool
3. Runs one decode step for all selected sequences simultaneously
4. Frees blocks from completed sequences

**vLLM API usage**:
```python
from vllm import LLM, SamplingParams

llm = LLM(model="...", gpu_memory_utilization=0.85)
out = llm.generate(["Article 1...", "Article 2..."], SamplingParams(max_tokens=128))
```

**OpenAI-compatible server**:
```bash
python -m vllm.entrypoints.openai.api_server --model ./models/lora_merged --port 8000
curl http://localhost:8000/v1/chat/completions -d '{"model":"...","messages":[...]}'
```

### 6.3 SGLang — Structured Generation Language

SGLang adds a **programming model** on top of efficient inference, solving
problems vLLM alone doesn't address.

**RadixAttention — automatic prefix caching**:

```
Request 1: [System prompt (50 tokens)] + [Article A (200 tokens)]
Request 2: [System prompt (50 tokens)] + [Article B (180 tokens)]

SGLang's radix tree:
root
└── "You are an expert summariser..." (50 tokens, cached)
    ├── [Article A tokens...]
    └── [Article B tokens...]

→ System prompt KV computed ONCE, shared across all requests
→ Effective throughput doubles for shared-prefix workloads
```

**Constrained decoding — guaranteed valid JSON**:

```python
@sgl.function
def extract_entities(s, article):
    s += sgl.user(article)
    s += sgl.assistant(
        sgl.gen("output", json_schema={
            "type": "object",
            "properties": {
                "people":    {"type": "array", "items": {"type": "string"}},
                "locations": {"type": "array", "items": {"type": "string"}},
            }
        })
    )
```

SGLang uses a **finite state machine** built from the JSON schema to mask
the logit distribution at each step — only tokens that keep the output
valid according to the schema get non-zero probability.

**Fork/join parallelism**:

```python
@sgl.function
def multi_view(s, article):
    forks = s.fork(3)   # create 3 parallel branches from current state
    for fork, angle in zip(forks, ["technical", "business", "public"]):
        fork += sgl.assistant(sgl.gen(f"summary_{angle}", max_tokens=80))
    forks.join()         # synchronise, merge states
```

All 3 branches share the encoded article (cached) and generate in parallel.

**When to use which**:

| Scenario | Use |
|----------|-----|
| High-throughput API, drop-in OpenAI replacement | **vLLM** |
| Multi-step reasoning, RAG pipelines | **SGLang** |
| Guaranteed structured output (JSON/XML) | **SGLang** |
| Shared system prompts across many requests | **SGLang** |
| Multi-GPU tensor parallelism | **vLLM** |

---

## 7. Step 5 — FastAPI and Docker

### 7.1 FastAPI Endpoint Design

**File**: [api/main.py](api/main.py)

FastAPI is an **ASGI** (Asynchronous Server Gateway Interface) web framework.
Unlike WSGI (Flask, Django), ASGI handles concurrent requests without threads,
using Python's `asyncio` event loop.

**Lifespan pattern — load model once, not per request**:
```python
@asynccontextmanager
async def lifespan(app):
    # Runs at startup
    model = load_model()   ← expensive, done once
    yield                  ← server runs here, model in memory
    # Runs at shutdown
    del model              ← cleanup
```

**Endpoints**:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/summarise` | Single article → summary |
| POST | `/summarise/batch` | Up to 16 articles → summaries |
| GET | `/health` | Liveness probe (am I running?) |
| GET | `/ready` | Readiness probe (is model loaded?) |
| GET | `/metrics` | Prometheus scrape endpoint |
| GET | `/docs` | Auto-generated Swagger UI |

**Why separate `/health` and `/ready`?**

```
/health  → Kubernetes restarts the Pod if this fails
           Use for: deadlocks, infinite loops, fatal errors
           Do NOT fail during model loading (it would restart before finishing)

/ready   → Kubernetes removes Pod from load balancer if this fails
           Use for: "model not loaded yet", "overwhelmed with requests"
           Pod stays alive but gets no new traffic until ready again
```

### 7.2 Prometheus Metrics in the API

The API uses `prometheus_client` to expose metrics at `/metrics`:

```python
# Counter: only increases — total requests, total errors
REQUEST_COUNT = Counter(
    "summarise_requests_total",
    "Total number of summarisation requests",
    ["endpoint", "status"],   # ← labels allow filtering in Grafana
)

# Histogram: records distribution of values
# Automatically creates:
#   summarise_request_latency_seconds_bucket  ← per-bucket counts
#   summarise_request_latency_seconds_count   ← total observations
#   summarise_request_latency_seconds_sum     ← sum of all latencies
REQUEST_LATENCY = Histogram(
    "summarise_request_latency_seconds",
    "Latency of summarisation requests",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
```

**At the end of every request**:
```python
latency = time.perf_counter() - t0
REQUEST_COUNT.labels(endpoint="summarise", status="success").inc()
REQUEST_LATENCY.labels(endpoint="summarise").observe(latency)
```

### 7.3 Dockerfile — Multi-Stage Build

**File**: [docker/Dockerfile](docker/Dockerfile)

Multi-stage builds keep the final image lean by separating build-time
tools from the runtime image:

```
Stage 1 (builder):
  FROM python:3.11-slim
  - Install gcc, g++ (needed to compile some packages)
  - pip install -r requirements.txt --prefix=/install
  → All Python packages compiled and stored in /install

Stage 2 (runtime):
  FROM python:3.11-slim  ← fresh image, no build tools
  - COPY --from=builder /install /usr/local  ← only the packages, not gcc
  - COPY api/ ./api/
  - Create non-root user (security best practice)
  → Final image is hundreds of MB smaller
```

**Key Dockerfile concepts**:

```dockerfile
# Layer caching: Docker caches each RUN/COPY as a layer.
# Put rarely-changing instructions first (package install),
# frequently-changing ones last (application code).
# This way, code changes don't invalidate the package install cache.

COPY requirements.txt .    ← cached unless requirements change
RUN pip install ...        ← only re-runs if requirements.txt changed
COPY api/ .                ← re-runs on every code change (ok, it's fast)
```

```dockerfile
# HEALTHCHECK: Docker will mark container "unhealthy" if this fails.
# Orchestrators (including Kubernetes via Docker Compose) use this.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
```

### 7.4 Docker Compose — Full Local Stack

**File**: [docker/docker-compose.yml](docker/docker-compose.yml)

Docker Compose defines a **multi-container application** as a single YAML file.

```
docker compose up --build
         │
         ▼
Networks created:
  - backend  (api ↔ prometheus ↔ grafana — internal)
  - frontend (nginx ↔ grafana — external)

Containers started:
  api (×3 replicas)
  nginx
  prometheus
  grafana
```

**Service discovery**: In Docker Compose, containers reach each other by
**service name** (e.g. `http://api:8080`). Docker's embedded DNS resolver
maps service names to container IPs automatically.

**Volumes**: Persistent data is stored in named volumes (`prometheus_data`,
`grafana_data`) that survive container restarts.

### 7.5 NGINX — Reverse Proxy and Load Balancer

**File**: [docker/nginx.conf](docker/nginx.conf)

NGINX sits in front of the API replicas, providing:

**1. Reverse proxy** — hides backend topology from clients:
```
Client sees: POST http://myservice.com/summarise
NGINX forwards to: http://api-replica-2:8080/summarise
Client never knows the backend IP or port
```

**2. Load balancing** — distributes requests across replicas:

```
upstream summarise_backend {
    least_conn;          ← algorithm choice
    server api:8080;
    keepalive 32;
}
```

**Load balancing algorithms**:

| Algorithm | How it works | Best for |
|-----------|-------------|----------|
| `round_robin` | Rotate through servers equally | Uniform request cost |
| `least_conn` | Route to server with fewest active connections | Variable-cost requests |
| `ip_hash` | Hash client IP → always same server | Session stickiness |
| `ewma` | Exponential weighted moving average latency | Latency-sensitive |

For ML inference, `least_conn` is ideal because inference time varies
significantly with input length.

**3. Rate limiting** — protect backend from traffic spikes:
```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=20r/s;
# 10m = 10MB memory for tracking IPs (stores ~160,000 IPs)
# rate=20r/s = max 20 requests/second per IP

location /summarise {
    limit_req zone=api_limit burst=10 nodelay;
    # burst=10: allow up to 10 requests to queue before rejecting
    # nodelay: process burst immediately (don't add artificial delay)
}
```

**4. Timeout configuration**:
```nginx
proxy_read_timeout 300s;   # ML inference can take 30+ seconds
```

**5. Request flow through NGINX**:
```
Client request
    │
    ▼
NGINX: check rate limit → reject if exceeded (HTTP 429)
    │
    ▼
NGINX: select backend via least_conn algorithm
    │
    ▼
NGINX: add X-Real-IP, X-Forwarded-For headers
    │
    ▼
Backend (FastAPI): process request, return response
    │
    ▼
NGINX: pass response back to client
```

---

## 8. Step 6 — Kubernetes

**Files**: [k8s/manifests/](k8s/manifests/)

### 8.1 Core Concepts and Architecture

Kubernetes (k8s) is a **container orchestration platform** — it manages
where containers run, how many copies run, how they communicate, and
how they self-heal.

**Control Plane** (the "brain"):
```
┌─────────────────────────────────────────────────────────┐
│                    Control Plane                        │
│                                                         │
│  ┌──────────────┐  ┌───────────┐  ┌────────────────┐  │
│  │  API Server  │  │  etcd     │  │  Scheduler     │  │
│  │  (REST API)  │  │ (state DB)│  │  (Pod→Node)    │  │
│  └──────────────┘  └───────────┘  └────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Controller Manager                        │  │
│  │  (Deployment ctrl, ReplicaSet ctrl, HPA ctrl...) │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Worker Nodes** (where workloads run):
```
┌────────────────────────────────────────────────────────┐
│                    Worker Node                         │
│                                                        │
│  ┌──────────┐  ┌──────────────────────────────────┐  │
│  │ kubelet  │  │         Pods                      │  │
│  │(Pod mgr) │  │  ┌──────────┐  ┌──────────┐      │  │
│  └──────────┘  │  │Container │  │Container │  ... │  │
│  ┌──────────┐  │  └──────────┘  └──────────┘      │  │
│  │kube-proxy│  └──────────────────────────────────┘  │
│  │(iptables)│                                         │
│  └──────────┘                                         │
└────────────────────────────────────────────────────────┘
```

**How a Pod gets scheduled**:
```
1. You submit Deployment YAML → API Server stores in etcd

2. Scheduler watches for unscheduled Pods:
   - Filters nodes that meet resource requests (CPU/memory)
   - Filters nodes that satisfy affinity/anti-affinity rules
   - Scores remaining nodes (most available resources wins)
   - Binds Pod to winning Node

3. kubelet on the Node:
   - Sees the Pod assigned to it
   - Pulls the container image
   - Creates the container (via containerd/cri-o)
   - Starts liveness/readiness probes

4. kube-proxy on every Node:
   - Watches for Service changes
   - Programs iptables/IPVS rules so ClusterIP → Pod IP traffic works
```

### 8.2 Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: summarisation
```

A Namespace is a **virtual partition** inside the cluster. It provides:
- **Isolation**: resources in different namespaces don't collide by name
- **RBAC scope**: you can grant permissions per namespace
- **Quotas**: limit total CPU/memory a namespace can use
- **Network policies**: control which namespaces can communicate

All our resources (`kubectl apply -f ...`) include `namespace: summarisation`
so they're logically grouped and isolated.

### 8.3 ConfigMap and Secret

**ConfigMap** — non-sensitive configuration:
```yaml
data:
  MODEL_PATH: "/app/models/lora_merged"
  PORT: "8080"
```

Consumed by Pods as environment variables:
```yaml
envFrom:
  - configMapRef:
      name: summarisation-config
```

**Secret** — sensitive values (passwords, API keys):
```yaml
# Values are base64-encoded (NOT encrypted by default in k8s)
data:
  api_key: dGhpcyBpcyBhIHNlY3JldA==   # echo -n "this is a secret" | base64
```

**Difference**: Secrets have stricter RBAC by default and can be encrypted
at rest with etcd encryption. Never put secrets in ConfigMaps.

### 8.4 Deployment, ReplicaSet, and Pod

**Hierarchy**:
```
Deployment  (desired state: "3 replicas of this Pod template")
    │
    └── ReplicaSet  (current state manager: "run exactly 3 Pods")
            │
            ├── Pod 1  (container runtime instance)
            ├── Pod 2
            └── Pod 3
```

**Deployment controller reconciliation loop**:
```
while true:
    desired  = read Deployment spec from etcd
    actual   = count running Pods with matching labels
    if actual < desired:
        create (desired - actual) new Pods
    elif actual > desired:
        delete (actual - desired) excess Pods
    sleep(1s)
```

**Rolling update** — zero-downtime deploys:
```
Before: [Pod v1] [Pod v1] [Pod v1]   (3 running)

Step 1: Create [Pod v2]             (4 total, maxSurge=1)
        [Pod v1] [Pod v1] [Pod v1] [Pod v2]

Step 2: Wait for Pod v2 to pass readinessProbe

Step 3: Delete one [Pod v1]         (back to 3 total)
        [Pod v1] [Pod v1] [Pod v2]

Repeat until all v1 Pods replaced
After:  [Pod v2] [Pod v2] [Pod v2]  (3 running, zero downtime)
```

**Liveness probe** — detects and recovers from failures:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 60   # model takes ~60s to load
  periodSeconds: 15         # check every 15s
  failureThreshold: 3       # restart after 3 consecutive failures
```

If the container is stuck in an infinite loop or deadlock, it still
responds to OS process signals but can't respond to HTTP — the liveness
probe detects this and triggers a container restart.

**Readiness probe** — traffic gating:
```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 60
  periodSeconds: 10
  failureThreshold: 3
```

If the model is still loading (Python is slow to import PyTorch + load
2GB of weights), the readiness probe fails and the Pod is removed from
the Service's load balancer. No traffic is sent until the model is ready.

### 8.5 Service — Virtual IP and Load Balancing

```yaml
apiVersion: v1
kind: Service
metadata:
  name: summarisation-service
spec:
  type: ClusterIP
  selector:
    app: summarisation-api   # routes to ALL Pods with this label
  ports:
    - port: 8080
      targetPort: 8080
```

**How kube-proxy implements a Service**:

```
Service: summarisation-service → ClusterIP: 10.96.45.12:8080

kube-proxy programs iptables rules on every Node:

  PREROUTING → KUBE-SERVICES → KUBE-SVC-XYZ
  KUBE-SVC-XYZ:
    33% probability → DNAT to Pod1 IP:8080
    33% probability → DNAT to Pod2 IP:8080
    33% probability → DNAT to Pod3 IP:8080

  Traffic to 10.96.45.12:8080 is randomly sent to one of the 3 Pods.
  Failed/unready Pods are removed from the iptables rules automatically.
```

DNS inside the cluster:
```
Other pods reach this service at:
  summarisation-service              (within same namespace)
  summarisation-service.summarisation  (cross-namespace)
  summarisation-service.summarisation.svc.cluster.local  (full FQDN)
```

**Service types**:

| Type | Access | Use case |
|------|--------|----------|
| `ClusterIP` | Internal only | Backend services |
| `NodePort` | Node IP + port 30000-32767 | Development/Docker Desktop |
| `LoadBalancer` | Cloud load balancer IP | Production in cloud |
| `ExternalName` | DNS alias | External services |

### 8.6 Ingress and Ingress Controller

**The problem**: Services are the k8s primitive for routing traffic, but
`ClusterIP` is internal-only and `NodePort` exposes an ugly high port number.
For HTTP/HTTPS traffic you need hostname-based routing, TLS termination,
and path-based rules — that's what Ingress provides.

**Two parts**:

1. **Ingress resource** — the routing *rules* (just YAML, no actual proxy):
```yaml
rules:
  - host: api.example.com
    http:
      paths:
        - path: /summarise
          backend:
            service: {name: summarisation-service, port: 8080}
```

2. **Ingress Controller** — the actual proxy that reads the rules and
   handles traffic (NGINX, Traefik, Envoy, HAProxy, etc.):
```
kubectl apply -f ingress-nginx-controller.yaml
→ Creates a Deployment running NGINX
→ That NGINX watches k8s for Ingress objects
→ When Ingress is created/updated, NGINX reloads its config
→ Traffic flows: client → NGINX Pod → Service → App Pod
```

**Why not just use NGINX directly (like in Docker Compose)?**

The Kubernetes Ingress Controller is Kubernetes-native: it automatically
discovers Services and Pods, handles health checks, and supports hot-reload
without restart. It also integrates with cert-manager for automatic TLS.

**Request flow with Ingress**:
```
Browser: GET https://api.example.com/summarise
   │
   ▼  DNS resolves to Node IP (or cloud LoadBalancer IP)
   │
   ▼  NodePort / LoadBalancer routes to NGINX Ingress Controller Pod
   │
   ▼  NGINX reads Ingress rule: /summarise → summarisation-service:8080
   │
   ▼  NGINX proxies to one of the 3 API Pods (via Service ClusterIP)
   │
   ▼  API Pod processes request, returns response
   │
   ▼  NGINX returns response to browser
```

### 8.7 HorizontalPodAutoscaler

HPA automatically adjusts `spec.replicas` in the Deployment based on metrics.

**Scale-up formula**:
```
desiredReplicas = ceil(currentReplicas × (currentMetric / targetMetric))

Example:
  currentReplicas = 3
  currentCPU      = 80%  (measured across all Pods)
  targetCPU       = 60%

  desiredReplicas = ceil(3 × 80/60) = ceil(4.0) = 4

Kubernetes creates 1 new Pod → 4 Pods now share the load
```

**Scale-down has a stabilization window** (300s default):
```
If metric drops below target, Kubernetes waits 5 minutes before scaling down.
Why? To avoid "flapping" — rapid up/down/up/down under fluctuating load.
```

**Custom metrics via Prometheus Adapter**:
```
prometheus-adapter reads Prometheus metrics and exposes them
as Kubernetes custom metrics API.

HPA query:   "what is summarise_requests_per_second per Pod?"
Prometheus:  sum(rate(summarise_requests_total[1m])) / count(pods)
HPA action:  scale up if > 10 req/s per Pod
```

### 8.8 Persistent Volumes

The model weights (~2.8 GB) should not be in the container image
(makes it too large and couples the model version to the image version).
Instead, mount them as a **PersistentVolume**:

```
PersistentVolume (PV) — the actual storage resource
  ├── On-premise: NFS server, local disk
  └── Cloud: AWS EFS, GCP Persistent Disk, Azure Disk

PersistentVolumeClaim (PVC) — a request for storage (how much, access mode)

Pod spec:
  volumes:
    - name: model-weights
      persistentVolumeClaim:
        claimName: model-weights-pvc
  containers:
    - volumeMounts:
        - name: model-weights
          mountPath: /app/models
          readOnly: true    ← multiple Pods can read simultaneously (RWX mode)
```

For Docker Desktop (local development), we use `hostPath`:
```yaml
volumes:
  - name: model-weights
    hostPath:
      path: /Users/yourname/models   # absolute path on your Mac
```

### 8.9 RBAC — Role-Based Access Control

Prometheus needs to query the Kubernetes API to discover Pod endpoints.
RBAC controls what each service account is allowed to do:

```
ServiceAccount "prometheus"
   │
   ▼ (bound via ClusterRoleBinding)
ClusterRole "prometheus"
   │ rules:
   ├── get, list, watch → pods
   ├── get, list, watch → services
   ├── get, list, watch → endpoints
   └── get              → /metrics (non-resource URL)
```

Without this RBAC setup, Prometheus would get HTTP 403 when trying to
list Pods for service discovery.

---

## 9. Step 7 — Prometheus and Grafana

### 9.1 How Prometheus Works

Prometheus uses a **pull model** (opposite of most monitoring systems
that push metrics to a central server):

```
Every scrape_interval (15s):

Prometheus ──► GET http://api-pod-1:8080/metrics
         ◄──── text/plain response:
               summarise_requests_total{endpoint="summarise",status="success"} 142
               summarise_request_latency_seconds_bucket{le="0.5"} 89
               summarise_request_latency_seconds_bucket{le="1.0"} 130
               ...

Prometheus stores these as time-series:
  (metric_name, labels, timestamp) → value

Queried later via PromQL.
```

**Why pull instead of push?**
- Prometheus controls the scrape rate (no agent overwhelm)
- Easy to see if a target is down (scrape fails)
- No need to configure agents with Prometheus endpoint
- Service discovery handles dynamic infrastructure

**Kubernetes service discovery**:
```yaml
kubernetes_sd_configs:
  - role: pod
relabel_configs:
  - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
    action: keep
    regex: "true"
```

Prometheus watches the Kubernetes API for Pods and automatically scrapes
any Pod annotated with `prometheus.io/scrape: "true"`.

### 9.2 Metric Types Explained

**Counter** — only increases, never decreases:
```
Use for: total requests, errors, bytes processed
Query:   rate(counter[5m])  ← per-second rate over 5-minute window
         increase(counter[1h])  ← total increase over last hour

Example: summarise_requests_total
  t=0:   5
  t=30s: 12
  t=60s: 19
  rate over [1m] = (19-5)/60 = 0.23 req/s
```

**Gauge** — can go up or down:
```
Use for: current memory, active connections, queue depth
Query:   model_memory_mb  ← current value
         avg_over_time(model_memory_mb[1h])  ← average

Example: model_memory_mb = 2847.3
```

**Histogram** — samples values into pre-defined buckets:
```
Configured buckets: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, +Inf]

When a request takes 0.73 seconds:
  bucket{le="0.1"} unchanged   (0.73 > 0.1)
  bucket{le="0.5"} unchanged   (0.73 > 0.5)
  bucket{le="1.0"} +1          (0.73 ≤ 1.0) ← first bucket that fits
  bucket{le="2.0"} +1          (cumulative — all larger buckets also increment)
  bucket{le="5.0"} +1
  bucket{le="+Inf"} +1
  _count +1
  _sum   +0.73

Over time, the buckets accumulate:
  le="0.1":  5   (5 requests took ≤ 0.1s)
  le="0.5": 23   (23 requests took ≤ 0.5s)
  le="1.0": 87   (87 requests took ≤ 1.0s)
  le="2.0": 95
  le="+Inf": 100  (all 100 requests)
```

### 9.3 PromQL — Computing p90 and p95

**The key formula**:

```promql
histogram_quantile(
  0.95,                                                -- the quantile (95th percentile)
  sum(
    rate(
      summarise_request_latency_seconds_bucket[5m]     -- bucket counter rates over 5min
    )
  ) by (le)                                            -- group by bucket boundary label
)
```

**How `histogram_quantile` works**:

```
We have bucket counts at time T:
  le=0.1 → 5 req/s
  le=0.5 → 23 req/s
  le=1.0 → 87 req/s
  le=2.0 → 95 req/s
  le=+∞  → 100 req/s

To find p95: find value V where 95% of requests took ≤ V seconds.
  95% of 100 = 95th sample
  87 samples are ≤ 1.0s
  95 samples are ≤ 2.0s
  95th sample falls between le=1.0 and le=2.0
  
  Linear interpolation:
  p95 = 1.0 + (95-87)/(95-87) × (2.0-1.0) = 1.0 + 1.0 = 2.0s
```

**Common PromQL queries**:

```promql
-- Request rate (requests per second)
sum(rate(summarise_requests_total[1m])) by (endpoint)

-- p50 latency (median)
histogram_quantile(0.50, sum(rate(summarise_request_latency_seconds_bucket[5m])) by (le))

-- p90 latency
histogram_quantile(0.90, sum(rate(summarise_request_latency_seconds_bucket[5m])) by (le))

-- p95 latency  ← most common SLA metric
histogram_quantile(0.95, sum(rate(summarise_request_latency_seconds_bucket[5m])) by (le))

-- p99 latency  ← catches worst-case outliers
histogram_quantile(0.99, sum(rate(summarise_request_latency_seconds_bucket[5m])) by (le))

-- Error rate as a percentage
100 * sum(rate(summarise_requests_total{status="error"}[5m]))
    / sum(rate(summarise_requests_total[5m]))

-- Apdex score (user satisfaction: satisfied < 1s, tolerating < 4s)
(
  sum(rate(summarise_request_latency_seconds_bucket{le="1.0"}[5m]))
  + sum(rate(summarise_request_latency_seconds_bucket{le="4.0"}[5m])) / 2
) / sum(rate(summarise_request_latency_seconds_count[5m]))
```

**What p95 and p90 mean in practice**:
```
p50 = 0.8s: half your users wait ≤ 0.8s
p90 = 2.1s: 90% of users wait ≤ 2.1s (10% wait longer)
p95 = 4.3s: 95% of users wait ≤ 4.3s (5% wait longer)
p99 = 12s:  99% of users wait ≤ 12s  (1% wait longer)

SLA example: "p95 latency must be below 5 seconds"
→ Set Grafana alert: p95 > 5s → PagerDuty page
```

### 9.4 Grafana Dashboards

**File**: [monitoring/grafana/dashboards/summarisation_dashboard.json](monitoring/grafana/dashboards/summarisation_dashboard.json)

The dashboard is **auto-provisioned** — it loads from disk at Grafana startup
via the provisioning YAML, no manual import needed.

**Panels**:

| Panel | Type | PromQL | What it shows |
|-------|------|--------|--------------|
| Request throughput | Time series | `rate(summarise_requests_total[1m])` | req/s by endpoint |
| p50/p90/p95/p99 latency | Time series | `histogram_quantile(...)` | Latency distribution over time |
| Error rate | Time series | errors / total | % of failed requests |
| Model memory | Gauge | `model_memory_mb` | Current memory with thresholds |
| Token throughput | Time series | `rate(tokens_processed_total[1m])` | Input/output tokens per second |
| p95 Stat | Stat (number) | instant p95 | Single number, red if > 5s |
| p90 Stat | Stat (number) | instant p90 | Single number, red if > 3s |

---

## 10. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Kubernetes Cluster                                │
│  Namespace: summarisation                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Data Plane (Worker Nodes)                       │   │
│  │                                                                     │   │
│  │  ┌──────────────┐                                                   │   │
│  │  │   Ingress    │  ← NGINX Ingress Controller                       │   │
│  │  │  Controller  │    reads Ingress YAML rules                       │   │
│  │  └──────┬───────┘    handles TLS, routing, rate-limiting            │   │
│  │         │                                                           │   │
│  │         │ routes to ClusterIP Service                               │   │
│  │         ▼                                                           │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │          summarisation-service (ClusterIP)                    │  │   │
│  │  │     kube-proxy iptables: 33% each to 3 Pods                  │  │   │
│  │  └────────┬──────────────────┬──────────────────┬───────────────┘  │   │
│  │           │                  │                  │                   │   │
│  │           ▼                  ▼                  ▼                   │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐          │   │
│  │  │   API Pod 1    │ │   API Pod 2    │ │   API Pod 3    │          │   │
│  │  │  FastAPI:8080  │ │  FastAPI:8080  │ │  FastAPI:8080  │          │   │
│  │  │  Model loaded  │ │  Model loaded  │ │  Model loaded  │          │   │
│  │  │  /metrics ─────┼─┼──────────────►Prometheus         │          │   │
│  │  └────────────────┘ └────────────────┘ └────────────────┘          │   │
│  │           ▲               ▲                                         │   │
│  │           │               │ mounts                                  │   │
│  │  ┌────────────────────────────────────────────┐                    │   │
│  │  │     PersistentVolume: /app/models           │                   │   │
│  │  │     (model weights, read-only, shared)      │                   │   │
│  │  └────────────────────────────────────────────┘                    │   │
│  │                                                                     │   │
│  │  ┌──────────────┐    scrapes /metrics    ┌──────────────────────┐  │   │
│  │  │  Prometheus  │◄──────────────────────│    API Pods (×3)      │  │   │
│  │  │  :30090      │                        └──────────────────────┘  │   │
│  │  └──────┬───────┘                                                   │   │
│  │         │ datasource                                                 │   │
│  │         ▼                                                            │   │
│  │  ┌──────────────┐                                                   │   │
│  │  │   Grafana    │  ← dashboards, alerts, p95/p90 panels             │   │
│  │  │   :30030     │                                                   │   │
│  │  └──────────────┘                                                   │   │
│  │                                                                     │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │         HPA (HorizontalPodAutoscaler)                        │  │   │
│  │  │  watches CPU/memory → adjusts replicas 2–10                  │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
         ▲
         │ NodePort :30080
         │
    ┌────────────┐
    │   Client   │  curl / browser / load test
    └────────────┘
```

---

## 11. Inference Request Flow — End to End

Here is the complete journey of a single `POST /summarise` request:

```
① Client sends HTTP POST
   curl -X POST http://localhost:30080/summarise \
     -d '{"article": "NASA found water on Mars..."}'

② NodePort → NGINX Ingress Controller Pod
   The request arrives at port 30080 on the Node.
   kube-proxy's iptables rules forward it to the NGINX Ingress Controller Pod.

③ NGINX reads Ingress rules
   Rule: path=/summarise → Service: summarisation-service:8080
   NGINX selects one of the 3 API Pods via least_conn algorithm.
   NGINX adds X-Real-IP, X-Forwarded-For headers.

④ NGINX → summarisation-service (ClusterIP)
   Traffic goes to ClusterIP 10.96.x.x:8080.
   kube-proxy DNAT's it to one of the 3 Pod IPs.

⑤ FastAPI receives request in API Pod
   uvicorn ASGI server accepts the TCP connection.
   FastAPI routes POST /summarise → summarise() function.
   Pydantic validates the request body.
   t0 = time.perf_counter()

⑥ Model inference
   tokenizer(article) → input_ids tensor (512 tokens max)
   model.generate(input_ids) → [beam search over vocab at each step]
   tokenizer.decode(out_ids) → "NASA's Perseverance rover..."

⑦ Prometheus metrics updated
   REQUEST_COUNT.labels("summarise", "success").inc()
   REQUEST_LATENCY.labels("summarise").observe(latency_seconds)
   TOKENS_PROCESSED.labels("input").inc(input_len)
   TOKENS_PROCESSED.labels("output").inc(output_len)

⑧ FastAPI returns JSON response
   {"summary": "...", "latency_ms": 843.2, "input_tokens": 412, "output_tokens": 67}

⑨ Response travels back: Pod → ClusterIP → NGINX → NodePort → Client

⑩ Meanwhile, Prometheus scrapes /metrics every 15 seconds
   Reads the histogram buckets, counters, gauges.
   Stores as time-series in its TSDB.

⑪ Grafana queries Prometheus via PromQL
   histogram_quantile(0.95, ...) = 2.1 seconds
   Displays on the p95 panel. If > 5s, fires alert.

⑫ HPA checks metrics every 30 seconds
   If avg CPU > 60%, adds more Pods (up to 10).
   New Pods pass readiness probe → added to Service rotation.
```

---

## 12. Running the Project

See [RUNBOOK.md](RUNBOOK.md) for the complete step-by-step commands.

**Quick start**:

```bash
# Step 1 — Data
python data/download_data.py

# Step 2 — Train (needs GPU for full training, CPU for quick test)
pip install -r requirements.txt
jupyter lab notebooks/01_training_lora_mora.ipynb

# Step 5 — Local Docker stack (no GPU needed for inference with small model)
docker compose -f docker/docker-compose.yml up --build

# Test the API
curl -X POST http://localhost/summarise \
  -H "Content-Type: application/json" \
  -d '{"article": "Scientists have discovered a new species of deep sea creature..."}'

# View Grafana dashboard
open http://localhost:3000   # admin / admin

# Step 6 — Kubernetes (Docker Desktop)
kubectl apply -f k8s/manifests/
kubectl get pods -n summarisation -w

# Step 7 — Prometheus
open http://localhost:30090
# Query: histogram_quantile(0.95, sum(rate(summarise_request_latency_seconds_bucket[5m])) by (le))
```
