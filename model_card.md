# HamsAI RAG Model Card

## Model Overview

| Component     | Model                   | Type                    |
| ------------- | ----------------------- | ----------------------- |
| Embedding     | BAAI/bge-m3             | Fine-tuned              |
| Reranker      | BAAI/bge-reranker-v2-m3 | Fine-tuned              |
| Generator LLM | qwen2.5:3b (Ollama)     | Off-the-shelf quantized |

---

## Embedding Model

**Base model:** BAAI/bge-m3
**HuggingFace:** https://huggingface.co/BAAI/bge-m3
**License:** MIT — commercial use allowed
**Fine-tuned model:** `ani122312/bge-m3-hamsai-finetuned` (private HuggingFace repo; reviewer access required)

### Fine-tuning Details

- **Method:** Full fine-tuning with TripletLoss
- **Layers fine-tuned:** All transformer layers (no freezing)
- **Loss function:** TripletLoss with explicit hard negatives
- **Hard negative strategy:** Same document category, different product/city/fact — forces the model to distinguish semantically similar but factually distinct passages
- **Training objective:** Maximize cosine similarity between query and positive chunk; minimize with hard negative chunk
- **Epochs:** 3
- **Batch size:** 8
- **Learning rate:** 2e-5
- **Warmup ratio:** 10%
- **Weight decay:** 0.01
- **Max sequence length:** 512
- **FP16:** Yes (on GPU)
- **Training time:** ~11 minutes on T4 GPU

### Fine-tuning Results

**In-model discriminability (same 176-pair held-out set, measured by `train_embeddings.py`):**

| Metric | Base BAAI/bge-m3 | Fine-tuned | Gain |
|---|---|---|---|
| Triplet accuracy | 0.7955 | **0.8920** | +9.65pp |

Triplet accuracy measures whether the model ranks the positive chunk above the hard-negative chunk for each test query. This is the cleanest apples-to-apples comparison because both before and after are evaluated on the same 176-pair held-out set with the same in-memory candidate pool.

**Full-corpus retrieval (hybrid BM25 + dense + rerank pipeline, `benchmark_retrieval.py`):**

| Metric | Base model (176-pair set) | Fine-tuned model (200-pair set) |
|---|---|---|
| Hybrid+Rerank Recall@5 (overall) | 0.6477 | 0.4650 |
| Hybrid+Rerank Recall@5 (AR) | 0.6125 | 0.4944 |
| Hybrid+Rerank Recall@5 (EN) | 0.6875 | 0.4691 |
| Hybrid+Rerank Recall@5 (cross-lingual) | 0.5625 | 0.3704 |
| Hybrid+Rerank MRR (overall) | 0.5284 | 0.4482 |
| Hybrid+Rerank nDCG@10 (overall) | 0.5876 | 0.4566 |

> **Important note on test set difference:** The base model was benchmarked on a 176-pair held-out set; the fine-tuned model on a 200-pair set (the final dataset expansion added 24 additional hard negative pairs after baseline was measured). These sets are not identical, so the full-corpus numbers cannot be directly subtracted to measure fine-tuning gain. A Colab GPU re-run of `benchmark_retrieval.py` with the base model on the same 200-pair set will provide an apples-to-apples comparison. The triplet accuracy above (same test set, both models) is the reliable fine-tuning signal.

### Dataset

- **Source:** Synthetic — generated using Gemini 3.5 Flash
- **Language coverage:** Arabic (MSA), English, Mixed Arabic/English, Cross-lingual
- **Training pairs:** 700 (80% of dataset)
- **Validation pairs:** 120 (used for early stopping and evaluator during training)
- **Held-out retrieval test pairs:** 200 (not seen during training)
- **QA evaluation pairs:** 160, including 20 not-found cases
- **Hard negatives:** 1 per query (explicit, same-category different-fact)
- **License:** CC-BY-4.0

### Arabic Text Normalization

- Diacritics (tashkeel) removal
- Alef variants normalization (أإآٱ → ا)
- Ya normalization (ى → ي)
- Teh marbuta normalization (ة → ه)
- Tatweel removal (ـ)
- Eastern Arabic numerals → Western numerals

### Model Size

- **Disk:** ~2.3 GB
- **VRAM:** ~2.5 GB (fp32), ~1.3 GB (fp16)
- **CPU inference:** Yes (slow)
- **GPU inference:** Yes
- **Quantization:** None applied to embedding model

### Known Limitations

- Training dataset is synthetic and should be reviewed before production use
- May underperform on Saudi dialect queries
- Cross-lingual retrieval accuracy depends on query complexity

---

## Reranker Model

**Base model:** BAAI/bge-reranker-v2-m3
**HuggingFace:** https://huggingface.co/BAAI/bge-reranker-v2-m3
**License:** MIT — commercial use allowed
**Fine-tuned model:** `ani122312/bge-reranker-hamsai-finetuned` (private HuggingFace repo; reviewer access required)

### Fine-tuning Details

- **Method:** Full fine-tuning, binary cross-entropy
- **Loss function:** BCE (positive pair label=1.0, negative pair label=0.0)
- **Training pairs:** 1400 (700 positive + 700 negative from same training triplets)
- **Epochs:** 2
- **Batch size:** 2 (gradient checkpointing enabled due to T4 VRAM constraint)
- **Learning rate:** 1e-5
- **Warmup steps:** 50
- **Training time:** ~20 minutes on T4 GPU

### Fine-tuning Results

Pair accuracy measures whether the model scores a (query, positive) pair higher than (query, hard-negative) for each test pair. Both base and fine-tuned models evaluated on the same 352 held-out pairs (176 positive + 176 negative from the 200-pair retrieval test set).

| Metric | Base BAAI/bge-reranker-v2-m3 | Fine-tuned | Gain |
|---|---|---|---|
| Pair accuracy (test set) | 0.8352 | **0.9034** | +6.82pp |
| Recall@1 boost on hybrid pipeline | 0.235 | **0.430** | +19.5pp |
| MRR boost on hybrid pipeline | 0.326 | **0.448** | +12.2pp |

### Model Size

- **Disk:** ~2.3 GB (full precision weights)
- **VRAM:** ~2.5 GB fp32 / ~1.3 GB fp16
- **CPU inference:** Yes (slow — ~10s per query on CPU, ~60ms on L4 GPU)
- **GPU inference:** Yes

---

## Generator LLM

**Model:** qwen2.5:3b
**Served via:** Ollama
**License:** Apache 2.0 — commercial use allowed
**Quantization:** Q4_K_M (GGUF via Ollama)

### Model Size

- **Disk:** ~2.0 GB
- **VRAM:** ~2.0 GB (Q4_K_M quantized)
- **CPU inference:** Yes
- **GPU inference:** Yes

### Known Limitations

- 3B parameter model — may produce less fluent Arabic than larger models
- Hallucination risk increases when retrieved context is sparse
- System prompt enforces grounded answers only

---

## Infrastructure

| Component      | Technology           |
| -------------- | -------------------- |
| Vector store   | Qdrant (self-hosted) |
| Keyword engine | Elasticsearch 8.15   |
| Session memory | Redis 7.4            |
| API framework  | FastAPI              |
| LLM serving    | Ollama               |

---

## Hardware Requirements

| Component | Minimum             | Recommended (assessment) |
| --------- | ------------------- | ------------------------ |
| GPU       | NVIDIA RTX 3050 4GB | NVIDIA L4 24GB           |
| RAM       | 16 GB               | 32 GB                    |
| Storage   | 20 GB               | 50 GB                    |

---

## Possible Improvements

- Increase training dataset size to 500+ pairs
- Add back-translation augmentation for cross-lingual pairs
- Fine-tune generator LLM with LoRA for grounded Arabic answering
- Add Saudi dialect support via dialect normalization layer
- Use larger generator model (7B+) on L4 GPU
