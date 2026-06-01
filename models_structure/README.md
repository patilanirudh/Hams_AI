# Models

Model weights are **not stored in this repository** due to GitHub's 100 MB file size limit.
Both fine-tuned models are hosted on HuggingFace and must be downloaded before running the system.

## HuggingFace Links

| Model | HuggingFace Repo | Base Model | Size |
|---|---|---|---|
| Fine-tuned Embedding | [`ani122312/bge-m3-hamsai-finetuned`](https://huggingface.co/ani122312/bge-m3-hamsai-finetuned) | BAAI/bge-m3 | ~2.2 GB |
| Fine-tuned Reranker | [`ani122312/bge-reranker-hamsai-finetuned`](https://huggingface.co/ani122312/bge-reranker-hamsai-finetuned) | BAAI/bge-reranker-v2-m3 | ~2.2 GB |

## Folder Structure (after download)

```
models/
├── bge-m3-hamsai-finetuned/
│   ├── 1_Pooling/
│   │   └── config.json
│   ├── config.json
│   ├── config_sentence_transformers.json
│   ├── model.safetensors          ← 2.2 GB — download from HuggingFace
│   ├── modules.json
│   ├── README.md
│   ├── sentence_bert_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
└── bge-reranker-hamsai-finetuned/
    ├── eval/
    │   └── CrossEncoderRerankingEvaluator_hamsai_reranker_val_results_@10.csv
    ├── config.json
    ├── config_sentence_transformers.json
    ├── model.safetensors          ← 2.2 GB — download from HuggingFace
    ├── modules.json
    ├── README.md
    ├── sentence_bert_config.json
    ├── special_tokens_map.json
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Download Instructions

```bash
# Option 1 — huggingface_hub (recommended)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ani122312/bge-m3-hamsai-finetuned',    local_dir='models/bge-m3-hamsai-finetuned')
snapshot_download('ani122312/bge-reranker-hamsai-finetuned', local_dir='models/bge-reranker-hamsai-finetuned')
"

# Option 2 — git lfs
git lfs install
git clone https://huggingface.co/ani122312/bge-m3-hamsai-finetuned    models/bge-m3-hamsai-finetuned
git clone https://huggingface.co/ani122312/bge-reranker-hamsai-finetuned models/bge-reranker-hamsai-finetuned
```

After downloading, the paths in `configs/serving_config.yaml` should match automatically.
