import json
import os
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

# ─── Dependencies ─────────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
    from torch.utils.data import DataLoader
    import torch
except ImportError:
    print("ERROR: Required packages not installed. Run: pip install sentence-transformers torch")
    sys.exit(1)

# ─── Load Config ──────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ─── Load Training Data ───────────────────────────────────────────────────────

def load_training_pairs(train_file: str) -> list:
    with open(train_file, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} training pairs from {train_file}")
    return pairs

# ─── Build InputExamples ──────────────────────────────────────────────────────

def build_examples(pairs: list) -> list:
    examples = []
    skipped = 0
    for item in pairs:
        query    = item.get('query', '').strip()
        positive = item.get('positive', '').strip()
        negative = item.get('hard_negative', '').strip()

        if not query or not positive or not negative:
            skipped += 1
            continue

        examples.append(InputExample(texts=[query, positive, negative]))

    print(f"Built {len(examples)} examples | Skipped {skipped} incomplete entries")
    return examples

# ─── Evaluator ────────────────────────────────────────────────────────────────

def build_evaluator(val_file: str, model: SentenceTransformer):
    if not Path(val_file).exists():
        print(f"WARNING: Validation file not found at {val_file}, skipping evaluator.")
        return None

    with open(val_file, 'r', encoding='utf-8') as f:
        val_pairs = json.load(f)

    sentences1 = []
    sentences2 = []
    scores = []

    for item in val_pairs:
        query    = item.get('query', '').strip()
        positive = item.get('positive', '').strip()
        negative = item.get('hard_negative', '').strip()

        if not query or not positive or not negative:
            continue

        # positive pair
        sentences1.append(query)
        sentences2.append(positive)
        scores.append(1.0)

        # negative pair
        sentences1.append(query)
        sentences2.append(negative)
        scores.append(0.0)

    if not sentences1:
        return None

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        name="hamsai_val"
    )
    print(f"Built evaluator with {len(sentences1)} sentence pairs")
    return evaluator

# ─── Train ────────────────────────────────────────────────────────────────────

def train(config: dict):
    model_cfg    = config['model']
    training_cfg = config['training']
    data_cfg     = config['data']

    base_model  = model_cfg['base_model']
    output_path = model_cfg['output_path']
    max_seq_len = model_cfg['max_seq_length']

    print(f"Base model      : {base_model}")
    print(f"Output path     : {output_path}")
    print(f"Max seq length  : {max_seq_len}")
    print(f"Epochs          : {training_cfg['epochs']}")
    print(f"Batch size      : {training_cfg['batch_size']}")
    print(f"Learning rate   : {training_cfg['learning_rate']}")
    print(f"Loss function   : {training_cfg['loss']}")

    # Load model
    print("\nLoading base model...")
    model = SentenceTransformer(base_model)
    model.max_seq_length = max_seq_len

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device          : {device}")

    # Load data
    train_pairs = load_training_pairs(data_cfg['train_file'])
    train_examples = build_examples(train_pairs)

    if not train_examples:
        print("ERROR: No valid training examples found.")
        sys.exit(1)

    # DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=training_cfg['batch_size']
    )

    # Loss
    loss_name = training_cfg['loss']
    if loss_name == "TripletLoss":
        loss_fn = losses.TripletLoss(model=model)
    elif loss_name == "MultipleNegativesRankingLoss":
        loss_fn = losses.MultipleNegativesRankingLoss(model=model)
    else:
        print(f"WARNING: Unknown loss {loss_name}, defaulting to TripletLoss")
        loss_fn = losses.TripletLoss(model=model)

    print(f"Loss function   : {loss_fn.__class__.__name__}")

    # Evaluator
    val_file  = str(Path(data_cfg['train_file']).parent / 'val_split.json')
    evaluator = build_evaluator(val_file, model)

    # Warmup steps
    total_steps   = len(train_dataloader) * training_cfg['epochs']
    warmup_steps  = training_cfg.get('warmup_steps', max(1, total_steps // 10))

    print(f"\nTotal steps     : {total_steps}")
    print(f"Warmup steps    : {warmup_steps}")
    print("\nStarting training...\n")

    # Train — version-aware API
    import sentence_transformers as _st
    st_major = int(_st.__version__.split('.')[0])

    if st_major >= 3:
        # New SentenceTransformers v3+ API (SentenceTransformerTrainer)
        try:
            from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
            from datasets import Dataset as HFDataset

            train_data = {"anchor": [], "positive": [], "negative": []}
            for ex in train_examples:
                train_data["anchor"].append(ex.texts[0])
                train_data["positive"].append(ex.texts[1])
                train_data["negative"].append(ex.texts[2])
            train_dataset = HFDataset.from_dict(train_data)

            args = SentenceTransformerTrainingArguments(
                output_dir=output_path,
                num_train_epochs=training_cfg['epochs'],
                per_device_train_batch_size=training_cfg['batch_size'],
                learning_rate=training_cfg['learning_rate'],
                warmup_ratio=training_cfg.get('warmup_ratio', 0.1),
                weight_decay=training_cfg.get('weight_decay', 0.01),
                fp16=training_cfg.get('fp16', False) and torch.cuda.is_available(),
                eval_strategy="epoch" if evaluator else "no",
                save_strategy="best",
                load_best_model_at_end=True if evaluator else False,
                logging_steps=50,
            )
            trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                loss=loss_fn,
                evaluator=evaluator,
            )
            trainer.train()
            model.save_pretrained(output_path)
            print(f"\nTraining complete (ST v3 API). Model saved to: {output_path}")
        except Exception as e:
            print(f"ST v3 trainer failed ({e}), falling back to model.fit()")
            st_major = 2  # trigger fallback below

    if st_major < 3:
        # Legacy SentenceTransformers v2 API
        model.fit(
            train_objectives=[(train_dataloader, loss_fn)],
            evaluator=evaluator,
            epochs=training_cfg['epochs'],
            warmup_steps=warmup_steps,
            optimizer_params={"lr": training_cfg['learning_rate']},
            weight_decay=training_cfg.get('weight_decay', 0.01),
            output_path=output_path,
            save_best_model=training_cfg.get('save_best_model', True),
            show_progress_bar=True,
            use_amp=training_cfg.get('fp16', False) and torch.cuda.is_available()
        )
        print(f"\nTraining complete (ST v2 API). Model saved to: {output_path}")

    return model

# ─── Before / After Evaluation ────────────────────────────────────────────────

import math

def _recall_at_k(retrieved_ids, positive_id, k):
    return 1.0 if positive_id in retrieved_ids[:k] else 0.0

def _mrr(retrieved_ids, positive_id):
    for rank, cid in enumerate(retrieved_ids):
        if cid == positive_id:
            return 1.0 / (rank + 1)
    return 0.0

def _ndcg_at_k(retrieved_ids, positive_id, k):
    for i, cid in enumerate(retrieved_ids[:k]):
        if cid == positive_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def evaluate_retrieval(model: SentenceTransformer, test_file: str, label: str):
    """
    Evaluates triplet accuracy AND Recall@1/5/10, MRR, nDCG@10.
    Builds an in-memory corpus from all unique positive + negative chunks
    in the test set, then retrieves via cosine similarity.
    """
    if not Path(test_file).exists():
        print(f"WARNING: Test file not found at {test_file}, skipping evaluation.")
        return {}

    with open(test_file, 'r', encoding='utf-8') as f:
        test_pairs = json.load(f)

    # Build in-memory corpus of unique chunks (positives + negatives)
    corpus_map = {}   # chunk_text → unique id
    for item in test_pairs:
        for key in ('positive', 'hard_negative'):
            text = item.get(key, '').strip()
            if text and text not in corpus_map:
                corpus_map[text] = len(corpus_map)

    corpus_texts = list(corpus_map.keys())
    print(f"  Encoding {len(corpus_texts)} corpus chunks...")
    corpus_embs = model.encode(corpus_texts, normalize_embeddings=True,
                                batch_size=32, show_progress_bar=False)

    import numpy as np
    corpus_embs = np.array(corpus_embs)

    triplet_correct = 0
    r1, r5, r10, mrr_scores, ndcg10 = [], [], [], [], []

    for item in tqdm(test_pairs, desc=f"Evaluating [{label}]"):
        query    = item.get('query', '').strip()
        positive = item.get('positive', '').strip()
        negative = item.get('hard_negative', '').strip()

        if not query or not positive or not negative:
            continue

        q_emb = model.encode(query, normalize_embeddings=True)
        sims  = corpus_embs @ q_emb                    # cosine similarity to all corpus chunks
        top_ids = np.argsort(sims)[::-1]               # sorted descending
        retrieved_texts = [corpus_texts[i] for i in top_ids[:20]]

        pos_id = corpus_map.get(positive, -1)
        neg_id = corpus_map.get(negative, -1)

        # Triplet accuracy
        if pos_id >= 0 and neg_id >= 0:
            if sims[pos_id] > sims[neg_id]:
                triplet_correct += 1

        # Recall / MRR / nDCG — using text match against retrieved
        pos_in_top = [t for t in retrieved_texts]
        r1.append(_recall_at_k(pos_in_top, positive, 1))
        r5.append(_recall_at_k(pos_in_top, positive, 5))
        r10.append(_recall_at_k(pos_in_top, positive, 10))
        mrr_scores.append(_mrr(pos_in_top, positive))
        ndcg10.append(_ndcg_at_k(pos_in_top, positive, 10))

    total    = len(r1)
    accuracy = triplet_correct / total if total > 0 else 0.0
    avg      = lambda lst: round(sum(lst) / len(lst), 4) if lst else 0.0

    result = {
        "label"           : label,
        "triplet_accuracy": round(accuracy, 4),
        "recall@1"        : avg(r1),
        "recall@5"        : avg(r5),
        "recall@10"       : avg(r10),
        "mrr"             : avg(mrr_scores),
        "ndcg@10"         : avg(ndcg10),
        "total_evaluated" : total,
    }

    print(f"[{label}] Triplet accuracy : {accuracy:.4f} ({triplet_correct}/{total})")
    print(f"[{label}] Recall@1         : {result['recall@1']}")
    print(f"[{label}] Recall@5         : {result['recall@5']}")
    print(f"[{label}] Recall@10        : {result['recall@10']}")
    print(f"[{label}] MRR              : {result['mrr']}")
    print(f"[{label}] nDCG@10          : {result['ndcg@10']}")
    return result

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== HamsAI Embedding Fine-tuning ===\n")

    config_path = './configs/training_config.yaml'
    if not Path(config_path).exists():
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Evaluate BEFORE fine-tuning
    print("--- Baseline evaluation (before fine-tuning) ---")
    base_model = SentenceTransformer(config['model']['base_model'])
    before_results = evaluate_retrieval(
        base_model,
        config['data']['test_file'],
        label="before_finetuning"
    )

    # Save before results
    import json as _json
    before_path = './results/before_finetuning_metrics.json'
    with open(before_path, 'w') as f:
        _json.dump(before_results, f, indent=2)
    print(f"Saved before metrics to {before_path}")

    # Fine-tune
    print("\n--- Fine-tuning ---")
    finetuned_model = train(config)

    # Evaluate AFTER fine-tuning
    print("\n--- Evaluation after fine-tuning ---")
    after_results = evaluate_retrieval(
        finetuned_model,
        config['data']['test_file'],
        label="after_finetuning"
    )

    # Save after results
    after_path = './results/after_finetuning_metrics.json'
    with open(after_path, 'w') as f:
        _json.dump(after_results, f, indent=2)
    print(f"Saved after metrics to {after_path}")

    # Print comparison
    print("\n--- Before vs After Fine-tuning ---")
    print(f"Before accuracy : {before_results.get('accuracy', 0):.4f}")
    print(f"After accuracy  : {after_results.get('accuracy', 0):.4f}")
    print(f"Improvement     : {after_results.get('accuracy', 0) - before_results.get('accuracy', 0):.4f}")

    print("\n=== train_embeddings.py complete ===")