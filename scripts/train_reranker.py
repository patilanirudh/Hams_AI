import json
import os
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

# ─── Dependencies ─────────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
    import torch
except ImportError:
    print("ERROR: Required packages not installed. Run: pip install sentence-transformers torch")
    sys.exit(1)

# ─── Load Config ──────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ─── Load Training Data ───────────────────────────────────────────────────────

def load_reranker_pairs(train_file: str) -> tuple[list, list]:
    with open(train_file, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    positive_examples = []
    negative_examples = []
    skipped = 0

    for item in pairs:
        query    = item.get('query', '').strip()
        positive = item.get('positive', '').strip()
        negative = item.get('hard_negative', '').strip()

        if not query or not positive or not negative:
            skipped += 1
            continue

        # Positive pair — label 1
        positive_examples.append(InputExample(
            texts=[query, positive],
            label=1.0
        ))

        # Hard negative pair — label 0
        negative_examples.append(InputExample(
            texts=[query, negative],
            label=0.0
        ))

    print(f"Positive pairs  : {len(positive_examples)}")
    print(f"Negative pairs  : {len(negative_examples)}")
    print(f"Skipped         : {skipped}")

    return positive_examples, negative_examples

# ─── Build Evaluator ──────────────────────────────────────────────────────────

def build_reranker_evaluator(val_file: str) -> CERerankingEvaluator | None:
    if not Path(val_file).exists():
        print(f"WARNING: Validation file not found at {val_file}, skipping evaluator.")
        return None

    with open(val_file, 'r', encoding='utf-8') as f:
        val_pairs = json.load(f)

    samples = {}
    for i, item in enumerate(val_pairs):
        query    = item.get('query', '').strip()
        positive = item.get('positive', '').strip()
        negative = item.get('hard_negative', '').strip()

        if not query or not positive or not negative:
            continue

        samples[str(i)] = {
            'query'    : query,
            'positive' : [positive],
            'negative' : [negative]
        }

    if not samples:
        return None

    evaluator = CERerankingEvaluator(samples=samples, name="hamsai_reranker_val")
    print(f"Built reranker evaluator with {len(samples)} samples")
    return evaluator

# ─── Evaluate Reranker ────────────────────────────────────────────────────────

def evaluate_reranker(model: CrossEncoder, test_file: str, label: str) -> dict:
    if not Path(test_file).exists():
        print(f"WARNING: Test file not found at {test_file}, skipping.")
        return {}

    with open(test_file, 'r', encoding='utf-8') as f:
        test_pairs = json.load(f)

    correct = 0
    total   = 0

    for item in tqdm(test_pairs, desc=f"Evaluating reranker [{label}]"):
        query    = item.get('query', '').strip()
        positive = item.get('positive', '').strip()
        negative = item.get('hard_negative', '').strip()

        if not query or not positive or not negative:
            continue

        scores = model.predict([
            [query, positive],
            [query, negative]
        ])

        if scores[0] > scores[1]:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"[{label}] Reranker accuracy (pos > neg): {correct}/{total} = {accuracy:.4f}")

    return {
        "label"    : label,
        "accuracy" : accuracy,
        "correct"  : correct,
        "total"    : total
    }

# ─── Train Reranker ───────────────────────────────────────────────────────────

def train_reranker(config: dict) -> CrossEncoder:
    reranker_cfg = config['reranker']
    data_cfg     = config['data']

    base_model  = reranker_cfg['base_model']
    output_path = reranker_cfg['output_path']
    epochs      = reranker_cfg['epochs']
    batch_size  = reranker_cfg['batch_size']
    lr          = reranker_cfg['learning_rate']

    print(f"Base model      : {base_model}")
    print(f"Output path     : {output_path}")
    print(f"Epochs          : {epochs}")
    print(f"Batch size      : {batch_size}")
    print(f"Learning rate   : {lr}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device          : {device}")

    # Load cross-encoder
    print("\nLoading base reranker model...")
    model = CrossEncoder(
        base_model,
        num_labels=1,
        max_length=512,
        device=device
    )

    # Load training pairs
    print("\nLoading training data...")
    positive_examples, negative_examples = load_reranker_pairs(data_cfg['train_file'])
    all_examples = positive_examples + negative_examples

    if not all_examples:
        print("ERROR: No valid training examples found.")
        sys.exit(1)

    # Evaluator
    val_file  = str(Path(data_cfg['train_file']).parent / 'val_split.json')
    evaluator = build_reranker_evaluator(val_file)

    # Warmup
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        all_examples,
        shuffle=True,
        batch_size=batch_size
    )
    total_steps  = len(train_dataloader) * epochs
    warmup_steps = max(1, total_steps // 10)

    print(f"\nTotal steps     : {total_steps}")
    print(f"Warmup steps    : {warmup_steps}")
    print("\nStarting reranker training...\n")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=output_path,
        show_progress_bar=True
    )

    print(f"\nReranker training complete. Model saved to: {output_path}")
    return model

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== HamsAI Reranker Fine-tuning ===\n")

    config_path = './configs/training_config.yaml'
    if not Path(config_path).exists():
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    test_file = config['data']['test_file']

    # Evaluate BEFORE fine-tuning
    print("--- Baseline evaluation (before fine-tuning) ---")
    base_model = CrossEncoder(
        config['reranker']['base_model'],
        num_labels=1,
        max_length=512
    )
    before_results = evaluate_reranker(
        base_model,
        test_file,
        label="before_finetuning"
    )

    # Save before results
    before_path = './results/before_finetuning_metrics.json'
    existing = {}
    if Path(before_path).exists():
        with open(before_path, 'r') as f:
            existing = json.load(f)
    existing['reranker'] = before_results
    with open(before_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Saved before reranker metrics to {before_path}")

    # Fine-tune
    print("\n--- Fine-tuning reranker ---")
    finetuned_model = train_reranker(config)

    # Evaluate AFTER fine-tuning
    print("\n--- Evaluation after fine-tuning ---")
    after_results = evaluate_reranker(
        finetuned_model,
        test_file,
        label="after_finetuning"
    )

    # Save after results
    after_path = './results/after_finetuning_metrics.json'
    existing = {}
    if Path(after_path).exists():
        with open(after_path, 'r') as f:
            existing = json.load(f)
    existing['reranker'] = after_results
    with open(after_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Saved after reranker metrics to {after_path}")

    # Print comparison
    print("\n--- Before vs After Reranker ---")
    print(f"Before accuracy : {before_results.get('accuracy', 0):.4f}")
    print(f"After accuracy  : {after_results.get('accuracy', 0):.4f}")
    print(f"Improvement     : {after_results.get('accuracy', 0) - before_results.get('accuracy', 0):.4f}")

    print("\n=== train_reranker.py complete ===")