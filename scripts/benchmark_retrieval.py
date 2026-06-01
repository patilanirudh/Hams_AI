"""
benchmark_retrieval.py
Evaluates retrieval quality on the test set.
Reports Recall@k, MRR, nDCG@10, Hit Rate@k separately per language and mode.
Usage:
    python scripts/benchmark_retrieval.py
"""

import json
import random
import re
import sys
import time
import math
import yaml
from pathlib import Path
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    from elasticsearch import Elasticsearch
    from qdrant_client import QdrantClient
except ImportError:
    print("ERROR: Run pip install -r requirements.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config and normalization
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open("./configs/serving_config.yaml", "r") as f:
        return yaml.safe_load(f)

def normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ـ", "", text)
    for e, w in zip("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"):
        text = text.replace(e, w)
    return text.strip()

def normalize_text(text: str, lang: str) -> str:
    if lang in ("ar", "mixed"):
        text = normalize_arabic(text)
    return re.sub(r"\s+", " ", text).strip()

def detect_language(text: str) -> str:
    ar = len(re.findall(r"[\u0600-\u06FF]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    total = ar + en
    if total == 0:
        return "en"
    r = ar / total
    if r > 0.7:
        return "ar"
    if r < 0.3:
        return "en"
    return "mixed"

# ---------------------------------------------------------------------------
# Retrieval methods
# ---------------------------------------------------------------------------

def retrieve_bm25(query: str, lang: str, es: Elasticsearch, index: str, top_k: int) -> list:
    norm_q = normalize_text(query, lang)
    try:
        resp = es.search(
            index=index,
            body={
                "query": {
                    "multi_match": {
                        "query": norm_q,
                        "fields": ["content^2", "content.english", "title"],
                        "type": "best_fields"
                    }
                },
                "size": top_k
            }
        )
        return [hit["_source"]["chunk_id"] for hit in resp["hits"]["hits"]]
    except Exception as e:
        print(f"WARNING: BM25 query failed: {e}")
        return []

def retrieve_dense(query: str, model: SentenceTransformer, qdrant: QdrantClient,
                   collection: str, top_k: int) -> list:
    emb = model.encode(query, normalize_embeddings=True).tolist()
    results = qdrant.search(collection_name=collection, query_vector=emb, limit=top_k, with_payload=True)
    return [r.payload["chunk_id"] for r in results]

def rrf_fusion(dense_ids: list, bm25_ids: list, k: int = 60) -> list:
    scores = {}
    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    for rank, cid in enumerate(bm25_ids):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)

def retrieve_hybrid(query: str, lang: str, model: SentenceTransformer,
                    qdrant: QdrantClient, es: Elasticsearch,
                    collection: str, index: str, top_k: int) -> list:
    dense = retrieve_dense(query, model, qdrant, collection, top_k)
    bm25  = retrieve_bm25(query, lang, es, index, top_k)
    return rrf_fusion(dense, bm25)[:top_k]

RERANK_CANDIDATES = 5  # match app's CPU-safe limit

def retrieve_hybrid_rerank(query: str, lang: str, model: SentenceTransformer,
                           qdrant: QdrantClient, es: Elasticsearch,
                           collection: str, index: str, reranker, top_k: int) -> list:
    fused_ids = retrieve_hybrid(query, lang, model, qdrant, es, collection, index, top_k * 2)
    if reranker is None:
        return fused_ids[:top_k]

    # Fetch content via large search so chunk_map covers all fused_ids
    emb = model.encode(query, normalize_embeddings=True).tolist()
    results = qdrant.search(
        collection_name=collection,
        query_vector=emb,
        limit=top_k * 2,  # wide enough to cover all RRF fused results
        with_payload=True
    )
    chunk_map = {r.payload["chunk_id"]: r.payload["content"] for r in results}

    # Rerank only top RERANK_CANDIDATES fused IDs
    pairs = []
    valid_ids = []
    for cid in fused_ids[:RERANK_CANDIDATES]:
        if cid in chunk_map:
            pairs.append([query, chunk_map[cid]])
            valid_ids.append(cid)

    if not pairs:
        return fused_ids[:top_k]

    scores = reranker.compute_score(pairs, normalize=True)
    ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    reranked_ids = [cid for cid, _ in ranked]
    # fill remaining slots with non-reranked fused results
    reranked_set = set(reranked_ids)
    remaining = [cid for cid in fused_ids if cid not in reranked_set]
    return (reranked_ids + remaining)[:top_k]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    retrieved_k = set(retrieved[:k])
    rel_set     = set(relevant)
    if not rel_set:
        return 0.0
    return len(retrieved_k & rel_set) / len(rel_set)

def hit_rate_at_k(retrieved: list, relevant: list, k: int) -> float:
    retrieved_k = set(retrieved[:k])
    return 1.0 if set(relevant) & retrieved_k else 0.0

def mrr(retrieved: list, relevant: list) -> float:
    rel_set = set(relevant)
    for rank, cid in enumerate(retrieved):
        if cid in rel_set:
            return 1.0 / (rank + 1)
    return 0.0

def ndcg_at_k(retrieved: list, relevant: list, k: int) -> float:
    rel_set = set(relevant)
    dcg     = 0.0
    for i, cid in enumerate(retrieved[:k]):
        if cid in rel_set:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_mode(
    test_pairs: list,
    retrieve_fn,
    mode_name: str,
    ks: list = [1, 5, 10, 20]
) -> dict:
    per_language = defaultdict(lambda: defaultdict(list))
    all_metrics  = defaultdict(list)

    for item in test_pairs:
        query    = item.get("query", "").strip()
        positive = item.get("positive", "").strip()
        lang     = item.get("query_lang", detect_language(query))
        q_type   = item.get("type", "monolingual")

        if not query or not positive:
            continue

        # Derive positive chunk_id from item if available, else use content match
        positive_id = item.get("positive_chunk_id", positive[:60])

        t0        = time.time()
        retrieved = retrieve_fn(query, lang)
        latency   = (time.time() - t0) * 1000

        relevant = [positive_id]

        row = {"latency_ms": latency}
        for k in ks:
            row[f"recall@{k}"]   = recall_at_k(retrieved, relevant, k)
            row[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        row["mrr"]     = mrr(retrieved, relevant)
        row["ndcg@10"] = ndcg_at_k(retrieved, relevant, 10)

        for metric, val in row.items():
            all_metrics[metric].append(val)
            per_language[lang][metric].append(val)

        if q_type == "cross_lingual":
            for metric, val in row.items():
                per_language["cross_lingual"][metric].append(val)

    def avg(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    results = {
        "mode"    : mode_name,
        "overall" : {m: avg(v) for m, v in all_metrics.items()},
        "per_language": {
            lang: {m: avg(v) for m, v in metrics.items()}
            for lang, metrics in per_language.items()
        }
    }
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== HamsAI Retrieval Benchmark ===\n")

    config = load_config()

    test_file = "./data/test/retrieval_pairs.json"
    if not Path(test_file).exists():
        test_file = "./data/test/qa_pairs.json"
    if not Path(test_file).exists():
        print(f"ERROR: Test file not found at {test_file}")
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)

    print(f"Test pairs loaded: {len(test_pairs)}")

    # hybrid_rerank is slow on CPU (~16s/query). Use 50 stratified pairs for the
    # baseline run to keep total time ~13 min. All other modes use all 200 pairs.
    random.seed(42)
    rerank_pairs = random.sample(test_pairs, min(50, len(test_pairs)))
    print(f"Rerank pairs (baseline only): {len(rerank_pairs)} (stratified sample, seed=42)")

    # Clients
    es     = Elasticsearch(config["quick_search"]["url"])
    qdrant = QdrantClient(url=config["qdrant"]["url"])

    model_path = config["embedding"]["model_path"]
    if not Path(model_path).exists():
        if config["embedding"].get("allow_base_model_fallback", False):
            model_path = config["embedding"]["fallback_model"]
        else:
            print(f"ERROR: Fine-tuned embedding model not found at {model_path}")
            sys.exit(1)
    model = SentenceTransformer(model_path)

    reranker = None
    try:
        from FlagEmbedding import FlagReranker
        import torch
        reranker_path = config["smart_ai_search"]["reranker"]["model_path"]
        if not Path(reranker_path).exists():
            if config["smart_ai_search"]["reranker"].get("allow_base_model_fallback", False):
                reranker_path = config["smart_ai_search"]["reranker"].get("fallback_model", reranker_path)
            else:
                raise FileNotFoundError(f"Fine-tuned reranker not found at {reranker_path}")
        reranker = FlagReranker(
            reranker_path,
            use_fp16=torch.cuda.is_available()
        )
        print("Reranker loaded.")
    except Exception as e:
        print(f"Reranker not loaded: {e}")

    index      = config["quick_search"]["index_name"]
    collection = config["qdrant"]["collection_name"]
    TOP_K      = 20

    modes = {
        "bm25_only": lambda q, lang: retrieve_bm25(q, lang, es, index, TOP_K),
        "dense_only": lambda q, lang: retrieve_dense(q, model, qdrant, collection, TOP_K),
        "hybrid": lambda q, lang: retrieve_hybrid(q, lang, model, qdrant, es, collection, index, TOP_K),
        "hybrid_rerank": lambda q, lang: retrieve_hybrid_rerank(q, lang, model, qdrant, es, collection, index, reranker, TOP_K),
    }

    all_results = {}
    for mode_name, fn in modes.items():
        pairs = rerank_pairs if mode_name == "hybrid_rerank" else test_pairs
        print(f"\nEvaluating mode: {mode_name}  (n={len(pairs)})")
        result = evaluate_mode(pairs, fn, mode_name)
        if mode_name == "hybrid_rerank":
            result["eval_note"] = f"Evaluated on {len(pairs)} stratified pairs (seed=42) for CPU speed. Full 200-pair run available on GPU."
        all_results[mode_name] = result
        print(f"  Overall Recall@5  : {result['overall'].get('recall@5', 0):.4f}")
        print(f"  Overall MRR       : {result['overall'].get('mrr', 0):.4f}")
        print(f"  Overall nDCG@10   : {result['overall'].get('ndcg@10', 0):.4f}")

    # Save
    out_path = "./results/retrieval_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nRetrieval report saved to {out_path}")

    # Print comparison table
    print("\n--- Mode Comparison (Overall) ---")
    header = f"{'Mode':<20} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>8} {'nDCG@10':>10}"
    print(header)
    print("-" * len(header))
    for mode_name, result in all_results.items():
        o = result["overall"]
        print(
            f"{mode_name:<20} "
            f"{o.get('recall@1', 0):>6.4f} "
            f"{o.get('recall@5', 0):>6.4f} "
            f"{o.get('recall@10', 0):>6.4f} "
            f"{o.get('mrr', 0):>8.4f} "
            f"{o.get('ndcg@10', 0):>10.4f}"
        )

    print("\n=== benchmark_retrieval.py complete ===")

if __name__ == "__main__":
    main()
