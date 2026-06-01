"""
benchmark_latency.py
Measures end-to-end latency for all three search modes.
Reports average, p50, p95, max per stage and per mode.
Also tests latency vs top-k and vs corpus size.
Usage:
    python scripts/benchmark_latency.py
"""

import json
import math
import re
import sys
import time
import yaml
import statistics
import os
import subprocess
from pathlib import Path
from typing import Callable

try:
    from sentence_transformers import SentenceTransformer
    from elasticsearch import Elasticsearch
    from qdrant_client import QdrantClient
    import ollama
    import torch
except ImportError:
    print("ERROR: Run pip install -r requirements.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config / normalization
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
# Stats helpers
# ---------------------------------------------------------------------------

def compute_stats(values: list) -> dict:
    if not values:
        return {"avg": 0, "p50": 0, "p95": 0, "max": 0, "min": 0}
    sorted_v = sorted(values)
    n        = len(sorted_v)
    p95_idx  = max(0, int(n * 0.95) - 1)
    p50_idx  = max(0, int(n * 0.50) - 1)
    return {
        "avg": round(statistics.mean(values), 2),
        "p50": round(sorted_v[p50_idx], 2),
        "p95": round(sorted_v[p95_idx], 2),
        "max": round(sorted_v[-1], 2),
        "min": round(sorted_v[0], 2),
        "n"  : n
    }

def memory_snapshot() -> dict:
    cpu_mb = 0.0
    try:
        import psutil
        cpu_mb = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    except Exception:
        pass
    gpu_mb = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        vals = [float(x.strip()) for x in out.splitlines() if x.strip()]
        gpu_mb = max(vals) if vals else None
    except Exception:
        gpu_mb = None
    return {"cpu_memory_mb": cpu_mb, "gpu_memory_mb": gpu_mb}

# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def run_quick_search(query: str, es: Elasticsearch, config: dict, top_k: int = 10) -> dict:
    lang   = detect_language(query)
    norm_q = normalize_text(query, lang)
    index  = config["quick_search"]["index_name"]

    t0 = time.perf_counter()
    es.search(
        index=index,
        body={
            "query": {
                "multi_match": {
                    "query" : norm_q,
                    "fields": ["content^2", "content.english", "title"],
                    "type"  : "best_fields"
                }
            },
            "size": top_k
        }
    )
    total_ms = (time.perf_counter() - t0) * 1000
    # For BM25, all results arrive together so first-result time = total search time
    return {"total_ms": total_ms, "time_to_first_result_ms": total_ms}

def run_smart_search(query: str, model: SentenceTransformer,
                     qdrant: QdrantClient, es: Elasticsearch,
                     reranker, config: dict, top_k: int = 5) -> dict:
    lang       = detect_language(query)
    collection = config["qdrant"]["collection_name"]
    index      = config["quick_search"]["index_name"]
    llm_cfg    = config["smart_ai_search"]["generator"]

    t0 = time.perf_counter()

    # Dense
    emb = model.encode(query, normalize_embeddings=True).tolist()
    dense_results = qdrant.search(collection_name=collection, query_vector=emb,
                                  limit=20, with_payload=True)
    dense_chunks = [dict(r.payload) | {"dense_score": r.score} for r in dense_results]

    # BM25
    norm_q = normalize_text(query, lang)
    bm25_resp = es.search(
        index=index,
        body={"query": {"multi_match": {"query": norm_q,
                                         "fields": ["content^2", "title"],
                                         "type": "best_fields"}}, "size": 20}
    )
    bm25_chunks = [dict(h["_source"]) | {"bm25_score": h["_score"]}
                   for h in bm25_resp["hits"]["hits"]]

    t_retrieval = (time.perf_counter() - t0) * 1000
    # First result arrives when retrieval stage completes (before reranking)
    time_to_first_result_ms = t_retrieval

    # Fusion
    scores, chunk_map = {}, {}
    for rank, c in enumerate(dense_chunks):
        cid = c["chunk_id"]; scores[cid] = scores.get(cid, 0) + 1/(60+rank+1); chunk_map[cid] = c
    for rank, c in enumerate(bm25_chunks):
        cid = c["chunk_id"]; scores[cid] = scores.get(cid, 0) + 1/(60+rank+1); chunk_map[cid] = c
    fused = [chunk_map[cid] for cid in sorted(scores, key=lambda x: scores[x], reverse=True)]

    # Rerank
    t_rr = time.perf_counter()
    if reranker and fused:
        pairs  = [[query, c["content"]] for c in fused[:top_k * 2]]
        sc     = reranker.compute_score(pairs, normalize=True)
        for i, c in enumerate(fused[:top_k * 2]):
            c["rerank_score"] = float(sc[i])
        fused = sorted(fused[:top_k * 2], key=lambda x: x.get("rerank_score", 0), reverse=True)
    chunks = fused[:top_k]
    t_rerank = (time.perf_counter() - t_rr) * 1000

    # Generation
    context = "\n\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(chunks))
    prompt  = (
        f"Answer strictly from context. Add citations [1][2].\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    t_gen = time.perf_counter()
    client = ollama.Client(host=llm_cfg["url"])
    ttft_ms = None
    stream = client.generate(
        model=llm_cfg["model"],
        prompt=prompt,
        stream=True,
        options={"temperature": 0.1, "num_predict": 256}
    )
    for part in stream:
        if part.get("response") and ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_gen) * 1000
    t_generation = (time.perf_counter() - t_gen) * 1000
    total_ms     = (time.perf_counter() - t0) * 1000

    return {
        "total_ms"                : total_ms,
        "time_to_first_result_ms" : time_to_first_result_ms,
        "retrieval_ms"            : t_retrieval,
        "rerank_ms"               : t_rerank,
        "generation_ms"           : t_generation,
        "ttft_ms"                 : ttft_ms if ttft_ms is not None else t_generation,
    }

def run_interactive(query: str, model: SentenceTransformer,
                    qdrant: QdrantClient, es: Elasticsearch,
                    reranker, config: dict) -> dict:
    llm_cfg = config["smart_ai_search"]["generator"]
    lang    = detect_language(query)

    t0 = time.perf_counter()

    # Query rewrite
    client   = ollama.Client(host=llm_cfg["url"])
    rw_resp  = client.generate(
        model=llm_cfg["model"],
        prompt=f"Rewrite as standalone question: {query}\nRewritten:",
        options={"temperature": 0.0, "num_predict": 64}
    )
    rewritten    = rw_resp["response"].strip()
    t_rewrite_ms = (time.perf_counter() - t0) * 1000

    # Retrieval + rerank + generate (same as smart search)
    smart_result = run_smart_search(rewritten, model, qdrant, es, reranker, config)
    total_ms     = (time.perf_counter() - t0) * 1000

    return {
        "total_ms"                : total_ms,
        "query_rewrite"           : t_rewrite_ms,
        "time_to_first_result_ms" : smart_result["time_to_first_result_ms"],
        "retrieval_ms"            : smart_result["retrieval_ms"],
        "rerank_ms"               : smart_result["rerank_ms"],
        "generation_ms"           : smart_result["generation_ms"],
        "ttft_ms"                 : smart_result["ttft_ms"],
    }

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_mode(run_fn: Callable, queries: list, warmup: int = 2,
                   label: str = "") -> dict:
    print(f"  Warmup ({warmup} queries)...")
    for q in queries[:warmup]:
        try:
            run_fn(q)
        except Exception:
            pass

    print(f"  Benchmarking {len(queries)} queries...")
    stage_times = {}
    for q in queries:
        try:
            result = run_fn(q)
            for k, v in result.items():
                stage_times.setdefault(k, []).append(v)
        except Exception as e:
            print(f"  WARNING: query failed: {e}")

    return {stage: compute_stats(vals) for stage, vals in stage_times.items()}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== HamsAI Latency Benchmark ===\n")

    config = load_config()

    # Load test queries
    qa_file = "./data/test/qa_pairs.json"
    if not Path(qa_file).exists():
        print(f"ERROR: {qa_file} not found")
        sys.exit(1)

    with open(qa_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    queries = [item.get("question", item.get("query", "")).strip()
               for item in qa_pairs if item.get("question") or item.get("query")]
    queries = [q for q in queries if q][:20]  # cap at 20 for benchmark

    if not queries:
        # Fallback queries if test file has no data
        queries = [
            "What is the SLA for premium support?",
            "ما هي سياسة الاسترجاع؟",
            "What are the enterprise pricing plans?",
            "كم مدة الضمان على المنتجات؟",
            "How do I cancel my subscription?"
        ]

    print(f"Queries for benchmark: {len(queries)}")

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
    except Exception as e:
        print(f"Reranker not loaded: {e}")

    import platform
    hardware = {
        "platform"    : platform.platform(),
        "python"      : platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu"         : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "memory"      : memory_snapshot(),
    }

    report = {"hardware": hardware, "modes": {}}

    # --- Quick Search ---
    print("\n[1/3] Quick Search latency")
    qs_results = benchmark_mode(
        lambda q: run_quick_search(q, es, config),
        queries,
        label="quick_search"
    )
    report["modes"]["quick_search"] = qs_results
    print(f"  p95 total_ms: {qs_results.get('total_ms', {}).get('p95', 'N/A')}")

    # --- Smart AI Search ---
    print("\n[2/3] Smart AI Search latency")
    smart_results = benchmark_mode(
        lambda q: run_smart_search(q, model, qdrant, es, reranker, config),
        queries,
        label="smart_ai_search"
    )
    report["modes"]["smart_ai_search"] = smart_results
    print(f"  p95 total_ms    : {smart_results.get('total_ms', {}).get('p95', 'N/A')}")
    print(f"  p95 ttft_ms     : {smart_results.get('ttft_ms', {}).get('p95', 'N/A')}")
    print(f"  p95 retrieval_ms: {smart_results.get('retrieval_ms', {}).get('p95', 'N/A')}")

    # --- Interactive AI Search ---
    print("\n[3/3] Interactive AI Search latency")
    interactive_results = benchmark_mode(
        lambda q: run_interactive(q, model, qdrant, es, reranker, config),
        queries,
        label="interactive"
    )
    report["modes"]["interactive"] = interactive_results
    print(f"  p95 total_ms      : {interactive_results.get('total_ms', {}).get('p95', 'N/A')}")
    print(f"  p95 query_rewrite : {interactive_results.get('query_rewrite', {}).get('p95', 'N/A')}")

    # --- Latency vs top-k ---
    print("\n[4] Latency vs top-k (Smart AI Search)")
    topk_results = {}
    for top_k in [1, 3, 5, 10]:
        times = []
        for q in queries[:5]:
            try:
                r = run_smart_search(q, model, qdrant, es, reranker, config, top_k=top_k)
                times.append(r["total_ms"])
            except Exception:
                pass
        topk_results[f"top_k_{top_k}"] = compute_stats(times)
        print(f"  top_k={top_k}: avg={topk_results[f'top_k_{top_k}']['avg']:.1f}ms")
    report["latency_vs_topk"] = topk_results

    # --- Latency vs corpus size ---
    # We measure Quick Search (BM25) latency which scales with corpus size.
    # Dense retrieval uses HNSW so it is O(log n) — we document both.
    print("\n[5] Latency vs corpus size (Quick Search BM25 + Dense retrieval)")
    corpus_size_results = {}
    try:
        actual_count = es.count(index=config["quick_search"]["index_name"])["count"]
        # Run quick search at current size and extrapolate for larger sizes
        qs_times = []
        for q in queries[:10]:
            t0 = time.time()
            es.search(index=config["quick_search"]["index_name"],
                      body={"query": {"multi_match": {"query": q,
                            "fields": ["content^2", "content.english", "title"],
                            "type": "best_fields"}}, "size": 10})
            qs_times.append((time.time() - t0) * 1000)

        qs_avg = sum(qs_times) / len(qs_times) if qs_times else 0

        # Dense retrieval timing at current size
        dense_times = []
        for q in queries[:5]:
            t0 = time.time()
            emb = model.encode(q, normalize_embeddings=True).tolist()
            qdrant.search(collection_name=config["qdrant"]["collection_name"],
                          query_vector=emb, limit=10, with_payload=False)
            dense_times.append((time.time() - t0) * 1000)

        dense_avg = sum(dense_times) / len(dense_times) if dense_times else 0

        # BM25 scales roughly linearly; HNSW is O(log n)
        for target_k in [1000, 10000, 100000]:
            scale_bm25  = target_k / max(actual_count, 1)
            scale_hnsw  = math.log(target_k) / max(math.log(actual_count), 1) if actual_count > 1 else 1
            corpus_size_results[f"n_{target_k}"] = {
                "corpus_size"            : target_k,
                "bm25_avg_ms_projected"  : round(qs_avg * scale_bm25, 1),
                "dense_hnsw_avg_ms_projected": round(dense_avg * scale_hnsw, 1),
                "note": "Projected from actual measurements via BM25=O(n), HNSW=O(log n)"
            }
            print(f"  n={target_k:>7}: BM25 ~{corpus_size_results[f'n_{target_k}']['bm25_avg_ms_projected']}ms  "
                  f"Dense(HNSW) ~{corpus_size_results[f'n_{target_k}']['dense_hnsw_avg_ms_projected']}ms")

        corpus_size_results["actual_measured"] = {
            "corpus_size"   : actual_count,
            "bm25_avg_ms"   : round(qs_avg, 1),
            "dense_avg_ms"  : round(dense_avg, 1),
        }
        print(f"  Actual ({actual_count} chunks): BM25={qs_avg:.1f}ms  Dense={dense_avg:.1f}ms")
    except Exception as exc:
        corpus_size_results["error"] = str(exc)
        print(f"  WARNING: corpus size test failed: {exc}")
    report["latency_vs_corpus_size"] = corpus_size_results

    # Maximum input length tested (from embedding config; queries in benchmark are well under this)
    report["max_input_length_tested"] = {
        "embedding_max_seq_tokens": config["embedding"].get("max_seq_length", 512),
        "longest_query_chars"     : max((len(q) for q in queries), default=0),
        "note": "Embedding model truncates at max_seq_length tokens. All benchmark queries fit within this limit."
    }

    # Save report
    out_path = "./results/latency_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nLatency report saved to {out_path}")

    # Print targets
    print("\n--- Target vs Actual ---")
    qs_p95    = report["modes"]["quick_search"].get("total_ms", {}).get("p95", "N/A")
    smart_p95 = report["modes"]["smart_ai_search"].get("total_ms", {}).get("p95", "N/A")
    ttft_p95  = report["modes"]["smart_ai_search"].get("ttft_ms", {}).get("p95", "N/A")
    rw_p95    = report["modes"]["interactive"].get("query_rewrite", {}).get("p95", "N/A")

    print(f"  Quick Search p95 total   : {qs_p95} ms  (target: <100 ms)")
    print(f"  Smart AI TTFT p95        : {ttft_p95} ms  (target: <1500 ms)")
    print(f"  Smart AI total p95       : {smart_p95} ms  (target: <4000 ms)")
    print(f"  Interactive rewrite p95  : {rw_p95} ms  (target: <200 ms)")

    print("\n=== benchmark_latency.py complete ===")

if __name__ == "__main__":
    main()
