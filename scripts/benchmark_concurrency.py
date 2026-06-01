"""
benchmark_concurrency.py
Tests the API under concurrent load at 1, 2, 4, 8, and 16 concurrent users.
Reports average latency, p95 latency, throughput (QPS), and failure rate per mode.
Requires the FastAPI server to be running at the configured host:port.
Usage:
    python scripts/benchmark_concurrency.py
"""

import json
import sys
import time
import threading
import statistics
import yaml
import os
import subprocess
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: Run pip install httpx")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open("./configs/serving_config.yaml", "r") as f:
        return yaml.safe_load(f)

CONFIG   = load_config()
API_HOST = f"http://{CONFIG['api']['host']}:{CONFIG['api']['port']}"
if CONFIG["api"]["host"] == "0.0.0.0":
    API_HOST = f"http://127.0.0.1:{CONFIG['api']['port']}"

TIMEOUT = 60  # seconds per request

# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

QUICK_QUERIES = [
    "returns policy",
    "سياسة الاسترجاع",
    "SLA premium support",
    "warranty period",
    "installation fees",
    "كم مدة الضمان",
    "subscription cancellation",
    "enterprise pricing",
    "رسوم التركيب",
    "support tiers"
]

SMART_QUERIES = [
    "What is the SLA for premium support tickets?",
    "ما هي سياسة الاسترجاع للطلبات المتأخرة؟",
    "How much does installation cost for large furniture items?",
    "كم مدة الضمان على غرف النوم؟",
    "What are the cancellation fees for enterprise subscriptions?",
    "أبغى أعرف الـ pricing للباقة المؤسسية",
    "What are the infrastructure requirements for HamsAI Core CRM?",
    "ما هي خطط الدعم الفني المتاحة؟",
    "How do I integrate using OAuth2?",
    "What is the data retention policy?"
]

INTERACTIVE_QUERIES = [
    ("new",   "What support plans are available?"),
    (None,    "What is the price for the premium plan?"),
    (None,    "How do I upgrade my plan?"),
    ("new",   "ما هي سياسة الضمان؟"),
    (None,    "وكم مدة الضمان للأجهزة؟"),
]

# ---------------------------------------------------------------------------
# Request runners
# ---------------------------------------------------------------------------

def send_quick(client: httpx.Client, query: str) -> dict:
    t0   = time.perf_counter()
    resp = client.post(f"{API_HOST}/search/quick", json={"query": query, "top_k": 5}, timeout=TIMEOUT)
    ms   = (time.perf_counter() - t0) * 1000
    return {"latency_ms": ms, "status": resp.status_code, "ok": resp.status_code == 200}

def send_smart(client: httpx.Client, query: str) -> dict:
    t0   = time.perf_counter()
    resp = client.post(f"{API_HOST}/search/smart", json={"query": query, "top_k": 3}, timeout=TIMEOUT)
    ms   = (time.perf_counter() - t0) * 1000
    return {"latency_ms": ms, "status": resp.status_code, "ok": resp.status_code == 200}

def send_interactive(client: httpx.Client, query: str, session_id: str) -> dict:
    t0   = time.perf_counter()
    resp = client.post(f"{API_HOST}/search/interactive",
                       json={"query": query, "session_id": session_id or "new"},
                       timeout=TIMEOUT)
    ms   = (time.perf_counter() - t0) * 1000
    return {"latency_ms": ms, "status": resp.status_code, "ok": resp.status_code == 200}

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(values: list) -> dict:
    if not values:
        return {"avg": 0, "p50": 0, "p95": 0, "max": 0, "n": 0}
    s       = sorted(values)
    n       = len(s)
    p95_idx = max(0, int(n * 0.95) - 1)
    p50_idx = max(0, int(n * 0.50) - 1)
    return {
        "avg": round(statistics.mean(s), 2),
        "p50": round(s[p50_idx], 2),
        "p95": round(s[p95_idx], 2),
        "max": round(s[-1], 2),
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
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
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
# Concurrency runner
# ---------------------------------------------------------------------------

def run_concurrent(send_fn, queries: list, n_users: int, n_requests: int = 10) -> dict:
    results_lock = threading.Lock()
    all_results  = []
    errors       = []

    def worker(worker_queries):
        with httpx.Client() as client:
            for q in worker_queries:
                try:
                    r = send_fn(client, q)
                    with results_lock:
                        all_results.append(r)
                except Exception as e:
                    with results_lock:
                        errors.append(str(e))

    # Distribute queries across workers
    queries_per_worker = []
    extended = (queries * ((n_requests * n_users // len(queries)) + 1))[:n_requests * n_users]
    chunk    = len(extended) // n_users
    for i in range(n_users):
        queries_per_worker.append(extended[i * chunk:(i + 1) * chunk])

    threads = [threading.Thread(target=worker, args=(queries_per_worker[i],))
               for i in range(n_users)]

    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_start

    latencies  = [r["latency_ms"] for r in all_results if r["ok"]]
    failed     = len(errors) + sum(1 for r in all_results if not r["ok"])
    total_reqs = len(all_results) + len(errors)
    qps        = len(all_results) / elapsed if elapsed > 0 else 0

    return {
        "n_users"     : n_users,
        "total_requests": total_reqs,
        "successful"  : len(latencies),
        "failed"      : failed,
        "failure_rate": round(failed / total_reqs, 4) if total_reqs else 0,
        "qps"         : round(qps, 2),
        "latency"     : compute_stats(latencies),
        "memory"      : memory_snapshot(),
        "elapsed_s"   : round(elapsed, 2)
    }

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_health() -> bool:
    try:
        resp = httpx.get(f"{API_HOST}/health", timeout=10)
        data = resp.json()
        print(f"  API status      : {data.get('status')}")
        print(f"  Elasticsearch   : {data.get('elasticsearch')}")
        print(f"  Qdrant          : {data.get('qdrant')}")
        print(f"  Redis           : {data.get('redis')}")
        print(f"  Embedding model : {data.get('embedding_model')}")
        return resp.status_code == 200
    except Exception as e:
        print(f"  ERROR: Cannot reach API at {API_HOST}: {e}")
        return False

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== HamsAI Concurrency Benchmark ===\n")
    print(f"API endpoint: {API_HOST}\n")

    print("--- Health Check ---")
    if not check_health():
        print(f"\nERROR: API is not running. Start it with:\n  python demo/app.py")
        sys.exit(1)

    concurrency_levels = [1, 2, 4, 8, 16]
    n_requests         = 5   # requests per user per level

    report = {
        "api_endpoint"  : API_HOST,
        "n_requests_per_user": n_requests,
        "modes"         : {}
    }

    # --- Quick Search ---
    print("\n--- Quick Search Concurrency ---")
    qs_results = {}
    for n in concurrency_levels:
        print(f"  Users: {n}")
        result = run_concurrent(
            lambda client, q: send_quick(client, q),
            QUICK_QUERIES,
            n_users=n,
            n_requests=n_requests
        )
        qs_results[f"users_{n}"] = result
        print(f"    QPS={result['qps']:.1f} | p95={result['latency']['p95']:.0f}ms | failures={result['failed']}")
    report["modes"]["quick_search"] = qs_results

    # --- Smart AI Search ---
    print("\n--- Smart AI Search Concurrency ---")
    smart_results = {}
    for n in concurrency_levels:
        print(f"  Users: {n}")
        result = run_concurrent(
            lambda client, q: send_smart(client, q),
            SMART_QUERIES,
            n_users=n,
            n_requests=n_requests
        )
        smart_results[f"users_{n}"] = result
        print(f"    QPS={result['qps']:.1f} | p95={result['latency']['p95']:.0f}ms | failures={result['failed']}")
    report["modes"]["smart_ai_search"] = smart_results

    # --- Interactive AI Search ---
    print("\n--- Interactive AI Search Concurrency ---")
    session_counter = [0]
    session_lock    = threading.Lock()

    def send_interactive_new(client, q):
        with session_lock:
            session_counter[0] += 1
            sid = f"bench_sess_{session_counter[0]}"
        return send_interactive(client, q, sid)

    interactive_results = {}
    for n in concurrency_levels:
        print(f"  Users: {n}")
        result = run_concurrent(
            send_interactive_new,
            [q for _, q in INTERACTIVE_QUERIES],
            n_users=n,
            n_requests=n_requests
        )
        interactive_results[f"users_{n}"] = result
        print(f"    QPS={result['qps']:.1f} | p95={result['latency']['p95']:.0f}ms | failures={result['failed']}")
    report["modes"]["interactive"] = interactive_results

    # Save
    out_path = "./results/concurrency_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nConcurrency report saved to {out_path}")

    # Summary table
    print("\n--- Summary Table ---")
    header = f"{'Mode':<22} {'Users':>6} {'QPS':>8} {'p95 ms':>10} {'Failures':>10}"
    print(header)
    print("-" * len(header))
    for mode_name, mode_data in report["modes"].items():
        for level_key, result in mode_data.items():
            print(
                f"{mode_name:<22} "
                f"{result['n_users']:>6} "
                f"{result['qps']:>8.2f} "
                f"{result['latency']['p95']:>10.0f} "
                f"{result['failed']:>10}"
            )

    print("\n=== benchmark_concurrency.py complete ===")

if __name__ == "__main__":
    main()
