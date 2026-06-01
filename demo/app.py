"""
HamsAI RAG API
Exposes three search modes through a single FastAPI service:
  - /search/quick       Quick Search (BM25 keyword)
  - /search/smart       Smart AI Search (hybrid RAG + LLM)
  - /search/interactive Interactive AI Search (multi-turn conversation)
  - /ingest             Document ingestion endpoint
  - /health             Health check
"""

import json
import os
import re
import sys
import time
import uuid
import yaml
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("hamsai")

# ---------------------------------------------------------------------------
# Lazy imports (fail loudly if packages are missing)
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
    from elasticsearch import Elasticsearch
    from qdrant_client import QdrantClient
    import ollama
    import redis as redis_lib
    import torch
except ImportError as e:
    logger.error("Missing dependency: %s. Run: pip install -r requirements.txt", e)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = str(_PROJECT_ROOT / "configs" / "serving_config.yaml")

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["quick_search"]["url"] = os.getenv("ELASTICSEARCH_URL", cfg["quick_search"]["url"])
    cfg["qdrant"]["url"] = os.getenv("QDRANT_URL", cfg["qdrant"]["url"])
    cfg["interactive_search"]["session"]["url"] = os.getenv("REDIS_URL", cfg["interactive_search"]["session"]["url"])
    cfg["smart_ai_search"]["generator"]["url"] = os.getenv("OLLAMA_URL", cfg["smart_ai_search"]["generator"]["url"])
    cfg["smart_ai_search"]["generator"]["model"] = os.getenv("LLM_MODEL", cfg["smart_ai_search"]["generator"]["model"])
    cfg["embedding"]["model_path"] = os.getenv("EMBEDDING_MODEL_PATH", cfg["embedding"]["model_path"])
    cfg["smart_ai_search"]["reranker"]["model_path"] = os.getenv(
        "RERANKER_MODEL_PATH",
        cfg["smart_ai_search"]["reranker"]["model_path"],
    )
    return cfg

CONFIG = load_config()

# ---------------------------------------------------------------------------
# Arabic / text normalization
# ---------------------------------------------------------------------------

def normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ـ", "", text)
    eastern = "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
    western = "01234567890123456789"
    for e, w in zip(eastern, western):
        text = text.replace(e, w)
    return text.strip()

def normalize_text(text: str, language: str) -> str:
    if language in ("ar", "mixed"):
        text = normalize_arabic(text)
    return re.sub(r"\s+", " ", text).strip()

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    arabic_chars  = len(re.findall(r"[\u0600-\u06FF]", text))
    english_chars = len(re.findall(r"[a-zA-Z]", text))
    total = arabic_chars + english_chars
    if total == 0:
        return "en"
    ratio = arabic_chars / total
    if ratio > 0.7:
        return "ar"
    if ratio < 0.3:
        return "en"
    return "mixed"

# ---------------------------------------------------------------------------
# Singleton clients (initialized on startup)
# ---------------------------------------------------------------------------

class AppState:
    es: Elasticsearch = None
    qdrant: QdrantClient = None
    redis: redis_lib.Redis = None
    embed_model: SentenceTransformer = None
    reranker = None

state = AppState()

def init_clients():
    cfg = CONFIG

    # Elasticsearch
    es_url = cfg["quick_search"]["url"]
    state.es = Elasticsearch(es_url)
    if not state.es.ping():
        logger.warning("Elasticsearch not reachable at %s — Quick Search will fail", es_url)
    else:
        logger.info("Elasticsearch connected: %s", es_url)

    # Qdrant
    qdrant_url = cfg["qdrant"]["url"]
    state.qdrant = QdrantClient(url=qdrant_url)
    logger.info("Qdrant connected: %s", qdrant_url)

    # Redis
    redis_url = cfg["interactive_search"]["session"]["url"]
    state.redis = redis_lib.from_url(redis_url)
    try:
        state.redis.ping()
        logger.info("Redis connected: %s", redis_url)
    except Exception:
        logger.warning("Redis not reachable at %s — Interactive mode will use in-memory sessions", redis_url)
        state.redis = None

    # Embedding model
    model_path = cfg["embedding"]["model_path"]
    abs_model_path = Path(model_path) if Path(model_path).is_absolute() else _PROJECT_ROOT / model_path
    if not abs_model_path.exists():
        if cfg["embedding"].get("allow_base_model_fallback", False):
            model_path = cfg["embedding"]["fallback_model"]
            logger.info("Fine-tuned model not found, using fallback: %s", model_path)
        else:
            raise RuntimeError(
                f"Fine-tuned embedding model not found at {abs_model_path}. "
                "Run scripts/train_embeddings.py or download the HF artifact first."
            )
    else:
        model_path = str(abs_model_path)
        logger.info("Loading fine-tuned embedding model: %s", model_path)
    state.embed_model = SentenceTransformer(model_path)

    # Reranker (optional)
    try:
        from FlagEmbedding import FlagReranker
        reranker_path = cfg["smart_ai_search"]["reranker"]["model_path"]
        abs_reranker_path = Path(reranker_path) if Path(reranker_path).is_absolute() else _PROJECT_ROOT / reranker_path
        if not abs_reranker_path.exists():
            if cfg["smart_ai_search"]["reranker"].get("allow_base_model_fallback", False):
                reranker_path = cfg["smart_ai_search"]["reranker"].get("fallback_model", reranker_path)
            else:
                raise RuntimeError(
                    f"Fine-tuned reranker model not found at {reranker_path}. "
                    "Run scripts/train_reranker.py or download the HF artifact first."
                )
        state.reranker = FlagReranker(reranker_path, use_fp16=torch.cuda.is_available())
        logger.info("Reranker loaded: %s", reranker_path)
    except Exception as e:
        logger.warning("Reranker not loaded (%s) — results will not be reranked", e)
        state.reranker = None

# In-memory session fallback (used when Redis is unavailable)
_memory_sessions: dict = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HamsAI RAG API",
    description="Self-hosted bilingual Arabic/English Retrieval-Augmented Generation system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

_web_dir = Path(__file__).parent / "web"
if _web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_web_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index = Path(__file__).parent / "web" / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h2>HamsAI RAG API is running. See <a href='/docs'>/docs</a> for API.</h2>")

@app.on_event("startup")
def on_startup():
    logger.info("Initializing HamsAI API...")
    init_clients()
    logger.info("HamsAI API ready.")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QuickSearchRequest(BaseModel):
    query: str
    top_k: int = 10

class SmartSearchRequest(BaseModel):
    query: str
    top_k: int = 3

class InteractiveSearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = "new"

class IngestURLRequest(BaseModel):
    url: str
    category: str = "ingested"

# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def dense_retrieve(query: str, top_k: int = 20) -> list:
    embedding = state.embed_model.encode(query, normalize_embeddings=True).tolist()
    results   = state.qdrant.search(
        collection_name=CONFIG["qdrant"]["collection_name"],
        query_vector=embedding,
        limit=top_k,
        with_payload=True
    )
    chunks = []
    for r in results:
        payload = dict(r.payload)
        payload["dense_score"] = r.score
        chunks.append(payload)
    return chunks

def bm25_retrieve(query: str, lang: str, top_k: int = 20) -> list:
    norm_q = normalize_text(query, lang)
    try:
        response = state.es.search(
            index=CONFIG["quick_search"]["index_name"],
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
    except Exception as e:
        logger.warning("Elasticsearch query failed: %s", e)
        return []
    chunks = []
    for hit in response["hits"]["hits"]:
        src = dict(hit["_source"])
        src["bm25_score"] = hit["_score"]
        chunks.append(src)
    return chunks

def rrf_fusion(dense: list, bm25: list, k: int = 60) -> list:
    scores    = {}
    chunk_map = {}
    for rank, chunk in enumerate(dense):
        cid = chunk["chunk_id"]
        scores[cid]    = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk
    for rank, chunk in enumerate(bm25):
        cid = chunk["chunk_id"]
        scores[cid]    = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = []
    for cid in sorted_ids:
        c = chunk_map[cid]
        c["rrf_score"] = round(scores[cid], 6)
        fused.append(c)
    return fused

def rerank_chunks(query: str, chunks: list, top_k: int = 3) -> list:
    if state.reranker is None or not chunks:
        return chunks[:top_k]
    candidates = chunks[:5]  # only rerank top-5 to keep CPU latency manageable
    pairs  = [[query, c["content"]] for c in candidates]
    scores = state.reranker.compute_score(pairs, normalize=True)
    for i, chunk in enumerate(candidates):
        chunk["rerank_score"] = float(scores[i])
    reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
    return reranked[:top_k]

# ---------------------------------------------------------------------------
# LLM generation helpers
# ---------------------------------------------------------------------------

def build_prompt(query: str, chunks: list, lang: str) -> str:
    context = "\n\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(chunks))
    if lang == "ar":
        return (
            "أنت مساعد متخصص في الإجابة على الأسئلة بناءً على المستندات المرفقة فقط.\n"
            "قواعد صارمة — يجب الالتزام بها في كل إجابة:\n"
            "- أجب باللغة العربية فقط\n"
            "- استخدم المعلومات من السياق المرفق فقط، لا تضف معلومات من خارجه\n"
            "- إذا وجدت الإجابة، أجب مباشرة وبإيجاز\n"
            "- إذا لم تجد الإجابة في السياق إطلاقاً، قل فقط: \"لم أجد هذه المعلومات في قاعدة المعرفة\"\n"
            "- لا تجمع بين الإجابة وعبارة عدم الوجود في نفس الرد\n"
            "- ضع رقم المصدر مباشرةً بعد كل جملة تستخدم فيها معلومة، مثال: \"المدة هي 7 أيام [1].\"\n"
            "- إذا استخدمت مصدرين، اذكرهما هكذا: [1][2]\n\n"
            f"المصادر:\n{context}\n\n"
            f"السؤال: {query}\n\n"
            "الإجابة (مع أرقام المصادر بعد كل جملة):"
        )
    return (
        "You are a precise assistant that answers questions strictly from the provided source documents.\n"
        "Strict rules:\n"
        "- Answer in English only\n"
        "- Use ONLY information from the provided sources — never from memory\n"
        "- If you found the answer, respond directly and concisely\n"
        "- If the answer is truly not in the sources, say ONLY: \"This information was not found in the knowledge bank\"\n"
        "- Never mix a real answer with the not-found phrase in the same response\n"
        "- Place a citation number immediately after every sentence that uses information, like: \"The SLA is 2 hours [1].\"\n"
        "- If two sources support a claim, cite both: [1][2]\n\n"
        f"Sources:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer (with inline citation numbers after each sentence):"
    )

def llm_generate(prompt: str) -> tuple[str, float, float]:
    cfg    = CONFIG["smart_ai_search"]["generator"]
    client = ollama.Client(host=cfg["url"])
    t0     = time.time()
    chunks = []
    ttft = None
    stream = client.generate(
        model=cfg["model"],
        prompt=prompt,
        stream=True,
        options={"temperature": cfg.get("temperature", 0.1), "num_predict": cfg.get("max_tokens", 512)}
    )
    for part in stream:
        token = part.get("response", "")
        if token and ttft is None:
            ttft = time.time() - t0
        chunks.append(token)
    total = time.time() - t0
    return "".join(chunks).strip(), (ttft if ttft is not None else total), total

def extract_citations(answer: str, chunks: list) -> list:
    citations = []
    for i, chunk in enumerate(chunks):
        if f"[{i+1}]" in answer:
            citations.append({
                "id"       : i + 1,
                "doc_id"   : chunk.get("doc_id", ""),
                "chunk_id" : chunk.get("chunk_id", ""),
                "snippet"  : chunk["content"][:200] + ("..." if len(chunk["content"]) > 200 else ""),
                "page"     : chunk.get("page", 1),
                "relevance": round(
                    chunk.get("rerank_score",
                    chunk.get("rrf_score",
                    chunk.get("dense_score", 0.0))), 4
                )
            })
    return citations

def llm_rewrite_query(query: str, history: list, lang: str) -> str:
    if not history:
        return query
    history_text = "".join(
        f"User: {t['user']}\nAssistant: {t['assistant']}\n"
        for t in history[-5:]
    )
    cfg    = CONFIG["smart_ai_search"]["generator"]
    client = ollama.Client(host=cfg["url"])
    if lang == "ar":
        prompt = (
            "بناءً على سياق المحادثة التالية، أعد صياغة السؤال الأخير ليكون سؤالاً مستقلاً وواضحاً.\n"
            "أخرج السؤال المعاد صياغته فقط، بدون أي شرح إضافي.\n\n"
            f"سياق المحادثة:\n{history_text}\n"
            f"السؤال الأخير: {query}\n\n"
            "السؤال المعاد صياغته:"
        )
    else:
        prompt = (
            "Based on the conversation history below, rewrite the last question as a fully standalone question.\n"
            "Output only the rewritten question, nothing else.\n\n"
            f"Conversation history:\n{history_text}\n"
            f"Last question: {query}\n\n"
            "Rewritten question:"
        )
    resp = client.generate(
        model=cfg["model"],
        prompt=prompt,
        options={"temperature": 0.0, "num_predict": 128}
    )
    return resp["response"].strip()

def llm_suggest_followups(query: str, answer: str, lang: str) -> list:
    cfg    = CONFIG["smart_ai_search"]["generator"]
    client = ollama.Client(host=cfg["url"])
    if lang == "ar":
        prompt = (
            "بناءً على السؤال والإجابة التالية، اقترح 3 أسئلة متابعة مفيدة وقصيرة.\n"
            "أخرج الأسئلة فقط، كل سؤال في سطر منفصل، بدون ترقيم.\n\n"
            f"السؤال: {query}\nالإجابة: {answer}\n\nالأسئلة المقترحة:"
        )
    else:
        prompt = (
            "Based on the question and answer below, suggest 3 short useful follow-up questions.\n"
            "Output only the questions, one per line, no numbering.\n\n"
            f"Question: {query}\nAnswer: {answer}\n\nSuggested follow-ups:"
        )
    resp  = client.generate(
        model=cfg["model"],
        prompt=prompt,
        options={"temperature": 0.3, "num_predict": 128}
    )
    lines = [l.strip() for l in resp["response"].strip().split("\n") if l.strip()]
    return lines[:3]

# ---------------------------------------------------------------------------
# Session storage helpers
# ---------------------------------------------------------------------------

def session_get(session_id: str) -> dict:
    if state.redis:
        raw = state.redis.get(f"session:{session_id}")
        return json.loads(raw) if raw else {"history": [], "turn": 0}
    return _memory_sessions.get(session_id, {"history": [], "turn": 0})

def session_set(session_id: str, data: dict):
    ttl = CONFIG["interactive_search"]["session"]["ttl_seconds"]
    if state.redis:
        state.redis.setex(f"session:{session_id}", ttl, json.dumps(data, ensure_ascii=False))
    else:
        _memory_sessions[session_id] = data

# ---------------------------------------------------------------------------
# Route: Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    es_ok     = state.es.ping() if state.es else False
    redis_ok  = False
    if state.redis:
        try:
            state.redis.ping()
            redis_ok = True
        except Exception:
            pass
    return {
        "status"           : "ok",
        "elasticsearch"    : "connected" if es_ok else "unavailable",
        "qdrant"           : "connected",
        "redis"            : "connected" if redis_ok else "unavailable (using in-memory)",
        "embedding_model"  : "loaded" if state.embed_model else "not loaded",
        "reranker"         : "loaded" if state.reranker else "not loaded"
    }

# ---------------------------------------------------------------------------
# Route: Quick Search
# ---------------------------------------------------------------------------

@app.post("/search/quick")
def route_quick_search(req: QuickSearchRequest):
    if not state.es or not state.es.ping():
        raise HTTPException(status_code=503, detail="Elasticsearch is not available")

    t_start = time.time()
    lang    = detect_language(req.query)
    norm_q  = normalize_text(req.query, lang)
    t_pre   = int((time.time() - t_start) * 1000)

    t_s = time.time()
    try:
        response = state.es.search(
            index=CONFIG["quick_search"]["index_name"],
            body={
                "query": {
                    "multi_match": {
                        "query" : norm_q,
                        "fields": ["content^2", "content.english", "title"],
                        "type"  : "best_fields"
                    }
                },
                "size": req.top_k
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Elasticsearch error: {e}")

    t_search = int((time.time() - t_s) * 1000)
    t_total  = int((time.time() - t_start) * 1000)

    results = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        snippet = src["content"][:300] + ("..." if len(src["content"]) > 300 else "")
        results.append({
            "doc_id"  : src["doc_id"],
            "chunk_id": src["chunk_id"],
            "title"   : src.get("title", ""),
            "snippet" : snippet,
            "score"   : round(hit["_score"], 4),
            "source"  : src.get("source", ""),
            "page"    : src.get("page", 1)
        })

    return {
        "mode"             : "quick_search",
        "query"            : req.query,
        "language_detected": lang,
        "results"          : results,
        "latency_ms"       : {
            "preprocessing": t_pre,
            "search"       : t_search,
            "total"        : t_total
        }
    }

# ---------------------------------------------------------------------------
# Route: Smart AI Search
# ---------------------------------------------------------------------------

@app.post("/search/smart")
def route_smart_search(req: SmartSearchRequest):
    t_start = time.time()
    lang    = detect_language(req.query)
    cfg     = CONFIG["smart_ai_search"]

    # Retrieval
    dense_chunks = dense_retrieve(req.query, top_k=cfg["retrieval"]["dense_top_k"])
    bm25_chunks  = bm25_retrieve(req.query, lang, top_k=cfg["retrieval"]["bm25_top_k"])
    fused        = rrf_fusion(dense_chunks, bm25_chunks, k=cfg["retrieval"]["rrf_k"])
    t_retrieval  = int((time.time() - t_start) * 1000)

    # Reranking
    t_re_start = time.time()
    reranked   = rerank_chunks(req.query, fused, top_k=cfg["retrieval"]["final_top_k"])
    t_rerank   = int((time.time() - t_re_start) * 1000)

    # Generation
    prompt             = build_prompt(req.query, reranked, lang)
    answer, ttft, gen  = llm_generate(prompt)
    t_total            = int((time.time() - t_start) * 1000)

    citations = extract_citations(answer, reranked)

    return {
        "mode"             : "smart_ai_search",
        "query"            : req.query,
        "language_detected": lang,
        "answer"           : answer,
        "citations"        : citations,
        "latency_ms"       : {
            "retrieval"          : t_retrieval,
            "rerank"             : t_rerank,
            "time_to_first_token": int(ttft * 1000),
            "generation_total"   : int(gen * 1000),
            "total"              : t_total
        }
    }

# ---------------------------------------------------------------------------
# Route: Interactive AI Search
# ---------------------------------------------------------------------------

@app.post("/search/interactive")
def route_interactive_search(req: InteractiveSearchRequest):
    t_start    = time.time()
    lang       = detect_language(req.query)
    cfg        = CONFIG["smart_ai_search"]
    session_id = req.session_id if req.session_id and req.session_id != "new" else f"sess_{uuid.uuid4().hex[:8]}"

    session = session_get(session_id)
    session["turn"] = session.get("turn", 0) + 1
    turn = session["turn"]

    # Query rewriting
    t_rw_start      = time.time()
    query_rewritten = llm_rewrite_query(req.query, session.get("history", []), lang)
    t_rewrite       = int((time.time() - t_rw_start) * 1000)

    # Retrieval
    t_ret_start  = time.time()
    dense_chunks = dense_retrieve(query_rewritten, top_k=cfg["retrieval"]["dense_top_k"])
    bm25_chunks  = bm25_retrieve(query_rewritten, lang, top_k=cfg["retrieval"]["bm25_top_k"])
    fused        = rrf_fusion(dense_chunks, bm25_chunks, k=cfg["retrieval"]["rrf_k"])
    t_retrieval  = int((time.time() - t_ret_start) * 1000)

    # Reranking
    t_re_start = time.time()
    reranked   = rerank_chunks(query_rewritten, fused, top_k=cfg["retrieval"]["final_top_k"])
    t_rerank   = int((time.time() - t_re_start) * 1000)

    # Generation
    prompt            = build_prompt(query_rewritten, reranked, lang)
    answer, ttft, gen = llm_generate(prompt)
    t_total           = int((time.time() - t_start) * 1000)

    citations  = extract_citations(answer, reranked)
    followups  = llm_suggest_followups(req.query, answer, lang)

    # Update and persist session
    history = session.get("history", [])
    history.append({"user": req.query, "assistant": answer})
    if len(history) > 10:
        history = history[-10:]
    session["history"] = history
    session_set(session_id, session)

    return {
        "mode"              : "interactive",
        "session_id"        : session_id,
        "turn"              : turn,
        "query"             : req.query,
        "query_rewritten"   : query_rewritten,
        "language_detected" : lang,
        "answer"            : answer,
        "citations"         : citations,
        "suggested_followups": followups,
        "latency_ms"        : {
            "query_rewrite"      : t_rewrite,
            "retrieval"          : t_retrieval,
            "rerank"             : t_rerank,
            "time_to_first_token": int(ttft * 1000),
            "generation_total"   : int(gen * 1000),
            "total"              : t_total
        }
    }

# ---------------------------------------------------------------------------
# Route: Ingest document (file upload)
# ---------------------------------------------------------------------------

@app.post("/ingest/file")
async def ingest_file_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    allowed = {".pdf", ".docx", ".txt", ".html", ".htm"}
    ext     = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed)}"
        )

    # Save to temp location
    tmp_dir  = _PROJECT_ROOT / "data" / "corpus" / "ingested"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename

    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    background_tasks.add_task(_ingest_file_task, tmp_path)

    return {
        "status"  : "accepted",
        "filename": file.filename,
        "message" : "Document queued for ingestion. It will be available for search shortly."
    }

def _ingest_file_task(file_path: Path):
    """Background task: chunk, embed, and index a file using already-loaded models."""
    try:
        ext = file_path.suffix.lower()
        raw_text = ""

        if ext == ".txt":
            raw_text = file_path.read_text(encoding="utf-8")
        elif ext == ".pdf":
            from pypdf import PdfReader
            raw_text = "\n".join(p.extract_text() or "" for p in PdfReader(str(file_path)).pages)
        elif ext == ".docx":
            from docx import Document as DocxDocument
            raw_text = "\n".join(p.text for p in DocxDocument(str(file_path)).paragraphs if p.text.strip())
        elif ext in (".html", ".htm"):
            from bs4 import BeautifulSoup
            raw_text = BeautifulSoup(file_path.read_text(encoding="utf-8"), "html.parser").get_text(" ")

        if not raw_text.strip():
            logger.error("Ingest: empty content in %s — skipped", file_path.name)
            return

        lang = detect_language(raw_text)
        chunk_cfg = CONFIG.get("chunking", {"chunk_size": 400, "chunk_overlap": 50})
        chunk_size, overlap = chunk_cfg.get("chunk_size", 400), chunk_cfg.get("chunk_overlap", 50)

        words = raw_text.split()
        raw_chunks, start = [], 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 50:
                raw_chunks.append(chunk)
            start += chunk_size - overlap

        doc_id = file_path.stem
        chunks = [
            {
                "doc_id"  : doc_id,
                "chunk_id": f"{doc_id}#{i}",
                "title"   : file_path.stem.replace("_", " ").title(),
                "language": lang,
                "content" : chunk,
                "source"  : str(file_path),
                "category": "ingested",
                "page"    : i + 1,
            }
            for i, chunk in enumerate(raw_chunks)
        ]
        logger.info("Ingest: %d chunks from %s (lang=%s)", len(chunks), file_path.name, lang)

        # Get current vector count for sequential IDs
        try:
            coll = state.qdrant.get_collection(CONFIG["qdrant"]["collection_name"])
            start_id = coll.vectors_count or 0
        except Exception:
            start_id = 0

        # Embed using already-loaded model (no reload)
        texts = [normalize_text(c["content"], c["language"]) for c in chunks]
        embeddings = state.embed_model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)

        # Upsert to Qdrant
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=start_id + i,
                vector=emb.tolist(),
                payload={k: c[k] for k in ("doc_id", "chunk_id", "title", "language", "content", "source", "category", "page")}
            )
            for i, (emb, c) in enumerate(zip(embeddings, chunks))
        ]
        state.qdrant.upsert(collection_name=CONFIG["qdrant"]["collection_name"], points=points)
        logger.info("Ingest: %d vectors → Qdrant", len(points))

        # Bulk index to Elasticsearch
        from elasticsearch import helpers as es_helpers
        actions = [
            {
                "_index" : CONFIG["quick_search"]["index_name"],
                "_id"    : c["chunk_id"],
                "_source": {
                    "doc_id"  : c["doc_id"],
                    "chunk_id": c["chunk_id"],
                    "title"   : c["title"],
                    "language": c["language"],
                    "content" : normalize_text(c["content"], c["language"]),
                    "source"  : c["source"],
                    "category": c["category"],
                    "page"    : c["page"],
                },
            }
            for c in chunks
        ]
        es_helpers.bulk(state.es, actions)
        logger.info("Ingest: %d chunks → Elasticsearch — %s complete", len(chunks), file_path.name)

    except Exception as exc:
        logger.error("Ingest task failed for %s: %s", file_path, exc, exc_info=True)

# ---------------------------------------------------------------------------
# Route: List ingested documents
# ---------------------------------------------------------------------------

@app.get("/documents")
def list_documents():
    try:
        collection = state.qdrant.get_collection(CONFIG["qdrant"]["collection_name"])
        vector_count = collection.vectors_count
    except Exception:
        vector_count = -1

    try:
        es_count = state.es.count(index=CONFIG["quick_search"]["index_name"])["count"]
    except Exception:
        es_count = -1

    return {
        "qdrant_vectors"       : vector_count,
        "elasticsearch_chunks" : es_count,
        "collection_name"      : CONFIG["qdrant"]["collection_name"],
        "index_name"           : CONFIG["quick_search"]["index_name"]
    }

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=CONFIG["api"]["host"],
        port=CONFIG["api"]["port"],
        workers=CONFIG["api"]["workers"],
        reload=CONFIG["api"]["reload"],
        log_level=CONFIG["api"]["log_level"]
    )
