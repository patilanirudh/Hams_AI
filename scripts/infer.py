import json
import re
import sys
import time
import yaml
import argparse
import uuid
from pathlib import Path
from typing import Optional

# ─── Dependencies ─────────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    from elasticsearch import Elasticsearch
    from qdrant_client import QdrantClient
    import ollama
    import redis
    import torch
except ImportError:
    print("ERROR: Required packages not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# ─── Arabic Normalization ─────────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F]', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ـ', '', text)
    eastern = '٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹'
    western = '01234567890123456789'
    for e, w in zip(eastern, western):
        text = text.replace(e, w)
    return text.strip()

def normalize_text(text: str, language: str) -> str:
    if language in ('ar', 'mixed'):
        text = normalize_arabic(text)
    return re.sub(r'\s+', ' ', text).strip()

# ─── Language Detection ───────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    arabic_chars  = len(re.findall(r'[\u0600-\u06FF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total = arabic_chars + english_chars
    if total == 0:
        return 'en'
    ratio = arabic_chars / total
    if ratio > 0.7:
        return 'ar'
    elif ratio < 0.3:
        return 'en'
    return 'mixed'

# ─── Load Config ──────────────────────────────────────────────────────────────

def load_config(config_path: str = './configs/serving_config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ─── Clients ──────────────────────────────────────────────────────────────────

def get_es_client(config: dict) -> Elasticsearch:
    return Elasticsearch(config['quick_search']['url'])

def get_qdrant_client(config: dict) -> QdrantClient:
    return QdrantClient(url=config['qdrant']['url'])

def get_redis_client(config: dict) -> redis.Redis:
    return redis.from_url(config['interactive_search']['session']['url'])

def get_embedding_model(config: dict) -> SentenceTransformer:
    model_path = config['embedding']['model_path']
    if not Path(model_path).exists():
        if config['embedding'].get('allow_base_model_fallback', False):
            model_path = config['embedding']['fallback_model']
        else:
            raise FileNotFoundError(
                f"Fine-tuned embedding model not found at {model_path}. "
                "Run scripts/train_embeddings.py or place the HF artifact under models/."
            )
    return SentenceTransformer(model_path)

def get_reranker(config: dict):
    try:
        from FlagEmbedding import FlagReranker
        reranker_path = config['smart_ai_search']['reranker']['model_path']
        if not Path(reranker_path).exists():
            if config['smart_ai_search']['reranker'].get('allow_base_model_fallback', False):
                reranker_path = config['smart_ai_search']['reranker'].get('fallback_model', reranker_path)
            else:
                raise FileNotFoundError(
                    f"Fine-tuned reranker model not found at {reranker_path}. "
                    "Run scripts/train_reranker.py or place the HF artifact under models/."
                )
        return FlagReranker(reranker_path, use_fp16=True)
    except Exception as e:
        print(f"WARNING: Could not load reranker: {e}")
        return None

# ─── Quick Search ─────────────────────────────────────────────────────────────

def quick_search(
    query: str,
    es: Elasticsearch,
    config: dict,
    top_k: int = 10
) -> dict:
    t_start = time.time()

    lang     = detect_language(query)
    index    = config['quick_search']['index_name']
    norm_q   = normalize_text(query, lang)

    t_preprocess = int((time.time() - t_start) * 1000)

    t_search_start = time.time()
    response = es.search(
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
    t_search = int((time.time() - t_search_start) * 1000)
    t_total  = int((time.time() - t_start) * 1000)

    results = []
    for hit in response['hits']['hits']:
        src = hit['_source']
        results.append({
            "doc_id"  : src['doc_id'],
            "chunk_id": src['chunk_id'],
            "title"   : src.get('title', ''),
            "snippet" : src['content'][:300] + '...' if len(src['content']) > 300 else src['content'],
            "score"   : round(hit['_score'], 4),
            "source"  : src.get('source', ''),
            "page"    : src.get('page', 1)
        })

    return {
        "mode"             : "quick_search",
        "query"            : query,
        "language_detected": lang,
        "results"          : results,
        "latency_ms"       : {
            "preprocessing": t_preprocess,
            "search"       : t_search,
            "total"        : t_total
        }
    }

# ─── Dense Retrieval ──────────────────────────────────────────────────────────

def dense_retrieve(
    query: str,
    model: SentenceTransformer,
    qdrant: QdrantClient,
    collection_name: str,
    top_k: int = 20
) -> list[dict]:
    embedding = model.encode(query, normalize_embeddings=True).tolist()
    results   = qdrant.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=top_k,
        with_payload=True
    )
    chunks = []
    for r in results:
        payload = r.payload
        payload['dense_score'] = r.score
        chunks.append(payload)
    return chunks

# ─── BM25 Retrieval ───────────────────────────────────────────────────────────

def bm25_retrieve(
    query: str,
    lang: str,
    es: Elasticsearch,
    index_name: str,
    top_k: int = 20
) -> list[dict]:
    norm_q   = normalize_text(query, lang)
    response = es.search(
        index=index_name,
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
    chunks = []
    for hit in response['hits']['hits']:
        src = hit['_source']
        src['bm25_score'] = hit['_score']
        chunks.append(src)
    return chunks

# ─── RRF Fusion ───────────────────────────────────────────────────────────────

def rrf_fusion(
    dense_chunks: list[dict],
    bm25_chunks: list[dict],
    k: int = 60
) -> list[dict]:
    scores = {}
    chunk_map = {}

    for rank, chunk in enumerate(dense_chunks):
        cid = chunk['chunk_id']
        scores[cid]    = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_chunks):
        cid = chunk['chunk_id']
        scores[cid]    = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = []
    for cid in sorted_ids:
        chunk = chunk_map[cid]
        chunk['rrf_score'] = round(scores[cid], 6)
        fused.append(chunk)
    return fused

# ─── Reranking ────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    chunks: list[dict],
    reranker,
    top_k: int = 3
) -> list[dict]:
    if reranker is None or not chunks:
        return chunks[:top_k]

    pairs  = [[query, c['content']] for c in chunks]
    scores = reranker.compute_score(pairs, normalize=True)

    for i, chunk in enumerate(chunks):
        chunk['rerank_score'] = float(scores[i])

    reranked = sorted(chunks, key=lambda x: x.get('rerank_score', 0), reverse=True)
    return reranked[:top_k]

# ─── LLM Generation ───────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: list[dict], lang: str) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(f"[{i+1}] {chunk['content']}")
    context = '\n\n'.join(context_parts)

    if lang == 'ar':
        return f"""أنت مساعد متخصص في الإجابة على الأسئلة بناءً على المستندات المرفقة فقط.
قواعد صارمة — يجب الالتزام بها في كل إجابة:
- أجب باللغة العربية فقط
- استخدم المعلومات من السياق المرفق فقط، لا تضف معلومات من خارجه
- إذا لم تجد الإجابة في السياق، قل فقط: "لم أجد هذه المعلومات في قاعدة المعرفة"
- ضع رقم المصدر مباشرةً بعد كل جملة تستخدم فيها معلومة، مثال: "المدة هي 7 أيام [1]."
- إذا استخدمت مصدرين، اذكرهما هكذا: [1][2]

المصادر:
{context}

السؤال: {query}

الإجابة (مع أرقام المصادر بعد كل جملة):"""
    else:
        return f"""You are a precise assistant that answers questions strictly from the provided source documents.
Strict rules — follow every one:
- Answer in English only
- Use ONLY information from the provided sources below — never from memory
- If the answer is not in the sources, respond only with: "This information was not found in the knowledge bank"
- Place a citation number immediately after every sentence that uses information, like: "The SLA is 2 hours [1]."
- If two sources support a claim, cite both: [1][2]

Sources:
{context}

Question: {query}

Answer (with inline citation numbers after each sentence):"""

def generate_answer(
    query: str,
    chunks: list[dict],
    lang: str,
    llm_model: str,
    ollama_url: str
) -> tuple[str, float, float]:
    prompt = build_prompt(query, chunks, lang)

    t_start = time.time()
    client  = ollama.Client(host=ollama_url)

    tokens = []
    ttft = None
    stream = client.generate(
        model=llm_model,
        prompt=prompt,
        stream=True,
        options={"temperature": 0.1, "num_predict": 512}
    )
    for part in stream:
        token = part.get('response', '')
        if token and ttft is None:
            ttft = time.time() - t_start
        tokens.append(token)
    answer  = ''.join(tokens).strip()
    t_total = time.time() - t_start

    return answer, (ttft if ttft is not None else t_total), t_total

# ─── Citation Extraction ──────────────────────────────────────────────────────

def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    citations = []
    for i, chunk in enumerate(chunks):
        marker = f"[{i+1}]"
        if marker in answer:
            citations.append({
                "id"       : i + 1,
                "doc_id"   : chunk.get('doc_id', ''),
                "chunk_id" : chunk.get('chunk_id', ''),
                "snippet"  : chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content'],
                "page"     : chunk.get('page', 1),
                "relevance": round(chunk.get('rerank_score', chunk.get('rrf_score', chunk.get('dense_score', 0.0))), 4)
            })
    return citations

# ─── Smart AI Search ──────────────────────────────────────────────────────────

def smart_ai_search(
    query: str,
    es: Elasticsearch,
    qdrant: QdrantClient,
    model: SentenceTransformer,
    reranker,
    config: dict
) -> dict:
    t_start = time.time()
    lang    = detect_language(query)
    cfg     = config['smart_ai_search']

    # Dense retrieval
    dense_chunks = dense_retrieve(
        query, model, qdrant,
        config['qdrant']['collection_name'],
        top_k=cfg['retrieval']['dense_top_k']
    )

    # BM25 retrieval
    bm25_chunks = bm25_retrieve(
        query, lang, es,
        config['quick_search']['index_name'],
        top_k=cfg['retrieval']['bm25_top_k']
    )

    # RRF fusion
    fused = rrf_fusion(dense_chunks, bm25_chunks, k=cfg['retrieval']['rrf_k'])
    t_retrieval = int((time.time() - t_start) * 1000)

    # Reranking
    t_rerank_start = time.time()
    reranked = rerank(query, fused, reranker, top_k=cfg['retrieval']['final_top_k'])
    t_rerank = int((time.time() - t_rerank_start) * 1000)

    # Generation
    llm_model  = cfg['generator']['model']
    ollama_url = cfg['generator']['url']
    answer, ttft, gen_total = generate_answer(query, reranked, lang, llm_model, ollama_url)

    t_total = int((time.time() - t_start) * 1000)

    # Citations
    citations = extract_citations(answer, reranked)

    return {
        "mode"             : "smart_ai_search",
        "query"            : query,
        "language_detected": lang,
        "answer"           : answer,
        "citations"        : citations,
        "latency_ms"       : {
            "retrieval"       : t_retrieval,
            "rerank"          : t_rerank,
            "time_to_first_token": int(ttft * 1000),
            "generation_total": int(gen_total * 1000),
            "total"           : t_total
        }
    }

# ─── Query Rewriting ──────────────────────────────────────────────────────────

def rewrite_query(
    query: str,
    history: list[dict],
    lang: str,
    llm_model: str,
    ollama_url: str
) -> str:
    if not history:
        return query

    history_text = ''
    for turn in history[-5:]:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    if lang == 'ar':
        prompt = f"""بناءً على سياق المحادثة التالية، أعد صياغة السؤال الأخير ليكون سؤالاً مستقلاً وواضحاً.
أخرج السؤال المعاد صياغته فقط، بدون أي شرح إضافي.

سياق المحادثة:
{history_text}

السؤال الأخير: {query}

السؤال المعاد صياغته:"""
    else:
        prompt = f"""Based on the conversation history below, rewrite the last question as a fully standalone question.
Output only the rewritten question, nothing else.

Conversation history:
{history_text}

Last question: {query}

Rewritten question:"""

    client   = ollama.Client(host=ollama_url)
    response = client.generate(
        model=llm_model,
        prompt=prompt,
        options={"temperature": 0.0, "num_predict": 128}
    )
    return response['response'].strip()

# ─── Suggested Followups ──────────────────────────────────────────────────────

def generate_followups(
    query: str,
    answer: str,
    lang: str,
    llm_model: str,
    ollama_url: str
) -> list[str]:
    if lang == 'ar':
        prompt = f"""بناءً على السؤال والإجابة التالية، اقترح 3 أسئلة متابعة مفيدة وقصيرة.
أخرج الأسئلة فقط، كل سؤال في سطر منفصل، بدون ترقيم.

السؤال: {query}
الإجابة: {answer}

الأسئلة المقترحة:"""
    else:
        prompt = f"""Based on the question and answer below, suggest 3 short useful follow-up questions.
Output only the questions, one per line, no numbering.

Question: {query}
Answer: {answer}

Suggested follow-ups:"""

    client   = ollama.Client(host=ollama_url)
    response = client.generate(
        model=llm_model,
        prompt=prompt,
        options={"temperature": 0.3, "num_predict": 128}
    )
    lines = [l.strip() for l in response['response'].strip().split('\n') if l.strip()]
    return lines[:3]

# ─── Interactive AI Search ────────────────────────────────────────────────────

def interactive_search(
    query: str,
    session_id: Optional[str],
    es: Elasticsearch,
    qdrant: QdrantClient,
    model: SentenceTransformer,
    reranker,
    redis_client: redis.Redis,
    config: dict
) -> dict:
    t_start = time.time()
    lang    = detect_language(query)
    cfg     = config['smart_ai_search']
    i_cfg   = config['interactive_search']

    # Session management
    if not session_id or session_id == 'new':
        session_id = f"sess_{uuid.uuid4().hex[:8]}"

    session_key  = f"session:{session_id}"
    session_data = redis_client.get(session_key)

    if session_data:
        session = json.loads(session_data)
    else:
        session = {"history": [], "turn": 0}

    session['turn'] += 1
    turn = session['turn']

    # Query rewriting
    t_rewrite_start = time.time()
    llm_model       = cfg['generator']['model']
    ollama_url      = cfg['generator']['url']

    query_rewritten = rewrite_query(
        query,
        session['history'],
        lang,
        llm_model,
        ollama_url
    )
    t_rewrite = int((time.time() - t_rewrite_start) * 1000)

    # Retrieval
    t_retrieval_start = time.time()
    dense_chunks = dense_retrieve(
        query_rewritten, model, qdrant,
        config['qdrant']['collection_name'],
        top_k=cfg['retrieval']['dense_top_k']
    )
    bm25_chunks = bm25_retrieve(
        query_rewritten, lang, es,
        config['quick_search']['index_name'],
        top_k=cfg['retrieval']['bm25_top_k']
    )
    fused = rrf_fusion(dense_chunks, bm25_chunks, k=cfg['retrieval']['rrf_k'])
    t_retrieval = int((time.time() - t_retrieval_start) * 1000)

    # Reranking
    t_rerank_start = time.time()
    reranked = rerank(
        query_rewritten, fused, reranker,
        top_k=cfg['retrieval']['final_top_k']
    )
    t_rerank = int((time.time() - t_rerank_start) * 1000)

    # Generation
    answer, ttft, gen_total = generate_answer(
        query_rewritten, reranked, lang, llm_model, ollama_url
    )

    t_total = int((time.time() - t_start) * 1000)

    # Citations
    citations = extract_citations(answer, reranked)

    # Suggested followups
    followups = generate_followups(query, answer, lang, llm_model, ollama_url)

    # Update session history
    session['history'].append({
        "user"     : query,
        "assistant": answer
    })

    # Keep last 10 turns only
    if len(session['history']) > 10:
        session['history'] = session['history'][-10:]

    # Save session
    ttl = i_cfg['session']['ttl_seconds']
    redis_client.setex(session_key, ttl, json.dumps(session, ensure_ascii=False))

    return {
        "mode"              : "interactive",
        "session_id"        : session_id,
        "turn"              : turn,
        "query"             : query,
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
            "generation_total"   : int(gen_total * 1000),
            "total"              : t_total
        }
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HamsAI Inference")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['quick_search', 'smart_ai_search', 'interactive'],
                        help='Search mode')
    parser.add_argument('--query', type=str, required=True,
                        help='Query string')
    parser.add_argument('--session', type=str, default='new',
                        help='Session ID for interactive mode (use "new" to start fresh)')
    args = parser.parse_args()

    config = load_config()

    # Initialize clients
    es           = get_es_client(config)
    qdrant       = get_qdrant_client(config)
    embed_model  = get_embedding_model(config)
    reranker     = get_reranker(config)
    redis_client = get_redis_client(config)

    if args.mode == 'quick_search':
        result = quick_search(args.query, es, config)

    elif args.mode == 'smart_ai_search':
        result = smart_ai_search(args.query, es, qdrant, embed_model, reranker, config)

    elif args.mode == 'interactive':
        result = interactive_search(
            args.query, args.session,
            es, qdrant, embed_model, reranker,
            redis_client, config
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
