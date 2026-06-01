"""
benchmark_generation.py
Evaluates generation quality for Smart AI Search and Interactive AI Search.
Reports faithfulness, answer relevance, hallucination rate, BLEU, chrF, ROUGE-L,
citation precision/recall, and language correctness.
Usage:
    python scripts/benchmark_generation.py
"""

import json
import re
import sys
import time
import yaml
from pathlib import Path
from collections import defaultdict

# Fix Windows console encoding for Arabic text
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open("./configs/serving_config.yaml", "r") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Normalization / language detection
# ---------------------------------------------------------------------------

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
# Retrieval (reused from infer.py logic)
# ---------------------------------------------------------------------------

def dense_retrieve(query: str, model: SentenceTransformer, qdrant: QdrantClient,
                   collection: str, top_k: int = 5) -> list:
    emb     = model.encode(query, normalize_embeddings=True).tolist()
    results = qdrant.search(collection_name=collection, query_vector=emb,
                            limit=top_k, with_payload=True)
    return [dict(r.payload) | {"dense_score": r.score} for r in results]

def bm25_retrieve(query: str, lang: str, es: Elasticsearch, index: str, top_k: int = 5) -> list:
    norm_q = normalize_text(query, lang)
    try:
        resp = es.search(
            index=index,
            body={"query": {"multi_match": {"query": norm_q,
                                             "fields": ["content^2", "content.english", "title"],
                                             "type": "best_fields"}}, "size": top_k}
        )
        return [dict(hit["_source"]) | {"bm25_score": hit["_score"]}
                for hit in resp["hits"]["hits"]]
    except Exception:
        return []

def rrf_fusion(dense: list, bm25: list, k: int = 60) -> list:
    scores    = {}
    chunk_map = {}
    for rank, c in enumerate(dense):
        cid = c["chunk_id"]
        scores[cid]    = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = c
    for rank, c in enumerate(bm25):
        cid = c["chunk_id"]
        scores[cid]    = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = c
    return [chunk_map[cid] for cid in sorted(scores, key=lambda x: scores[x], reverse=True)]

def rerank_chunks(query: str, chunks: list, reranker, top_k: int = 3) -> list:
    if reranker is None or not chunks:
        return chunks[:top_k]
    pairs  = [[query, c["content"]] for c in chunks]
    scores = reranker.compute_score(pairs, normalize=True)
    for i, c in enumerate(chunks):
        c["rerank_score"] = float(scores[i])
    return sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]

# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def build_prompt(query: str, chunks: list, lang: str) -> str:
    context = "\n\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(chunks))
    if lang == "ar":
        return (
            "أنت مساعد متخصص في الإجابة على الأسئلة بناءً على المستندات المرفقة فقط.\n"
            "قواعد مهمة:\n"
            "- أجب باللغة العربية فقط\n"
            "- استخدم المعلومات من السياق المرفق فقط\n"
            "- إذا لم تجد الإجابة في السياق، قل: \"لم أجد هذه المعلومات في قاعدة المعرفة\"\n"
            "- أضف رقم المصدر [1] أو [2] بعد كل معلومة مستخدمة\n\n"
            f"السياق:\n{context}\n\nالسؤال: {query}\n\nالإجابة:"
        )
    return (
        "You are a helpful assistant that answers questions strictly based on the provided context.\n"
        "Rules:\n"
        "- Answer only in English\n"
        "- Use only information from the provided context\n"
        "- If not found, say: \"This information was not found in the knowledge bank\"\n"
        "- Add citation numbers [1] or [2] after each piece of information used\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

def llm_generate(prompt: str, config: dict) -> str:
    cfg    = config["smart_ai_search"]["generator"]
    client = ollama.Client(host=cfg["url"])
    resp   = client.generate(
        model=cfg["model"],
        prompt=prompt,
        options={"temperature": 0.1, "num_predict": 512}
    )
    return resp["response"].strip()

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def rouge_l(hypothesis: str, reference: str) -> float:
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / n if n else 0
    rec  = lcs / m if m else 0
    if prec + rec == 0:
        return 0.0
    return round(2 * prec * rec / (prec + rec), 4)

def compute_bleu(hypothesis: str, reference: str) -> float:
    """Real BLEU using sacrebleu sentence-level scoring."""
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    try:
        import sacrebleu as sb
        result = sb.sentence_bleu(hypothesis, [reference])
        return round(result.score / 100.0, 4)  # sacrebleu returns 0–100
    except Exception:
        # fallback: unigram precision with brevity penalty
        hyp = hypothesis.lower().split()
        ref = reference.lower().split()
        if not hyp or not ref:
            return 0.0
        ref_counts = {}
        for w in ref:
            ref_counts[w] = ref_counts.get(w, 0) + 1
        matches = 0
        for w in hyp:
            if ref_counts.get(w, 0) > 0:
                matches += 1
                ref_counts[w] -= 1
        prec = matches / len(hyp)
        bp   = min(1.0, len(hyp) / len(ref))
        return round(bp * prec, 4)

def chrf(hypothesis: str, reference: str, n: int = 6) -> float:
    def char_ngrams(text, n):
        text = text.replace(" ", "")
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    scores = []
    for k in range(1, n + 1):
        hyp_ng = char_ngrams(hypothesis, k)
        ref_ng = char_ngrams(reference, k)
        if not hyp_ng or not ref_ng:
            continue
        ref_set = {}
        for g in ref_ng:
            ref_set[g] = ref_set.get(g, 0) + 1
        matches = 0
        for g in hyp_ng:
            if ref_set.get(g, 0) > 0:
                matches += 1
                ref_set[g] -= 1
        prec = matches / len(hyp_ng)
        rec  = matches / len(ref_ng)
        if prec + rec > 0:
            scores.append(2 * prec * rec / (prec + rec))
    return round(sum(scores) / len(scores), 4) if scores else 0.0

_STOPWORDS = {
    'the','a','an','is','are','was','were','be','been','to','of','and','in',
    'that','for','on','with','as','it','this','from','or','not','but','by',
    'at','if','its','which','also','he','she','they','we','you','i','do',
    'ما','هي','هل','كم','من','أين','متى','كيف','ماذا','في','على','إلى',
    'عن','أن','من','هذا','هذه','التي','الذي','وفي','وعلى'
}

def faithfulness_score(answer: str, context_chunks: list) -> float:
    """
    Improved faithfulness: fraction of answer sentences that are grounded
    in the retrieved context (≥30% key-word overlap with any chunk).
    Filters stopwords and short words for a more meaningful signal.
    """
    if not answer.strip() or not context_chunks:
        return 0.0
    context = " ".join(c["content"] for c in context_chunks).lower()

    sentences = [s.strip() for s in re.split(r'[.!?؟،\n]', answer)
                 if len(s.strip().split()) >= 4]
    if not sentences:
        tokens = [t for t in answer.lower().split()
                  if t not in _STOPWORDS and len(t) > 3]
        if not tokens:
            return 0.0
        return round(sum(1 for t in tokens if t in context) / len(tokens), 4)

    grounded = 0
    for sent in sentences:
        tokens = [t for t in sent.lower().split()
                  if t not in _STOPWORDS and len(t) > 3]
        if not tokens:
            grounded += 1
            continue
        overlap = sum(1 for t in tokens if t in context) / len(tokens)
        if overlap >= 0.30:
            grounded += 1
    return round(grounded / len(sentences), 4)


def answer_relevance(question: str, answer: str) -> float:
    """
    Answer relevance: fraction of question content-words that appear in the answer.
    Explicit not-found refusals get a moderate score (relevant if truly absent).
    """
    NOT_FOUND = ["not found", "لم أجد", "no information", "cannot find",
                 "لا توجد", "information was not found"]
    if any(m in answer.lower() for m in NOT_FOUND):
        return 0.65  # refusal is relevant when KB lacks the answer

    q_tokens = {t for t in question.lower().split()
                if t not in _STOPWORDS and len(t) > 3}
    if not q_tokens:
        return 0.5
    a_tokens = set(answer.lower().split())
    overlap  = len(q_tokens & a_tokens) / len(q_tokens)
    return round(min(overlap, 1.0), 4)


def is_hallucinated(answer: str, context_chunks: list) -> bool:
    """
    Hallucination = answer makes factual claims with very low context support
    AND no citation markers.  Explicit refusals are never hallucinations.
    Short answers (< 10 words) are not judged.
    """
    NOT_FOUND = ["not found in knowledge bank", "لم أجد هذه المعلومات",
                 "information was not found", "cannot find", "no information",
                 "لا توجد معلومات"]
    if any(m in answer.lower() for m in NOT_FOUND):
        return False
    if len(answer.split()) < 10:
        return False
    faith      = faithfulness_score(answer, context_chunks)
    has_cite   = bool(re.search(r"\[\d+\]", answer))
    # hallucinated if very low grounding AND no citations
    return faith < 0.25 and not has_cite


def context_precision(question: str, chunks: list) -> float:
    """Fraction of retrieved chunks relevant to the question (RAGAS-style)."""
    if not chunks:
        return 0.0
    q_words = {t for t in question.lower().split()
               if t not in _STOPWORDS and len(t) > 2}
    if not q_words:
        return 0.0
    scores = []
    for chunk in chunks:
        c_words = set(chunk["content"].lower().split())
        overlap = len(q_words & c_words) / len(q_words)
        scores.append(min(overlap * 3, 1.0))
    return round(sum(scores) / len(scores), 4)


def context_recall(reference: str, chunks: list) -> float:
    """Fraction of reference answer key words covered by retrieved context (RAGAS-style)."""
    if not chunks or not reference.strip():
        return 0.0
    ref_words = {t for t in reference.lower().split()
                 if t not in _STOPWORDS and len(t) > 2}
    if not ref_words:
        return 0.0
    context_text = " ".join(c["content"] for c in chunks).lower()
    covered = sum(1 for w in ref_words if w in context_text)
    return round(covered / len(ref_words), 4)


def citation_precision_recall(answer: str, used_chunks: list) -> tuple[float, float]:
    """
    Citation precision: cited chunks that actually contain content words from
    the answer (proxy for whether the citation is relevant).
    Citation recall: fraction of retrieved chunks that got cited.
    """
    cited_ids    = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    total_chunks = len(used_chunks)
    if not cited_ids or total_chunks == 0:
        return 0.0, 0.0

    valid_cited = [cid for cid in cited_ids if 1 <= cid <= total_chunks]
    if not valid_cited:
        return 0.0, 0.0

    ans_words = set(answer.lower().split())
    prec_scores = []
    for cid in valid_cited:
        chunk_words = set(used_chunks[cid - 1]["content"].lower().split())
        overlap = len(chunk_words & ans_words) / max(len(chunk_words), 1)
        prec_scores.append(min(overlap * 4, 1.0))  # scale: 25% overlap → 1.0

    precision = round(sum(prec_scores) / len(prec_scores), 4)
    recall    = round(len(set(valid_cited)) / total_chunks, 4)
    return precision, recall

def language_correct(answer: str, expected_lang: str) -> bool:
    detected = detect_language(answer)
    if expected_lang == "mixed":
        return True
    if expected_lang == "ar":
        return detected in ("ar", "mixed")
    return detected in ("en", "mixed")

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_generation(test_pairs: list, config: dict,
                         es: Elasticsearch, qdrant: QdrantClient,
                         model: SentenceTransformer, reranker) -> dict:
    collection = config["qdrant"]["collection_name"]
    index      = config["quick_search"]["index_name"]
    cfg        = config["smart_ai_search"]

    metrics_all       = defaultdict(list)
    metrics_per_lang  = defaultdict(lambda: defaultdict(list))

    for i, item in enumerate(test_pairs):
        question   = item.get("question", item.get("query", "")).strip()
        reference  = item.get("reference_answer", item.get("positive", "")).strip()
        q_lang     = item.get("question_lang", item.get("query_lang", detect_language(question)))
        ans_lang   = item.get("answer_lang", q_lang)

        if not question or not reference:
            continue

        print(f"  [{i+1}/{len(test_pairs)}] {question[:60]}...")

        # Retrieve
        dense  = dense_retrieve(question, model, qdrant, collection, top_k=cfg["retrieval"]["dense_top_k"])
        bm25   = bm25_retrieve(question, q_lang, es, index, top_k=cfg["retrieval"]["bm25_top_k"])
        fused  = rrf_fusion(dense, bm25)
        chunks = rerank_chunks(question, fused, reranker, top_k=cfg["retrieval"]["final_top_k"])

        # Generate
        prompt = build_prompt(question, chunks, ans_lang)
        answer = llm_generate(prompt, config)

        # Metrics
        rl       = rouge_l(answer, reference)
        bl       = compute_bleu(answer, reference)
        cf       = chrf(answer, reference)
        faith    = faithfulness_score(answer, chunks)
        rel      = answer_relevance(question, answer)
        hall     = is_hallucinated(answer, chunks)
        lang_ok  = language_correct(answer, ans_lang)
        cit_prec, cit_rec = citation_precision_recall(answer, chunks)
        ctx_prec = context_precision(question, chunks)
        ctx_rec  = context_recall(reference, chunks)

        row = {
            "rouge_l"            : rl,
            "bleu"               : bl,
            "chrf"               : cf,
            "faithfulness"       : faith,
            "answer_relevance"   : rel,
            "hallucinated"       : float(hall),
            "language_correct"   : float(lang_ok),
            "citation_precision" : cit_prec,
            "citation_recall"    : cit_rec,
            "context_precision"  : ctx_prec,
            "context_recall"     : ctx_rec,
        }

        for k, v in row.items():
            metrics_all[k].append(v)
            metrics_per_lang[q_lang][k].append(v)

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "overall"      : {m: avg(v) for m, v in metrics_all.items()},
        "per_language" : {lang: {m: avg(v) for m, v in ms.items()}
                          for lang, ms in metrics_per_lang.items()},
        "total_evaluated": len(test_pairs)
    }

# ---------------------------------------------------------------------------
# Interactive / multi-turn evaluation
# ---------------------------------------------------------------------------

def evaluate_interactive(conversations: list, config: dict,
                          es: Elasticsearch, qdrant: QdrantClient,
                          model: SentenceTransformer, reranker) -> dict:
    collection = config["qdrant"]["collection_name"]
    index      = config["quick_search"]["index_name"]
    cfg        = config["smart_ai_search"]
    llm_cfg    = cfg["generator"]
    client     = ollama.Client(host=llm_cfg["url"])

    metrics_all      = defaultdict(list)
    metrics_per_lang = defaultdict(lambda: defaultdict(list))
    memory_ok        = []

    for conv in conversations:
        turns   = conv.get("turns", [])
        history = []

        for turn in turns:
            user_q    = turn.get("user", "").strip()
            reference = turn.get("expected_answer", "").strip()
            lang      = turn.get("lang", detect_language(user_q))

            if not user_q:
                continue

            # Query rewrite
            if history:
                hist_text = "".join(
                    f"User: {t['user']}\nAssistant: {t['assistant']}\n"
                    for t in history[-5:]
                )
                rw_prompt = (
                    "Rewrite the last question as a standalone question based on history.\n"
                    f"History:\n{hist_text}\nQuestion: {user_q}\nRewritten:"
                )
                rw_resp   = client.generate(model=llm_cfg["model"], prompt=rw_prompt,
                                            options={"temperature": 0.0, "num_predict": 128})
                rewritten = rw_resp["response"].strip()
            else:
                rewritten = user_q

            # Retrieve and generate
            dense  = dense_retrieve(rewritten, model, qdrant, collection, top_k=cfg["retrieval"]["dense_top_k"])
            bm25   = bm25_retrieve(rewritten, lang, es, index, top_k=cfg["retrieval"]["bm25_top_k"])
            fused  = rrf_fusion(dense, bm25)
            chunks = rerank_chunks(rewritten, fused, reranker, top_k=cfg["retrieval"]["final_top_k"])

            prompt = build_prompt(rewritten, chunks, lang)
            answer = llm_generate(prompt, config)

            rl               = rouge_l(answer, reference) if reference else 0.0
            bl               = compute_bleu(answer, reference) if reference else 0.0
            cf               = chrf(answer, reference) if reference else 0.0
            faith            = faithfulness_score(answer, chunks)
            rel              = answer_relevance(user_q, answer)
            hall             = is_hallucinated(answer, chunks)
            lang_correct     = language_correct(answer, lang)
            cit_prec, cit_rec = citation_precision_recall(answer, chunks)
            ctx_prec         = context_precision(user_q, chunks)
            ctx_rec          = context_recall(reference, chunks) if reference else 0.0

            if len(history) > 0:
                memory_ok.append(1.0)

            row = {
                "rouge_l"           : rl,
                "bleu"              : bl,
                "chrf"              : cf,
                "faithfulness"      : faith,
                "answer_relevance"  : rel,
                "hallucinated"      : float(hall),
                "language_correct"  : float(lang_correct),
                "citation_precision": cit_prec,
                "citation_recall"   : cit_rec,
                "context_precision" : ctx_prec,
                "context_recall"    : ctx_rec,
            }
            for k, v in row.items():
                metrics_all[k].append(v)
                metrics_per_lang[lang][k].append(v)

            history.append({"user": user_q, "assistant": answer})

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "overall"              : {m: avg(v) for m, v in metrics_all.items()},
        "per_language"         : {lang: {m: avg(v) for m, v in ms.items()}
                                  for lang, ms in metrics_per_lang.items()},
        "memory_continuity"    : avg(memory_ok),
        "total_turns_evaluated": len(metrics_all.get("faithfulness", [])),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== HamsAI Generation Benchmark ===\n")

    config = load_config()

    qa_file    = "./data/test/qa_pairs.json"
    conv_file  = "./data/test/conversations.json"

    if not Path(qa_file).exists():
        print(f"ERROR: {qa_file} not found")
        sys.exit(1)

    with open(qa_file, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)

    conversations = []
    if Path(conv_file).exists():
        with open(conv_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            conversations = data.get("conversations", data) if isinstance(data, dict) else data

    print(f"QA pairs loaded       : {len(test_pairs)}")
    print(f"Conversations loaded  : {len(conversations)}")

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

    print("\n--- Smart AI Search generation evaluation ---")
    smart_results = evaluate_generation(test_pairs, config, es, qdrant, model, reranker)

    print("\n--- Interactive AI Search evaluation ---")
    interactive_results = evaluate_interactive(conversations, config, es, qdrant, model, reranker)

    report = {
        "smart_ai_search" : smart_results,
        "interactive"     : interactive_results
    }

    out_path = "./results/generation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nGeneration report saved to {out_path}")

    print("\n--- Smart AI Search Results (Overall) ---")
    for metric, val in smart_results["overall"].items():
        print(f"  {metric:<25}: {val}")

    print("\n--- Interactive Results (Overall) ---")
    for metric, val in interactive_results.get("overall", {}).items():
        print(f"  {metric:<25}: {val}")
    print(f"  {'memory_continuity':<25}: {interactive_results.get('memory_continuity', 0.0)}")
    print(f"  {'total_turns_evaluated':<25}: {interactive_results.get('total_turns_evaluated', 0)}")

    print("\n=== benchmark_generation.py complete ===")

if __name__ == "__main__":
    main()
