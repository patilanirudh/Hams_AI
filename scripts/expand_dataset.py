"""
Build a larger reproducible synthetic dataset for the HamsAI assessment.

The generated files keep the existing repo schema, but add the fields needed for
real retrieval benchmarks: positive_chunk_id, source_doc_id, language labels, and
answerability. The documents are deliberately one chunk each with the current
400-word chunk size, so doc_id#0 is the relevant chunk id.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

CORPUS_DIR = Path("data/corpus")
TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")


TOPICS = [
    {
        "key": "returns",
        "category": "policy",
        "ar_title": "سياسة الاسترجاع والاستبدال",
        "en_title": "Returns and Exchange Policy",
        "terms": ["return window", "refund", "defect", "shipping fee"],
        "ar_terms": ["مهلة الاسترجاع", "استرداد المبلغ", "عيب مصنعي", "رسوم الشحن"],
    },
    {
        "key": "sla",
        "category": "support",
        "ar_title": "اتفاقية مستوى الخدمة للدعم الفني",
        "en_title": "Technical Support SLA",
        "terms": ["Severity 1", "first response", "resolution time", "service credit"],
        "ar_terms": ["الأولوية الحرجة", "وقت الاستجابة", "وقت الحل", "رصيد الخدمة"],
    },
    {
        "key": "pricing",
        "category": "commercial",
        "ar_title": "أسعار الاشتراكات والباقات",
        "en_title": "Subscription Pricing",
        "terms": ["SaaS Pro", "Enterprise Suite", "additional user", "annual billing"],
        "ar_terms": ["باقة SaaS Pro", "باقة المؤسسات", "مستخدم إضافي", "الفوترة السنوية"],
    },
    {
        "key": "warranty",
        "category": "hardware",
        "ar_title": "ضمان الأجهزة والأثاث",
        "en_title": "Hardware and Furniture Warranty",
        "terms": ["limited warranty", "extended warranty", "technician visit", "replacement"],
        "ar_terms": ["الضمان المحدود", "تمديد الضمان", "زيارة الفني", "الاستبدال"],
    },
    {
        "key": "delivery",
        "category": "operations",
        "ar_title": "التوصيل والتركيب",
        "en_title": "Delivery and Installation",
        "terms": ["installation", "site readiness", "delivery fee", "rescheduling"],
        "ar_terms": ["التركيب", "جاهزية الموقع", "رسوم التوصيل", "إعادة الجدولة"],
    },
    {
        "key": "privacy",
        "category": "security",
        "ar_title": "خصوصية البيانات وأمنها",
        "en_title": "Data Privacy and Security",
        "terms": ["AES-256", "TLS 1.3", "RBAC", "MFA"],
        "ar_terms": ["تشفير AES-256", "بروتوكول TLS 1.3", "التحكم بالصلاحيات", "المصادقة متعددة العوامل"],
    },
    {
        "key": "onboarding",
        "category": "implementation",
        "ar_title": "خطة التهيئة والتدريب",
        "en_title": "Onboarding and Training Plan",
        "terms": ["discovery", "migration", "UAT", "go-live"],
        "ar_terms": ["مرحلة الاكتشاف", "ترحيل البيانات", "اختبار القبول", "الإطلاق"],
    },
    {
        "key": "api",
        "category": "technical",
        "ar_title": "دليل واجهات API",
        "en_title": "API Integration Guide",
        "terms": ["OAuth2", "rate limit", "production key", "sandbox"],
        "ar_terms": ["مصادقة OAuth2", "حد الطلبات", "مفتاح الإنتاج", "بيئة الاختبار"],
    },
    {
        "key": "infrastructure",
        "category": "technical",
        "ar_title": "متطلبات البنية التحتية",
        "en_title": "Infrastructure Requirements",
        "terms": ["CPU cores", "RAM", "PostgreSQL", "Docker"],
        "ar_terms": ["أنوية المعالج", "الذاكرة", "PostgreSQL", "Docker"],
    },
    {
        "key": "billing",
        "category": "finance",
        "ar_title": "الفوترة والمدفوعات",
        "en_title": "Billing and Payments",
        "terms": ["invoice", "Net 30", "late fee", "suspension"],
        "ar_terms": ["الفاتورة", "الدفع خلال 30 يوم", "غرامة التأخير", "تعليق الخدمة"],
    },
]

PRODUCTS = [
    ("HamsAI Cloud ERP", "نظام HamsAI Cloud ERP"),
    ("HamsAI Core CRM", "نظام HamsAI Core CRM"),
    ("HamsAI SmartDesk", "مكتب HamsAI SmartDesk"),
    ("HamsAI ServerRack X1", "خزانة HamsAI ServerRack X1"),
    ("Ergonomic Chair Pro", "كرسي Ergonomic Chair Pro"),
]

LOCATIONS = ["Riyadh", "Jeddah", "Dammam", "الرياض", "جدة", "الدمام"]


def ensure_dirs() -> None:
    for path in (CORPUS_DIR, TRAIN_DIR, TEST_DIR, Path("results")):
        path.mkdir(parents=True, exist_ok=True)


def en_doc(topic: dict, idx: int) -> dict:
    product_en, _ = PRODUCTS[idx % len(PRODUCTS)]
    location = LOCATIONS[idx % 3]
    first = 1 + (idx % 6)
    second = first + 3
    amount = 150 + (idx * 75) % 4800
    eastern_hint = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"][idx % 10]
    content = (
        f"{topic['en_title']} section {idx} applies to {product_en} customers in {location}. "
        f"The primary rule for {topic['terms'][0]} is {first} business days, while the escalation "
        f"or completion target for {topic['terms'][1]} is {second} business days. Customers must "
        f"submit the request through the HamsAI portal and include the contract id, invoice date, "
        f"customer name, city, and affected service. The standard administrative amount for this "
        f"case is SAR {amount}, and the equivalent reference value is also recorded with Eastern "
        f"Arabic numeral marker {eastern_hint}. Exceptions require written approval from the "
        f"enterprise success manager. This section is intentionally close to other {topic['key']} "
        f"sections so retrieval must distinguish numbers, product names, dates, and negation. "
        f"The policy is effective from 30 May 2026 and remains valid until replaced by a newer "
        f"document. Source evidence must cite this exact document when answering."
    )
    return {
        "doc_id": f"{topic['key']}_en_{idx:03d}",
        "title": f"{topic['en_title']} - Section {idx}",
        "language": "en",
        "content": content,
        "source": f"kb/generated/{topic['key']}_en_{idx:03d}.txt",
        "category": topic["category"],
    }


def ar_doc(topic: dict, idx: int) -> dict:
    _, product_ar = PRODUCTS[idx % len(PRODUCTS)]
    location = LOCATIONS[3 + (idx % 3)]
    first = 2 + (idx % 7)
    second = first + 4
    amount = 200 + (idx * 90) % 5200
    eastern_amount = str(amount).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    content = (
        f"ينطبق قسم {topic['ar_title']} رقم {idx} على عملاء {product_ar} في {location}. "
        f"القاعدة الأساسية الخاصة بـ {topic['ar_terms'][0]} هي {first} أيام عمل، بينما تكون "
        f"مهلة {topic['ar_terms'][1]} {second} أيام عمل من تاريخ قبول الطلب. يجب على العميل "
        f"إرسال الطلب من خلال بوابة HamsAI مع رقم العقد وتاريخ الفاتورة واسم العميل والمدينة "
        f"والخدمة المتأثرة. تبلغ الرسوم الإدارية لهذه الحالة {amount} ريال سعودي ({eastern_amount} ر.س). "
        f"لا يتم تطبيق الاستثناءات إلا بعد موافقة مكتوبة من مدير نجاح العملاء. تم تصميم هذا النص "
        f"ليكون قريباً دلالياً من أقسام أخرى في {topic['key']} حتى يختبر النظام الأرقام والتواريخ "
        f"والأسماء وحالات النفي بدقة. تسري السياسة اعتباراً من 30 مايو 2026 ويجب الاستشهاد بهذا "
        f"المستند تحديداً عند الإجابة."
    )
    return {
        "doc_id": f"{topic['key']}_ar_{idx:03d}",
        "title": f"{topic['ar_title']} - قسم {idx}",
        "language": "ar",
        "content": content,
        "source": f"kb/generated/{topic['key']}_ar_{idx:03d}.txt",
        "category": topic["category"],
    }


def mixed_doc(topic: dict, idx: int) -> dict:
    product_en, product_ar = PRODUCTS[idx % len(PRODUCTS)]
    limit = 20 + (idx % 9) * 5
    amount = 300 + (idx * 110) % 6000
    eastern_limit = str(limit).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    content = (
        f"يوضح هذا المستند المختلط Mixed Policy رقم {idx} قواعد {topic['ar_title']} الخاصة بـ "
        f"{product_ar} / {product_en}. The operational threshold for {topic['terms'][2]} is "
        f"{limit} requests per month ({eastern_limit} طلب)، and the fee for exceeding it is "
        f"SAR {amount}. يجب استخدام المصطلح التقني {topic['terms'][3]} كما هو في تذاكر الدعم "
        f"لأن فرق التشغيل في الرياض وجدة والدمام تعتمد عليه في التصنيف. If the customer asks in "
        f"Arabic, the answer may still need to retrieve this mixed document because the product "
        f"name, API code, or SLA label appears in English. لا يجوز تعميم هذه القاعدة على منتجات "
        f"أخرى، ولا تنطبق عند وجود إعفاء مكتوب في العقد. Effective date: 30 May 2026."
    )
    return {
        "doc_id": f"{topic['key']}_mixed_{idx:03d}",
        "title": f"{topic['ar_title']} - {topic['en_title']} {idx}",
        "language": "mixed",
        "content": content,
        "source": f"kb/generated/{topic['key']}_mixed_{idx:03d}.txt",
        "category": topic["category"],
    }


def build_docs() -> list[dict]:
    docs: list[dict] = []
    for topic in TOPICS:
        for idx in range(1, 35):  # 10 topics × 34 variants × 3 langs = 1020 chunks
            docs.append(ar_doc(topic, idx))
            docs.append(en_doc(topic, idx))
            docs.append(mixed_doc(topic, idx))
    return docs


def chunked_from_docs(docs: list[dict]) -> list[dict]:
    return [
        {
            "doc_id": doc["doc_id"],
            "chunk_id": f"{doc['doc_id']}#0",
            "title": doc["title"],
            "language": doc["language"],
            "content": doc["content"],
            "source": doc["source"],
            "category": doc["category"],
            "page": 1,
        }
        for doc in docs
    ]


def query_for_doc(doc: dict, variant: int, target_lang: str | None = None) -> str:
    lang = target_lang or doc["language"]
    if lang == "ar":
        return random.choice(
            [
                f"ما هي القاعدة الأساسية في مستند {doc['title']}؟",
                f"كم تبلغ الرسوم أو المهلة المذكورة في {doc['doc_id']}؟",
                f"ما الشرط الذي ينطبق على العميل حسب {doc['title']}؟",
                f"اشرح سياسة {doc['category']} المرتبطة بهذا المستند.",
            ]
        )
    if lang == "mixed":
        return random.choice(
            [
                f"ما تفاصيل {doc['category']} for {doc['doc_id']}؟",
                f"What is the SAR fee and Arabic condition in {doc['title']}?",
                f"اذكر threshold و الرسوم في {doc['doc_id']}.",
            ]
        )
    return random.choice(
        [
            f"What is the main rule in {doc['title']}?",
            f"What fee, date, or deadline is stated in {doc['doc_id']}?",
            f"Which customer condition applies according to {doc['title']}?",
            f"Summarize the {doc['category']} policy in this document.",
        ]
    )


def answer_for_doc(doc: dict, answer_lang: str | None = None) -> str:
    lang = answer_lang or doc["language"]
    snippet = " ".join(doc["content"].split()[:45])
    if lang == "ar":
        return f"توجد الإجابة في المستند {doc['doc_id']}: {snippet}. المصدر: {doc['doc_id']}#0"
    if lang == "mixed":
        return f"الإجابة موجودة في {doc['doc_id']}: {snippet}. Source: {doc['doc_id']}#0"
    return f"The answer is in document {doc['doc_id']}: {snippet}. Source: {doc['doc_id']}#0"


def make_pairs_for_chunks(split_chunks: list[dict], negative_pool: list[dict], prefix: str) -> list[dict]:
    pairs: list[dict] = []
    by_category: dict[str, list[dict]] = {}
    for c in negative_pool:
        by_category.setdefault(c["category"], []).append(c)

    for chunk in split_chunks:
        same_category = [c for c in by_category[chunk["category"]] if c["chunk_id"] != chunk["chunk_id"]]
        other_chunks = [c for c in negative_pool if c["chunk_id"] != chunk["chunk_id"]]
        negatives = same_category or other_chunks

        query_langs = [chunk["language"]]
        if chunk["language"] == "en":
            query_langs.append("ar")
        elif chunk["language"] == "ar":
            query_langs.append("en")
        else:
            query_langs.extend(["ar", "en"])

        for v, q_lang in enumerate(query_langs[:3]):
            for repeat in range(2):
                neg = random.choice(negatives)
                q_type = "cross_lingual" if q_lang in ("ar", "en") and chunk["language"] in ("ar", "en") and q_lang != chunk["language"] else "monolingual"
                if chunk["language"] == "mixed" or q_lang == "mixed":
                    q_type = "mixed"
                pairs.append(
                    {
                        "id": f"{prefix}_{len(pairs)+1:04d}",
                        "query": query_for_doc(chunk, v + repeat, q_lang),
                        "positive": chunk["content"],
                        "hard_negative": neg["content"],
                        "positive_chunk_id": chunk["chunk_id"],
                        "hard_negative_chunk_id": neg["chunk_id"],
                        "source_doc_id": chunk["doc_id"],
                        "query_lang": q_lang if q_lang in ("ar", "en") else "mixed",
                        "positive_lang": chunk["language"],
                        "type": q_type,
                    }
                )

    random.shuffle(pairs)
    return pairs


def build_pairs(chunks: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    shuffled = chunks[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    train_chunks = shuffled[:train_end]
    val_chunks   = shuffled[train_end:val_end]
    test_chunks  = shuffled[val_end:]

    train = make_pairs_for_chunks(train_chunks, train_chunks, "train_pair")[:700]
    val   = make_pairs_for_chunks(val_chunks,   val_chunks,   "val_pair")[:120]
    test  = make_pairs_for_chunks(test_chunks,  test_chunks,  "test_pair")[:200]
    return train, val, test


def build_qa(chunks: list[dict], limit: int = 140) -> list[dict]:
    selected = chunks[:]
    random.shuffle(selected)
    qa: list[dict] = []
    for chunk in selected[:limit]:
        q_lang = random.choice(["ar", "en", chunk["language"]])
        if q_lang == "mixed":
            q_lang = "ar"
        q_type = "cross_lingual" if q_lang in ("ar", "en") and chunk["language"] in ("ar", "en") and q_lang != chunk["language"] else "monolingual"
        if chunk["language"] == "mixed":
            q_type = "mixed"
        qa.append(
            {
                "id": f"qa_{len(qa)+1:04d}",
                "query": query_for_doc(chunk, 0, q_lang),
                "question": query_for_doc(chunk, 1, q_lang),
                "positive": chunk["content"],
                "reference_answer": answer_for_doc(chunk, q_lang),
                "positive_chunk_id": chunk["chunk_id"],
                "source_doc_id": chunk["doc_id"],
                "question_lang": q_lang,
                "query_lang": q_lang,
                "answer_lang": q_lang,
                "type": q_type,
                "answerable": True,
            }
        )

    for idx in range(20):
        lang = "ar" if idx % 2 == 0 else "en"
        qa.append(
            {
                "id": f"qa_not_found_{idx+1:03d}",
                "query": "ما سياسة تأجير السيارات للموظفين؟" if lang == "ar" else "What is the employee car rental policy?",
                "question": "ما سياسة تأجير السيارات للموظفين؟" if lang == "ar" else "What is the employee car rental policy?",
                "positive": "",
                "reference_answer": "لم أجد هذه المعلومات في قاعدة المعرفة" if lang == "ar" else "This information was not found in the knowledge bank",
                "positive_chunk_id": None,
                "source_doc_id": None,
                "question_lang": lang,
                "query_lang": lang,
                "answer_lang": lang,
                "type": "not_found",
                "answerable": False,
            }
        )
    return qa


def build_conversations(chunks: list[dict]) -> dict:
    convs = []
    topic_groups: dict[str, list[dict]] = {}
    for c in chunks:
        topic_key = c["doc_id"].split("_", 1)[0]
        topic_groups.setdefault(topic_key, []).append(c)

    for idx, (category, group) in enumerate(list(topic_groups.items())[:10], start=1):
        turns = []
        for turn_no, chunk in enumerate(group[:5], start=1):
            lang = "ar" if (idx + turn_no) % 2 == 0 else "en"
            user = query_for_doc(chunk, turn_no, lang)
            turns.append(
                {
                    "turn": turn_no,
                    "user": user,
                    "standalone_query": user,
                    "expected_answer": answer_for_doc(chunk, lang),
                    "source_doc_id": chunk["doc_id"],
                    "positive_chunk_id": chunk["chunk_id"],
                    "lang": lang,
                }
            )
        convs.append(
            {
                "conversation_id": f"conv_{idx:03d}",
                "topic": category,
                "turns": turns,
            }
        )
    return {"conversations": convs}


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    ensure_dirs()
    docs = build_docs()
    ar_docs = [d for d in docs if d["language"] == "ar"]
    en_docs = [d for d in docs if d["language"] == "en"]
    mixed_docs = [d for d in docs if d["language"] == "mixed"]

    chunks = chunked_from_docs(docs)
    train, val, test_pairs = build_pairs(chunks)
    qa = build_qa(chunks)
    conversations = build_conversations(chunks)

    write_json(CORPUS_DIR / "arabic_policies.json", ar_docs)
    write_json(CORPUS_DIR / "english_policies.json", en_docs)
    write_json(CORPUS_DIR / "mixed_docs.json", mixed_docs)
    write_json(CORPUS_DIR / "chunked_corpus.json", chunks)
    write_json(TRAIN_DIR / "training_pairs.json", train + val + test_pairs)
    write_json(TRAIN_DIR / "train_split.json", train)
    write_json(TRAIN_DIR / "val_split.json", val)
    write_json(TEST_DIR / "qa_pairs.json", qa)
    write_json(TEST_DIR / "retrieval_pairs.json", test_pairs)
    write_json(TEST_DIR / "conversations.json", conversations)

    summary = {
        "seed": RANDOM_SEED,
        "documents": len(docs),
        "chunks": len(chunks),
        "train_pairs": len(train),
        "val_pairs": len(val),
        "retrieval_test_pairs": len(test_pairs),
        "qa_pairs": len(qa),
        "conversations": len(conversations["conversations"]),
        "license": "Synthetic dataset for assessment use; document as CC-BY-4.0 in README if submitted.",
    }
    write_json(Path("results/dataset_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
