import json
import re
import os
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

# ─── Arabic Normalization ───────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F]', '', text)
    # Normalize alef variants → ا
    text = re.sub(r'[أإآٱ]', 'ا', text)
    # Normalize ya → ي
    text = re.sub(r'ى', 'ي', text)
    # Normalize teh marbuta → ه
    text = re.sub(r'ة', 'ه', text)
    # Remove tatweel
    text = re.sub(r'ـ', '', text)
    # Normalize Eastern Arabic numerals → Western
    eastern = '٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹'
    western = '01234567890123456789'
    for e, w in zip(eastern, western):
        text = text.replace(e, w)
    return text.strip()

def normalize_english(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text(text: str, language: str) -> str:
    if language == 'ar':
        return normalize_arabic(text)
    elif language == 'en':
        return normalize_english(text)
    else:
        # mixed — apply both
        text = normalize_arabic(text)
        text = normalize_english(text)
        return text

# ─── Chunking ───────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        if len(chunk.strip()) > 50:  # skip tiny chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ─── Load Corpus ────────────────────────────────────────────────────────────

def load_corpus(corpus_dir: str) -> list[dict]:
    corpus_files = [
        'arabic_policies.json',
        'english_policies.json',
        'mixed_docs.json'
    ]
    all_docs = []
    for fname in corpus_files:
        fpath = Path(corpus_dir) / fname
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping.")
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} docs from {fname}")
    return all_docs

# ─── Build Chunked Corpus ───────────────────────────────────────────────────

def build_chunked_corpus(docs: list[dict]) -> list[dict]:
    chunked = []
    for doc in tqdm(docs, desc="Chunking documents"):
        lang = doc.get('language', 'en')
        content = doc.get('content', '')
        normalized = normalize_text(content, lang)
        chunks = chunk_text(normalized)
        for i, chunk in enumerate(chunks):
            chunked.append({
                'doc_id': doc['doc_id'],
                'chunk_id': f"{doc['doc_id']}#{i}",
                'title': doc.get('title', ''),
                'language': lang,
                'content': chunk,
                'source': doc.get('source', ''),
                'category': doc.get('category', ''),
                'page': i + 1
            })
    return chunked

# ─── Split Training Data ────────────────────────────────────────────────────

def split_training_data(train_file: str, output_dir: str):
    with open(train_file, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    total = len(pairs)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train = pairs[:train_end]
    val = pairs[train_end:val_end]
    test = pairs[val_end:]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/train_split.json", 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(f"{output_dir}/val_split.json", 'w', encoding='utf-8') as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from expand_dataset import main as build_expanded_dataset

        print("=== HamsAI Expanded Data Preparation ===")
        build_expanded_dataset()
        print("\n prepare_data.py complete.")
        raise SystemExit(0)
    except ImportError:
        print("WARNING: expand_dataset.py not found; falling back to legacy preparation.")

    print("=== HamsAI Data Preparation ===")

    # Load and chunk corpus
    docs = load_corpus('./data/corpus')
    print(f"\nTotal documents: {len(docs)}")

    chunked = build_chunked_corpus(docs)
    print(f"Total chunks: {len(chunked)}")

    # Save chunked corpus
    out_path = './data/corpus/chunked_corpus.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)
    print(f"Saved chunked corpus → {out_path}")

    # Split training data
    print("\n=== Splitting Training Data ===")
    split_training_data(
        train_file='./data/train/training_pairs.json',
        output_dir='./data/train'
    )

    print("\n prepare_data.py complete.")
