import json
import re
import os
import sys
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

# ─── Dependencies ─────────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    from elasticsearch import Elasticsearch, helpers
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
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
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total = arabic_chars + english_chars
    if total == 0:
        return 'en'
    arabic_ratio = arabic_chars / total
    if arabic_ratio > 0.7:
        return 'ar'
    elif arabic_ratio < 0.3:
        return 'en'
    else:
        return 'mixed'

# ─── Document Parsers ─────────────────────────────────────────────────────────

def parse_txt(file_path: Path) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_pdf(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text
    except Exception as e:
        print(f"ERROR reading PDF {file_path}: {e}")
        return ''

def parse_docx(file_path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(file_path))
        return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"ERROR reading DOCX {file_path}: {e}")
        return ''

def parse_html(file_path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        return soup.get_text(separator=' ')
    except Exception as e:
        print(f"ERROR reading HTML {file_path}: {e}")
        return ''

def parse_document(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == '.txt':
        return parse_txt(file_path)
    elif ext == '.pdf':
        return parse_pdf(file_path)
    elif ext == '.docx':
        return parse_docx(file_path)
    elif ext in ('.html', '.htm'):
        return parse_html(file_path)
    else:
        print(f"WARNING: Unsupported file type {ext}, skipping {file_path.name}")
        return ''

# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ─── Index to Elasticsearch ───────────────────────────────────────────────────

def index_to_elasticsearch(
    es: Elasticsearch,
    index_name: str,
    chunks: list[dict]
) -> int:
    actions = []
    for chunk in chunks:
        normalized = normalize_text(chunk['content'], chunk['language'])
        actions.append({
            "_index": index_name,
            "_id"   : chunk['chunk_id'],
            "_source": {
                "doc_id"  : chunk['doc_id'],
                "chunk_id": chunk['chunk_id'],
                "title"   : chunk['title'],
                "language": chunk['language'],
                "content" : normalized,
                "source"  : chunk['source'],
                "category": chunk['category'],
                "page"    : chunk['page']
            }
        })

    if not actions:
        return 0

    success, failed = helpers.bulk(es, actions, stats_only=True)
    return success

# ─── Index to Qdrant ──────────────────────────────────────────────────────────

def index_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer,
    chunks: list[dict],
    start_id: int = 0
) -> int:
    texts = [normalize_text(c['content'], c['language']) for c in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32
    )

    points = []
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        points.append(PointStruct(
            id=start_id + i,
            vector=embedding.tolist(),
            payload={
                "doc_id"  : chunk['doc_id'],
                "chunk_id": chunk['chunk_id'],
                "title"   : chunk['title'],
                "language": chunk['language'],
                "content" : chunk['content'],
                "source"  : chunk['source'],
                "category": chunk['category'],
                "page"    : chunk['page']
            }
        ))

    client.upsert(collection_name=collection_name, points=points)
    return len(points)

# ─── Ingest Single File ───────────────────────────────────────────────────────

def ingest_file(
    file_path: Path,
    es: Elasticsearch,
    qdrant_client: QdrantClient,
    model: SentenceTransformer,
    config: dict,
    existing_count: int = 0
) -> int:
    print(f"\nIngesting: {file_path.name}")

    # Parse
    raw_text = parse_document(file_path)
    if not raw_text.strip():
        print(f"WARNING: Empty content in {file_path.name}, skipping.")
        return 0

    # Detect language
    language = detect_language(raw_text)
    print(f"  Language detected : {language}")

    # Chunk
    chunk_cfg  = config['chunking']
    raw_chunks = chunk_text(
        raw_text,
        chunk_size=chunk_cfg['chunk_size'],
        overlap=chunk_cfg['chunk_overlap']
    )
    print(f"  Chunks created    : {len(raw_chunks)}")

    # Build chunk dicts
    doc_id = file_path.stem
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunks.append({
            "doc_id"  : doc_id,
            "chunk_id": f"{doc_id}#{i}",
            "title"   : file_path.stem.replace('_', ' ').title(),
            "language": language,
            "content" : chunk,
            "source"  : str(file_path),
            "category": "ingested",
            "page"    : i + 1
        })

    # Index to Elasticsearch
    index_name = config['quick_search']['index_name']
    es_count   = index_to_elasticsearch(es, index_name, chunks)
    print(f"  Elasticsearch     : {es_count} chunks indexed")

    # Index to Qdrant
    collection_name = config['qdrant']['collection_name']
    qdrant_count    = index_to_qdrant(
        qdrant_client,
        collection_name,
        model,
        chunks,
        start_id=existing_count
    )
    print(f"  Qdrant            : {qdrant_count} vectors uploaded")

    return len(chunks)

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HamsAI Document Ingestion")
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Path to a file or directory to ingest (PDF, DOCX, TXT, HTML)'
    )
    args = parser.parse_args()

    print("=== HamsAI Ingest Pipeline ===\n")

    # Load config
    config_path = './configs/serving_config.yaml'
    if not Path(config_path).exists():
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Connect Elasticsearch
    es_url = config['quick_search']['url']
    es     = Elasticsearch(es_url)
    if not es.ping():
        print(f"ERROR: Cannot connect to Elasticsearch at {es_url}")
        sys.exit(1)
    print(f"Elasticsearch connected : {es_url}")

    # Connect Qdrant
    qdrant_url    = config['qdrant']['url']
    qdrant_client = QdrantClient(url=qdrant_url)
    print(f"Qdrant connected        : {qdrant_url}")

    # Load embedding model
    model_path = config['embedding']['model_path']
    if not Path(model_path).exists():
        if config['embedding'].get('allow_base_model_fallback', False):
            print(f"Fine-tuned model not found at {model_path}")
            print(f"Using fallback: {config['embedding']['fallback_model']}")
            model_path = config['embedding']['fallback_model']
        else:
            print(f"ERROR: Fine-tuned embedding model not found at {model_path}")
            sys.exit(1)

    print(f"Loading embedding model : {model_path}")
    model = SentenceTransformer(model_path)

    # Resolve input path
    input_path = Path(args.path)
    if not input_path.exists():
        print(f"ERROR: Path does not exist: {input_path}")
        sys.exit(1)

    # Collect files
    supported = {'.pdf', '.docx', '.txt', '.html', '.htm'}
    if input_path.is_file():
        files = [input_path] if input_path.suffix.lower() in supported else []
    else:
        files = [f for f in input_path.rglob('*') if f.suffix.lower() in supported]

    if not files:
        print(f"ERROR: No supported files found at {input_path}")
        sys.exit(1)

    print(f"\nFiles to ingest: {len(files)}")

    # Ingest
    total_chunks = 0
    for file_path in tqdm(files, desc="Ingesting files"):
        count = ingest_file(
            file_path,
            es,
            qdrant_client,
            model,
            config,
            existing_count=total_chunks
        )
        total_chunks += count

    print(f"\nTotal chunks ingested: {total_chunks}")
    print("\n=== ingest.py complete ===")
