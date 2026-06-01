import json
import re
import os
import sys
from pathlib import Path
from tqdm import tqdm

# ─── Arabic Normalization (same as prepare_data.py) ─────────────────────────

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

# ─── Elasticsearch Index ─────────────────────────────────────────────────────

def build_elasticsearch_index(chunks: list[dict], es_url: str, index_name: str):
    from elasticsearch import Elasticsearch, helpers

    es = Elasticsearch(es_url)

    if not es.ping():
        print(f"ERROR: Cannot connect to Elasticsearch at {es_url}")
        sys.exit(1)

    # Delete existing index
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Deleted existing index: {index_name}")

    # Create index with Arabic analyzer settings
    mappings = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "arabic_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "arabic_normalization", "arabic_stem"]
                    },
                    "english_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "porter_stem"]
                    }
                }
            },
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "doc_id":    {"type": "keyword"},
                "chunk_id":  {"type": "keyword"},
                "title":     {"type": "text", "analyzer": "arabic_analyzer"},
                "language":  {"type": "keyword"},
                "content":   {
                    "type": "text",
                    "analyzer": "arabic_analyzer",
                    "fields": {
                        "english": {"type": "text", "analyzer": "english_analyzer"}
                    }
                },
                "source":    {"type": "keyword"},
                "category":  {"type": "keyword"},
                "page":      {"type": "integer"}
            }
        }
    }

    es.indices.create(index=index_name, body=mappings)
    print(f"Created index: {index_name}")

    # Bulk index documents
    actions = []
    for chunk in chunks:
        normalized_content = normalize_text(chunk['content'], chunk['language'])
        actions.append({
            "_index": index_name,
            "_id": chunk['chunk_id'],
            "_source": {
                "doc_id":   chunk['doc_id'],
                "chunk_id": chunk['chunk_id'],
                "title":    chunk['title'],
                "language": chunk['language'],
                "content":  normalized_content,
                "source":   chunk['source'],
                "category": chunk['category'],
                "page":     chunk['page']
            }
        })

    success, failed = helpers.bulk(es, actions, stats_only=True)
    print(f"Indexed {success} chunks into Elasticsearch | Failed: {failed}")
    return success

# ─── Qdrant Vector Index ─────────────────────────────────────────────────────

def build_qdrant_index(chunks: list[dict], qdrant_url: str, collection_name: str, model_path: str):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer

    client = QdrantClient(url=qdrant_url)

    # Load embedding model
    print(f"Loading embedding model from: {model_path}")
    model = SentenceTransformer(model_path)
    vector_size = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {vector_size}")

    # Recreate collection (delete if exists, then create)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Created Qdrant collection: {collection_name}")

    # Encode and upload in batches
    batch_size = 32
    points = []

    texts = [normalize_text(c['content'], c['language']) for c in chunks]

    print(f"Encoding {len(texts)} chunks...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_texts = texts[i:i + batch_size]
        batch_chunks = chunks[i:i + batch_size]
        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        for j, (embedding, chunk) in enumerate(zip(embeddings, batch_chunks)):
            points.append(PointStruct(
                id=i + j,
                vector=embedding.tolist(),
                payload={
                    "doc_id":   chunk['doc_id'],
                    "chunk_id": chunk['chunk_id'],
                    "title":    chunk['title'],
                    "language": chunk['language'],
                    "content":  chunk['content'],
                    "source":   chunk['source'],
                    "category": chunk['category'],
                    "page":     chunk['page']
                }
            ))

    client.upsert(collection_name=collection_name, points=points)
    print(f"Uploaded {len(points)} vectors to Qdrant")
    return len(points)

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    print("=== HamsAI Index Builder ===")

    # Load config
    with open('./configs/serving_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load chunked corpus
    corpus_path = './data/corpus/chunked_corpus.json'
    if not Path(corpus_path).exists():
        print(f"ERROR: {corpus_path} not found. Run prepare_data.py first.")
        sys.exit(1)

    with open(corpus_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # Build Elasticsearch index
    print("\n--- Building Elasticsearch Index (Quick Search) ---")
    es_url        = config['quick_search']['url']
    index_name    = config['quick_search']['index_name']
    build_elasticsearch_index(chunks, es_url, index_name)

    # Build Qdrant index
    print("\n--- Building Qdrant Vector Index (Smart AI + Interactive) ---")
    qdrant_url      = config['qdrant']['url']
    collection_name = config['qdrant']['collection_name']

    model_path = config['embedding']['model_path']
    if not Path(model_path).exists():
        if config['embedding'].get('allow_base_model_fallback', False):
            print(f"Fine-tuned model not found at {model_path}, using fallback: {config['embedding']['fallback_model']}")
            model_path = config['embedding']['fallback_model']
        else:
            print(f"ERROR: Fine-tuned embedding model not found at {model_path}. Run train_embeddings.py or download the HF artifact first.")
            sys.exit(1)

    build_qdrant_index(chunks, qdrant_url, collection_name, model_path)

    print("\n=== Index build complete ===")
