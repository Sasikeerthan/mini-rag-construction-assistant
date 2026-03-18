"""FAISS-based retrieval module for the RAG pipeline."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import faiss

from rag.chunker import chunk_documents
from rag.embedder import get_embeddings

CACHE_DIR = os.environ.get("RAG_CACHE_DIR", "/app/.cache")


def _docs_hash(doc_dir: str) -> str:
    """Compute a hash of all markdown files to detect changes."""
    h = hashlib.sha256()
    for p in sorted(Path(doc_dir).glob("*.md")):
        h.update(p.read_bytes())
    return h.hexdigest()


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from normalized embeddings.

    Since the embeddings are L2-normalized, inner product equals cosine similarity.

    Args:
        embeddings: np.ndarray of shape (n, dim) with normalized vectors.

    Returns:
        A faiss.IndexFlatIP index ready for search.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def search(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Embed a query, search the FAISS index, and return top-k chunks with scores.

    Args:
        query: The search query string.
        index: A FAISS IndexFlatIP built from chunk embeddings.
        chunks: The list of chunk dicts (from chunker.chunk_documents).
        top_k: Number of results to return.

    Returns:
        List of chunk dicts, each augmented with a "score" key (cosine similarity).
    """
    query_embedding = get_embeddings([query])
    top_k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        result = dict(chunks[idx])
        result["score"] = float(score)
        results.append(result)

    return results


def _load_cache(doc_dir: str) -> tuple[faiss.IndexFlatIP, list[dict]] | None:
    """Try to load cached index and chunks if documents haven't changed."""
    cache_path = Path(CACHE_DIR)
    hash_file = cache_path / "docs_hash.txt"
    index_file = cache_path / "faiss.index"
    chunks_file = cache_path / "chunks.json"

    if not all(f.exists() for f in [hash_file, index_file, chunks_file]):
        return None

    cached_hash = hash_file.read_text().strip()
    if cached_hash != _docs_hash(doc_dir):
        return None

    index = faiss.read_index(str(index_file))
    chunks = json.loads(chunks_file.read_text())
    return index, chunks


def _save_cache(doc_dir: str, index: faiss.IndexFlatIP, chunks: list[dict]):
    """Save index and chunks to disk cache."""
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)

    (cache_path / "docs_hash.txt").write_text(_docs_hash(doc_dir))
    faiss.write_index(index, str(cache_path / "faiss.index"))
    (cache_path / "chunks.json").write_text(json.dumps(chunks))


def build_rag_index(doc_dir: str) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Chunk docs, embed, build index — with disk caching.

    On first run, embeds all chunks and saves the index to disk.
    On subsequent runs, loads from cache if documents haven't changed.

    Args:
        doc_dir: Path to directory containing .md files.

    Returns:
        Tuple of (faiss index, list of chunk dicts).
    """
    cached = _load_cache(doc_dir)
    if cached is not None:
        return cached

    chunks = chunk_documents(doc_dir)
    texts = [chunk["content"] for chunk in chunks]
    embeddings = get_embeddings(texts)
    index = build_index(embeddings)

    _save_cache(doc_dir, index, chunks)
    return index, chunks
