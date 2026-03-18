"""Embedding module using sentence-transformers for the RAG pipeline."""

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load and cache the sentence-transformer model as a singleton."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Embed a list of texts and return normalized embeddings as a numpy array.

    Args:
        texts: List of strings to embed.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim) with normalized vectors.
    """
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)
