"""
Document and query embeddings using sentence-transformers.

An embedding is a fixed-size numeric vector that captures semantic meaning of text.
Similar meanings → vectors that are close together (measured by cosine similarity).
Text → embedding vector (e.g. 384 floats for all-MiniLM-L6-v2).
"""

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME
from src.utils import setup_logging

logger = setup_logging()

_model: SentenceTransformer | None = None


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load the embedding model once and reuse it (avoids slow reloads)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model ready")
    return _model


def embed_documents(
    chunks: list[dict[str, Any]],
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    """
    Embed all chunk texts. Returns shape (num_chunks, embedding_dim).
    """
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)

    st_model = model or load_embedding_model()
    texts = [c["text"] for c in chunks]
    logger.info("Embedding %s document chunks", len(texts))
    vectors = st_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(
    query: str,
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    """
    Embed a single user question. Returns shape (embedding_dim,).
    """
    st_model = model or load_embedding_model()
    vec = st_model.encode(
        query,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(vec, dtype=np.float32)
