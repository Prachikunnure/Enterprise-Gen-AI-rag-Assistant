"""
FAISS vector index with JSON metadata aligned by row index.
"""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.config import EMBEDDING_DIMENSION
from src.utils import setup_logging

logger = setup_logging()


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for cosine similarity on L2-normalized vectors (inner product).
    """
    if embeddings.size == 0:
        raise ValueError("Cannot build index: no embeddings provided.")

    dim = embeddings.shape[1]
    if dim != EMBEDDING_DIMENSION:
        logger.warning(
            "Embedding dimension %s differs from config %s; using actual dim %s",
            dim,
            EMBEDDING_DIMENSION,
            dim,
        )

    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    # IndexFlatIP = inner product; with normalized vectors this equals cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built with %s vectors, dim=%s", index.ntotal, dim)
    return index


def save_index(index: faiss.Index, path: str | Path) -> None:
    """Persist FAISS index to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info("Saved FAISS index to %s", path)


def load_index(path: str | Path) -> faiss.Index:
    """Load FAISS index from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector index not found: {path}")
    index = faiss.read_index(str(path))
    logger.info("Loaded FAISS index from %s (%s vectors)", path, index.ntotal)
    return index


def save_metadata(chunks: list[dict[str, Any]], path: str | Path) -> None:
    """Save chunk metadata; list index must match FAISS row index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Saved metadata for %s chunks to %s", len(chunks), path)


def load_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Load chunk metadata from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded metadata for %s chunks", len(chunks))
    return chunks


def search_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the index. Returns (scores, indices) for top_k neighbors.
    Scores are cosine similarity when vectors are normalized.
    """
    if index.ntotal == 0:
        return np.array([]), np.array([])

    q = np.ascontiguousarray(query_embedding.astype(np.float32).reshape(1, -1))
    k = min(top_k, index.ntotal)
    scores, indices = index.search(q, k)
    return scores[0], indices[0]
