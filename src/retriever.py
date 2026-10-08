"""
Retrieve relevant document chunks for a user question.
"""

from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DEFAULT_SIMILARITY_THRESHOLD
from src.embeddings import embed_query
from src.vector_store import search_index
from src.utils import setup_logging

logger = setup_logging()


def retrieve_relevant_chunks(
    query: str,
    index: faiss.Index,
    chunks: list[dict[str, Any]],
    embedding_model: SentenceTransformer,
    top_k: int = 5,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    1. Embed the question
    2. Search FAISS for top-k
    3. Map indices to chunk metadata and filter by threshold
    """
    if not query.strip():
        return []

    if index is None or index.ntotal == 0:
        logger.warning("Retrieval skipped: empty index")
        return []

    if len(chunks) != index.ntotal:
        logger.error(
            "Metadata/index mismatch: %s chunks vs %s index vectors",
            len(chunks),
            index.ntotal,
        )
        raise ValueError("Chunk metadata does not match FAISS index size.")

    query_vec = embed_query(query, embedding_model)
    scores, indices = search_index(index, query_vec, top_k=top_k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        idx = int(idx)
        if idx >= len(chunks):
            continue
        sim = float(score)
        if sim < similarity_threshold:
            logger.debug("Filtered chunk %s with score %.3f below threshold", idx, sim)
            continue
        chunk = chunks[idx]
        results.append(
            {
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "unknown"),
                "page": chunk.get("page", 0),
                "chunk_id": chunk.get("chunk_id", idx),
                "score": sim,
            }
        )

    logger.info("Retrieval returned %s chunks (top_k=%s)", len(results), top_k)
    return results
