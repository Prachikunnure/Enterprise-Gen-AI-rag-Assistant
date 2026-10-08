"""Tests for FAISS retrieval structure."""

import numpy as np

from src.retriever import retrieve_relevant_chunks
from src.vector_store import build_index


class _FakeModel:
    """Deterministic tiny embeddings for tests (no model download)."""

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            # Simple hash-based 384-d vector
            v = np.zeros(384, dtype=np.float32)
            for i, ch in enumerate(t[:384]):
                v[i % 384] += ord(ch) * 0.001
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            out.append(v)
        if len(out) == 1 and isinstance(texts, list) and len(texts) == 1:
            return out[0]
        return np.stack(out)


def test_retrieval_returns_expected_structure():
    chunks = [
        {"chunk_id": 0, "text": "leave policy annual days", "source": "handbook.pdf", "page": 3},
        {"chunk_id": 1, "text": "remote work two days per week", "source": "remote.pdf", "page": 1},
        {"chunk_id": 2, "text": "cafeteria menu lunch", "source": "misc.pdf", "page": 9},
    ]
    texts = [c["text"] for c in chunks]
    model = _FakeModel()
    embeddings = np.stack([model.encode(t) for t in texts])
    index = build_index(embeddings)

    results = retrieve_relevant_chunks(
        query="remote work policy",
        index=index,
        chunks=chunks,
        embedding_model=model,
        top_k=2,
        similarity_threshold=0.0,
    )

    assert len(results) <= 2
    assert len(results) >= 1
    for r in results:
        assert "text" in r
        assert "source" in r
        assert "page" in r
        assert "score" in r
