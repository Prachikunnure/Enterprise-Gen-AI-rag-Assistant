"""Tests for text chunking."""

from src.text_processor import clean_text, create_chunks


def test_clean_text_collapses_whitespace():
    raw = "Hello   world\n\n\n\nNext line"
    cleaned = clean_text(raw)
    assert "Hello world" in cleaned
    assert "\n\n\n" not in cleaned


def test_create_chunks_preserves_metadata():
    pages = [
        {"text": "word " * 600, "page": 2, "source": "policy.pdf"},
    ]
    chunks = create_chunks(pages, chunk_size=100, overlap=10)
    assert len(chunks) >= 2
    assert all(c["source"] == "policy.pdf" for c in chunks)
    assert all(c["page"] == 2 for c in chunks)
    assert chunks[0]["chunk_id"] == 0
    assert "text" in chunks[0] and len(chunks[0]["text"]) > 0


def test_chunk_ids_increment():
    pages = [{"text": "a b c d e f g h i j " * 50, "page": 1, "source": "a.pdf"}]
    chunks = create_chunks(pages, chunk_size=20, overlap=5)
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))
