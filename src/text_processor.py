"""
Text cleaning and chunking for RAG.
"""

import re
from typing import Any

from src.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS
from src.utils import setup_logging

logger = setup_logging()


def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving sentence meaning.
    """
    if not text:
        return ""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces/tabs on each line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Remove repeated blank lines
    cleaned_lines: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank
    return "\n".join(cleaned_lines).strip()


def _words(text: str) -> list[str]:
    return text.split()


def _take_words(words: list[str], start: int, count: int) -> str:
    return " ".join(words[start : start + count])


def create_chunks(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """
    Split page text into overlapping word-based chunks.

    We use words (not characters) because chunk_size=500 is easier to reason about
    for beginners and roughly maps to token counts for English prose.

    Each chunk keeps source filename and page number from the page it was built from.
    When a page is longer than chunk_size, multiple chunks share the same page metadata.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
    chunk_id = 0
    step = chunk_size - overlap

    for page in pages:
        text = clean_text(page.get("text", ""))
        if not text:
            continue
        source = page.get("source", "unknown")
        page_num = int(page.get("page", 0))
        words = _words(text)
        if not words:
            continue

        start = 0
        while start < len(words):
            piece = _take_words(words, start, chunk_size)
            if piece.strip():
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": piece,
                        "source": source,
                        "page": page_num,
                    }
                )
                chunk_id += 1
            if start + chunk_size >= len(words):
                break
            start += step

    logger.info("Created %s chunks from %s pages", len(chunks), len(pages))
    return chunks
