"""
End-to-end RAG: retrieve context, call LLM, return answer and sources.
"""

from typing import Any

import faiss
from sentence_transformers import SentenceTransformer

from src.llm import LLMConfigurationError, format_context, generate_answer
from src.retriever import retrieve_relevant_chunks
from src.utils import setup_logging

logger = setup_logging()

NOT_FOUND_ANSWER = "I could not find this information in the uploaded documents."


def _unique_sources(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate sources by (source, page) preserving order."""
    seen: set[tuple[str, int]] = set()
    sources: list[dict[str, Any]] = []
    for ch in retrieved:
        key = (str(ch.get("source", "")), int(ch.get("page", 0)))
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": key[0], "page": key[1]})
    return sources


def answer_question(
    question: str,
    index: faiss.Index,
    chunks: list[dict[str, Any]],
    embedding_model: SentenceTransformer,
    top_k: int = 5,
    similarity_threshold: float = 0.35,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Full RAG pipeline: embed query → retrieve → prompt LLM → answer + citations.
    """
    retrieved = retrieve_relevant_chunks(
        query=question,
        index=index,
        chunks=chunks,
        embedding_model=embedding_model,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    if not retrieved:
        logger.info("No relevant chunks above threshold")
        return {
            "answer": NOT_FOUND_ANSWER,
            "sources": [],
            "retrieved_chunks": [],
        }

    context = format_context(retrieved)

    try:
        answer = generate_answer(
            question=question,
            context=context,
            conversation_history=conversation_history,
        )
    except LLMConfigurationError as exc:
        return {
            "answer": str(exc),
            "sources": _unique_sources(retrieved),
            "retrieved_chunks": retrieved,
            "error": "llm_config",
        }
    except RuntimeError as exc:
        return {
            "answer": str(exc),
            "sources": _unique_sources(retrieved),
            "retrieved_chunks": retrieved,
            "error": "llm_api",
        }

    return {
        "answer": answer,
        "sources": _unique_sources(retrieved),
        "retrieved_chunks": retrieved,
    }


def process_documents_pipeline(
    pdf_paths: list[str],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> tuple[faiss.Index, list[dict[str, Any]], SentenceTransformer]:
    """
    Ingest PDFs: load → chunk → embed → build FAISS index.
    Saves index and metadata to configured paths.
    """
    from src.config import (
        CHUNK_OVERLAP_WORDS,
        CHUNK_SIZE_WORDS,
        METADATA_PATH,
        VECTOR_INDEX_PATH,
    )
    from src.embeddings import embed_documents, load_embedding_model
    from src.pdf_loader import load_multiple_pdfs
    from src.text_processor import create_chunks
    from src.vector_store import build_index, save_index, save_metadata

    size = chunk_size if chunk_size is not None else CHUNK_SIZE_WORDS
    ov = overlap if overlap is not None else CHUNK_OVERLAP_WORDS

    pages = load_multiple_pdfs(pdf_paths)
    doc_chunks = create_chunks(pages, chunk_size=size, overlap=ov)
    model = load_embedding_model()
    embeddings = embed_documents(doc_chunks, model)
    index = build_index(embeddings)
    save_index(index, VECTOR_INDEX_PATH)
    save_metadata(doc_chunks, METADATA_PATH)
    return index, doc_chunks, model
