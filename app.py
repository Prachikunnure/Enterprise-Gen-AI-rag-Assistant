"""
Streamlit UI for Enterprise GenAI RAG Assistant.
Run: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when Streamlit runs this file
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    METADATA_PATH,
    VECTOR_INDEX_PATH,
    ensure_directories,
)
from src.embeddings import load_embedding_model
from src.llm import get_llm_status_message
from src.pdf_loader import PDFLoadError
from src.rag_pipeline import answer_question, process_documents_pipeline
from src.utils import setup_logging, validate_question
from src.vector_store import load_index, load_metadata

logger = setup_logging()


def init_session_state() -> None:
    defaults = {
        "messages": [],  # chat history: {"role": "user"|"assistant", "content": str}
        "index_ready": False,
        "faiss_index": None,
        "chunks": [],
        "embedding_model": None,
        "last_sources": [],
        "process_status": "Upload PDFs and click Process Documents.",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def try_load_existing_index() -> None:
    """Load saved FAISS index and metadata if present."""
    if st.session_state.get("index_ready") and st.session_state.get("faiss_index") is not None:
        return
    if not VECTOR_INDEX_PATH.exists() or not METADATA_PATH.exists():
        return
    try:
        st.session_state["faiss_index"] = load_index(VECTOR_INDEX_PATH)
        st.session_state["chunks"] = load_metadata(METADATA_PATH)
        st.session_state["embedding_model"] = load_embedding_model()
        st.session_state["index_ready"] = True
        st.session_state["process_status"] = (
            f"Loaded existing index ({st.session_state['faiss_index'].ntotal} chunks)."
        )
        logger.info("Restored index from disk")
    except Exception as exc:
        logger.error("Could not load existing index: %s", exc)
        st.session_state["process_status"] = f"Could not load saved index: {exc}"


def save_uploaded_files(uploaded_files) -> list[Path]:
    ensure_directories()
    paths: list[Path] = []
    for uf in uploaded_files:
        dest = PROJECT_ROOT / "data" / "uploads" / uf.name
        dest.write_bytes(uf.getbuffer())
        paths.append(dest)
    return paths


def render_sidebar() -> tuple[int, float]:
    st.sidebar.header("Documents")
    uploaded = st.sidebar.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    top_k = st.sidebar.slider("Top K (retrieved chunks)", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    threshold = st.sidebar.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULT_SIMILARITY_THRESHOLD),
        step=0.05,
        help="Chunks below this cosine similarity are not sent to the LLM.",
    )

    if st.sidebar.button("Process Documents", type="primary", use_container_width=True):
        if not uploaded:
            st.sidebar.error("Please upload at least one PDF first.")
        else:
            with st.sidebar.status("Processing...", expanded=True) as status:
                try:
                    paths = save_uploaded_files(uploaded)
                    for p in paths:
                        st.write(f"Saved {p.name}")
                    index, chunks, model = process_documents_pipeline([str(p) for p in paths])
                    st.session_state["faiss_index"] = index
                    st.session_state["chunks"] = chunks
                    st.session_state["embedding_model"] = model
                    st.session_state["index_ready"] = True
                    st.session_state["messages"] = []
                    st.session_state["process_status"] = (
                        f"Processed {len(paths)} file(s), {len(chunks)} chunks, index ready."
                    )
                    status.update(label="Done", state="complete")
                    st.sidebar.success(st.session_state["process_status"])
                except (PDFLoadError, FileNotFoundError, ValueError) as exc:
                    status.update(label="Failed", state="error")
                    st.sidebar.error(str(exc))
                except Exception as exc:
                    status.update(label="Failed", state="error")
                    logger.exception("Processing failed")
                    st.sidebar.error(f"Processing failed: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader("System status")
    if st.session_state.get("index_ready"):
        n = st.session_state["faiss_index"].ntotal if st.session_state.get("faiss_index") else 0
        st.sidebar.success("Documents processed")
        st.sidebar.success("Embeddings in index")
        st.sidebar.success(f"Vector index ready ({n} vectors)")
    else:
        st.sidebar.warning("Index not ready")
    st.sidebar.caption(st.session_state.get("process_status", ""))
    st.sidebar.caption(get_llm_status_message())

    return top_k, threshold


def render_chat(top_k: int, threshold: float) -> None:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                st.markdown("**Sources:**")
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"{i}. `{src['source']}` — Page {src['page']}")

    question = st.chat_input("Ask a question about your documents...")
    if question:
        err = validate_question(question)
        if err:
            st.warning(err)
            return
        if not st.session_state.get("index_ready") or st.session_state.get("faiss_index") is None:
            st.error("Please upload PDFs and click **Process Documents** before asking questions.")
            return

        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        history_for_llm = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["messages"][:-1]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    if st.session_state.get("embedding_model") is None:
                        st.session_state["embedding_model"] = load_embedding_model()
                    result = answer_question(
                        question=question,
                        index=st.session_state["faiss_index"],
                        chunks=st.session_state["chunks"],
                        embedding_model=st.session_state["embedding_model"],
                        top_k=top_k,
                        similarity_threshold=threshold,
                        conversation_history=history_for_llm,
                    )
                except Exception as exc:
                    logger.exception("RAG failed")
                    result = {"answer": f"Something went wrong: {exc}", "sources": [], "retrieved_chunks": []}

            st.markdown(result["answer"])
            sources = result.get("sources") or []
            if sources:
                st.markdown("**Sources:**")
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"{i}. `{src['source']}` — Page {src['page']}")

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": sources,
            }
        )


def main() -> None:
    st.set_page_config(
        page_title="Enterprise GenAI RAG Assistant",
        page_icon="📄",
        layout="wide",
    )
    ensure_directories()
    init_session_state()
    try_load_existing_index()

    st.title("Enterprise GenAI RAG Assistant")
    st.caption("Ask questions about your uploaded documents using Retrieval-Augmented Generation.")

    top_k, threshold = render_sidebar()
    render_chat(top_k, threshold)


if __name__ == "__main__":
    main()
