"""
streamlit_app.py
===================
Streamlit front-end for the RAG application.

Wires together every numbered pipeline stage (01-07) into an interactive UI:
    1. Upload documents (.txt, .md, .pdf, .docx)
    2. Build the index (preprocess -> chunk -> embed -> store in ChromaDB)
    3. Ask a question
    4. See a grounded, cited answer plus the retrieved evidence behind it

API key handling
-----------------
Following the Student RAG Project Instructions exactly: the real API key is
never hardcoded. Locally it is read from a `.env` file (via
`OPENROUTER_API_KEY` / `OPENROUTER_MODEL` environment variables). On
Streamlit Cloud it is read from `st.secrets` using the same fallback pattern
shown in the instructions PDF.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from typing import Any, Dict, List

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

documents_module = importlib.import_module("01_documents")
preprocessing_module = importlib.import_module("02_preprocessing")
chunking_module = importlib.import_module("03_chunking")
vector_module = importlib.import_module("04_vector_representation")
store_module = importlib.import_module("05_create_chroma_store")
retrieve_module = importlib.import_module("06_retrieve_context")
prompting_module = importlib.import_module("07_prompting")

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")


def _load_api_key_from_secrets() -> None:
    """
    Fill in OPENROUTER_API_KEY / OPENROUTER_MODEL from Streamlit Secrets if
    they were not already provided as environment variables.

    This mirrors the exact fallback pattern given in the Student RAG Project
    Instructions PDF.
    """
    try:
        if not prompting_module.OPENROUTER_API_KEY:
            prompting_module.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        prompting_module.OPENROUTER_MODEL = st.secrets.get(
            "OPENROUTER_MODEL", prompting_module.OPENROUTER_MODEL
        )
    except Exception:  # noqa: BLE001 - st.secrets raises if no secrets.toml exists locally
        pass


_load_api_key_from_secrets()


def _init_session_state() -> None:
    """Initialize all Streamlit session_state keys used by this app."""
    defaults = {
        "collection": None,
        "index_built": False,
        "num_documents": 0,
        "num_chunks": 0,
        "chat_history": [],  # list of {"query": ..., "result": ...}
        "persist_dir": tempfile.mkdtemp(prefix="rag_chroma_"),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


def build_index_from_uploads(uploaded_files: List[Any]) -> None:
    """
    Run the full pipeline (load -> preprocess -> chunk -> embed -> store) on
    a set of uploaded files and save the resulting collection in session state.

    Args:
        uploaded_files: Files from `st.file_uploader`.
    """
    with st.spinner("Loading documents..."):
        documents = documents_module.load_documents_from_uploads(uploaded_files)

    with st.spinner("Cleaning text..."):
        documents = preprocessing_module.preprocess_documents(documents)

    with st.spinner("Chunking documents..."):
        chunks = chunking_module.chunk_documents(documents, adaptive=True)

    with st.spinner("Generating embeddings (first run downloads the model, ~80MB)..."):
        collection = store_module.build_vector_store(
            chunks,
            persist_directory=st.session_state["persist_dir"],
            collection_name="rag_documents",
            reset=True,
        )

    st.session_state["collection"] = collection
    st.session_state["index_built"] = True
    st.session_state["num_documents"] = len(documents)
    st.session_state["num_chunks"] = len(chunks)
    st.session_state["chat_history"] = []


def build_index_from_samples() -> None:
    """Build the index from the bundled sample corpus (data/raw/)."""
    with st.spinner("Building index from the sample corpus..."):
        collection, num_documents, num_chunks = store_module.run_full_pipeline(
            persist_directory=st.session_state["persist_dir"],
            collection_name="rag_documents",
        )
    st.session_state["collection"] = collection
    st.session_state["index_built"] = True
    st.session_state["num_documents"] = num_documents
    st.session_state["num_chunks"] = num_chunks
    st.session_state["chat_history"] = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 RAG Knowledge Assistant")
    st.caption("Retrieval-Augmented Generation over your own documents")

    st.subheader("1. Add documents")
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, .pdf, or .docx files",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Build index", type="primary", disabled=not uploaded_files):
            try:
                build_index_from_uploads(uploaded_files)
                st.success("Index built.")
            except (ValueError, RuntimeError) as exc:
                st.error(f"Failed to build index: {exc}")
    with col_b:
        if st.button("Use sample docs"):
            try:
                build_index_from_samples()
                st.success("Sample index built.")
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                st.error(f"Failed to build sample index: {exc}")

    st.divider()
    st.subheader("2. Retrieval settings")
    retrieval_k = st.slider("Chunks to retrieve (top-K)", min_value=3, max_value=15, value=8)
    max_context_chunks = st.slider("Max chunks used as evidence", min_value=1, max_value=6, value=3)
    word_budget = st.slider("Context word budget", min_value=80, max_value=500, value=220, step=20)
    prefer_current = st.checkbox("Prefer current sources over outdated ones", value=True)

    st.divider()
    st.subheader("Status")
    if st.session_state["index_built"]:
        st.success(
            f"Indexed {st.session_state['num_documents']} document(s), "
            f"{st.session_state['num_chunks']} chunk(s)."
        )
    else:
        st.info("No index built yet. Upload documents or use the sample corpus.")

    if not prompting_module.OPENROUTER_API_KEY:
        st.warning(
            "OPENROUTER_API_KEY is not set. Add it to a local `.env` file "
            "(see `example.env`) or to Streamlit Secrets when deployed."
        )
    st.caption(f"LLM model: `{prompting_module.OPENROUTER_MODEL}`")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Ask your documents")
st.write(
    "Upload policies, guides, or procedures in the sidebar, build the index, "
    "then ask a question. Answers are grounded only in the retrieved evidence "
    "and cite their sources."
)

question = st.text_input(
    "Your question",
    placeholder="e.g. How much of my tuition can I get back if I withdraw?",
    disabled=not st.session_state["index_built"],
)

ask_clicked = st.button("Ask", type="primary", disabled=not st.session_state["index_built"])

if ask_clicked:
    if not question or not question.strip():
        st.error("Please enter a question.")
    else:
        try:
            with st.spinner("Retrieving relevant context..."):
                context_package = retrieve_module.build_context_package(
                    st.session_state["collection"],
                    question,
                    retrieval_k=retrieval_k,
                    max_context_chunks=max_context_chunks,
                    word_budget=word_budget,
                    prefer_current=prefer_current,
                )

            with st.spinner("Generating answer..."):
                result = prompting_module.generate_answer(context_package)

            st.session_state["chat_history"].insert(
                0, {"query": question, "result": result, "context_package": context_package}
            )
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
        except RuntimeError as exc:
            st.error(f"Something went wrong: {exc}")

for turn in st.session_state["chat_history"]:
    st.markdown(f"### 🧑 {turn['query']}")

    if not turn["result"]["has_evidence"]:
        st.warning(turn["result"]["answer"])
    else:
        st.markdown(turn["result"]["answer"])

        sources = turn["result"]["sources"]
        if sources:
            with st.expander(f"📎 View {len(sources)} retrieved source(s)"):
                for position, source in enumerate(sources, start=1):
                    currency_label = "🟢 CURRENT" if source.get("is_current") else "🟠 OUTDATED"
                    st.markdown(
                        f"**[Source {position}] {source.get('title', 'Untitled')}** "
                        f"— {source.get('effective_date', 'unknown date')} — {currency_label} "
                        f"— similarity `{source.get('score', 0):.2f}`"
                    )
                    st.text(source.get("chunk_text", ""))

    st.divider()
