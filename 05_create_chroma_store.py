"""
05_create_chroma_store.py
============================
Vector database stage of the RAG pipeline.

Responsibility
--------------
Take the chunked, embedded corpus and persist it into a local ChromaDB
collection so it can be queried later without re-embedding anything.

Design decisions
------------------
- **ChromaDB PersistentClient**: writes to a local on-disk directory
  (`chroma_db/` by default), which works both locally and on Streamlit
  Cloud's ephemeral filesystem (rebuilt each time the app boots from
  uploaded files, since Streamlit Cloud does not guarantee persistent
  storage across deploys).
- **Cosine similarity space**: matches the `normalize_embeddings=True`
  vectors produced by `04_vector_representation.py`, so cosine similarity
  is equivalent to a dot product (fast and numerically stable).
- **Pre-computed embeddings**: this module stores vectors computed by
  `sentence-transformers` directly (via `collection.add(embeddings=...)`)
  instead of letting Chroma re-embed text with its own default embedding
  function. This guarantees the exact same model is used for both indexing
  and querying, which Lab 7 stresses is required for a valid comparison
  ("If they use different models, their vectors cannot be compared.").
- **Batched writes**: `add_chunks_to_store` writes in batches to stay well
  under Chroma's per-call payload limits for larger corpora.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

DEFAULT_PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
DEFAULT_COLLECTION_NAME = "rag_documents"
DEFAULT_BATCH_SIZE = 100


def get_chroma_client(persist_directory: str = DEFAULT_PERSIST_DIRECTORY):
    """
    Create (or connect to) a persistent ChromaDB client.

    Args:
        persist_directory: Local folder where Chroma stores its data.

    Returns:
        A `chromadb.PersistentClient` instance.

    Raises:
        RuntimeError: If chromadb is not installed.
    """
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "chromadb is required. Install it with `pip install chromadb`."
        ) from exc

    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(path=persist_directory)


def get_or_create_collection(client, collection_name: str = DEFAULT_COLLECTION_NAME):
    """
    Get an existing collection or create a new one with cosine similarity.

    Args:
        client: A ChromaDB client (see `get_chroma_client`).
        collection_name: Name of the collection to fetch/create.

    Returns:
        A ChromaDB `Collection` object.
    """
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(client, collection_name: str = DEFAULT_COLLECTION_NAME):
    """
    Delete a collection if it exists, then recreate it empty.

    Used when rebuilding the index from scratch (e.g. new document upload)
    so stale chunks from a previous corpus never leak into retrieval.

    Args:
        client: A ChromaDB client.
        collection_name: Name of the collection to reset.

    Returns:
        A fresh, empty ChromaDB `Collection`.
    """
    try:
        client.delete_collection(name=collection_name)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass
    return get_or_create_collection(client, collection_name)


def _chunk_metadata_for_storage(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the metadata dict Chroma will store alongside each chunk.

    Chroma metadata values must be str/int/float/bool, so this filters the
    chunk dict down to that subset.

    Args:
        chunk: A chunk dictionary produced by `03_chunking.py`.

    Returns:
        A flat metadata dict safe for Chroma to store.
    """
    allowed_keys = [
        "document_id", "title", "doc_type", "source", "effective_date",
        "is_current", "chunk_index", "word_count",
    ]
    metadata = {key: chunk[key] for key in allowed_keys if key in chunk}
    # Chroma does not accept Python bools reliably across all backends in
    # older SDK versions; storing as int is the safe, explicit choice.
    if "is_current" in metadata:
        metadata["is_current"] = int(bool(metadata["is_current"]))
    return metadata


def add_chunks_to_store(
    collection,
    chunks: List[Dict[str, Any]],
    embeddings,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """
    Add chunks (with their embeddings) to a Chroma collection in batches.

    Args:
        collection: A ChromaDB collection (see `get_or_create_collection`).
        chunks: List of chunk dicts, in the same order as `embeddings`.
        embeddings: NumPy array or list of embedding vectors, aligned 1:1 with `chunks`.
        batch_size: Number of chunks written per `collection.add` call.

    Returns:
        The total number of chunks added.

    Raises:
        ValueError: If `chunks` and `embeddings` have mismatched lengths.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must be the same length."
        )
    if not chunks:
        raise ValueError("No chunks provided to store.")

    total_added = 0
    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start:start + batch_size]
        batch_embeddings = embeddings[start:start + batch_size]

        collection.add(
            ids=[chunk["chunk_id"] for chunk in batch_chunks],
            embeddings=[list(map(float, vector)) for vector in batch_embeddings],
            documents=[chunk["chunk_text"] for chunk in batch_chunks],
            metadatas=[_chunk_metadata_for_storage(chunk) for chunk in batch_chunks],
        )
        total_added += len(batch_chunks)

    return total_added


def build_vector_store(
    chunks: List[Dict[str, Any]],
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset: bool = True,
):
    """
    End-to-end helper: embed chunk search_text and persist everything to Chroma.

    Args:
        chunks: List of chunk dicts from `03_chunking.py` (must include "search_text").
        persist_directory: Local folder for the Chroma store.
        collection_name: Name of the Chroma collection to (re)build.
        reset: If True, wipe any existing collection with the same name first.
               Set False to incrementally add chunks to an existing store.

    Returns:
        The populated ChromaDB collection.
    """
    import importlib
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    vector_module = importlib.import_module("04_vector_representation")

    search_texts = [chunk["search_text"] for chunk in chunks]
    embeddings = vector_module.embed_texts(search_texts)

    client = get_chroma_client(persist_directory)
    collection = reset_collection(client, collection_name) if reset else get_or_create_collection(client, collection_name)

    add_chunks_to_store(collection, chunks, embeddings)
    return collection


def run_full_pipeline(
    documents_directory: Optional[str] = None,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """
    Run the complete offline pipeline: load -> preprocess -> chunk -> embed -> store.

    This is what a scheduled indexing job (or the CLI entry point below)
    would call to (re)build the vector store from a folder of source files.

    Args:
        documents_directory: Folder of source documents. Defaults to the
                              bundled `data/raw` sample corpus.
        persist_directory: Local folder for the Chroma store.
        collection_name: Name of the Chroma collection to build.

    Returns:
        A tuple of (collection, num_documents, num_chunks).
    """
    import importlib
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    documents_module = importlib.import_module("01_documents")
    preprocessing_module = importlib.import_module("02_preprocessing")
    chunking_module = importlib.import_module("03_chunking")

    documents_directory = documents_directory or os.path.join(
        os.path.dirname(__file__), "data", "raw"
    )

    documents = documents_module.load_documents_from_directory(documents_directory)
    documents = preprocessing_module.preprocess_documents(documents)
    chunks = chunking_module.chunk_documents(documents, adaptive=True)

    collection = build_vector_store(
        chunks, persist_directory=persist_directory, collection_name=collection_name, reset=True
    )

    return collection, len(documents), len(chunks)


if __name__ == "__main__":
    try:
        collection, num_documents, num_chunks = run_full_pipeline()
        print(
            f"Indexed {num_documents} document(s) into {num_chunks} chunk(s). "
            f"Collection now contains {collection.count()} item(s)."
        )
    except RuntimeError as exc:
        print(f"Pipeline failed: {exc}")
