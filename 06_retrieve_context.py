"""
06_retrieve_context.py
=========================
Retrieval and context-building stage of the RAG pipeline.

Responsibility
--------------
1. Embed the user's query and run top-K similarity search against the
   Chroma collection built in `05_create_chroma_store.py`.
2. Turn the raw candidate chunks into a clean "context package": the
   deduplicated, current-preferring, word-budgeted evidence that actually
   gets shown to the LLM, following `build_context_package` from Lab 8
   (Section 13) and Lab 9 (Section 8).

Why context building is a separate step from retrieval
--------------------------------------------------------
Lab 8 makes this point directly: "If you paste raw retrieval output into a
prompt, the model may use an outdated source as if it were current, repeat
the same fact from multiple chunks, see irrelevant chunks that dilute
attention, or exceed the token budget." This module implements exactly the
filters the labs use to prevent that:

    - **Score floor** (`min_score`): drop chunks that are not similar enough
      to the query to be trustworthy evidence.
    - **Currency preference** (`prefer_current`): rank `is_current=True`
      chunks ahead of outdated ones with the same or lower relevance.
    - **Deduplication**: drop chunks whose normalized text has already been
      selected.
    - **Per-document cap** (`max_chunks_per_document`): stop one very long,
      highly relevant document from crowding out other sources.
    - **Word budget**: stop adding chunks once the context would get too
      long for a concise, focused prompt.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))
vector_module = importlib.import_module("04_vector_representation")

DEFAULT_RETRIEVAL_K = 8
DEFAULT_MAX_CONTEXT_CHUNKS = 3
DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 1
DEFAULT_WORD_BUDGET = 220
DEFAULT_MIN_SCORE = 0.15


def retrieve_top_k(collection, query: str, k: int = DEFAULT_RETRIEVAL_K) -> List[Dict[str, Any]]:
    """
    Run top-K semantic retrieval against a Chroma collection.

    Args:
        collection: A ChromaDB collection populated by `05_create_chroma_store.py`.
        query: The user's natural language question.
        k: Number of candidate chunks to retrieve.

    Returns:
        A list of candidate dicts, each with chunk_id, chunk_text, score
        (cosine similarity, higher is better), and the stored metadata
        fields (title, doc_type, effective_date, is_current, document_id).

    Raises:
        ValueError: If the query is empty.
        RuntimeError: If the collection is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query text cannot be empty.")

    if collection.count() == 0:
        raise RuntimeError(
            "The vector store is empty. Upload documents and build the index first."
        )

    query_embedding = vector_module.embed_query(query)
    k = min(k, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    candidates: List[Dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for chunk_id, chunk_text, metadata, distance in zip(ids, documents, metadatas, distances):
        # Chroma's cosine "distance" is (1 - cosine_similarity); converting
        # back to similarity keeps scores in the intuitive [0, 1]-ish range
        # used throughout Labs 7-9.
        similarity = 1.0 - distance
        candidate = dict(metadata)
        candidate["chunk_id"] = chunk_id
        candidate["chunk_text"] = chunk_text
        candidate["score"] = similarity
        candidate["is_current"] = bool(candidate.get("is_current", 1))
        candidates.append(candidate)

    return candidates


def build_context_package(
    collection,
    query: str,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    max_context_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS,
    max_chunks_per_document: int = DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    word_budget: int = DEFAULT_WORD_BUDGET,
    prefer_current: bool = True,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Dict[str, Any]:
    """
    Retrieve candidates and reduce them to a clean, citable context package.

    Args:
        collection: A ChromaDB collection.
        query: The user's natural language question.
        retrieval_k: How many raw candidates to pull from the vector store.
        max_context_chunks: Maximum number of chunks to keep for the final prompt.
        max_chunks_per_document: Maximum chunks allowed from a single source document.
        word_budget: Maximum total words across all selected chunks.
        prefer_current: Rank `is_current=True` chunks ahead of outdated ones.
        min_score: Minimum cosine similarity a chunk must have to be considered.

    Returns:
        A dict with:
            - "query": the original query
            - "candidates": all raw retrieved candidates (for debugging/UI)
            - "selected_chunks": the chunks kept for the prompt
            - "context_text": the formatted, labeled context block
            - "used_words": total word count of the selected chunks
            - "has_evidence": whether any chunk passed all the filters
    """
    candidates = retrieve_top_k(collection, query, k=retrieval_k)

    if prefer_current:
        candidates = sorted(
            candidates,
            key=lambda c: (c["is_current"], c["score"], c.get("effective_date", "")),
            reverse=True,
        )
    else:
        candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)

    selected_chunks: List[Dict[str, Any]] = []
    seen_texts = set()
    per_document_counts: Dict[Any, int] = {}
    used_words = 0

    for candidate in candidates:
        if candidate["score"] < min_score:
            continue

        normalized_text = re.sub(r"\s+", " ", candidate["chunk_text"]).strip().lower()
        if normalized_text in seen_texts:
            continue

        document_id = candidate.get("document_id")
        doc_count = per_document_counts.get(document_id, 0)
        if doc_count >= max_chunks_per_document:
            continue

        chunk_words = len(candidate["chunk_text"].split())
        if selected_chunks and used_words + chunk_words > word_budget:
            continue

        selected_chunks.append(candidate)
        seen_texts.add(normalized_text)
        per_document_counts[document_id] = doc_count + 1
        used_words += chunk_words

        if len(selected_chunks) >= max_context_chunks:
            break

    context_blocks = []
    for position, chunk in enumerate(selected_chunks, start=1):
        currency_label = "CURRENT" if chunk["is_current"] else "OUTDATED"
        context_blocks.append(
            f"[Source {position}] {chunk.get('title', 'Untitled')} | "
            f"{chunk.get('effective_date', 'unknown date')} | {currency_label}\n"
            f"{chunk['chunk_text']}"
        )

    return {
        "query": query,
        "candidates": candidates,
        "selected_chunks": selected_chunks,
        "context_text": "\n\n".join(context_blocks),
        "used_words": used_words,
        "has_evidence": len(selected_chunks) > 0,
    }


if __name__ == "__main__":
    create_store_module = importlib.import_module("05_create_chroma_store")

    print("Building the vector store from the bundled sample documents...")
    collection, num_documents, num_chunks = create_store_module.run_full_pipeline()
    print(f"Indexed {num_documents} document(s) into {num_chunks} chunk(s).\n")

    demo_query = "How much of my tuition can I get back if I withdraw?"
    package = build_context_package(collection, demo_query)

    print(f"Query: {demo_query}")
    print(f"Selected {len(package['selected_chunks'])} chunk(s), {package['used_words']} word(s)\n")
    print("=== Context Package ===")
    print(package["context_text"] or "(no evidence passed the filters)")
