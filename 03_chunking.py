"""
03_chunking.py
================
Adaptive chunking stage of the RAG pipeline.

Responsibility
--------------
Split each document's text into overlapping, retrievable chunks and attach
the metadata every later stage depends on (title, doc_type, effective_date,
is_current, chunk_index, search_text).

Chunking algorithm
-------------------
The core splitter is the fixed-size, overlapping word-window chunker taught
in Lab 6 (`chunk_text_by_words`) and reused unchanged in Lab 8 and Lab 9
(`chunk_text`). It is simple, fast, has no external dependencies, and is easy
to reason about and debug -- all properties the labs call out explicitly as
reasons to prefer it over more complex splitters for a teaching-grade RAG
system.

Why "adaptive"
--------------
The labs always call this function with one fixed (chunk_size, overlap) pair
for the whole corpus. This module adds one genuinely adaptive decision on
top of that: `determine_adaptive_chunk_params` inspects each document's
length and selects a chunk size that fits it, instead of using one global
constant for documents that might be a single paragraph or several pages:

    - Very short documents (< 60 words) are kept as a single chunk. Splitting
      a short policy into two overlapping halves only hurts retrieval
      precision (Lab 9's "Rule / exception split" failure mode) for no
      benefit.
    - Short-to-medium documents use the same chunk_size/overlap as the labs
      (38 words / 10 words overlap), which the labs found gives a good
      balance between chunk cohesion and retrieval granularity.
    - Long documents get slightly larger chunks (60 words) with
      proportionally larger overlap (15 words) so that a single procedure or
      exception is less likely to be split across a chunk boundary, while
      still keeping enough chunks to make retrieval selective.

The 10/38 ratio (~26% overlap) from the labs is preserved across tiers so
each configuration keeps roughly the same relative continuity between
adjacent chunks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def chunk_text_fixed(text: str, chunk_size: int = 38, overlap: int = 10) -> List[str]:
    """
    Split text into overlapping, fixed-size word windows.

    This is the exact chunking algorithm used in Labs 6, 8, and 9.

    Args:
        text: The text to split.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words repeated between consecutive chunks.

    Returns:
        A list of text chunks. Returns an empty list for empty/whitespace-only text.

    Raises:
        ValueError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


def determine_adaptive_chunk_params(text: str) -> Dict[str, int]:
    """
    Choose (chunk_size, overlap) based on document length.

    Args:
        text: The full document text.

    Returns:
        A dict with "chunk_size" and "overlap" keys.
    """
    word_count = len(text.split())

    if word_count < 60:
        # Keep short documents whole: one chunk, no splitting needed.
        return {"chunk_size": max(word_count, 1), "overlap": 0}

    if word_count <= 300:
        # Matches the chunk_size/overlap used throughout Labs 6, 8, and 9.
        return {"chunk_size": 38, "overlap": 10}

    # Long documents: larger windows so a single step/exception is less
    # likely to be split across chunk boundaries, same ~26% overlap ratio.
    return {"chunk_size": 60, "overlap": 15}


def chunk_document(document: Dict[str, Any], adaptive: bool = True) -> List[Dict[str, Any]]:
    """
    Chunk a single document and attach retrieval metadata to each chunk.

    Args:
        document: A document dict with at least "document_id", "title",
                  "doc_type", "effective_date", "is_current", and "text".
        adaptive: If True, choose chunk_size/overlap per document length via
                  `determine_adaptive_chunk_params`. If False, use the fixed
                  Lab 8/9 defaults (chunk_size=38, overlap=10) for every
                  document.

    Returns:
        A list of chunk dictionaries, each carrying the parent document's
        metadata plus chunk-specific fields (chunk_id, chunk_index,
        chunk_text, word_count, search_text).

    Raises:
        KeyError: If a required document field is missing.
    """
    required_fields = ["document_id", "title", "doc_type", "effective_date", "is_current", "text"]
    missing = [field for field in required_fields if field not in document]
    if missing:
        raise KeyError(f"Document is missing required field(s): {missing}")

    if adaptive:
        params = determine_adaptive_chunk_params(document["text"])
    else:
        params = {"chunk_size": 38, "overlap": 10}

    raw_chunks = chunk_text_fixed(
        document["text"], chunk_size=params["chunk_size"], overlap=params["overlap"]
    )

    chunk_rows: List[Dict[str, Any]] = []
    for chunk_index, chunk_text_value in enumerate(raw_chunks):
        chunk_rows.append(
            {
                "chunk_id": f"doc{document['document_id']}_chunk{chunk_index}",
                "document_id": document["document_id"],
                "title": document["title"],
                "doc_type": document["doc_type"],
                "source": document.get("source", ""),
                "effective_date": document["effective_date"],
                "is_current": bool(document["is_current"]),
                "chunk_index": chunk_index,
                "chunk_text": chunk_text_value,
                "word_count": len(chunk_text_value.split()),
                "chunk_size_used": params["chunk_size"],
                "overlap_used": params["overlap"],
                # search_text prepends title/doc_type so both lexical and
                # semantic retrieval benefit from document-level context,
                # matching the approach in Lab 8, Section 5.
                "search_text": f"{document['title']} {document['doc_type']} {chunk_text_value}",
            }
        )

    return chunk_rows


def chunk_documents(
    documents: List[Dict[str, Any]], adaptive: bool = True
) -> List[Dict[str, Any]]:
    """
    Chunk every document in a corpus.

    Args:
        documents: List of document dicts (see `chunk_document`).
        adaptive: Whether to use per-document adaptive chunk sizing.

    Returns:
        A flat list of chunk dictionaries across all documents.

    Raises:
        ValueError: If `documents` is empty.
    """
    if not documents:
        raise ValueError("No documents provided to chunk.")

    all_chunks: List[Dict[str, Any]] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, adaptive=adaptive))

    return all_chunks


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    import importlib

    documents_module = importlib.import_module("01_documents")

    sample_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    try:
        docs = documents_module.load_documents_from_directory(sample_dir)
        chunks = chunk_documents(docs, adaptive=True)
        print(f"Loaded {len(docs)} document(s) -> produced {len(chunks)} chunk(s)\n")
        for chunk in chunks:
            print(
                f"[{chunk['chunk_id']}] words={chunk['word_count']} "
                f"(chunk_size={chunk['chunk_size_used']}, overlap={chunk['overlap_used']}) "
                f"-> {chunk['chunk_text'][:80]}..."
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not run demo: {exc}")
