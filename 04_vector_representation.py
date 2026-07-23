"""
04_vector_representation.py
=============================
Embedding generation stage of the RAG pipeline.

Responsibility
--------------
Convert chunk text (and later, query text) into dense vector representations
using the same embedding model taught in Lab 7 and used again in Lab 8:
`sentence-transformers/all-MiniLM-L6-v2`.

Why this model
--------------
- Strong, well-established open-source sentence embedding model (384
  dimensions), small enough (~80MB) to run comfortably on Streamlit Cloud's
  free tier with no GPU.
- Directly demonstrated in Lab 7 (`retrieve_top_k_semantic`) and Lab 8, so
  using it here keeps the production pipeline consistent with what was
  taught and evaluated in the labs.
- Normalizing embeddings to unit length (`normalize_embeddings=True`) makes
  cosine similarity equal to a dot product, which is faster and is exactly
  what ChromaDB's default "cosine" space expects.

Scalability
-----------
`embed_texts` batches encoding calls and exposes a `batch_size` parameter so
larger corpora do not need to be embedded in a single call. The model is
loaded once and cached at module level (`get_embedding_model`) so repeated
calls (e.g. from the Streamlit app) do not reload the model from disk.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load and cache the sentence embedding model.

    Using `lru_cache` ensures the (relatively expensive) model load happens
    only once per process, which matters a lot in a Streamlit app that
    reruns the script on every interaction.

    Returns:
        A loaded `SentenceTransformer` instance.

    Raises:
        RuntimeError: If sentence-transformers is not installed or the model
                      cannot be downloaded/loaded.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "sentence-transformers is required. Install it with "
            "`pip install sentence-transformers`."
        ) from exc

    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {exc}"
        ) from exc


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of texts into normalized dense vectors.

    Args:
        texts: List of strings to embed (e.g. chunk `search_text` values).
        batch_size: Number of texts encoded per batch (tune down on
                    memory-constrained deployments).

    Returns:
        A NumPy array of shape (len(texts), EMBEDDING_DIMENSIONS), with each
        row normalized to unit length.

    Raises:
        ValueError: If `texts` is empty.
    """
    if not texts:
        raise ValueError("No texts provided to embed.")

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query using the same model used for chunk embeddings.

    Args:
        query: The user's natural language question.

    Returns:
        A 1D NumPy array of shape (EMBEDDING_DIMENSIONS,).

    Raises:
        ValueError: If the query is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("Query text cannot be empty.")

    return embed_texts([query])[0]


if __name__ == "__main__":
    demo_texts = [
        "Tuition Refund Policy Students who withdraw during the first 7 days may get an 80 percent refund.",
        "Password Reset for Students requires the self-service portal and a verification code.",
    ]
    demo_query = "How do I get my tuition money back?"

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    chunk_vectors = embed_texts(demo_texts)
    query_vector = embed_query(demo_query)

    print(f"Chunk embeddings shape: {chunk_vectors.shape}")
    print(f"Query embedding shape:  {query_vector.shape}")

    similarities = chunk_vectors @ query_vector
    for text, score in zip(demo_texts, similarities):
        print(f"  similarity={score:.4f}  ->  {text[:60]}...")
