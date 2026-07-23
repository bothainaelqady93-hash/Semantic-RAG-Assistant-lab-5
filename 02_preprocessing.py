"""
02_preprocessing.py
====================
Text preprocessing stage of the RAG pipeline.

Responsibility
--------------
Provide every classical text preprocessing technique covered in Lab 5
(lowercasing, punctuation removal, number removal, whitespace normalization,
URL removal, tokenization, stopword removal, stemming, lemmatization) as
small, composable, well-documented functions, plus one configurable
`preprocess_text()` pipeline that mirrors the lab's implementation exactly.

Design decision: which profile does the RAG pipeline actually use?
--------------------------------------------------------------------
Lab 5's own conclusion (see "Recommended mindset" in the notebook) is that
preprocessing should match the task, not be applied blindly. For a RAG system
built on top of a *dense sentence embedding model* (Lab 7/8's
`all-MiniLM-L6-v2`), aggressive normalization is actively harmful:

    - Embedding models are pretrained on natural, unaltered text, so
      lowercasing/stemming/stopword removal mostly discards signal the model
      already knows how to use, without shrinking the vocabulary problem that
      motivated those techniques for classical TF-IDF/BM25 in the first place.
    - Negation words, numbers (prices, dates, versions), and punctuation such
      as "C++" or "10/10" frequently carry the exact meaning a support/RAG
      question depends on (see Lab 5's own decision guide table).

So this module ships the full, configurable toolkit (for teaching purposes
and for anyone who wants to build a classical TF-IDF/BM25 index as a
secondary retriever), but the RAG pipeline itself calls `clean_for_embedding`,
a minimal, safe profile that only removes URLs and normalizes whitespace
before chunking and embedding.
"""

from __future__ import annotations

import re
import string
from typing import Dict, List

import nltk

# ---------------------------------------------------------------------------
# Step 1: Import libraries and prepare safe fallbacks (mirrors Lab 5)
# ---------------------------------------------------------------------------


def _ensure_nltk_resources() -> None:
    """
    Download the NLTK resources this module needs, tolerating environments
    without internet access (in which case safe fallbacks below are used).
    """
    resources = [
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception:  # noqa: BLE001 - offline environments are fine
                pass


_ensure_nltk_resources()

from nltk.corpus import stopwords  # noqa: E402
from nltk.corpus import wordnet as wn  # noqa: E402
from nltk.stem import PorterStemmer, WordNetLemmatizer  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except ImportError:  # pragma: no cover - scikit-learn is a required dependency
    ENGLISH_STOP_WORDS = frozenset()


def _nltk_resource_available(resource_path: str) -> bool:
    """Check whether an NLTK resource is available locally."""
    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        return False


HAS_PUNKT = _nltk_resource_available("tokenizers/punkt")
HAS_STOPWORDS = _nltk_resource_available("corpora/stopwords")

try:
    wn.ensure_loaded()
    HAS_WORDNET = bool(wn.synsets("dog"))
except Exception:  # noqa: BLE001
    HAS_WORDNET = False

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

if HAS_STOPWORDS:
    STOP_WORDS = set(stopwords.words("english"))
else:
    STOP_WORDS = set(ENGLISH_STOP_WORDS)

# Negation words are protected by default during stopword removal because
# removing them silently flips the meaning of a sentence (Lab 5, Method 6).
NEGATION_WORDS = {"no", "not", "nor", "never"}

_FALLBACK_LEMMA_MAP = {
    ("loved", "v"): "love",
    ("loving", "v"): "love",
    ("studies", "v"): "study",
    ("studying", "v"): "study",
    ("running", "v"): "run",
    ("better", "a"): "good",
    ("cars", "n"): "car",
    ("mice", "n"): "mouse",
    ("children", "n"): "child",
}

_PUNCTUATION_TRANSLATOR = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# Step 2: Individual preprocessing methods
# ---------------------------------------------------------------------------


def safe_word_tokenize(text: str) -> List[str]:
    """
    Tokenize text into words, falling back to a regex tokenizer if NLTK's
    punkt resource is not available in the current environment.

    Args:
        text: Input text.

    Returns:
        A list of word tokens. Empty input returns an empty list.
    """
    if not text.strip():
        return []
    if HAS_PUNKT:
        return word_tokenize(text)
    return re.findall(r"\b\w+\b", text)


def safe_lemmatize(token: str, pos: str = "n") -> str:
    """
    Lemmatize a single token, falling back to a small rule-based map when
    WordNet is unavailable.

    Args:
        token: The word to lemmatize.
        pos: WordNet part-of-speech tag ("n" for noun, "v" for verb, etc.).

    Returns:
        The lemmatized token in lowercase.
    """
    token = token.lower()

    if HAS_WORDNET:
        return lemmatizer.lemmatize(token, pos=pos)

    if (token, pos) in _FALLBACK_LEMMA_MAP:
        return _FALLBACK_LEMMA_MAP[(token, pos)]

    if pos == "n":
        if token.endswith("ies") and len(token) > 3:
            return token[:-3] + "y"
        if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            return token[:-1]

    if pos == "v":
        if token.endswith("ing") and len(token) > 4:
            return token[:-3]
        if token.endswith("ed") and len(token) > 3:
            return token[:-2]

    return token


def remove_urls(text: str) -> str:
    """Remove http(s) and www links from text."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_punctuation(text: str) -> str:
    """Remove all standard punctuation characters from text."""
    return text.translate(_PUNCTUATION_TRANSLATOR)


def remove_numbers(text: str) -> str:
    """Remove digit sequences from text. Use with care: prices, dates, and
    version numbers are often meaningful (see module docstring)."""
    return re.sub(r"\d+", "", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace (spaces, tabs, newlines) into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Step 3: Configurable preprocessing pipeline (same shape as Lab 5)
# ---------------------------------------------------------------------------


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_url: bool = True,
    remove_punct: bool = True,
    remove_num: bool = False,
    normalize_space: bool = True,
    remove_stop_words: bool = False,
    preserve_negation: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = False,
) -> str:
    """
    Fully configurable preprocessing pipeline, matching Lab 5's `preprocess_text`.

    Args:
        text: Raw input text.
        lowercase: Lowercase the text.
        remove_url: Strip http(s)/www links.
        remove_punct: Strip punctuation characters.
        remove_num: Strip digit sequences.
        normalize_space: Collapse repeated whitespace.
        remove_stop_words: Remove English stopwords after tokenization.
        preserve_negation: When removing stopwords, keep "no"/"not"/"nor"/"never".
        use_stemming: Apply Porter stemming to tokens.
        use_lemmatization: Apply WordNet lemmatization to tokens.

    Returns:
        The processed text, re-joined with single spaces.

    Raises:
        ValueError: If both `use_stemming` and `use_lemmatization` are True.
    """
    if use_stemming and use_lemmatization:
        raise ValueError("Use either stemming or lemmatization, not both at the same time.")

    if lowercase:
        text = text.lower()
    if remove_url:
        text = remove_urls(text)
    if remove_punct:
        text = remove_punctuation(text)
    if remove_num:
        text = remove_numbers(text)
    if normalize_space:
        text = normalize_whitespace(text)

    if not text:
        return ""

    tokens = safe_word_tokenize(text)

    if remove_stop_words:
        if preserve_negation:
            tokens = [t for t in tokens if (t not in STOP_WORDS) or (t in NEGATION_WORDS)]
        else:
            tokens = [t for t in tokens if t not in STOP_WORDS]

    if use_stemming:
        tokens = [stemmer.stem(t) for t in tokens]

    if use_lemmatization:
        tokens = [safe_lemmatize(t, pos="v") for t in tokens]

    return " ".join(tokens)


# Named profiles, identical in spirit to Lab 5's comparison table.
PREPROCESSING_PROFILES: Dict[str, Dict[str, bool]] = {
    "minimal_clean": dict(
        lowercase=True, remove_url=True, remove_punct=True, remove_num=False,
        normalize_space=True, remove_stop_words=False, preserve_negation=True,
        use_stemming=False, use_lemmatization=False,
    ),
    "stopword_reduced": dict(
        lowercase=True, remove_url=True, remove_punct=True, remove_num=False,
        normalize_space=True, remove_stop_words=True, preserve_negation=True,
        use_stemming=False, use_lemmatization=False,
    ),
    "aggressive_stemmed": dict(
        lowercase=True, remove_url=True, remove_punct=True, remove_num=True,
        normalize_space=True, remove_stop_words=True, preserve_negation=False,
        use_stemming=True, use_lemmatization=False,
    ),
    "readable_lemmatized": dict(
        lowercase=True, remove_url=True, remove_punct=True, remove_num=False,
        normalize_space=True, remove_stop_words=True, preserve_negation=True,
        use_stemming=False, use_lemmatization=True,
    ),
}


def clean_for_embedding(text: str) -> str:
    """
    The profile actually used before chunking/embedding in this RAG pipeline.

    Only removes URLs and normalizes whitespace. Case, punctuation, numbers,
    and stopwords are preserved because the embedding model
    (`all-MiniLM-L6-v2`) was trained on natural text and performs best on
    inputs that look like natural text. See the module docstring for the
    full rationale.

    Args:
        text: Raw document or chunk text.

    Returns:
        Lightly cleaned text, safe to embed and safe to show to a human.
    """
    text = remove_urls(text)
    text = normalize_whitespace(text)
    return text


def preprocess_documents(documents: List[Dict]) -> List[Dict]:
    """
    Apply `clean_for_embedding` to the `text` field of every document.

    Args:
        documents: List of document dictionaries, each with a "text" field
                   (as produced by `01_documents.py`).

    Returns:
        A new list of documents with cleaned text. Other fields are preserved.

    Raises:
        ValueError: If a document is missing the required "text" field.
    """
    cleaned_documents = []
    for document in documents:
        if "text" not in document:
            raise ValueError(f"Document is missing a 'text' field: {document}")
        cleaned_document = dict(document)
        cleaned_document["text"] = clean_for_embedding(document["text"])
        cleaned_documents.append(cleaned_document)
    return cleaned_documents


if __name__ == "__main__":
    sample_sentences = [
        "I LOVE this library!!!",
        "Visit https://university.edu/library right now.",
        "The refund is 199.99 dollars, not 99.99.",
    ]

    print("=== Preprocessing profile comparison ===")
    for sentence in sample_sentences:
        print(f"\nOriginal: {sentence}")
        for profile_name, profile_kwargs in PREPROCESSING_PROFILES.items():
            print(f"  {profile_name:20s}: {preprocess_text(sentence, **profile_kwargs)}")
        print(f"  {'clean_for_embedding':20s}: {clean_for_embedding(sentence)}")
