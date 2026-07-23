# RAG Knowledge Assistant

A complete Retrieval-Augmented Generation (RAG) application built from Python
scripts (no notebooks), following the pipeline taught in Labs 5-9:

```
documents -> preprocessing -> chunking -> vector representation -> vector store
          -> context retrieval -> prompting -> Streamlit UI
```

Upload your own documents (policies, guides, FAQs, procedures — `.txt`,
`.md`, `.pdf`, `.docx`), and ask questions in plain English. Answers are
grounded only in the retrieved evidence and cite their sources.

---

## 1. Project overview

| Stage | File | What it does |
|---|---|---|
| Document loading | `01_documents.py` | Loads `.txt`/`.md`/`.pdf`/`.docx` files (uploaded or from disk) into a common schema |
| Preprocessing | `02_preprocessing.py` | Full Lab 5 toolkit (lowercasing, punctuation/number/URL removal, stopwords, stemming, lemmatization) plus the embedding-safe default profile actually used by the pipeline |
| Chunking | `03_chunking.py` | Adaptive, overlapping word-window chunking (Labs 6/8/9 style) |
| Vector representation | `04_vector_representation.py` | Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2` (Lab 7/8) |
| Vector store | `05_create_chroma_store.py` | Persists chunks + embeddings + metadata in ChromaDB |
| Context retrieval | `06_retrieve_context.py` | Top-K similarity search + context packaging (dedup, currency preference, word budget) — Lab 8/9 style |
| Prompting | `07_prompting.py` | Grounded, citation-enforcing prompt + OpenRouter LLM call |
| UI | `streamlit_app.py` | Upload, index, ask, and view cited answers |

## 2. Architecture

```
                 ┌──────────────────┐
  user uploads → │  01_documents.py │  → normalized documents (title, doc_type,
                 └────────┬─────────┘     effective_date, is_current, text)
                          ▼
                 ┌───────────────────┐
                 │ 02_preprocessing  │  → cleaned text (URL removal, whitespace)
                 └────────┬──────────┘
                          ▼
                 ┌───────────────────┐
                 │  03_chunking.py   │  → overlapping chunks + metadata
                 └────────┬──────────┘
                          ▼
                 ┌────────────────────────┐
                 │ 04_vector_representation│ → 384-dim normalized embeddings
                 └────────┬────────────────┘
                          ▼
                 ┌──────────────────────┐
                 │ 05_create_chroma_store│ → persistent ChromaDB collection
                 └────────┬──────────────┘
                          ▼  (at question time)
                 ┌──────────────────────┐
  question → →   │ 06_retrieve_context  │ → context package (top-K, deduped,
                 └────────┬──────────────┘    word-budgeted, current-preferred)
                          ▼
                 ┌──────────────────┐
                 │  07_prompting.py │  → grounded prompt → OpenRouter LLM
                 └────────┬──────────┘
                          ▼
                 ┌──────────────────┐
                 │ streamlit_app.py │  → cited answer shown to the user
                 └──────────────────┘
```

## 3. Requirements

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (free tier available)

## 4. Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Running locally

1. Copy `example.env` to `.env` and add your real key:

   ```
   OPENROUTER_API_KEY=sk-or-...
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ```

2. (Optional) Rebuild the sample index from the command line to sanity-check
   the pipeline end to end:

   ```bash
   python 05_create_chroma_store.py
   python 06_retrieve_context.py
   python 07_prompting.py
   ```

3. Launch the app:

   ```bash
   streamlit run streamlit_app.py
   ```

4. In the sidebar, either upload your own files and click **Build index**,
   or click **Use sample docs** to try it immediately with the bundled
   `data/raw/` corpus (a tuition refund policy, a password reset procedure,
   and a campus printing guide).

## 6. Running on Streamlit Cloud

1. Push this repository to GitHub — **make sure `.env` is not included**
   (it is already excluded via `.gitignore`).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `streamlit_app.py`.
3. Open your app → **Manage app** → **Secrets**, and add:

   ```toml
   OPENROUTER_API_KEY = "your_openrouter_key_here"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```

4. `streamlit_app.py` reads these secrets automatically at startup:

   ```python
   try:
       if not rag.OPENROUTER_API_KEY:
           rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
       rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
   except Exception:
       pass
   ```

## 7. Folder structure

```
.
├── 01_documents.py
├── 02_preprocessing.py
├── 03_chunking.py
├── 04_vector_representation.py
├── 05_create_chroma_store.py
├── 06_retrieve_context.py
├── 07_prompting.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── example.env
└── data/
    ├── raw/                # bundled sample documents
    │   ├── tuition_refund_policy.txt
    │   ├── password_reset_procedure.txt
    │   └── campus_printing_guide.txt
    └── chroma_db/           # generated at runtime (git-ignored)
```

## 8. Design decisions and rationale

- **Chunking (`03_chunking.py`)**: uses the fixed-size overlapping
  word-window chunker from Labs 6/8/9 (`chunk_size=38`, `overlap=10` for
  medium documents), because the labs found this gives a good balance
  between chunk cohesion and retrieval granularity. It is made *adaptive* on
  top of that: very short documents (< 60 words) are kept whole to avoid
  splitting a short policy in half for no benefit, and long documents
  (> 300 words) use larger windows (60/15) so a rule and its exception are
  less likely to land in different chunks — while keeping the same ~26%
  overlap ratio the labs use throughout.
- **Preprocessing (`02_preprocessing.py`)**: implements the entire Lab 5
  toolkit, but the pipeline itself only applies a minimal
  `clean_for_embedding` profile (URL removal + whitespace normalization)
  before chunking/embedding. Aggressive normalization (lowercasing,
  stopword removal, stemming) is built for classical TF-IDF/BM25 retrieval
  and is counter-productive for a pretrained sentence embedding model, which
  performs best on natural, unaltered text — and RAG answers frequently
  depend on exact numbers, negation, and product codes that aggressive
  cleaning would destroy (see the module docstring for the full argument,
  which follows Lab 5's own "Recommended mindset" section).
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, the exact model
  used in Labs 7 and 8 — small, fast, free, and good enough to run on
  Streamlit Cloud's free tier with no GPU.
- **Vector database**: ChromaDB with a persistent, cosine-similarity
  collection, storing pre-computed embeddings (rather than Chroma's default
  embedding function) so indexing and querying are guaranteed to use the
  same model — a point Lab 7 stresses directly.
- **Context building (`06_retrieve_context.py`)**: ports
  `build_context_package` from Labs 8 and 9 — deduplication, a per-document
  chunk cap, a word budget, and current-vs-outdated preference — so the LLM
  never sees redundant, irrelevant, or stale evidence.
- **Prompting (`07_prompting.py`)**: uses Lab 8's "strict" prompt style (the
  best-performing of the three compared in that lab), which forbids outside
  knowledge, gives an exact refusal sentence for missing evidence, handles
  CURRENT vs OUTDATED conflicts explicitly, and forces a two-section
  `Answer:` / `Sources:` output.
- **LLM provider**: OpenRouter (`openai/gpt-4o-mini` by default), per the
  Student RAG Project Instructions — this replaces Lab 9's local Ollama call
  (which only works on a student's own machine) with a cloud API that works
  from a deployed Streamlit app, while keeping the same grounded prompt and
  `<think>`-tag cleanup logic from that lab.

## 9. Screenshots

*(Add screenshots of the running app here before submission, e.g.)*

- `screenshots/sidebar-upload.png` — sidebar with file upload and index status
- `screenshots/answer-with-sources.png` — a question, its grounded answer, and cited sources

## 10. Future improvements

- Add a hybrid (TF-IDF/BM25 + embeddings) retriever, as compared in Lab 7/8,
  for corpora where exact terminology matters more than semantic similarity.
- Stream LLM tokens into the UI instead of waiting for the full response.
- Support persistent storage across Streamlit Cloud restarts (e.g. an
  external vector database) instead of rebuilding the index from uploads
  each session.
- Add automatic retrieval evaluation (precision@K, recall@K, hit rate,
  reciprocal rank — as implemented in Labs 6-9) against a small labeled
  query set, surfaced in the sidebar.
- Multi-turn conversational memory (currently each question is answered
  independently).
