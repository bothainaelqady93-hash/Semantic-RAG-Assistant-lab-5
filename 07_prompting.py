"""
07_prompting.py
==================
Prompt engineering and answer generation stage of the RAG pipeline.

Responsibility
--------------
1. Turn a context package (from `06_retrieve_context.py`) into a grounded
   prompt that prevents hallucination, forces citation, and handles missing
   or conflicting evidence explicitly.
2. Send that prompt to an LLM and return a clean final answer.

Prompt design
-------------
This combines the two prompt patterns taught in the labs:

- Lab 8's "strict" prompt (Section 17), which is the strongest of the three
  styles compared there (weak / better / strict) specifically because it:
    1. forbids outside knowledge,
    2. gives an exact refusal sentence for missing evidence,
    3. tells the model how to handle an OUTDATED source,
    4. forces a fixed two-section output (`Answer:` / `Sources:`).
- Lab 9's `build_grounded_prompt`, which keeps the same rules in a more
  compact form for a single end-to-end generation call.

Why OpenRouter (not the labs' local Ollama)
---------------------------------------------
Lab 9 explicitly uses a local Ollama server because it is free and needs no
API key for a teaching notebook. That does not work for a Streamlit Cloud
deployment, which has no access to the student's local machine. The official
Student RAG Project Instructions require OpenRouter with Streamlit Secrets
instead, so this module swaps `ask_ollama` for `call_openrouter` while
keeping everything else (the grounded prompt, the `<think>`-tag cleanup from
`extract_final_answer`) identical to Lab 9's implementation, since reasoning
models available through OpenRouter can also emit these tags.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# These are read at import time from environment variables (see example.env).
# streamlit_app.py additionally overrides them from st.secrets when deployed,
# exactly as described in the Student RAG Project Instructions.
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The provided sources do not contain enough information to answer this question."
)


def build_grounded_prompt(query: str, context_text: str) -> str:
    """
    Build the strict, citation-enforcing RAG prompt.

    Args:
        query: The user's natural language question.
        context_text: The formatted context block from `build_context_package`
                       (labeled "[Source N] ... CURRENT/OUTDATED").

    Returns:
        The full prompt string to send to the LLM.
    """
    return f"""You are a careful, grounded RAG assistant.

Rules:
1. Use only the provided context. Never add outside or background knowledge.
2. If the answer is not in the context, say exactly: "{INSUFFICIENT_EVIDENCE_MESSAGE}"
3. If a source is marked OUTDATED, do not use it as the primary answer. Mention it only to note a conflict.
4. If a CURRENT and an OUTDATED source disagree, state the conflict and use the CURRENT source.
5. Cite the source numbers (e.g. [Source 1]) that support each part of your answer.
6. Keep the answer concise but complete. Output exactly two sections:
Answer: <your grounded answer, with inline [Source N] citations>
Sources: <comma-separated list of the source numbers you actually used>

Question:
{query}

Context:
{context_text if context_text.strip() else "(no context was retrieved)"}
"""


def call_openrouter(prompt: str, temperature: float = 0.0, timeout: int = 60) -> str:
    """
    Send a prompt to an LLM through the OpenRouter chat completions API.

    Args:
        prompt: The full grounded prompt to send.
        temperature: Sampling temperature (0 = deterministic, preferred for RAG QA).
        timeout: Request timeout in seconds.

    Returns:
        The raw text returned by the model.

    Raises:
        RuntimeError: If no API key is configured, or the request fails.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file locally, "
            "or to Streamlit Secrets when deployed."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    try:
        response = requests.post(
            OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=timeout
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response format: {data}") from exc


def extract_final_answer(raw_text: str) -> str:
    """
    Clean a raw LLM response: strip hidden reasoning blocks and extra blank lines.

    Some reasoning models (available through OpenRouter) emit a `<think>...</think>`
    block before the actual answer. This mirrors Lab 9's `extract_final_answer`.

    Args:
        raw_text: The raw text returned by the LLM.

    Returns:
        Cleaned text, safe to display to the user.
    """
    if not raw_text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()
    return cleaned


def generate_answer(context_package: Dict[str, Any], temperature: float = 0.0) -> Dict[str, Any]:
    """
    Full generation step: build the grounded prompt, call the LLM, clean the answer.

    Args:
        context_package: The dict returned by `build_context_package`
                          (must contain "query" and "context_text").
        temperature: Sampling temperature passed to the LLM.

    Returns:
        A dict with:
            - "answer": the cleaned final answer text
            - "prompt": the exact prompt sent to the LLM (useful for debugging/UI)
            - "sources": the list of selected chunk dicts used as context
            - "has_evidence": whether any context was available at all
    """
    query = context_package["query"]
    context_text = context_package["context_text"]

    if not context_package.get("has_evidence", bool(context_text.strip())):
        return {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "prompt": None,
            "sources": [],
            "has_evidence": False,
        }

    prompt = build_grounded_prompt(query, context_text)
    raw_answer = call_openrouter(prompt, temperature=temperature)
    final_answer = extract_final_answer(raw_answer)

    return {
        "answer": final_answer,
        "prompt": prompt,
        "sources": context_package.get("selected_chunks", []),
        "has_evidence": True,
    }


if __name__ == "__main__":
    demo_context_package = {
        "query": "How much of my tuition can I get back if I withdraw in the first week?",
        "context_text": (
            "[Source 1] Tuition Refund Policy | 2026-08-01 | CURRENT\n"
            "Students who withdraw from all courses during the first 7 calendar days of the "
            "semester may receive an 80 percent tuition refund."
        ),
        "selected_chunks": [],
        "has_evidence": True,
    }

    print("=== Grounded prompt preview ===")
    print(build_grounded_prompt(demo_context_package["query"], demo_context_package["context_text"]))

    if OPENROUTER_API_KEY:
        result = generate_answer(demo_context_package)
        print("\n=== Answer ===")
        print(result["answer"])
    else:
        print("\n(Skipping live call: OPENROUTER_API_KEY is not set in this environment.)")
