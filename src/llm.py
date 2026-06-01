"""
llm.py — LLM integration for RAG response generation.

Default: Groq (FREE — llama-3.3-70b-versatile, 6000 req/day)
Optional: Anthropic Claude (set LLM_PROVIDER=anthropic in .env)

Prompt forces citation-grounded answers with uncertainty acknowledgement.
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()

# ── Provider selection ────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()   # "groq" | "anthropic"

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise, citation-driven research assistant.

Rules:
1. Answer ONLY using the provided context chunks. Do not use outside knowledge.
2. After each key claim, cite the source in brackets: [source.pdf, p.3] or [url].
3. If the context does not contain enough information to answer confidently, say:
   "I don't have enough information in the provided documents to answer this."
4. Be concise but complete. Use bullet points for multi-part answers.
5. Never fabricate facts, statistics, or quotes.
"""

def _format_context(chunks: List[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["source"]
        page   = f", p.{chunk['page']}" if chunk.get("page") else ""
        score  = chunk["score"]
        parts.append(
            f"[Chunk {i} | {source}{page} | relevance: {score}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Groq (FREE default) ───────────────────────────────────────────────────────

def _generate_groq(query: str, chunks: List[dict], model: str, max_tokens: int) -> dict:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Get a free key at console.groq.com")

    client = Groq(api_key=api_key)
    context = _format_context(chunks)
    user_msg = f"Context documents:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    return {
        "answer":  response.choices[0].message.content,
        "sources": list({c["source"] for c in chunks}),
        "model":   model,
        "tokens": {
            "input":  response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
    }


# ── Anthropic Claude (optional) ───────────────────────────────────────────────

def _generate_anthropic(query: str, chunks: List[dict], model: str, max_tokens: int) -> dict:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set.")

    client  = anthropic.Anthropic(api_key=api_key)
    context = _format_context(chunks)
    user_msg = f"Context documents:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return {
        "answer":  response.content[0].text,
        "sources": list({c["source"] for c in chunks}),
        "model":   model,
        "tokens": {
            "input":  response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    }


# ── Public interface ──────────────────────────────────────────────────────────

_DEFAULT_MODELS = {
    "groq":      "llama-3.3-70b-versatile",
    "anthropic": "claude-3-5-haiku-20241022",
}

def generate_answer(
    query:      str,
    chunks:     List[dict],
    model:      str | None = None,
    max_tokens: int = 1024,
) -> dict:
    """
    Generate a cited answer from retrieved chunks.
    Provider is selected via LLM_PROVIDER env var (default: groq).
    """
    if not chunks:
        return {
            "answer":  "No relevant documents found for your query.",
            "sources": [],
            "model":   "none",
            "tokens":  {"input": 0, "output": 0},
        }

    provider = LLM_PROVIDER
    resolved_model = model or _DEFAULT_MODELS.get(provider, "llama-3.3-70b-versatile")

    if provider == "anthropic":
        return _generate_anthropic(query, chunks, resolved_model, max_tokens)
    else:
        return _generate_groq(query, chunks, resolved_model, max_tokens)
