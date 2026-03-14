"""Embedding utilities for semantic deduplication in PhoneClaw.

Uses an OpenAI-compatible embeddings API to compute dense vector representations
of text, enabling semantic similarity checks that catch paraphrases and
near-duplicates that exact string matching would miss.

API key priority (first found wins):
  1. OPENAI_API_KEY  → uses https://api.openai.com/v1
  2. OPENROUTER_API_KEY → uses https://openrouter.ai/api/v1
  3. Neither available → falls back to normalised-token Jaccard similarity

Embeddings for known texts are cached in-process to avoid redundant API calls.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from typing import Optional

# Cosine similarity above this threshold → treat as semantic duplicate.
# 0.88 works well for short factual phrases in Chinese / English.
SIMILARITY_THRESHOLD = 0.88

# Jaccard similarity threshold used when the embedding API is unavailable.
JACCARD_THRESHOLD = 0.50

# Model used for embedding.  text-embedding-3-small is cheap, fast, and
# works with both OpenAI and OpenRouter.
EMBED_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Module-level lazy state
# ---------------------------------------------------------------------------
_client = None          # openai.OpenAI instance (or None)
_cache: dict[str, list[float]] = {}   # in-process cache: text → vector


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_client():
    """Return a lazy-initialised OpenAI-compatible client, or None."""
    global _client
    if _client is not None:
        return _client

    try:
        import openai
    except ImportError:
        return None

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if openai_key:
        _client = openai.OpenAI(api_key=openai_key)
    elif openrouter_key:
        _client = openai.OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


def _normalise(text: str) -> str:
    """Lower-case, Unicode-normalise, and strip punctuation."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[\s\W_]+", " ", text)
    return text.strip()


def _jaccard_tokens(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    ta = set(_normalise(a).split())
    tb = set(_normalise(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cosine(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(text: str) -> Optional[list[float]]:
    """Return the embedding vector for *text*.

    Returns None if the embedding API is unavailable or the call fails.
    Results are cached in-process for the lifetime of the interpreter.
    """
    text = text.strip()
    if not text:
        return None

    if text in _cache:
        return _cache[text]

    client = _get_client()
    if client is None:
        return None

    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        vec: list[float] = resp.data[0].embedding
        _cache[text] = vec
        return vec
    except Exception:
        return None


def similarity(text_a: str, text_b: str) -> float:
    """Return a [0, 1] similarity score between two texts.

    Tries the embedding API first; falls back to Jaccard if unavailable.
    """
    va = embed(text_a)
    vb = embed(text_b)
    if va is not None and vb is not None:
        return _cosine(va, vb)
    return _jaccard_tokens(text_a, text_b)


def is_semantic_duplicate(
    text: str,
    candidates: list[str],
    threshold: float = SIMILARITY_THRESHOLD,
) -> int:
    """Check whether *text* is semantically equivalent to any string in
    *candidates*.

    Uses embedding cosine similarity when the API is reachable, otherwise
    falls back to normalised-token Jaccard similarity with a lower threshold.

    Returns:
        Index (≥ 0) of the best-matching candidate above threshold,
        or -1 if no duplicate was found.
    """
    text = text.strip()
    if not text:
        return -1

    using_embeddings = embed(text) is not None

    if using_embeddings:
        sim_threshold = threshold
    else:
        # Jaccard is a weaker signal; use a lower threshold
        sim_threshold = JACCARD_THRESHOLD

    best_idx = -1
    best_sim = sim_threshold - 1e-6  # must strictly exceed threshold

    for i, c in enumerate(candidates):
        if using_embeddings:
            score = similarity(text, c)
        else:
            score = _jaccard_tokens(text, c)

        if score > best_sim:
            best_sim = score
            best_idx = i

    return best_idx
