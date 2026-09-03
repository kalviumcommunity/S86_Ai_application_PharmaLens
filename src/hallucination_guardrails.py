"""Retrieval quality checks used to prevent unsupported generation."""

from __future__ import annotations

from typing import Any


DEFAULT_RELEVANCE_THRESHOLD = 0.65
DEFAULT_MIN_RELEVANT_CHUNKS = 1


def assess_retrieval_quality(
    chunks: list[dict[str, Any]],
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    min_relevant_chunks: int = DEFAULT_MIN_RELEVANT_CHUNKS,
) -> dict[str, Any]:
    """Assess whether retrieved chunks are strong enough for generation."""
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance_threshold must be between 0 and 1.")
    if min_relevant_chunks <= 0:
        raise ValueError("min_relevant_chunks must be greater than 0.")

    relevant_chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk.get("score"), (int, float))
        and chunk["score"] >= relevance_threshold
        and bool(str(chunk.get("text", "")).strip())
    ]

    if not chunks:
        reason = "no_context"
    elif len(relevant_chunks) < min_relevant_chunks:
        reason = "insufficient_relevance"
    else:
        reason = "sufficient_relevance"

    numeric_scores = [
        chunk["score"]
        for chunk in chunks
        if isinstance(chunk.get("score"), (int, float))
    ]

    return {
        "is_sufficient": reason == "sufficient_relevance",
        "reason": reason,
        "relevance_threshold": relevance_threshold,
        "min_relevant_chunks": min_relevant_chunks,
        "retrieved_count": len(chunks),
        "relevant_count": len(relevant_chunks),
        "max_score": max(numeric_scores, default=0.0),
    }