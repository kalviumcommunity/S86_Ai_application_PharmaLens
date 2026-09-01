"""
Re-ranking module for precision-focused chunk retrieval.

Re-ranking takes a larger candidate set from initial vector retrieval
and scores each candidate more carefully against the query, allowing
the most relevant chunks to bubble to the top before sending to the LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


def rerank_score_with_llm(
    client: OpenAI,
    model: str,
    query: str,
    chunk: dict[str, Any],
) -> float:
    """
    Score a chunk's relevance to a query using an LLM.

    The LLM provides a 0-10 relevance score based on how directly
    the chunk answers or relates to the query.

    Args:
        client: OpenAI-compatible client.
        model: Chat model name.
        query: User query string.
        chunk: Dictionary with "text" and optional "metadata" keys.

    Returns:
        A relevance score (0.0 to 10.0).
    """
    chunk_text = chunk.get("text", "")
    
    prompt = f"""Score how relevant this chunk is to the query on a scale of 0 to 10.

Query: {query}

Chunk: {chunk_text}

Return only the numeric score (0-10) as a single line."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance scoring assistant. "
                        "Rate chunks on how directly they answer the query. "
                        "Return only a number from 0 to 10."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        
        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        
        # Clamp score to 0-10 range
        score = max(0.0, min(10.0, score))
        return score
        
    except (ValueError, IndexError) as e:
        logger.warning(
            "Failed to parse LLM score for chunk. Returning 0. Error: %s", e
        )
        return 0.0
    except Exception as e:
        logger.warning("LLM scoring failed for chunk: %s", e)
        return 0.0


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    final_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Re-rank a candidate set using LLM scoring and return top-k re-ranked results.

    The re-ranking process:
    1. Score each candidate against the query using an LLM
    2. Sort by re-rank score (descending)
    3. Return the top-k results

    Args:
        query: User query string.
        candidates: List of chunk dictionaries from vector retrieval.
        client: OpenAI-compatible client.
        model: Chat model name.
        final_k: Number of top results to return after re-ranking.

    Returns:
        List of top-k candidates, sorted by re-rank score descending,
        with a "rerank_score" field added to each.
    """
    if not candidates:
        return []

    if final_k <= 0:
        raise ValueError("final_k must be a positive integer.")

    logger.info(
        "Re-ranking %d candidates for query: %s",
        len(candidates),
        query[:50],
    )

    reranked = []
    for i, chunk in enumerate(candidates):
        rerank_score = rerank_score_with_llm(client, model, query, chunk)
        reranked.append(
            {
                **chunk,
                "rerank_score": rerank_score,
            }
        )
        logger.debug("Scored candidate %d/%d: %.2f", i + 1, len(candidates), rerank_score)

    # Sort by rerank score descending
    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    result = reranked[:final_k]
    logger.info("Selected top %d candidates after re-ranking", len(result))
    return result


def rerank_and_compare(
    query: str,
    candidates: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    final_k: int = 3,
) -> dict[str, Any]:
    """
    Re-rank candidates and return a comparison of before/after ordering.

    Useful for diagnostics and understanding re-ranking impact.

    Args:
        query: User query string.
        candidates: List of chunk dictionaries from vector retrieval.
        client: OpenAI-compatible client.
        model: Chat model name.
        final_k: Number of results to show in comparison.

    Returns:
        Dictionary with "before", "after", and metadata.
    """
    if not candidates:
        return {
            "query": query,
            "before": [],
            "after": [],
            "candidate_count": 0,
            "final_k": final_k,
        }

    # Show initial order
    before = candidates[:final_k]

    # Re-rank
    reranked_full = rerank_candidates(query, candidates, client, model, len(candidates))
    after = reranked_full[:final_k]

    return {
        "query": query,
        "before": before,
        "after": after,
        "candidate_count": len(candidates),
        "final_k": final_k,
    }


def display_comparison(
    comparison: dict[str, Any],
    show_text_length: int = 120,
) -> str:
    """
    Format a re-ranking comparison for display.

    Args:
        comparison: Output from rerank_and_compare().
        show_text_length: Number of characters to display from chunk text.

    Returns:
        Formatted string for printing.
    """
    lines = []
    lines.append(f"\nQuery: {comparison['query']}")
    lines.append(f"Candidates: {comparison['candidate_count']}")
    lines.append(f"Final K: {comparison['final_k']}")
    lines.append("\n" + "=" * 80)
    lines.append("BEFORE RE-RANKING (initial vector retrieval order)")
    lines.append("=" * 80)

    for rank, item in enumerate(comparison["before"], start=1):
        lines.append(f"\nRank: {rank}")
        vector_score = item.get('score', 'N/A')
        if isinstance(vector_score, (int, float)):
            lines.append(f"  Vector Score: {vector_score:.4f}")
        else:
            lines.append(f"  Vector Score: {vector_score}")
        lines.append(f"  Source: {item.get('metadata', {}).get('source', 'unknown')}")
        text_preview = item.get("text", "")[:show_text_length]
        lines.append(f"  Text: {text_preview}...")

    lines.append("\n" + "=" * 80)
    lines.append("AFTER RE-RANKING (LLM-scored order)")
    lines.append("=" * 80)

    for rank, item in enumerate(comparison["after"], start=1):
        lines.append(f"\nRank: {rank}")
        vector_score = item.get('score', 'N/A')
        if isinstance(vector_score, (int, float)):
            lines.append(f"  Vector Score: {vector_score:.4f}")
        else:
            lines.append(f"  Vector Score: {vector_score}")
        
        rerank_score = item.get('rerank_score', 'N/A')
        if isinstance(rerank_score, (int, float)):
            lines.append(f"  Rerank Score: {rerank_score:.2f}")
        else:
            lines.append(f"  Rerank Score: {rerank_score}")
        
        lines.append(f"  Source: {item.get('metadata', {}).get('source', 'unknown')}")
        text_preview = item.get("text", "")[:show_text_length]
        lines.append(f"  Text: {text_preview}...")

    return "\n".join(lines)
