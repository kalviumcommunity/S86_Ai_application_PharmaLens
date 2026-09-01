"""
Demo: Chunk re-ranking for precision-focused retrieval.

This demo shows how re-ranking improves the relevance of retrieved chunks
by scoring them more carefully against the query after initial vector retrieval.

Pattern:
  1. Retrieve a larger candidate set from vector search (e.g., k=10)
  2. Re-rank all candidates using LLM scoring
  3. Keep top-k for final context (e.g., k=3)
  4. Compare before and after ordering
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import load_settings
from src.retrieval import build_demo_chunk_records, retrieve_top_k, embed_query
from src.reranking import rerank_and_compare, display_comparison

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "reranking_demo_results.json"

# Sample query for demonstration
SAMPLE_QUERY = "What adverse events were reported during the clinical trial?"


def run_reranking_demo() -> dict[str, Any]:
    """
    Run the re-ranking demo: retrieve candidates, re-rank, compare results.

    Returns:
        Dictionary with retrieval stats, before/after ranking, and timing info.
    """
    # Load configuration
    settings = load_settings(require_chat=True, require_embedding=True)

    # Initialize client
    client = OpenAI(
        base_url=settings["openai_base_url"] or None,
        api_key=settings["openai_api_key"],
    )

    # Load sample chunks
    chunk_records = build_demo_chunk_records()
    print(f"\nLoaded {len(chunk_records)} demo chunks.")

    # Embed the query
    print(f"Embedding query: {SAMPLE_QUERY}")
    query_embedding = embed_query(client, settings["embed_model"], SAMPLE_QUERY)

    # Step 1: Retrieve larger candidate set (k=10)
    k_candidates = 10
    print(f"\nStep 1: Retrieving {k_candidates} candidates from vector search...")
    start_time = time.time()
    candidates = retrieve_top_k(
        SAMPLE_QUERY,
        query_embedding,
        chunk_records,
        k=k_candidates,
    )
    retrieval_time = time.time() - start_time
    print(f"  ✓ Retrieved {len(candidates)} candidates in {retrieval_time:.4f}s")

    # Step 2: Re-rank candidates to select top-3
    final_k = 3
    print(f"\nStep 2: Re-ranking {len(candidates)} candidates for top-{final_k}...")
    start_time = time.time()
    comparison = rerank_and_compare(
        SAMPLE_QUERY,
        candidates,
        client,
        settings["chat_model"],
        final_k=final_k,
    )
    reranking_time = time.time() - start_time
    print(f"  ✓ Re-ranking completed in {reranking_time:.4f}s")

    # Display results
    display_text = display_comparison(comparison)
    print(display_text)

    # Step 3: Cost and latency analysis
    print("\n" + "=" * 80)
    print("COST & LATENCY ANALYSIS")
    print("=" * 80)

    num_llm_calls = len(candidates)
    print(f"\nRetrieval phase:")
    print(f"  - Vector search: {retrieval_time:.4f}s (1 DB query)")
    print(f"  - Result: {len(candidates)} candidates")

    print(f"\nRe-ranking phase:")
    print(f"  - LLM calls: {num_llm_calls} (one per candidate)")
    print(f"  - Total time: {reranking_time:.4f}s")
    print(f"  - Per-call: {reranking_time / num_llm_calls:.4f}s")

    total_time = retrieval_time + reranking_time
    print(f"\nTotal end-to-end time: {total_time:.4f}s")

    print(f"\nTrade-offs:")
    print(f"  ✓ Precision: Re-ranking improves ranking quality (more relevant chunks at top)")
    print(f"  ✗ Latency: +{reranking_time:.4f}s due to LLM scoring")
    print(f"  ✗ Cost: {num_llm_calls} additional LLM calls beyond retrieval")

    # Build output structure
    result = {
        "query": SAMPLE_QUERY,
        "embedding_model": settings["embed_model"],
        "chat_model": settings["chat_model"],
        "retrieval_config": {
            "candidate_count": len(candidates),
            "final_k": final_k,
        },
        "before_reranking": [
            {
                "rank": i + 1,
                "vector_score": round(item.get("score", 0.0), 6),
                "source": item.get("metadata", {}).get("source", "unknown"),
                "text": item.get("text", "")[:120],
            }
            for i, item in enumerate(comparison["before"])
        ],
        "after_reranking": [
            {
                "rank": i + 1,
                "vector_score": round(item.get("score", 0.0), 6),
                "rerank_score": round(item.get("rerank_score", 0.0), 2),
                "source": item.get("metadata", {}).get("source", "unknown"),
                "text": item.get("text", "")[:120],
            }
            for i, item in enumerate(comparison["after"])
        ],
        "timing": {
            "retrieval_seconds": round(retrieval_time, 4),
            "reranking_seconds": round(reranking_time, 4),
            "total_seconds": round(total_time, 4),
            "llm_calls": num_llm_calls,
        },
        "analysis": {
            "precision_improvement": (
                "Chunks are re-sorted by relevance to query; "
                "highest-scoring chunks move to top regardless of initial vector ranking."
            ),
            "cost_latency_tradeoff": (
                f"Adds {num_llm_calls} LLM calls (+{reranking_time:.2f}s latency) "
                f"to improve ranking precision. Worth it when answer quality matters more than speed."
            ),
        },
    }

    return result


def main() -> None:
    """Run the re-ranking demo and save results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = run_reranking_demo()
        
        # Save to JSON
        OUTPUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n✓ Results saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error running re-ranking demo: {e}")
        raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    main()
