"""
Simple example of re-ranking in action.

This example demonstrates the re-ranking workflow without requiring
external API calls, using mock data to show before/after ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


def create_mock_rerank_client():
    """Create a mock OpenAI client for demonstration."""
    client = MagicMock()

    # Mock LLM scoring: return varying scores based on chunk content
    scores = {
        "adverse events": 9.0,
        "eligibility": 4.0,
        "protocol": 6.0,
    }

    def mock_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        if messages:
            content = messages[-1].get("content", "")
            # Simple heuristic: look for keywords in the prompt
            if "adverse" in content.lower():
                score = 9.0
            elif "eligibility" in content.lower():
                score = 4.0
            elif "protocol" in content.lower():
                score = 6.0
            else:
                score = 5.0
        else:
            score = 5.0

        response = MagicMock()
        response.choices[0].message.content = str(score)
        return response

    client.chat.completions.create = mock_create
    return client


def example_simple_reranking():
    """Example 1: Simple re-ranking with mock data."""
    print("=" * 80)
    print("EXAMPLE 1: Simple Re-Ranking Pattern")
    print("=" * 80)

    from reranking import rerank_candidates, display_comparison

    # Mock client
    mock_client = create_mock_rerank_client()

    # Sample chunks retrieved by vector search (in initial order)
    query = "What adverse events were reported during the clinical trial?"
    candidates = [
        {
            "text": "Clinical trial overview: adverse events included headache, nausea, and fatigue during the treatment period.",
            "metadata": {"source": "clinical_trial_overview.txt", "chunk_index": 0},
            "score": 0.85,  # Initial vector similarity score
        },
        {
            "text": "Eligibility criteria: adults with moderate disease and no prior therapy were eligible for the study.",
            "metadata": {"source": "eligibility_criteria.md", "chunk_index": 1},
            "score": 0.72,
        },
        {
            "text": "Study protocol: treatment goals included improvement in disease severity and patient-reported symptoms.",
            "metadata": {"source": "study_protocol.txt", "chunk_index": 2},
            "score": 0.68,
        },
    ]

    print(f"\nQuery: {query}")
    print(f"Candidates: {len(candidates)}")

    # Re-rank to get top-3
    reranked = rerank_candidates(
        query,
        candidates,
        mock_client,
        "gpt-4",
        final_k=3,
    )

    print("\nRe-ranked results:")
    for rank, chunk in enumerate(reranked, start=1):
        print(f"\n{rank}. Score: {chunk['rerank_score']:.1f}")
        print(f"   Source: {chunk['metadata']['source']}")
        print(f"   Text: {chunk['text'][:80]}...")


def example_before_and_after():
    """Example 2: Compare before and after re-ranking."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Before & After Comparison")
    print("=" * 80)

    from reranking import rerank_and_compare, display_comparison

    mock_client = create_mock_rerank_client()

    query = "What adverse events were reported during the clinical trial?"
    candidates = [
        {
            "text": "Eligibility criteria: adults with moderate disease were eligible.",
            "metadata": {"source": "eligibility.txt"},
            "score": 0.75,
        },
        {
            "text": "Clinical trial adverse events: headache (10%), nausea (8%), fatigue (12%).",
            "metadata": {"source": "trial_data.txt"},
            "score": 0.72,
        },
        {
            "text": "Study protocol version 2.1 outlines the trial methodology.",
            "metadata": {"source": "protocol.txt"},
            "score": 0.70,
        },
    ]

    # Show the comparison
    comparison = rerank_and_compare(
        query,
        candidates,
        mock_client,
        "gpt-4",
        final_k=2,
    )

    print(display_comparison(comparison, show_text_length=100))


def example_cost_analysis():
    """Example 3: Analyze cost and latency trade-offs."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Cost & Latency Trade-Off Analysis")
    print("=" * 80)

    print("\nScenario: Pharma Q&A with 10 candidates")
    print()

    # Simulated timings
    vector_search_time = 0.05  # 50ms for vector DB lookup
    candidates_count = 10
    llm_call_time = 0.15  # 150ms per LLM call

    print("WITHOUT Re-Ranking:")
    print(f"  • Vector search: {vector_search_time:.2f}s")
    print(f"  • LLM calls: 0 (just return top-3)")
    print(f"  • Total latency: {vector_search_time:.2f}s")
    print(f"  • Result quality: Good (vector similarity)")

    print("\nWITH Re-Ranking:")
    reranking_time = candidates_count * llm_call_time
    total_time = vector_search_time + reranking_time
    speedup = total_time / vector_search_time
    print(f"  • Vector search: {vector_search_time:.2f}s")
    print(f"  • Re-ranking: {reranking_time:.2f}s ({candidates_count} LLM calls × {llm_call_time:.2f}s)")
    print(f"  • Total latency: {total_time:.2f}s")
    print(f"  • Latency increase: {speedup:.1f}×")
    print(f"  • Result quality: Excellent (LLM-scored relevance)")

    print("\nCost Comparison (using OpenAI GPT-4 pricing):")
    base_cost = 0.001  # Placeholder: retrieval cost
    llm_scoring_cost = 0.03 * (candidates_count / 1000)  # Approximate
    print(f"  • Without re-ranking: ${base_cost:.4f} (minimal)")
    print(f"  • With re-ranking: ${base_cost + llm_scoring_cost:.4f} (+{candidates_count} LLM calls)")
    print(f"  • Cost multiplier: {(base_cost + llm_scoring_cost) / base_cost:.1f}×")

    print("\nWhen to Re-Rank:")
    print("  ✓ Medical/legal/financial content (precision critical)")
    print("  ✓ Complex questions needing careful matching")
    print("  ✓ When you have time budget (batch processing)")
    print("  ✗ Real-time chat (user waiting for response)")
    print("  ✗ Simple lookup queries")
    print("  ✗ Cost-constrained environments")


def example_retrieval_pipeline():
    """Example 4: Full RAG pipeline with re-ranking."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Full RAG Pipeline")
    print("=" * 80)

    print("""
The typical RAG flow:

1. User asks a question
   Query: "What adverse events were reported?"

2. Embed the query
   query_embedding = embed_query(query)

3. RETRIEVE: Vector search for candidates
   candidates = retrieve_top_k(query_embedding, k=10)
   Time: ~50ms | Quality: Good | Cost: Low

4. RE-RANK: Score candidates more carefully
   reranked = rerank_candidates(query, candidates, k=3)
   Time: ~1.5s | Quality: Excellent | Cost: Medium

5. BUILD CONTEXT: Select top-k chunks
   context = format_chunks(reranked[:3])

6. QUERY LLM: Use context to answer
   answer = llm.chat(context + query)

7. RETURN: Send answer to user
   return answer

Key insight:
  • Retrieval is fast but broad (finds "related" chunks)
  • Re-ranking is slow but precise (finds "relevant" chunks)
  • The two-phase approach balances speed and quality
""")


def example_configuration():
    """Example 5: Configuration patterns for re-ranking."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Configuration Patterns")
    print("=" * 80)

    config_examples = {
        "speed_optimized": {
            "candidate_k": 5,
            "final_k": 3,
            "rerank": False,
            "use_case": "Real-time chat, fast responses",
        },
        "balanced": {
            "candidate_k": 10,
            "final_k": 3,
            "rerank": True,
            "use_case": "Standard RAG, good quality/speed balance",
        },
        "precision_optimized": {
            "candidate_k": 20,
            "final_k": 5,
            "rerank": True,
            "rerank_model": "gpt-4",
            "use_case": "Medical/legal, precision critical",
        },
    }

    for name, config in config_examples.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        for key, value in config.items():
            print(f"  • {key}: {value}")


def main():
    """Run all examples."""
    example_simple_reranking()
    example_before_and_after()
    example_cost_analysis()
    example_retrieval_pipeline()
    example_configuration()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Re-ranking improves precision by scoring candidates more carefully after
vector retrieval. It's a trade-off:

  + Higher precision (more relevant chunks)
  + Better LLM answers
  - Increased latency
  - Increased cost

Use re-ranking when precision matters more than speed. For production
systems, consider:

  1. Profile your query latency budget
  2. Measure answer quality with/without re-ranking
  3. Use re-ranking selectively (e.g., only for complex queries)
  4. Consider cross-encoders for faster re-ranking
  5. Cache re-rank scores for common queries

See docs/reranking.md for full integration guide.
""")


if __name__ == "__main__":
    main()
