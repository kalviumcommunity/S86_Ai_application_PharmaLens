"""
5 Runnable Examples: Retrieval Evaluation (No API Key Required)

These examples show different patterns for measuring and improving retrieval.
Each example is self-contained and uses mock data.

Run any example:
  python -m src.retrieval_evaluation_examples
"""

from __future__ import annotations

import json
import time
from typing import Any


# ============================================================================
# EXAMPLE 1: Simple Evaluation Pattern
# ============================================================================

def example_1_simple_evaluation() -> dict[str, Any]:
    """
    Example 1: Evaluate a single query

    This shows the basic pattern for evaluating retrieval on one query.
    """
    from src.retrieval_evaluation import evaluate_retrieval

    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Evaluation Pattern")
    print("=" * 80)

    # Create sample chunks (in real usage, these come from your corpus)
    chunks = [
        {
            "text": "The clinical trial enrolled 100 patients.",
            "metadata": {"source": "trial.txt", "chunk_index": 0},
        },
        {
            "text": "Adverse events included headache and nausea.",
            "metadata": {"source": "trial.txt", "chunk_index": 1},
        },
        {
            "text": "Patients must be ages 18-65 to participate.",
            "metadata": {"source": "eligibility.txt", "chunk_index": 0},
        },
    ]

    # Query we want to evaluate
    query = "What adverse events were reported?"

    # Chunks we retrieved (e.g., from vector search)
    retrieved = chunks[:2]  # Got the first two chunks

    # Chunks we expect to find (ground truth)
    relevant = {"trial.txt:1"}  # Only the second chunk is relevant

    # Evaluate
    result = evaluate_retrieval(query, retrieved, relevant)

    print(f"\nQuery: {query}")
    print(f"Retrieved: {len(result['retrieved_ids'])} chunks")
    print(f"Relevant: {len(result['relevant_ids'])} chunks")
    print(f"Hits: {len(result['hits'])} chunks")
    print(f"\nRecall:    {result['recall']:.0%}")
    print(f"Precision: {result['precision']:.0%}")

    return result


# ============================================================================
# EXAMPLE 2: Comparing Before/After Re-Ranking
# ============================================================================

def example_2_before_after_reranking() -> dict[str, Any]:
    """
    Example 2: Measure impact of re-ranking on recall/precision

    This shows how to evaluate retrieval before and after applying re-ranking.
    """
    from src.retrieval_evaluation import evaluate_retrieval

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Before/After Re-Ranking Comparison")
    print("=" * 80)

    chunks = [
        {
            "text": "Adverse events reported include headache, nausea, and dizziness.",
            "metadata": {"source": "trial.txt", "chunk_index": 1},
        },
        {
            "text": "Patients were between 18 and 65 years old.",
            "metadata": {"source": "eligibility.txt", "chunk_index": 0},
        },
        {
            "text": "The study lasted 12 weeks.",
            "metadata": {"source": "protocol.txt", "chunk_index": 2},
        },
    ]

    query = "What adverse events were reported?"
    relevant = {"trial.txt:1"}

    # BEFORE re-ranking: retrieved in order (wrong order)
    print("\nBefore Re-Ranking:")
    before_result = evaluate_retrieval(query, chunks, relevant)
    print(f"  Retrieved: {before_result['retrieved_ids']}")
    print(f"  Recall:    {before_result['recall']:.0%}")
    print(f"  Precision: {before_result['precision']:.0%}")

    # AFTER re-ranking: reordered to put relevant chunk first
    reordered = [chunks[0], chunks[1], chunks[2]]  # trial.txt:1 is first
    print("\nAfter Re-Ranking:")
    after_result = evaluate_retrieval(query, reordered, relevant)
    print(f"  Retrieved: {after_result['retrieved_ids']}")
    print(f"  Recall:    {after_result['recall']:.0%}")
    print(f"  Precision: {after_result['precision']:.0%}")

    return {
        "before": before_result,
        "after": after_result,
        "recall_improvement": after_result["recall"] - before_result["recall"],
    }


# ============================================================================
# EXAMPLE 3: Metric Aggregation Across Multiple Queries
# ============================================================================

def example_3_metric_aggregation() -> dict[str, Any]:
    """
    Example 3: Evaluate multiple queries and aggregate metrics

    This shows how to measure overall retrieval quality across a set of queries.
    """
    from src.retrieval_evaluation import evaluate_retrieval, aggregate_metrics

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Metric Aggregation Across Multiple Queries")
    print("=" * 80)

    chunks = [
        {
            "text": "Adverse events included headache and nausea.",
            "metadata": {"source": "trial.txt", "chunk_index": 1},
        },
        {
            "text": "Patients must be ages 18-65.",
            "metadata": {"source": "eligibility.txt", "chunk_index": 0},
        },
        {
            "text": "The study protocol specified enrollment of 100 patients.",
            "metadata": {"source": "protocol.txt", "chunk_index": 2},
        },
    ]

    # Define test queries
    queries = [
        {
            "query": "What adverse events were reported?",
            "relevant_chunk_ids": {"trial.txt:1"},
            "retrieved": chunks[:2],  # Got relevant chunk
        },
        {
            "query": "Who is eligible to participate?",
            "relevant_chunk_ids": {"eligibility.txt:0"},
            "retrieved": chunks[:2],  # Got relevant chunk
        },
        {
            "query": "How many patients were enrolled?",
            "relevant_chunk_ids": {"protocol.txt:2"},
            "retrieved": chunks[1:],  # Missed relevant chunk
        },
    ]

    # Evaluate each query
    results = []
    for q in queries:
        result = evaluate_retrieval(q["query"], q["retrieved"], q["relevant_chunk_ids"])
        results.append(result)
        print(f"\n{q['query']}")
        print(f"  Recall:    {result['recall']:.0%}")
        print(f"  Precision: {result['precision']:.0%}")

    # Aggregate metrics
    metrics = aggregate_metrics(results)

    print("\n" + "-" * 80)
    print("AGGREGATED METRICS")
    print("-" * 80)
    print(f"Queries evaluated: {metrics['num_queries']}")
    print(f"Avg Recall:        {metrics['avg_recall']:.1%}")
    print(f"Avg Precision:     {metrics['avg_precision']:.1%}")
    print(f"Recall range:      {metrics['min_recall']:.1%} - {metrics['max_recall']:.1%}")

    return metrics


# ============================================================================
# EXAMPLE 4: Finding and Analyzing Failures
# ============================================================================

def example_4_failure_analysis() -> dict[str, Any]:
    """
    Example 4: Find queries that failed and analyze why

    This shows how to identify retrieval failures for debugging.
    """
    from src.retrieval_evaluation import evaluate_retrieval, find_failures

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Failure Analysis")
    print("=" * 80)

    chunks = [
        {
            "text": "Adverse events: headache, nausea, dizziness.",
            "metadata": {"source": "safety.txt", "chunk_index": 1},
        },
        {
            "text": "Eligibility: ages 18-65, no prior treatment.",
            "metadata": {"source": "inclusion.txt", "chunk_index": 0},
        },
        {
            "text": "Study design: 12-week double-blind trial.",
            "metadata": {"source": "protocol.txt", "chunk_index": 2},
        },
    ]

    # Test queries with known relevant chunks
    queries = [
        {
            "query": "What adverse events occurred?",
            "relevant": {"safety.txt:1"},
            "retrieved": [chunks[0]],  # Retrieved correct chunk
        },
        {
            "query": "Who can participate?",
            "relevant": {"inclusion.txt:0"},
            "retrieved": [chunks[1], chunks[2]],  # Retrieved correct chunk
        },
        {
            "query": "What was the study design?",
            "relevant": {"protocol.txt:2"},
            "retrieved": [chunks[0], chunks[1]],  # MISSED relevant chunk
        },
        {
            "query": "How long was the trial?",
            "relevant": {"protocol.txt:2"},
            "retrieved": [chunks[1]],  # MISSED relevant chunk
        },
    ]

    # Evaluate all queries
    results = []
    for q in queries:
        result = evaluate_retrieval(q["query"], q["retrieved"], q["relevant"])
        results.append(result)

    # Find failures (queries with recall < 100%)
    failures = find_failures(results, recall_threshold=1.0)

    print(f"\nTotal queries: {len(results)}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("\n" + "-" * 80)
        print("FAILURE ANALYSIS")
        print("-" * 80)

        for i, failure in enumerate(failures, 1):
            print(f"\n{i}. Query: {failure['query']}")
            print(f"   Expected: {failure['relevant_ids']}")
            print(f"   Retrieved: {failure['retrieved_ids']}")
            print(f"   Recall: {failure['recall']:.0%}")

            # Identify failure type
            if failure["recall"] == 0.0:
                cause = "Zero recall - completely missed relevant chunk"
            else:
                cause = f"Partial recall - missed {len(failure['relevant_ids']) - failure['num_hits']} chunks"

            print(f"   Cause: {cause}")

    return {"total": len(results), "failures": len(failures), "failed_queries": failures}


# ============================================================================
# EXAMPLE 5: Measuring k Values Trade-off
# ============================================================================

def example_5_k_values_tradeoff() -> dict[str, Any]:
    """
    Example 5: Measure recall/precision trade-off at different k values

    This shows how to find the optimal k value for your use case.
    """
    from src.retrieval_evaluation import evaluate_retrieval, aggregate_metrics

    print("\n" + "=" * 80)
    print("EXAMPLE 5: k Values Trade-off Analysis")
    print("=" * 80)

    # Create chunks with simulated relevance scores
    chunks = [
        {
            "text": "Adverse events: headache, nausea, dizziness.",
            "metadata": {"source": "safety.txt", "chunk_index": 1},
        },
        {
            "text": "Similar but less relevant information about events.",
            "metadata": {"source": "other.txt", "chunk_index": 2},
        },
        {
            "text": "More information about safety and side effects.",
            "metadata": {"source": "safety.txt", "chunk_index": 3},
        },
        {
            "text": "Unrelated clinical information.",
            "metadata": {"source": "clinical.txt", "chunk_index": 4},
        },
        {
            "text": "Additional safety notes.",
            "metadata": {"source": "safety.txt", "chunk_index": 5},
        },
    ]

    query = "What adverse events were reported?"
    relevant = {"safety.txt:1"}

    print(f"\nQuery: {query}")
    print(f"Relevant: {relevant}")

    results_by_k = {}

    for k in [1, 2, 3, 5]:
        retrieved = chunks[:k]
        result = evaluate_retrieval(query, retrieved, relevant)

        recall = result["recall"]
        precision = result["precision"]
        results_by_k[k] = result

        print(f"\nk={k}:")
        print(f"  Retrieved: {len(retrieved)} chunks")
        print(f"  Recall:    {recall:.0%}")
        print(f"  Precision: {precision:.0%}")

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print("As k increases:")
    print("  ✓ Recall improves (more chances to find relevant chunk)")
    print("  ✗ Precision declines (more irrelevant chunks included)")
    print("\nChoose k based on your use case:")
    print("  - High-precision (quality over coverage): k=3-5")
    print("  - Balanced: k=5-10")
    print("  - High-recall (coverage over quality): k=10-20")

    return results_by_k


# ============================================================================
# Main Runner
# ============================================================================

def main() -> None:
    """Run all examples."""
    start_time = time.time()

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION - 5 RUNNABLE EXAMPLES")
    print("=" * 80)
    print("\nThese examples show patterns for measuring and improving retrieval.")
    print("No API key required - all use mock data.")

    # Run examples
    example_1_simple_evaluation()
    example_2_before_after_reranking()
    example_3_metric_aggregation()
    example_4_failure_analysis()
    example_5_k_values_tradeoff()

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print(f"Execution time: {elapsed:.3f}s")
    print("\nNext steps:")
    print("  1. Read the quick start: docs/RETRIEVAL_EVALUATION_QUICKSTART.md")
    print("  2. Review the full guide: RETRIEVAL_EVALUATION_GUIDE.md")
    print("  3. Run the demo: python -m src.retrieval_evaluation_demo")
    print("  4. Build your own labelled queries")
    print("  5. Evaluate your retrieval system")


if __name__ == "__main__":
    main()
