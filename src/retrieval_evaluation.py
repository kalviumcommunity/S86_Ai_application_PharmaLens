"""
Retrieval evaluation module for measuring recall and precision.

This module provides tools to:
- Define labelled query sets with known relevant chunks
- Evaluate retrieval quality (recall, precision)
- Aggregate metrics across queries
- Inspect failures to identify improvement opportunities
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_chunk_id(metadata: dict[str, Any]) -> str:
    """
    Build a unique chunk ID from metadata.
    
    Combines source and chunk_index to create a unique identifier.
    Format: "source:chunk_index"
    
    Args:
        metadata: Chunk metadata dict containing "source" and optionally "chunk_index"
    
    Returns:
        String chunk ID (e.g., "clinical_trial.txt:2")
    """
    source = metadata.get("source", "unknown")
    chunk_index = metadata.get("chunk_index", 0)
    return f"{source}:{chunk_index}"


def evaluate_retrieval(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    relevant_chunk_ids: set[str],
) -> dict[str, Any]:
    """
    Evaluate retrieval results against known relevant chunks.
    
    Computes recall (fraction of relevant chunks retrieved) and
    precision (fraction of retrieved chunks that are relevant).
    
    Args:
        query: The query string
        retrieved_chunks: List of retrieved chunk dicts with "metadata" field
        relevant_chunk_ids: Set of chunk IDs that should have been retrieved
    
    Returns:
        Dict with:
        - query: The query string
        - retrieved_ids: List of retrieved chunk IDs
        - relevant_ids: Set of relevant chunk IDs
        - hits: List of chunk IDs that were both retrieved and relevant
        - recall: Fraction of relevant chunks that were retrieved (0-1)
        - precision: Fraction of retrieved chunks that are relevant (0-1)
        - num_relevant: Number of relevant chunks
        - num_retrieved: Number of retrieved chunks
    """
    # Extract retrieved chunk IDs
    retrieved_ids = [
        build_chunk_id(chunk["metadata"])
        for chunk in retrieved_chunks
    ]
    
    # Find hits (retrieved AND relevant)
    hits = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevant_chunk_ids]
    
    # Compute recall and precision
    num_relevant = len(relevant_chunk_ids)
    num_retrieved = len(retrieved_ids)
    num_hits = len(hits)
    
    recall = num_hits / num_relevant if num_relevant > 0 else 0.0
    precision = num_hits / num_retrieved if num_retrieved > 0 else 0.0
    
    return {
        "query": query,
        "retrieved_ids": retrieved_ids,
        "relevant_ids": sorted(relevant_chunk_ids),
        "hits": hits,
        "recall": float(recall),
        "precision": float(precision),
        "num_relevant": num_relevant,
        "num_retrieved": num_retrieved,
        "num_hits": num_hits,
    }


def evaluate_queries(
    labelled_queries: list[dict[str, Any]],
    retrieve_fn: callable,
    k: int = 5,
) -> list[dict[str, Any]]:
    """
    Evaluate retrieval on a set of labelled queries.
    
    Args:
        labelled_queries: List of dicts with:
            - "query": Query string
            - "relevant_chunk_ids": Set or list of chunk IDs that should be retrieved
        retrieve_fn: Function that takes (query, k) and returns list of retrieved chunks
        k: Number of results to retrieve per query
    
    Returns:
        List of evaluation results, one per query
    """
    results = []
    
    for item in labelled_queries:
        query = item["query"]
        relevant_ids = set(item["relevant_chunk_ids"])
        
        try:
            # Retrieve top-k chunks
            retrieved = retrieve_fn(query, k=k)
            
            # Evaluate
            evaluation = evaluate_retrieval(query, retrieved, relevant_ids)
            results.append(evaluation)
            
            logger.debug(
                f"Evaluated query '{query[:50]}': "
                f"recall={evaluation['recall']:.2f}, "
                f"precision={evaluation['precision']:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate query '{query}': {e}")
            # Add failed evaluation
            results.append({
                "query": query,
                "retrieved_ids": [],
                "relevant_ids": sorted(relevant_ids),
                "hits": [],
                "recall": 0.0,
                "precision": 0.0,
                "num_relevant": len(relevant_ids),
                "num_retrieved": 0,
                "num_hits": 0,
                "error": str(e),
            })
    
    return results


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate recall and precision metrics across all results.
    
    Args:
        results: List of evaluation results from evaluate_queries()
    
    Returns:
        Dict with aggregated metrics:
        - num_queries: Number of queries evaluated
        - avg_recall: Average recall across all queries
        - avg_precision: Average precision across all queries
        - recall_by_query: List of (query, recall) tuples
        - precision_by_query: List of (query, precision) tuples
        - min_recall: Minimum recall
        - max_recall: Maximum recall
    """
    if not results:
        return {
            "num_queries": 0,
            "avg_recall": 0.0,
            "avg_precision": 0.0,
            "recall_by_query": [],
            "precision_by_query": [],
            "min_recall": 0.0,
            "max_recall": 0.0,
            "min_precision": 0.0,
            "max_precision": 0.0,
        }
    
    recalls = [r["recall"] for r in results]
    precisions = [r["precision"] for r in results]
    
    return {
        "num_queries": len(results),
        "avg_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "avg_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall_by_query": [
            (r["query"], r["recall"]) for r in results
        ],
        "precision_by_query": [
            (r["query"], r["precision"]) for r in results
        ],
        "min_recall": min(recalls) if recalls else 0.0,
        "max_recall": max(recalls) if recalls else 0.0,
        "min_precision": min(precisions) if precisions else 0.0,
        "max_precision": max(precisions) if precisions else 0.0,
    }


def find_failures(
    results: list[dict[str, Any]],
    recall_threshold: float = 1.0,
    precision_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Find queries where recall or precision fell below thresholds.
    
    Args:
        results: List of evaluation results
        recall_threshold: Queries with recall < this fail (default 1.0 = perfect)
        precision_threshold: Queries with precision < this fail
    
    Returns:
        List of failed evaluation results
    """
    failures = [
        result
        for result in results
        if result["recall"] < recall_threshold
        or result["precision"] < precision_threshold
    ]
    return failures


def report_failures(failures: list[dict[str, Any]]) -> str:
    """
    Format failure cases for human review.
    
    Args:
        failures: List of failed evaluation results
    
    Returns:
        Formatted report string
    """
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"RETRIEVAL FAILURES ({len(failures)} queries)")
    lines.append(f"{'='*80}\n")
    
    for i, failure in enumerate(failures, 1):
        lines.append(f"{i}. Query: {failure['query']}")
        lines.append(f"   Recall: {failure['recall']:.1%} | Precision: {failure['precision']:.1%}")
        lines.append(f"   Expected ({failure['num_relevant']}): {failure['relevant_ids']}")
        lines.append(f"   Retrieved ({failure['num_retrieved']}): {failure['retrieved_ids']}")
        
        if failure["hits"]:
            lines.append(f"   ✓ Hits ({len(failure['hits'])}): {failure['hits']}")
        else:
            lines.append(f"   ✗ No hits (missed all relevant chunks)")
        
        if "error" in failure:
            lines.append(f"   Error: {failure['error']}")
        
        lines.append("")
    
    return "\n".join(lines)


def detailed_report(
    results: list[dict[str, Any]],
    include_all: bool = False,
) -> str:
    """
    Generate a detailed evaluation report.
    
    Args:
        results: List of evaluation results
        include_all: If True, show all queries; if False, show only failures
    
    Returns:
        Formatted report string
    """
    if not results:
        return "No results to report."
    
    metrics = aggregate_metrics(results)
    
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("RETRIEVAL EVALUATION REPORT")
    lines.append(f"{'='*80}")
    
    lines.append(f"\nSummary Metrics:")
    lines.append(f"  Queries evaluated: {metrics['num_queries']}")
    lines.append(f"  Avg Recall:        {metrics['avg_recall']:.1%}")
    lines.append(f"  Avg Precision:     {metrics['avg_precision']:.1%}")
    lines.append(f"  Min Recall:        {metrics['min_recall']:.1%}")
    lines.append(f"  Max Recall:        {metrics['max_recall']:.1%}")
    
    # Show failures
    failures = find_failures(results)
    if failures:
        lines.append(report_failures(failures))
    else:
        lines.append(f"\n✓ All queries achieved perfect recall!")
    
    # Show all if requested
    if include_all:
        lines.append(f"\n{'='*80}")
        lines.append("ALL QUERIES")
        lines.append(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result['query']}")
            lines.append(f"   Recall:    {result['recall']:.1%}")
            lines.append(f"   Precision: {result['precision']:.1%}")
            lines.append(f"   Relevant:  {result['num_relevant']} | "
                        f"Retrieved: {result['num_retrieved']} | "
                        f"Hits: {result['num_hits']}")
            lines.append("")
    
    return "\n".join(lines)


def recall_at_k_series(
    results: list[dict[str, Any]],
) -> dict[int, float]:
    """
    Compute average recall@k for k in [1, 3, 5, 10].
    
    Note: This uses the retrieved_ids list and assumes it's already
    limited to k. For true recall@k across multiple k values, you need
    to re-run evaluation with different k parameters.
    
    Args:
        results: List of evaluation results
    
    Returns:
        Dict mapping k to average recall at that k
    """
    if not results:
        return {}
    
    # Group by number retrieved to estimate k values
    recalls_by_count = {}
    for result in results:
        count = result["num_retrieved"]
        if count not in recalls_by_count:
            recalls_by_count[count] = []
        recalls_by_count[count].append(result["recall"])
    
    # Average recall for each count
    return {
        count: sum(recalls) / len(recalls)
        for count, recalls in recalls_by_count.items()
    }
