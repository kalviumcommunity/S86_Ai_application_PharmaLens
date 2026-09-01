"""
Demo: Retrieval evaluation with recall and precision measurement.

This demo shows how to:
1. Create a labelled query set with known relevant chunks
2. Evaluate retrieval quality
3. Aggregate metrics
4. Inspect failures to find improvement opportunities
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
from src.retrieval_evaluation import (
    evaluate_queries,
    aggregate_metrics,
    find_failures,
    detailed_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "retrieval_evaluation_results.json"


def build_labelled_queries() -> list[dict[str, Any]]:
    """
    Build a set of labelled queries for testing.
    
    Each query has a set of chunk IDs that should be retrieved.
    Format of chunk IDs: "source:chunk_index"
    """
    return [
        {
            "query": "What adverse events were reported during the clinical trial?",
            "relevant_chunk_ids": {"clinical_trial_overview.txt:2"},
        },
        {
            "query": "Who was eligible to participate in the study?",
            "relevant_chunk_ids": {"eligibility_criteria.md:1"},
        },
        {
            "query": "What were the treatment goals of the study?",
            "relevant_chunk_ids": {"study_protocol.txt:0"},
        },
        {
            "query": "Tell me about the clinical trial.",
            "relevant_chunk_ids": {
                "clinical_trial_overview.txt:2",
                "eligibility_criteria.md:1",
                "study_protocol.txt:0",
            },
        },
        {
            "query": "What happened in the study?",
            "relevant_chunk_ids": {
                "clinical_trial_overview.txt:2",
                "study_protocol.txt:0",
            },
        },
    ]


def demo_with_demo_data() -> dict[str, Any]:
    """
    Run evaluation using built-in demo data (no API required).
    """
    print("\n" + "="*80)
    print("RETRIEVAL EVALUATION DEMO (Demo Data)")
    print("="*80)
    
    # Load demo data
    chunk_records = build_demo_chunk_records()
    print(f"\nLoaded {len(chunk_records)} demo chunks")
    
    # Build labelled queries
    labelled_queries = build_labelled_queries()
    print(f"Loaded {len(labelled_queries)} labelled queries")
    
    # Create retriever function that works with demo data
    def retrieve_fn(query: str, k: int = 5) -> list[dict[str, Any]]:
        # For demo, use simple mock scores instead of embeddings
        # In real usage, this would embed the query and search vectors
        from src.embedding_demo import cosine_similarity
        
        # Use a mock embedding (in practice, would call embed_query)
        mock_embedding = [0.5, 0.5, 0.5]
        return retrieve_top_k(query, mock_embedding, chunk_records, k=k)
    
    # Evaluate queries
    print(f"\nEvaluating queries (k=3)...")
    start_time = time.time()
    results = evaluate_queries(labelled_queries, retrieve_fn, k=3)
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.4f}s")
    
    # Get metrics
    metrics = aggregate_metrics(results)
    
    # Show summary
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    print(f"Queries evaluated:  {metrics['num_queries']}")
    print(f"Avg Recall:         {metrics['avg_recall']:.1%}")
    print(f"Avg Precision:      {metrics['avg_precision']:.1%}")
    print(f"Recall range:       {metrics['min_recall']:.1%} - {metrics['max_recall']:.1%}")
    
    # Show failures
    failures = find_failures(results)
    if failures:
        print(f"\nFailed queries: {len(failures)}/{len(results)}")
        print("\nTop failures:")
        for failure in failures[:3]:
            print(f"\n  Query: {failure['query']}")
            print(f"  Expected: {failure['relevant_ids']}")
            print(f"  Retrieved: {failure['retrieved_ids']}")
            print(f"  Recall: {failure['recall']:.1%}, Precision: {failure['precision']:.1%}")
    else:
        print(f"\n✓ Perfect recall! All relevant chunks were retrieved.")
    
    # Show detailed report
    report = detailed_report(results, include_all=True)
    print(report)
    
    # Build result structure
    result = {
        "demo_type": "demo_data",
        "num_queries": len(labelled_queries),
        "k": 3,
        "chunk_count": len(chunk_records),
        "metrics": {
            "num_queries": metrics['num_queries'],
            "avg_recall": round(metrics['avg_recall'], 4),
            "avg_precision": round(metrics['avg_precision'], 4),
            "min_recall": round(metrics['min_recall'], 4),
            "max_recall": round(metrics['max_recall'], 4),
        },
        "failures_count": len(failures),
        "results": [
            {
                "query": r["query"],
                "recall": round(r["recall"], 4),
                "precision": round(r["precision"], 4),
                "num_relevant": r["num_relevant"],
                "num_retrieved": r["num_retrieved"],
                "num_hits": r["num_hits"],
            }
            for r in results
        ],
        "timing_seconds": round(eval_time, 4),
    }
    
    return result


def main() -> None:
    """Run the retrieval evaluation demo."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        result = demo_with_demo_data()
        
        # Save results
        OUTPUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n✓ Results saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"\nError running evaluation demo: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    main()
