"""
Tests for the retrieval evaluation module.

Verifies that recall and precision calculations are correct and that
failure detection works as expected.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.retrieval_evaluation import (
    build_chunk_id,
    evaluate_retrieval,
    evaluate_queries,
    aggregate_metrics,
    find_failures,
    detailed_report,
    recall_at_k_series,
)


class TestChunkID(unittest.TestCase):
    """Test chunk ID generation."""

    def test_chunk_id_basic(self) -> None:
        """Test basic chunk ID generation."""
        metadata = {"source": "trial.txt", "chunk_index": 2}
        chunk_id = build_chunk_id(metadata)
        self.assertEqual(chunk_id, "trial.txt:2")

    def test_chunk_id_missing_index(self) -> None:
        """Test chunk ID when chunk_index is missing."""
        metadata = {"source": "trial.txt"}
        chunk_id = build_chunk_id(metadata)
        self.assertEqual(chunk_id, "trial.txt:0")

    def test_chunk_id_missing_source(self) -> None:
        """Test chunk ID when source is missing."""
        metadata = {"chunk_index": 2}
        chunk_id = build_chunk_id(metadata)
        self.assertEqual(chunk_id, "unknown:2")


class TestEvaluateRetrieval(unittest.TestCase):
    """Test retrieval evaluation logic."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sample_chunks = [
            {
                "text": "Clinical trial adverse events",
                "metadata": {"source": "trial.txt", "chunk_index": 0},
            },
            {
                "text": "Eligibility criteria",
                "metadata": {"source": "eligibility.txt", "chunk_index": 1},
            },
            {
                "text": "Study protocol",
                "metadata": {"source": "protocol.txt", "chunk_index": 2},
            },
        ]
        self.query = "What adverse events occurred?"

    def test_perfect_recall(self) -> None:
        """Test evaluation with perfect recall."""
        relevant = {"trial.txt:0"}
        result = evaluate_retrieval(self.query, self.sample_chunks[:1], relevant)

        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["num_hits"], 1)

    def test_partial_recall(self) -> None:
        """Test evaluation with partial recall."""
        relevant = {"trial.txt:0", "protocol.txt:2"}
        result = evaluate_retrieval(
            self.query, self.sample_chunks[:1], relevant
        )

        self.assertEqual(result["recall"], 0.5)  # 1/2 relevant chunks
        self.assertEqual(result["precision"], 1.0)  # 1/1 retrieved were relevant
        self.assertEqual(result["num_hits"], 1)

    def test_zero_recall(self) -> None:
        """Test evaluation with zero recall."""
        relevant = {"trial.txt:0"}
        result = evaluate_retrieval(
            self.query, self.sample_chunks[1:], relevant
        )

        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["num_hits"], 0)

    def test_low_precision(self) -> None:
        """Test evaluation with low precision."""
        relevant = {"trial.txt:0"}
        result = evaluate_retrieval(self.query, self.sample_chunks, relevant)

        self.assertEqual(result["recall"], 1.0)  # Found the 1 relevant
        self.assertEqual(result["precision"], 1.0/3)  # Only 1/3 were relevant
        self.assertEqual(result["num_hits"], 1)

    def test_empty_retrieved(self) -> None:
        """Test evaluation with no retrieved chunks."""
        relevant = {"trial.txt:0"}
        result = evaluate_retrieval(self.query, [], relevant)

        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["num_retrieved"], 0)

    def test_empty_relevant(self) -> None:
        """Test evaluation with no relevant chunks."""
        relevant = set()
        result = evaluate_retrieval(self.query, self.sample_chunks, relevant)

        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["num_relevant"], 0)


class TestEvaluateQueries(unittest.TestCase):
    """Test evaluation of multiple queries."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sample_chunks = [
            {
                "text": "Trial events",
                "metadata": {"source": "trial.txt", "chunk_index": 0},
            },
            {
                "text": "Eligibility",
                "metadata": {"source": "eligibility.txt", "chunk_index": 1},
            },
        ]
        
        self.labelled_queries = [
            {
                "query": "What events?",
                "relevant_chunk_ids": {"trial.txt:0"},
            },
            {
                "query": "Who is eligible?",
                "relevant_chunk_ids": {"eligibility.txt:1"},
            },
        ]

    def test_evaluate_queries_basic(self) -> None:
        """Test evaluation of multiple queries."""
        def retrieve_fn(query: str, k: int = 5) -> list[dict]:
            return self.sample_chunks[:1]  # Always return first chunk

        results = evaluate_queries(self.labelled_queries, retrieve_fn, k=1)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["recall"], 1.0)  # First chunk matches
        self.assertEqual(results[1]["recall"], 0.0)  # First chunk doesn't match

    def test_evaluate_queries_with_error(self) -> None:
        """Test evaluation when retrieve function fails."""
        def retrieve_fn(query: str, k: int = 5) -> list[dict]:
            raise RuntimeError("Retrieval failed")

        results = evaluate_queries(self.labelled_queries, retrieve_fn, k=1)

        self.assertEqual(len(results), 2)
        self.assertTrue("error" in results[0])
        self.assertEqual(results[0]["recall"], 0.0)


class TestAggregateMetrics(unittest.TestCase):
    """Test metric aggregation."""

    def test_aggregate_metrics(self) -> None:
        """Test aggregating metrics from multiple results."""
        results = [
            {
                "query": "q1",
                "recall": 1.0,
                "precision": 1.0,
                "relevant_ids": [],
                "retrieved_ids": [],
                "hits": [],
                "num_relevant": 1,
                "num_retrieved": 1,
                "num_hits": 1,
            },
            {
                "query": "q2",
                "recall": 0.5,
                "precision": 0.5,
                "relevant_ids": [],
                "retrieved_ids": [],
                "hits": [],
                "num_relevant": 2,
                "num_retrieved": 2,
                "num_hits": 1,
            },
        ]

        metrics = aggregate_metrics(results)

        self.assertEqual(metrics["num_queries"], 2)
        self.assertEqual(metrics["avg_recall"], 0.75)
        self.assertEqual(metrics["avg_precision"], 0.75)
        self.assertEqual(metrics["min_recall"], 0.5)
        self.assertEqual(metrics["max_recall"], 1.0)

    def test_aggregate_metrics_empty(self) -> None:
        """Test aggregating empty results."""
        metrics = aggregate_metrics([])

        self.assertEqual(metrics["num_queries"], 0)
        self.assertEqual(metrics["avg_recall"], 0.0)
        self.assertEqual(metrics["avg_precision"], 0.0)


class TestFindFailures(unittest.TestCase):
    """Test failure detection."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.results = [
            {
                "query": "q1",
                "recall": 1.0,
                "precision": 1.0,
                "relevant_ids": [],
                "retrieved_ids": [],
                "hits": [],
                "num_relevant": 1,
                "num_retrieved": 1,
                "num_hits": 1,
            },
            {
                "query": "q2",
                "recall": 0.5,
                "precision": 0.5,
                "relevant_ids": [],
                "retrieved_ids": [],
                "hits": [],
                "num_relevant": 2,
                "num_retrieved": 2,
                "num_hits": 1,
            },
            {
                "query": "q3",
                "recall": 0.0,
                "precision": 0.0,
                "relevant_ids": [],
                "retrieved_ids": [],
                "hits": [],
                "num_relevant": 1,
                "num_retrieved": 0,
                "num_hits": 0,
            },
        ]

    def test_find_failures_perfect_recall(self) -> None:
        """Test finding failures when all have recall < 1.0."""
        failures = find_failures(self.results, recall_threshold=1.0)

        self.assertEqual(len(failures), 2)  # q2 and q3 fail
        self.assertEqual(failures[0]["query"], "q2")
        self.assertEqual(failures[1]["query"], "q3")

    def test_find_failures_threshold(self) -> None:
        """Test finding failures with custom threshold."""
        failures = find_failures(self.results, recall_threshold=0.6)

        self.assertEqual(len(failures), 2)  # q2 and q3 fail

    def test_find_failures_none(self) -> None:
        """Test when no queries fail."""
        failures = find_failures(self.results, recall_threshold=0.0)

        self.assertEqual(len(failures), 0)


class TestReports(unittest.TestCase):
    """Test report generation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.results = [
            {
                "query": "Test query",
                "recall": 0.5,
                "precision": 0.5,
                "relevant_ids": ["a:0", "b:1"],
                "retrieved_ids": ["a:0", "c:2"],
                "hits": ["a:0"],
                "num_relevant": 2,
                "num_retrieved": 2,
                "num_hits": 1,
            },
        ]

    def test_detailed_report(self) -> None:
        """Test detailed report generation."""
        report = detailed_report(self.results)

        self.assertIn("RETRIEVAL EVALUATION REPORT", report)
        self.assertIn("Test query", report)
        self.assertIn("50.0%", report)  # Recall percentage

    def test_detailed_report_empty(self) -> None:
        """Test report with empty results."""
        report = detailed_report([])

        self.assertIn("No results to report", report)

    def test_recall_at_k_series(self) -> None:
        """Test recall@k series calculation."""
        results = [
            {
                "query": "q1",
                "recall": 1.0,
                "num_retrieved": 5,
                "num_relevant": 1,
                "num_hits": 1,
                "retrieved_ids": [],
                "relevant_ids": [],
                "hits": [],
            },
            {
                "query": "q2",
                "recall": 0.5,
                "num_retrieved": 5,
                "num_relevant": 2,
                "num_hits": 1,
                "retrieved_ids": [],
                "relevant_ids": [],
                "hits": [],
            },
        ]

        series = recall_at_k_series(results)

        # Both have 5 retrieved, so should average to 0.75
        self.assertIn(5, series)
        self.assertEqual(series[5], 0.75)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_evaluate_retrieval_with_duplicates(self) -> None:
        """Test evaluation when chunk IDs appear multiple times."""
        chunks = [
            {"text": "A", "metadata": {"source": "a.txt", "chunk_index": 0}},
            {"text": "A", "metadata": {"source": "a.txt", "chunk_index": 0}},
        ]
        relevant = {"a.txt:0"}

        result = evaluate_retrieval("query", chunks, relevant)

        # Both chunks have same ID, so both are hits
        self.assertEqual(result["num_retrieved"], 2)
        self.assertEqual(result["num_hits"], 2)
        # But precision is still computed correctly
        self.assertEqual(result["precision"], 1.0)

    def test_evaluate_retrieval_special_characters(self) -> None:
        """Test evaluation with special characters in source names."""
        chunks = [
            {
                "text": "Content",
                "metadata": {"source": "trial-2024_v1.2.txt", "chunk_index": 0},
            },
        ]
        relevant = {"trial-2024_v1.2.txt:0"}

        result = evaluate_retrieval("query", chunks, relevant)

        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["num_hits"], 1)


if __name__ == "__main__":
    unittest.main()
