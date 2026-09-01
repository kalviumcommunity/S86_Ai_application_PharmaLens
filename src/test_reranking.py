"""
Tests for the re-ranking module.

Verifies that re-ranking correctly scores and re-orders chunks based on
relevance to the query.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from typing import Any

from src.reranking import (
    rerank_score_with_llm,
    rerank_candidates,
    rerank_and_compare,
    display_comparison,
)


class TestReranking(unittest.TestCase):
    """Test cases for re-ranking functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sample_chunks = [
            {
                "text": "Clinical trial adverse events include headache and nausea.",
                "metadata": {"source": "trial.txt", "chunk_index": 0},
                "score": 0.85,
            },
            {
                "text": "The study protocol outlines the methodology.",
                "metadata": {"source": "protocol.txt", "chunk_index": 1},
                "score": 0.72,
            },
            {
                "text": "Eligibility criteria include adults 18-65 years old.",
                "metadata": {"source": "eligibility.txt", "chunk_index": 2},
                "score": 0.65,
            },
        ]
        self.sample_query = "What adverse events were reported?"

    def test_rerank_score_with_llm(self) -> None:
        """Test LLM scoring of a chunk."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "8.5"
        mock_client.chat.completions.create.return_value = mock_response

        chunk = self.sample_chunks[0]
        score = rerank_score_with_llm(
            mock_client,
            "test-model",
            self.sample_query,
            chunk,
        )

        self.assertEqual(score, 8.5)
        mock_client.chat.completions.create.assert_called_once()

    def test_rerank_score_clamping(self) -> None:
        """Test that scores are clamped to 0-10 range."""
        mock_client = MagicMock()
        
        # Test high score
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "15.5"
        mock_client.chat.completions.create.return_value = mock_response

        score = rerank_score_with_llm(
            mock_client,
            "test-model",
            self.sample_query,
            self.sample_chunks[0],
        )
        self.assertEqual(score, 10.0)

        # Test low score
        mock_response.choices[0].message.content = "-5.0"
        score = rerank_score_with_llm(
            mock_client,
            "test-model",
            self.sample_query,
            self.sample_chunks[0],
        )
        self.assertEqual(score, 0.0)

    def test_rerank_candidates_empty(self) -> None:
        """Test re-ranking with empty candidate list."""
        mock_client = MagicMock()
        result = rerank_candidates(
            self.sample_query,
            [],
            mock_client,
            "test-model",
            final_k=3,
        )
        self.assertEqual(result, [])

    def test_rerank_candidates_invalid_k(self) -> None:
        """Test re-ranking with invalid final_k."""
        mock_client = MagicMock()
        with self.assertRaises(ValueError):
            rerank_candidates(
                self.sample_query,
                self.sample_chunks,
                mock_client,
                "test-model",
                final_k=0,
            )

    def test_rerank_candidates_reordering(self) -> None:
        """Test that re-ranking reorders chunks by re-rank score."""
        mock_client = MagicMock()

        # Set up mock to return scores: 9, 5, 7
        # So the middle chunk should move to the top
        scores = [9.0, 5.0, 7.0]
        call_count = [0]

        def side_effect(*args, **kwargs):
            response = MagicMock()
            response.choices[0].message.content = str(scores[call_count[0]])
            call_count[0] += 1
            return response

        mock_client.chat.completions.create.side_effect = side_effect

        result = rerank_candidates(
            self.sample_query,
            self.sample_chunks,
            mock_client,
            "test-model",
            final_k=3,
        )

        # First chunk should have score 9.0 (highest)
        self.assertEqual(result[0]["rerank_score"], 9.0)
        # Second should have score 7.0
        self.assertEqual(result[1]["rerank_score"], 7.0)
        # Third should have score 5.0 (lowest)
        self.assertEqual(result[2]["rerank_score"], 5.0)

    def test_rerank_candidates_respects_final_k(self) -> None:
        """Test that re-ranking returns only top-k results."""
        mock_client = MagicMock()
        scores = [3.0, 2.0, 1.0]
        call_count = [0]

        def side_effect(*args, **kwargs):
            response = MagicMock()
            response.choices[0].message.content = str(scores[call_count[0]])
            call_count[0] += 1
            return response

        mock_client.chat.completions.create.side_effect = side_effect

        result = rerank_candidates(
            self.sample_query,
            self.sample_chunks,
            mock_client,
            "test-model",
            final_k=2,
        )

        self.assertEqual(len(result), 2)

    def test_rerank_and_compare_structure(self) -> None:
        """Test that rerank_and_compare returns correct structure."""
        mock_client = MagicMock()
        scores = [8.0, 6.0, 4.0]
        call_count = [0]

        def side_effect(*args, **kwargs):
            response = MagicMock()
            response.choices[0].message.content = str(scores[call_count[0]])
            call_count[0] += 1
            return response

        mock_client.chat.completions.create.side_effect = side_effect

        result = rerank_and_compare(
            self.sample_query,
            self.sample_chunks,
            mock_client,
            "test-model",
            final_k=2,
        )

        self.assertIn("query", result)
        self.assertIn("before", result)
        self.assertIn("after", result)
        self.assertIn("candidate_count", result)
        self.assertIn("final_k", result)

        self.assertEqual(result["query"], self.sample_query)
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["final_k"], 2)
        self.assertEqual(len(result["before"]), 2)
        self.assertEqual(len(result["after"]), 2)

    def test_rerank_and_compare_empty(self) -> None:
        """Test rerank_and_compare with empty candidates."""
        mock_client = MagicMock()
        result = rerank_and_compare(
            self.sample_query,
            [],
            mock_client,
            "test-model",
            final_k=3,
        )

        self.assertEqual(len(result["before"]), 0)
        self.assertEqual(len(result["after"]), 0)
        self.assertEqual(result["candidate_count"], 0)

    def test_display_comparison_format(self) -> None:
        """Test that display_comparison produces readable output."""
        comparison = {
            "query": self.sample_query,
            "before": self.sample_chunks[:2],
            "after": list(reversed(self.sample_chunks[:2])),
            "candidate_count": 3,
            "final_k": 2,
        }

        output = display_comparison(comparison)

        self.assertIn(self.sample_query, output)
        self.assertIn("BEFORE RE-RANKING", output)
        self.assertIn("AFTER RE-RANKING", output)
        self.assertIn("Rank: 1", output)
        self.assertIn("Rank: 2", output)

    def test_display_comparison_shows_scores(self) -> None:
        """Test that display output includes both vector and rerank scores."""
        self.sample_chunks[0]["rerank_score"] = 8.5
        self.sample_chunks[1]["rerank_score"] = 6.2

        comparison = {
            "query": self.sample_query,
            "before": self.sample_chunks[:1],
            "after": self.sample_chunks[:2],
            "candidate_count": 2,
            "final_k": 2,
        }

        output = display_comparison(comparison)

        # Vector score should appear in both sections
        self.assertIn("Vector Score:", output)
        # Rerank score should appear in after section
        self.assertIn("Rerank Score:", output)


class TestRerankerEdgeCases(unittest.TestCase):
    """Edge case tests for re-ranker."""

    def test_rerank_with_missing_text_field(self) -> None:
        """Test handling of chunks without text field."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "5.0"
        mock_client.chat.completions.create.return_value = mock_response

        chunk = {"metadata": {"source": "test.txt"}}  # No "text" field
        score = rerank_score_with_llm(
            mock_client,
            "test-model",
            "What is this?",
            chunk,
        )

        # Should handle gracefully
        self.assertEqual(score, 5.0)

    def test_rerank_with_special_characters(self) -> None:
        """Test re-ranking with special characters in text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "7.0"
        mock_client.chat.completions.create.return_value = mock_response

        chunk = {
            "text": "Special chars: <>&\"'\\n\\t émoji🔬",
            "metadata": {"source": "special.txt"},
            "score": 0.5,
        }

        score = rerank_score_with_llm(
            mock_client,
            "test-model",
            "What about special chars?",
            chunk,
        )

        self.assertEqual(score, 7.0)


if __name__ == "__main__":
    unittest.main()
