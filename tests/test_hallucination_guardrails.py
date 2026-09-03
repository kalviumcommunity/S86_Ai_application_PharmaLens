import unittest

from src.hallucination_guardrails import assess_retrieval_quality


class HallucinationGuardrailTests(unittest.TestCase):
    def test_empty_results_are_rejected(self):
        result = assess_retrieval_quality([])

        self.assertFalse(result["is_sufficient"])
        self.assertEqual(result["reason"], "no_context")

    def test_low_scores_are_rejected(self):
        result = assess_retrieval_quality(
            [{"score": 0.64, "text": "Unrelated study details."}]
        )

        self.assertFalse(result["is_sufficient"])
        self.assertEqual(result["reason"], "insufficient_relevance")
        self.assertEqual(result["relevant_count"], 0)

    def test_too_few_relevant_chunks_are_rejected(self):
        result = assess_retrieval_quality(
            [
                {"score": 0.91, "text": "Strong supporting evidence."},
                {"score": 0.40, "text": "Weak supporting evidence."},
            ],
            min_relevant_chunks=2,
        )

        self.assertFalse(result["is_sufficient"])
        self.assertEqual(result["relevant_count"], 1)

    def test_strong_context_is_accepted(self):
        result = assess_retrieval_quality(
            [
                {"score": 0.91, "text": "Headache and nausea were reported."},
                {"score": 0.78, "text": "Fatigue was also reported."},
            ],
            min_relevant_chunks=2,
        )

        self.assertTrue(result["is_sufficient"])
        self.assertEqual(result["reason"], "sufficient_relevance")


if __name__ == "__main__":
    unittest.main()