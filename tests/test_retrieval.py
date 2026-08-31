import unittest

from src.retrieval import retrieve_top_k


class RetrievalTests(unittest.TestCase):
    def test_retrieve_top_k_returns_scores_and_metadata(self):
        chunks = [
            {
                "text": "Clinical trial adverse events included headache and nausea.",
                "metadata": {"source": "clinical_trial_overview.txt", "chunk_index": 0},
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "text": "Eligibility requires adult patients with moderate disease.",
                "metadata": {"source": "eligibility_criteria.md", "chunk_index": 1},
                "embedding": [0.0, 1.0, 0.0],
            },
            {
                "text": "The study protocol summary describes treatment goals.",
                "metadata": {"source": "study_protocol.txt", "chunk_index": 2},
                "embedding": [0.0, 0.0, 1.0],
            },
        ]

        query = "What adverse events were reported during the clinical trial?"
        query_embedding = [1.0, 0.1, 0.0]

        results = retrieve_top_k(query, query_embedding, chunks, k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["metadata"]["source"], "clinical_trial_overview.txt")
        self.assertGreater(results[0]["score"], results[1]["score"])
        self.assertIn("text", results[0])
        self.assertIn("metadata", results[0])


if __name__ == "__main__":
    unittest.main()
