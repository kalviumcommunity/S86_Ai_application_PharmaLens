import unittest

from src.filtered_retrieval import filter_chunk_records, hybrid_search


class FilteredRetrievalTests(unittest.TestCase):
    def test_filter_restricts_records_to_source(self):
        chunks = [
            {
                "text": "Clinical trial adverse events included headache and nausea.",
                "metadata": {"source": "clinical_trial_overview.txt", "section": "Safety"},
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "text": "This unrelated note discusses clinic operations and staffing.",
                "metadata": {"source": "operations_notes.txt", "section": "Operations"},
                "embedding": [0.0, 1.0, 0.0],
            },
        ]

        filtered = filter_chunk_records(chunks, {"source": "clinical_trial_overview.txt"})
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["metadata"]["source"], "clinical_trial_overview.txt")

    def test_hybrid_search_prioritizes_exact_term_matches(self):
        chunks = [
            {
                "text": "Adverse events included headache and nausea.",
                "metadata": {"source": "clinical_trial_overview.txt", "section": "Safety"},
                "embedding": [0.0, 1.0, 0.0],
            },
            {
                "text": "The drug label contains general usage instructions.",
                "metadata": {"source": "drug_label.txt", "section": "Label"},
                "embedding": [1.0, 0.0, 0.0],
            },
        ]

        results = hybrid_search(
            query="What adverse events were reported?",
            query_embedding=[0.9, 0.1, 0.0],
            chunk_records=chunks,
            keyword_terms=["adverse events"],
            k=2,
        )

        self.assertTrue(results[0]["text"].startswith("Adverse events"))
        self.assertGreater(results[0]["score"], results[1]["score"])


if __name__ == "__main__":
    unittest.main()
