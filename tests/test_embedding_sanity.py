import unittest

from src.embedding_sanity import build_sanity_report, rank_chunks, validate_chunk_records


class EmbeddingSanityTests(unittest.TestCase):
    def setUp(self):
        self.vectors = {
            "clinical_trial_overview.txt": [1.0, 0.0, 0.0],
            "eligibility_criteria.md": [0.0, 1.0, 0.0],
            "noisy_clinical_report.txt": [0.0, 0.0, 1.0],
        }
        self.records = [
            {
                "text": source,
                "metadata": {"source": source},
                "embedding": vector,
            }
            for source, vector in self.vectors.items()
        ]

    def fake_embed(self, texts):
        return [
            self.vectors["clinical_trial_overview.txt"]
            if "adverse events" in text
            else self.vectors["eligibility_criteria.md"]
            if "eligible" in text
            else self.vectors["noisy_clinical_report.txt"]
            for text in texts
        ]

    def test_related_source_ranks_first(self):
        ranked = rank_chunks(
            "What adverse events were reported during the clinical trial?",
            self.records,
            self.fake_embed,
        )

        self.assertEqual(
            ranked[0]["metadata"]["source"],
            "clinical_trial_overview.txt",
        )

    def test_report_counts_known_query_results(self):
        report = build_sanity_report("test-model", self.records, self.fake_embed)

        self.assertIn("Tests: 3 | Passed: 3 | Failed: 0", report)
        self.assertIn("Potential surprising case to inspect", report)

    def test_dimension_validation_rejects_mismatched_vectors(self):
        invalid_records = [
            *self.records,
            {
                "text": "invalid",
                "metadata": {"source": "invalid.txt"},
                "embedding": [1.0, 2.0],
            },
        ]

        with self.assertRaisesRegex(ValueError, "dimensions are inconsistent"):
            validate_chunk_records(invalid_records)


if __name__ == "__main__":
    unittest.main()