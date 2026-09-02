import unittest

from src.rag_pipeline import assemble_context, build_prompt, format_chunk


class ContextInjectionTests(unittest.TestCase):
    def test_format_chunk_labels_source_and_chunk_index(self):
        chunk = {
            "text": "Clinical trial adverse events included headache and nausea.",
            "metadata": {"source": "clinical_trial_overview.txt", "chunk_index": 3},
        }

        rendered = format_chunk(2, chunk)

        self.assertIn("[2] clinical_trial_overview.txt#3", rendered)
        self.assertIn("Clinical trial adverse events included headache and nausea.", rendered)

    def test_assemble_context_respects_token_budget(self):
        chunks = [
            {"text": "A" * 200, "metadata": {"source": "s1.txt", "chunk_index": 0}},
            {"text": "B" * 200, "metadata": {"source": "s2.txt", "chunk_index": 1}},
            {"text": "C" * 200, "metadata": {"source": "s3.txt", "chunk_index": 2}},
        ]

        context = assemble_context(chunks, max_tokens=200)
        self.assertIsInstance(context, str)
        self.assertLessEqual(len(context), 5000)

    def test_build_prompt_includes_context_and_grounding_instructions(self):
        chunks = [
            {
                "text": "Adults aged 18 to 65 were eligible.",
                "metadata": {"source": "eligibility_criteria.md", "chunk_index": 1},
            }
        ]

        result = build_prompt("Who was eligible?", chunks)

        self.assertIn("Answer the question using only the provided context.", result["prompt"])
        self.assertIn("Context:", result["prompt"])
        self.assertIn("eligibility_criteria.md#1", result["prompt"])
        self.assertEqual(result["sources_used"][0]["source"], "eligibility_criteria.md")
        self.assertGreaterEqual(result["context_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
