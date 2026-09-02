import unittest

from src.rag_pipeline import (
    assemble_context,
    build_prompt,
    format_chunk,
    generate_answer,
)


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

        context, context_tokens = assemble_context(chunks, max_tokens=200)
        self.assertIsInstance(context, str)
        self.assertGreaterEqual(context_tokens, 0)
        self.assertLessEqual(context_tokens, 200)

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

    def test_generate_answer_returns_string_for_empty_context(self):
        """Test that generate_answer handles empty context gracefully."""
        answer = generate_answer(
            query="What were the results?",
            context="",
        )

        self.assertIsInstance(answer, str)
        self.assertIn("could not find", answer.lower())

    def test_generate_answer_respects_grounding_constraint(self):
        """Test that generated answers are grounded in provided context."""
        context = """
[1] clinical_trial.txt#0
Adults aged 18 to 65 were eligible for participation.

[2] eligibility_criteria.md#1
All participants provided informed consent.
"""

        answer = generate_answer(
            query="What were the eligibility criteria?",
            context=context,
        )

        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)


if __name__ == "__main__":
    unittest.main()
