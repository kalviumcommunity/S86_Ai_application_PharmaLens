import unittest

from src.conversational_rag import (
    ConversationalRAG,
    rewrite_follow_up_question,
)


class ConversationalRAGTests(unittest.TestCase):
    def test_first_question_stays_standalone(self):
        query = rewrite_follow_up_question("What adverse events were reported?", [])

        self.assertEqual(query, "What adverse events were reported?")

    def test_follow_up_uses_previous_question(self):
        query = rewrite_follow_up_question(
            "Which one was most common?",
            [{"role": "user", "content": "What adverse events were reported?"}],
        )

        self.assertIn("What adverse events were reported?", query)
        self.assertIn("Which one was most common?", query)

    def test_retrieval_receives_rewritten_query_and_history_is_saved(self):
        retrieved_queries = []

        def retrieve(query):
            retrieved_queries.append(query)
            return [{"text": "Headache was most common.", "score": 0.92}]

        def generate(question, chunks, history):
            return f"{chunks[0]['text']} [1]"

        rag = ConversationalRAG(retrieve, generate)
        first = rag.ask("What adverse events were reported?")
        second = rag.ask("Which one was most common?")

        self.assertEqual(retrieved_queries[0], first["standalone_query"])
        self.assertEqual(retrieved_queries[1], second["standalone_query"])
        self.assertIn("What adverse events were reported?", retrieved_queries[1])
        self.assertEqual(len(second["history"]), 4)
        self.assertEqual(second["answer"], "Headache was most common. [1]")


if __name__ == "__main__":
    unittest.main()