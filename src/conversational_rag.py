"""Conversation-aware orchestration for retrieval-augmented answers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


History = list[dict[str, str]]
RetrieveFn = Callable[[str], list[dict[str, Any]]]
GenerateFn = Callable[[str, list[dict[str, Any]], History], str]
RewriteFn = Callable[[str, History], str]


def rewrite_follow_up_question(
    question: str,
    history: History,
) -> str:
    """Turn a follow-up into a standalone query using the last user turn."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    previous_questions = [
        turn["content"]
        for turn in history
        if turn.get("role") == "user" and turn.get("content", "").strip()
    ]
    if not previous_questions:
        return question.strip()

    return (
        f"Regarding the clinical research question '{previous_questions[-1]}', "
        f"answer this follow-up: {question.strip()}"
    )


class ConversationalRAG:
    """Keep dialogue state and retrieve each turn with a standalone query."""

    def __init__(
        self,
        retrieve_fn: RetrieveFn,
        generate_fn: GenerateFn,
        rewrite_fn: RewriteFn = rewrite_follow_up_question,
    ) -> None:
        self.retrieve_fn = retrieve_fn
        self.generate_fn = generate_fn
        self.rewrite_fn = rewrite_fn
        self.history: History = []

    def ask(self, question: str) -> dict[str, Any]:
        """Rewrite, retrieve, answer, and append both sides of the turn."""
        standalone_query = self.rewrite_fn(question, self.history)
        retrieved_chunks = self.retrieve_fn(standalone_query)
        answer = self.generate_fn(question, retrieved_chunks, self.history)

        self.history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )

        return {
            "question": question,
            "standalone_query": standalone_query,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "history": list(self.history),
        }