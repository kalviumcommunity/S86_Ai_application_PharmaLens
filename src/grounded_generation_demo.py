"""
Grounded Answer Generation Demo

This module demonstrates the final RAG step: generating answers grounded in
retrieved context versus ungrounded answers based on the model's memory.

A grounded answer should:
- Use only the retrieved context
- Be traceable to source documents
- Admit when supporting context is missing

An ungrounded answer may:
- Sound fluent and confident
- Use the model's internal knowledge
- Potentially contain hallucinations
- Lack source attribution
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.rag_pipeline import (
    answer_query,
    generate_ungrounded_answer,
)


# ============================================================
# PATHS & LOGGING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "grounded_generation_demo.log"


def setup_logging() -> None:
    """Configure logging to terminal and output file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                OUTPUT_FILE,
                mode="w",
                encoding="utf-8",
            ),
        ],
    )


# ============================================================
# DEMO QUERIES
# ============================================================

DEMO_QUERIES = [
    "What adverse events were reported in clinical studies?",
    "Who is eligible to participate in the trial?",
    "What is the weather forecast for next week?",  # Likely missing from corpus
]


# ============================================================
# GROUNDING CHECK
# ============================================================

def print_grounding_check(result: dict) -> str:
    """
    Display grounding verification for a RAG result.
    Shows that the answer is traceable to source documents.
    """
    lines = []

    lines.append("Answer:")
    lines.append(result["answer"])
    lines.append("")

    lines.append("Sources:")
    if result["sources"]:
        for source in result["sources"]:
            lines.append(f"  - {source}")
    else:
        lines.append("  (No supporting context found)")

    lines.append("")

    return "\n".join(lines)


# ============================================================
# COMPARISON REPORT
# ============================================================

def build_comparison_report(
    query: str,
    ungrounded_answer: str,
    grounded_result: dict,
) -> str:
    """Build a report comparing grounded and ungrounded answers."""
    lines = []

    lines.append("=" * 70)
    lines.append("GROUNDED ANSWER GENERATION COMPARISON")
    lines.append("=" * 70)

    lines.append("")

    # Query
    lines.append("QUESTION")
    lines.append("-" * 70)
    lines.append(query)
    lines.append("")

    # Ungrounded Answer
    lines.append("UNGROUNDED ANSWER (without retrieval)")
    lines.append("-" * 70)
    lines.append("Generated from model's internal knowledge only.")
    lines.append("No source attribution. May contain hallucinations.")
    lines.append("")
    lines.append(ungrounded_answer)
    lines.append("")

    # Grounded Answer
    lines.append("GROUNDED ANSWER (with retrieval)")
    lines.append("-" * 70)
    lines.append("Generated from retrieved context only.")
    lines.append("Traceable to source documents.")
    lines.append("")
    lines.append(print_grounding_check(grounded_result))

    # Comparison Summary
    lines.append("")
    lines.append("KEY DIFFERENCES")
    lines.append("-" * 70)

    if grounded_result["sources"]:
        lines.append(
            f"✓ Grounded answer is based on {len(grounded_result['retrieved_chunks'])} retrieved chunks"
        )
        lines.append(f"✓ Sources: {', '.join(grounded_result['sources'])}")
        lines.append(
            "✓ Grounded answer is verifiable against source documents"
        )
    else:
        lines.append(
            "✗ No supporting context found in corpus"
        )
        lines.append("✗ Grounded answer admits information gap")
        lines.append(
            "✓ This is better than hallucinating an ungrounded response"
        )

    lines.append("")
    lines.append("GROUNDING REDUCES HALLUCINATION")
    lines.append("-" * 70)
    lines.append(
        "By explicitly grounding answers in retrieved context, "
        "we ensure that:"
    )
    lines.append("  1. Answers reflect document content, not model memory")
    lines.append("  2. Source citations are accurate and verifiable")
    lines.append("  3. Gaps in knowledge are acknowledged instead of filled")
    lines.append("  4. Hallucinations are less likely to occur")

    lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the grounded generation demo."""
    setup_logging()

    print(
        "\n========== PHARMALENS GROUNDED GENERATION DEMO ==========\n"
    )

    logging.info(
        "Starting grounded answer generation demonstration..."
    )

    report_sections = []

    for query in DEMO_QUERIES:
        logging.info("Processing query: %s", query)

        try:
            # Generate ungrounded answer
            ungrounded = generate_ungrounded_answer(query)

            # Generate grounded answer via RAG pipeline
            grounded_result = answer_query(
                query=query,
                k=3,
            )

            # Build comparison report
            comparison = build_comparison_report(
                query=query,
                ungrounded_answer=ungrounded,
                grounded_result=grounded_result,
            )

            report_sections.append(comparison)

        except Exception as e:
            logging.error(
                "Error processing query '%s': %s",
                query,
                str(e),
            )

    # Combine all sections and save
    full_report = "\n".join(report_sections)

    print(full_report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(full_report, encoding="utf-8")

    print(f"\nDemo output saved to: {OUTPUT_FILE}")

    logging.info(
        "Grounded generation demonstration completed successfully."
    )


if __name__ == "__main__":
    main()
