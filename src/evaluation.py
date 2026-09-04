from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from src.rag_pipeline import answer_with_citations


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_SET_PATH = (
    ROOT_DIR
    / "data"
    / "evaluation_test_set.json"
)

RESULTS_PATH = (
    ROOT_DIR
    / "outputs"
    / "evaluation_results.json"
)

SUMMARY_PATH = (
    ROOT_DIR
    / "outputs"
    / "evaluation_summary.md"
)


# ---------------------------------------------------------
# LOAD TEST SET
# ---------------------------------------------------------

def load_test_set() -> list[dict[str, Any]]:
    """
    Load evaluation questions from JSON.
    """

    with open(
        TEST_SET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for simple keyword matching.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ---------------------------------------------------------
# CORRECTNESS
# ---------------------------------------------------------

def score_correctness(
    answer: str,
    expected_points: list[str],
) -> float:
    """
    Score how many expected answer points appear
    in the generated answer.

    Score range: 0.0 to 1.0
    """

    normalized_answer = normalize_text(
        answer
    )

    if not expected_points:
        return 0.0

    matched = 0

    for point in expected_points:

        normalized_point = normalize_text(
            point
        )

        if normalized_point in normalized_answer:
            matched += 1

    return matched / len(expected_points)


# ---------------------------------------------------------
# EXTRACT CITATIONS
# ---------------------------------------------------------

def extract_citation_markers(
    answer: str,
) -> list[str]:
    """
    Extract citations such as [1], [2], [3].
    """

    markers = re.findall(
        r"\[\d+\]",
        answer,
    )

    return list(
        dict.fromkeys(markers)
    )


# ---------------------------------------------------------
# GROUNDING
# ---------------------------------------------------------

def score_grounding(
    answer: str,
    citations: dict[str, dict[str, Any]],
) -> float:
    """
    Check whether the citations used in the answer
    point to retrieved chunks containing source text.

    Score:
        1.0 = all answer citations are grounded
        0.0 = no valid grounding
    """

    markers = extract_citation_markers(
        answer
    )

    # If the answer makes factual claims but
    # contains no citations, grounding fails.
    if not markers:
        return 0.0

    valid = 0

    for marker in markers:

        citation = citations.get(
            marker
        )

        if not citation:
            continue

        source = citation.get(
            "source"
        )

        text = citation.get(
            "text"
        )

        if source and text:
            valid += 1

    return valid / len(markers)


# ---------------------------------------------------------
# CITATION ACCURACY
# ---------------------------------------------------------

def score_citation_accuracy(
    answer: str,
    citations: dict[str, dict[str, Any]],
    expected_sources: list[str],
) -> float:
    """
    Check whether answer citations map to the
    expected source documents.
    """

    markers = extract_citation_markers(
        answer
    )

    if not markers:
        return 0.0

    if not expected_sources:
        return 0.0

    correct = 0

    for marker in markers:

        citation = citations.get(
            marker
        )

        if not citation:
            continue

        source = citation.get(
            "source"
        )

        if source in expected_sources:
            correct += 1

    return correct / len(markers)


# ---------------------------------------------------------
# SOURCE EXTRACTION
# ---------------------------------------------------------

def get_cited_sources(
    answer: str,
    citations: dict[str, dict[str, Any]],
) -> list[str]:
    """
    Return unique source documents cited by the answer.
    """

    markers = extract_citation_markers(
        answer
    )

    sources = []

    for marker in markers:

        citation = citations.get(
            marker
        )

        if not citation:
            continue

        source = citation.get(
            "source"
        )

        if source and source not in sources:
            sources.append(source)

    return sources


# ---------------------------------------------------------
# FAILURE ANALYSIS
# ---------------------------------------------------------

def determine_failure_causes(
    correctness: float,
    grounding: float,
    citation_accuracy: float,
) -> list[str]:
    """
    Identify likely causes based on the weakest
    evaluation dimensions.
    """

    causes = []

    if correctness < 1.0:
        causes.append(
            "Answer missed one or more expected answer points."
        )

    if grounding < 1.0:
        causes.append(
            "Answer contains claims without fully valid retrieved-source support."
        )

    if citation_accuracy < 1.0:
        causes.append(
            "One or more citations do not point to the expected supporting source."
        )

    return causes


# ---------------------------------------------------------
# SCORE ONE EXAMPLE
# ---------------------------------------------------------

def score_example(
    example: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the RAG system and score one test example.
    """

    result = answer_with_citations(
        example["question"],
        k=4,
    )

    answer = result.get(
        "answer",
        ""
    )

    citations = result.get(
        "citations",
        {}
    )

    correctness = score_correctness(
        answer=answer,
        expected_points=example[
            "expected_points"
        ],
    )

    grounding = score_grounding(
        answer=answer,
        citations=citations,
    )

    citation_accuracy = score_citation_accuracy(
        answer=answer,
        citations=citations,
        expected_sources=example[
            "expected_sources"
        ],
    )

    cited_sources = get_cited_sources(
        answer,
        citations,
    )

    failure_causes = determine_failure_causes(
        correctness,
        grounding,
        citation_accuracy,
    )

    return {
        "id": example["id"],
        "question": example["question"],
        "expected_points": example[
            "expected_points"
        ],
        "expected_sources": example[
            "expected_sources"
        ],
        "answer": answer,
        "correctness": round(
            correctness,
            2,
        ),
        "grounding": round(
            grounding,
            2,
        ),
        "citation_accuracy": round(
            citation_accuracy,
            2,
        ),
        "cited_sources": cited_sources,
        "citations": citations,
        "failures": failure_causes,
    }


# ---------------------------------------------------------
# EVALUATE COMPLETE TEST SET
# ---------------------------------------------------------

def evaluate_test_set(
    test_set: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Run all evaluation examples.
    """

    rows = []

    for index, example in enumerate(
        test_set,
        start=1,
    ):

        print(
            f"Evaluating {index}/{len(test_set)}: "
            f"{example['question']}"
        )

        result = score_example(
            example
        )

        rows.append(result)

    return rows


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

def build_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate overall evaluation metrics.
    """

    if not rows:
        return {
            "questions": 0,
            "avg_correctness": 0.0,
            "avg_grounding": 0.0,
            "avg_citation_accuracy": 0.0,
            "overall_score": 0.0,
            "failures": [],
        }

    avg_correctness = sum(
        row["correctness"]
        for row in rows
    ) / len(rows)

    avg_grounding = sum(
        row["grounding"]
        for row in rows
    ) / len(rows)

    avg_citation_accuracy = sum(
        row["citation_accuracy"]
        for row in rows
    ) / len(rows)

    overall_score = (
        avg_correctness
        + avg_grounding
        + avg_citation_accuracy
    ) / 3

    failures = [
        {
            "id": row["id"],
            "question": row["question"],
            "correctness": row[
                "correctness"
            ],
            "grounding": row[
                "grounding"
            ],
            "citation_accuracy": row[
                "citation_accuracy"
            ],
            "causes": row[
                "failures"
            ],
        }
        for row in rows
        if min(
            row["correctness"],
            row["grounding"],
            row["citation_accuracy"],
        ) < 1.0
    ]

    return {
        "questions": len(rows),
        "avg_correctness": round(
            avg_correctness,
            2,
        ),
        "avg_grounding": round(
            avg_grounding,
            2,
        ),
        "avg_citation_accuracy": round(
            avg_citation_accuracy,
            2,
        ),
        "overall_score": round(
            overall_score,
            2,
        ),
        "failures": failures,
    }


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_results(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """
    Save detailed evaluation results as JSON.
    """

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "summary": summary,
        "results": rows,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------
# SAVE HUMAN-READABLE SUMMARY
# ---------------------------------------------------------

def save_summary(
    summary: dict[str, Any],
) -> None:
    """
    Save a readable Markdown evaluation report.
    """

    lines = []

    lines.append(
        "# RAG Evaluation Summary"
    )

    lines.append("")

    lines.append(
        "## Overall Scores"
    )

    lines.append("")

    lines.append(
        f"- Questions evaluated: {summary['questions']}"
    )

    lines.append(
        f"- Average correctness: {summary['avg_correctness']:.2f}"
    )

    lines.append(
        f"- Average grounding: {summary['avg_grounding']:.2f}"
    )

    lines.append(
        f"- Average citation accuracy: "
        f"{summary['avg_citation_accuracy']:.2f}"
    )

    lines.append(
        f"- Overall score: {summary['overall_score']:.2f}"
    )

    lines.append("")

    lines.append(
        "## Failures"
    )

    lines.append("")

    if not summary["failures"]:

        lines.append(
            "No evaluation failures were detected."
        )

    else:

        for failure in summary[
            "failures"
        ]:

            lines.append(
                f"### {failure['id']}"
            )

            lines.append(
                f"**Question:** {failure['question']}"
            )

            lines.append(
                f"- Correctness: {failure['correctness']:.2f}"
            )

            lines.append(
                f"- Grounding: {failure['grounding']:.2f}"
            )

            lines.append(
                f"- Citation accuracy: "
                f"{failure['citation_accuracy']:.2f}"
            )

            lines.append(
                "**Likely causes:**"
            )

            for cause in failure[
                "causes"
            ]:

                lines.append(
                    f"- {cause}"
                )

            lines.append("")

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines)
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:

    print()
    print("=" * 70)
    print("PHARMALENS RAG EVALUATION")
    print("=" * 70)
    print()

    test_set = load_test_set()

    print(
        f"Loaded {len(test_set)} evaluation questions."
    )

    print()

    rows = evaluate_test_set(
        test_set
    )

    summary = build_summary(
        rows
    )

    save_results(
        rows,
        summary,
    )

    save_summary(
        summary
    )

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Questions: "
        f"{summary['questions']}"
    )

    print(
        f"Correctness: "
        f"{summary['avg_correctness']:.2f}"
    )

    print(
        f"Grounding: "
        f"{summary['avg_grounding']:.2f}"
    )

    print(
        f"Citation Accuracy: "
        f"{summary['avg_citation_accuracy']:.2f}"
    )

    print(
        f"Overall Score: "
        f"{summary['overall_score']:.2f}"
    )

    print()
    print(
        f"Results saved to: "
        f"{RESULTS_PATH}"
    )

    print(
        f"Summary saved to: "
        f"{SUMMARY_PATH}"
    )

    print()


if __name__ == "__main__":
    main()