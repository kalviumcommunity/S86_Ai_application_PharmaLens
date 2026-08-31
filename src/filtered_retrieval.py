from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.embedding_demo import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "filtered_retrieval_results.json"


def filter_chunk_records(
    chunk_records: list[dict[str, Any]],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Restrict retrieval to chunks matching a metadata filter."""
    if not chunk_records:
        return []

    return [
        record
        for record in chunk_records
        if all(
            record.get("metadata", {}).get(key) == value
            for key, value in filters.items()
        )
    ]


def hybrid_search(
    query: str,
    query_embedding: list[float],
    chunk_records: list[dict[str, Any]],
    keyword_terms: list[str] | None = None,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Combine vector similarity with optional keyword boosts for a hybrid score."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    keyword_terms = keyword_terms or []
    normalized_query = query.lower()
    scored: list[dict[str, Any]] = []

    for record in chunk_records:
        text = str(record.get("text", ""))
        metadata = dict(record.get("metadata", {}))
        embedding = record.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue

        vector_score = cosine_similarity(query_embedding, embedding)
        keyword_bonus = 0.0
        text_lower = text.lower()
        for term in keyword_terms:
            term_lower = term.lower()
            if term_lower in text_lower:
                keyword_bonus += 1.0

        final_score = vector_score + keyword_bonus
        scored.append(
            {
                "text": text,
                "metadata": metadata,
                "score": float(final_score),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:k]


def build_demo_chunk_records() -> list[dict[str, Any]]:
    """Return corpus chunks that demonstrate why metadata filters improve precision."""
    return [
        {
            "text": "Adverse events in the clinical study included headache, nausea, and fatigue.",
            "metadata": {"source": "clinical_trial_overview.txt", "section": "Safety", "document_type": "clinical_trial"},
            "embedding": [0.98, 0.12, 0.03],
        },
        {
            "text": "Adverse events were monitored throughout the study but no new serious safety events were noted.",
            "metadata": {"source": "safety_summary.txt", "section": "Safety", "document_type": "safety_report"},
            "embedding": [0.88, 0.27, 0.05],
        },
        {
            "text": "The hospital operations checklist covers staffing, room turnover, and scheduling.",
            "metadata": {"source": "operations_notes.txt", "section": "Operations", "document_type": "ops_log"},
            "embedding": [0.41, 0.77, 0.19],
        },
        {
            "text": "The patient label recommends taking the medication after meals and monitoring blood pressure.",
            "metadata": {"source": "drug_label.txt", "section": "Label", "document_type": "label"},
            "embedding": [0.62, 0.28, 0.71],
        },
    ]


def build_demo_results() -> dict[str, Any]:
    """Build a filtered-search comparison showing the value of metadata scoping."""
    query = "What adverse events were reported during the clinical trial?"
    query_embedding = [0.94, 0.18, 0.04]
    chunk_records = build_demo_chunk_records()

    unfiltered = hybrid_search(
        query=query,
        query_embedding=query_embedding,
        chunk_records=chunk_records,
        keyword_terms=["adverse events", "clinical"],
        k=3,
    )

    filtered = hybrid_search(
        query=query,
        query_embedding=query_embedding,
        chunk_records=filter_chunk_records(chunk_records, {"document_type": "clinical_trial"}),
        keyword_terms=["adverse events", "clinical"],
        k=3,
    )

    return {
        "query": query,
        "filter": {"document_type": "clinical_trial"},
        "unfiltered": [
            {
                "rank": index + 1,
                "score": round(item["score"], 6),
                "text": item["text"],
                "metadata": item["metadata"],
            }
            for index, item in enumerate(unfiltered)
        ],
        "filtered": [
            {
                "rank": index + 1,
                "score": round(item["score"], 6),
                "text": item["text"],
                "metadata": item["metadata"],
            }
            for index, item in enumerate(filtered)
        ],
        "precision_gain": "Filtered retrieval removes unrelated operations and label material, retaining the relevant clinical safety content.",
    }


def main() -> None:
    """Run the filtered retrieval demo and save the output JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_demo_results()
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
