from __future__ import annotations

from pathlib import Path
from typing import Callable

from openai import OpenAI

from src.config import load_settings
from src.corpus_ingestion import DATA_DIR, ingest_corpus
from src.embedding_demo import cosine_similarity, generate_embeddings, validate_dimensions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "embedding_sanity_report.md"

EmbeddingFunction = Callable[[list[str]], list[list[float]]]

TEST_CASES = [
    {
        "query": "What adverse events were reported during the clinical trial?",
        "expected_source": "clinical_trial_overview.txt",
    },
    {
        "query": "Which participants are eligible for this clinical study?",
        "expected_source": "eligibility_criteria.md",
    },
    {
        "query": "What was the primary objective of the trial?",
        "expected_source": "noisy_clinical_report.txt",
    },
]


def embed_with_client(
    client: OpenAI,
    model: str,
    texts: list[str],
) -> list[list[float]]:
    """Embed texts with the same provider and model for queries and chunks."""
    return generate_embeddings(client, model, texts)


def rank_chunks(
    query: str,
    chunk_records: list[dict],
    embed: EmbeddingFunction,
) -> list[dict]:
    """Return chunks ranked by cosine similarity to the query."""
    if not chunk_records:
        raise ValueError("Cannot rank an empty chunk collection.")

    query_embedding = embed([query])[0]
    ranked = []
    for chunk in chunk_records:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        ranked.append({**chunk, "score": score})

    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def validate_chunk_records(chunk_records: list[dict]) -> int:
    """Check vector dimensions, text/vector alignment, and source metadata."""
    if not chunk_records:
        raise ValueError("No chunk records were provided.")

    vectors = [record["embedding"] for record in chunk_records]
    dimension = validate_dimensions(vectors)

    for record in chunk_records:
        if not record.get("text", "").strip():
            raise ValueError("Every chunk record must contain non-empty text.")
        if not record.get("metadata", {}).get("source"):
            raise ValueError("Every chunk record must contain source metadata.")

    return dimension


def build_sanity_report(
    model: str,
    chunk_records: list[dict],
    embed: EmbeddingFunction,
) -> str:
    """Run known-query checks and render a concise retrieval sanity report."""
    dimension = validate_chunk_records(chunk_records)
    rows = []

    for case in TEST_CASES:
        ranked = rank_chunks(case["query"], chunk_records, embed)
        top = ranked[0]
        expected_rank = next(
            (
                index
                for index, item in enumerate(ranked, start=1)
                if item["metadata"]["source"] == case["expected_source"]
            ),
            None,
        )
        rows.append(
            {
                "query": case["query"],
                "expected_source": case["expected_source"],
                "top_source": top["metadata"]["source"],
                "top_score": round(top["score"], 4),
                "expected_rank": expected_rank,
                "passed": top["metadata"]["source"] == case["expected_source"],
                "top_preview": top["text"][:120].replace("\n", " "),
                "top_results": ranked[:3],
            }
        )

    passed = sum(row["passed"] for row in rows)
    failed = len(rows) - passed
    lines = [
        "# Embedding Quality Sanity Report",
        "",
        f"Embedding model: `{model}`",
        f"Chunk records: {len(chunk_records)}",
        f"Vector dimension: {dimension}",
        "",
        "## Known Query Checks",
        "",
        f"Tests: {len(rows)} | Passed: {passed} | Failed: {failed}",
        "",
        "| Query | Expected source | Top source | Expected rank | Top score | Result |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]

    for row in rows:
        result = "PASS" if row["passed"] else "CHECK"
        lines.append(
            f"| {row['query']} | {row['expected_source']} | "
            f"{row['top_source']} | {row['expected_rank'] or 'missing'} | "
            f"{row['top_score']:.4f} | {result} |"
        )

    lines.extend(
        [
            "",
            "## Inspection Notes",
            "",
            "The expected source should appear at rank 1 for this smoke test. "
            "A lower rank is a useful failure signal, not a model-quality benchmark.",
            "",
            "The pipeline uses one model for both query and chunk embeddings, "
            "checks equal vector dimensions, and keeps each vector paired with "
            "its original text and source metadata.",
            "",
            "Potential surprising case to inspect: a generic clinical chunk may "
            "rank above a more specific chunk because it shares broad terms such "
            "as `clinical`, `study`, or `safety`. Review the top three results "
            "below whenever a check is `CHECK`.",
            "",
            "## Top Results",
            "",
        ]
    )

    for row in rows:
        lines.extend(
            [
                f"### {row['query']}",
                "",
                f"Expected: `{row['expected_source']}`; top result: "
                f"`{row['top_source']}` ({row['top_score']:.4f})",
            ]
        )
        for rank, result in enumerate(row["top_results"], start=1):
            lines.append(
                f"{rank}. `{result['metadata']['source']}` "
                f"({result['score']:.4f}): {result['text'][:120].replace(chr(10), ' ')}..."
            )
        lines.append("")

    lines.extend(
        [
            "## Risks Checked",
            "",
            "- Model consistency: query and chunk calls use the same configured model.",
            "- Dimension consistency: all vectors must have one shared dimension.",
            "- Chunk alignment: vectors are attached in the original embedding response order.",
            "- Metadata integrity: every result must retain a non-empty source.",
            "- Metric: ranking uses cosine similarity, with larger scores first.",
            "",
        ]
    )
    return "\n".join(lines)


def create_embedding_records(
    client: OpenAI,
    model: str,
    chunks: list[dict],
) -> list[dict]:
    """Embed corpus chunks and retain text, vector, and source metadata together."""
    embeddings = embed_with_client(client, model, [chunk["text"] for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise ValueError("The embedding response count does not match the chunk count.")
    return [
        {**chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings)
    ]


def main() -> None:
    settings = load_settings(require_chat=False, require_embedding=True)
    client = OpenAI(
        base_url=settings["openai_base_url"] or None,
        api_key=settings["openai_api_key"],
    )
    ingestion = ingest_corpus(DATA_DIR / "sample_corpus")
    chunks = ingestion["chunks"]
    records = create_embedding_records(client, settings["embed_model"], chunks)
    report = build_sanity_report(
        settings["embed_model"],
        records,
        lambda texts: embed_with_client(client, settings["embed_model"], texts),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Saved sanity report to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()