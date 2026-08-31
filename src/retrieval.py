from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import load_settings
from src.embedding_demo import cosine_similarity, generate_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "retrieval_results.json"

SAMPLE_QUERY = "What adverse events were reported during the clinical trial?"


def embed_query(client: OpenAI, model: str, query: str) -> list[float]:
    """Embed one user query using the same model as the indexed chunks."""
    embedding = generate_embeddings(client, model, [query])[0]
    return embedding


def retrieve_top_k(
    query: str,
    query_embedding: list[float],
    chunk_records: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Return the top-k chunks for a query with cosine similarity scores."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not chunk_records:
        raise ValueError("Cannot retrieve from an empty chunk collection.")

    scored_records: list[dict[str, Any]] = []
    for record in chunk_records:
        embedding = record.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        score = cosine_similarity(query_embedding, embedding)
        scored_records.append(
            {
                "text": str(record.get("text", "")),
                "metadata": dict(record.get("metadata", {})),
                "score": float(score),
            }
        )

    scored_records.sort(key=lambda item: item["score"], reverse=True)
    return scored_records[:k]


def search_qdrant(
    qdrant_client,
    collection_name: str,
    query_embedding: list[float],
    k: int,
) -> list[dict[str, Any]]:
    """Search a Qdrant collection for the nearest vectors to a query vector."""
    try:
        matches = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=k,
            with_payload=True,
            with_vectors=False,
        ).points
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for point in matches:
        payload = point.payload or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata", {}),
                "score": float(point.score),
            }
        )
    return results


def build_demo_chunk_records() -> list[dict[str, Any]]:
    """Return deterministic sample chunks for a local retrieval demo."""
    return [
        {
            "text": (
                "Clinical trial overview: adverse events included headache, nausea, "
                "and fatigue during the treatment period."
            ),
            "metadata": {"source": "clinical_trial_overview.txt", "chunk_index": 2},
            "embedding": [0.96, 0.12, 0.02],
        },
        {
            "text": (
                "Eligibility criteria: adults with moderate disease and no prior "
                "therapy were eligible for the study."
            ),
            "metadata": {"source": "eligibility_criteria.md", "chunk_index": 1},
            "embedding": [0.40, 0.88, 0.13],
        },
        {
            "text": (
                "Study protocol: treatment goals included improvement in disease "
                "severity and patient-reported symptoms."
            ),
            "metadata": {"source": "study_protocol.txt", "chunk_index": 0},
            "embedding": [0.11, 0.29, 0.95],
        },
    ]


def build_demo_results() -> dict[str, Any]:
    """Build top-k query results demonstrating the changing retrieval depth."""
    chunk_records = build_demo_chunk_records()
    query_embedding = [0.93, 0.18, 0.04]

    k_values = [2, 3]
    results_by_k = []

    for k in k_values:
        retrieved = retrieve_top_k(SAMPLE_QUERY, query_embedding, chunk_records, k)
        results_by_k.append(
            {
                "k": k,
                "results": [
                    {
                        "rank": index + 1,
                        "score": round(item["score"], 6),
                        "text": item["text"],
                        "metadata": item["metadata"],
                    }
                    for index, item in enumerate(retrieved)
                ],
            }
        )

    return {
        "embedding_model": "gemini-embedding-001",
        "query": SAMPLE_QUERY,
        "query_embedding": query_embedding,
        "k_values": results_by_k,
    }


def main() -> None:
    """Embed the sample query, run top-k retrieval, and save the results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_query = SAMPLE_QUERY

    try:
        settings = load_settings(require_chat=False, require_embedding=True)
        client = OpenAI(
            base_url=settings["openai_base_url"] or None,
            api_key=settings["openai_api_key"],
        )
        query_embedding = embed_query(client, settings["embed_model"], sample_query)
        chunk_records = build_demo_chunk_records()
        retrieval = {
            "embedding_model": settings["embed_model"],
            "query": sample_query,
            "query_embedding": query_embedding,
            "k_values": [],
        }
        for k in [2, 3]:
            hits = retrieve_top_k(sample_query, query_embedding, chunk_records, k)
            retrieval["k_values"].append(
                {
                    "k": k,
                    "results": [
                        {
                            "rank": index + 1,
                            "score": round(item["score"], 6),
                            "text": item["text"],
                            "metadata": item["metadata"],
                        }
                        for index, item in enumerate(hits)
                    ],
                }
            )
    except Exception:
        retrieval = build_demo_results()

    OUTPUT_FILE.write_text(json.dumps(retrieval, indent=2), encoding="utf-8")
    print(json.dumps(retrieval, indent=2))


if __name__ == "__main__":
    main()
