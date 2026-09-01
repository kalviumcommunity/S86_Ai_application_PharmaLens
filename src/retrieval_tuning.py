from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, IsEmptyCondition


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_CACHE_FILE = OUTPUT_DIR / "retrieval_query_embeddings.json"
SUMMARY_FILE = OUTPUT_DIR / "retrieval_tuning.log"

COLLECTION_NAME_DEFAULT = "rag_chunks"
QDRANT_URL_DEFAULT = "http://localhost:6333"

VECTOR_DIMENSION_DEFAULT = 3072

# The corpus contains 8 real chunks.
# clinical-trial-demo.txt was only created for the vector DB readback test,
# so it should not participate in retrieval relevance evaluation.
TEST_RECORD_SOURCE = "clinical-trial-demo.txt"


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("retrieval_tuning")


def configure_logging() -> None:
    """
    Configure console + file logging.
    """

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        SUMMARY_FILE,
        mode="w",
        encoding="utf-8",
    )

    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# ============================================================================
# ENVIRONMENT / CONFIGURATION
# ============================================================================


def load_settings() -> dict[str, str]:
    """
    Load configuration from .env.

    This intentionally does not depend on the project's existing
    load_settings() signature so the retrieval experiment remains
    compatible with older versions of config.py.
    """

    load_dotenv(BASE_DIR / ".env")

    settings = {
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "").strip(),
        "openai_api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        "embed_model": os.getenv("EMBED_MODEL", "").strip(),
        "qdrant_url": os.getenv(
            "QDRANT_URL",
            QDRANT_URL_DEFAULT,
        ).strip(),
        "qdrant_collection": os.getenv(
            "QDRANT_COLLECTION",
            COLLECTION_NAME_DEFAULT,
        ).strip(),
        "vector_dimension": os.getenv(
            "VECTOR_DIMENSION",
            str(VECTOR_DIMENSION_DEFAULT),
        ).strip(),
    }

    required = [
        "OPENAI_API_KEY",
        "EMBED_MODEL",
    ]

    missing = []

    for variable in required:
        key = variable.lower()

        if variable == "OPENAI_API_KEY":
            value = settings["openai_api_key"]
        elif variable == "EMBED_MODEL":
            value = settings["embed_model"]
        else:
            value = settings.get(key, "")

        if not value:
            missing.append(variable)

    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )

    return settings


# ============================================================================
# TEST QUERIES
# ============================================================================


TEST_QUERIES = [
    {
        "query": "What did Study 001 evaluate?",
        "expected_source": "Study_001_Clinical_Report.pdf",
        "expected_chunk": "study-001-chunk-001",
    },
    {
        "query": "What was the primary endpoint in Study 001?",
        "expected_source": "Study_001_Clinical_Report.pdf",
        "expected_chunk": "study-001-chunk-002",
    },
    {
        "query": "How did Drug Y compare with placebo?",
        "expected_source": "Study_002_Clinical_Report.pdf",
        "expected_chunk": "study-002-chunk-001",
    },
    {
        "query": "What adverse event information is associated with Drug X?",
        "expected_source": "Drug_X_Safety_Bulletin.pdf",
        "expected_chunk": "safety-bulletin-001",
    },
]


# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================


RETRIEVAL_SETTINGS = [
    {
        "name": "baseline_k3",
        "k": 3,
        "min_score": 0.0,
    },
    {
        "name": "expanded_k5",
        "k": 5,
        "min_score": 0.0,
    },
    {
        "name": "strict_k3",
        "k": 3,
        "min_score": 0.70,
    },
]


# ============================================================================
# QUERY EMBEDDING CACHE
# ============================================================================


def load_query_cache() -> dict[str, list[float]]:
    """
    Load previously generated query embeddings.

    This prevents the same query from consuming embedding API quota
    on every retrieval-tuning run.
    """

    if not QUERY_CACHE_FILE.exists():
        return {}

    try:
        with QUERY_CACHE_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            logger.warning(
                "Query embedding cache has invalid format. Starting empty."
            )
            return {}

        return data

    except (json.JSONDecodeError, OSError) as error:
        logger.warning(
            "Could not read query embedding cache: %s",
            error,
        )
        return {}


def save_query_cache(
    cache: dict[str, list[float]],
) -> None:
    """
    Save query embeddings locally.
    """

    QUERY_CACHE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = QUERY_CACHE_FILE.with_suffix(".tmp")

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cache,
            file,
        )

    temporary_file.replace(QUERY_CACHE_FILE)


def get_query_embedding(
    client: OpenAI,
    model: str,
    query: str,
    cache: dict[str, list[float]],
) -> list[float]:
    """
    Return a cached query embedding when available.

    Otherwise generate it once and save it.
    """

    if query in cache:
        logger.info(
            "Using cached embedding: %s",
            query,
        )
        return cache[query]

    logger.info(
        "Generating embedding for query: %s",
        query,
    )

    try:
        response = client.embeddings.create(
            model=model,
            input=query,
        )

    except Exception as error:
        error_text = str(error)

        if "429" in error_text or "quota" in error_text.lower():
            raise RuntimeError(
                "Embedding API quota is currently unavailable. "
                "Wait for the Gemini embedding quota to reset, then "
                "run this script again. Previously cached queries will "
                "not consume additional embedding requests."
            ) from error

        raise

    if not response.data:
        raise RuntimeError(
            "Embedding API returned no embedding data."
        )

    embedding = response.data[0].embedding

    if not embedding:
        raise RuntimeError(
            "Embedding API returned an empty vector."
        )

    cache[query] = embedding

    save_query_cache(cache)

    logger.info(
        "Cached query embedding: %s",
        query,
    )

    return embedding


# ============================================================================
# QDRANT CONNECTION
# ============================================================================


def create_qdrant_client(
    qdrant_url: str,
) -> QdrantClient:
    """
    Create a Qdrant client.
    """

    client = QdrantClient(
        url=qdrant_url,
    )

    # Simple connectivity check.
    client.get_collections()

    return client


def verify_collection(
    client: QdrantClient,
    collection_name: str,
    expected_dimension: int,
) -> None:
    """
    Verify that the expected Qdrant collection exists
    and has the correct vector dimension.
    """

    collection = client.get_collection(
        collection_name=collection_name,
    )

    vectors_config = collection.config.params.vectors

    actual_dimension = vectors_config.size

    if actual_dimension != expected_dimension:
        raise ValueError(
            f"Vector dimension mismatch: "
            f"Qdrant={actual_dimension}, "
            f"expected={expected_dimension}"
        )

    logger.info(
        "Collection verified: %s",
        collection_name,
    )

    logger.info(
        "Vector dimension verified: %s",
        actual_dimension,
    )


# ============================================================================
# CORPUS FILTERING
# ============================================================================


def is_real_corpus_record(
    payload: dict[str, Any] | None,
) -> bool:
    """
    Return True for actual PharmaLens corpus records.

    The vector database contains one additional test record from the
    previous vector database assignment. That record is intentionally
    excluded from this experiment.
    """

    if not payload:
        return False

    metadata = payload.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        return False

    source = metadata.get(
        "source",
        "",
    )

    if source == TEST_RECORD_SOURCE:
        return False

    return bool(source)


def get_corpus_records(
    client: QdrantClient,
    collection_name: str,
) -> list[Any]:
    """
    Read all records from Qdrant and keep only actual corpus records.
    """

    records: list[Any] = []

    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            if is_real_corpus_record(point.payload):
                records.append(point)

        if next_offset is None:
            break

        offset = next_offset

    return records


# ============================================================================
# RETRIEVAL
# ============================================================================


def retrieve(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    k: int,
) -> list[Any]:
    """
    Retrieve the nearest chunks from Qdrant.

    We retrieve a few extra records and remove the old vector DB
    demonstration record before applying the final k.
    """

    # Retrieve all current corpus records.

    records = get_corpus_records(
        client=client,
        collection_name=collection_name,
    )

    if not records:
        return []

    # Since this assignment corpus is small, calculate similarity using
    # Qdrant's search API against the stored vectors.

    search_results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k + 3,
        with_payload=True,
        with_vectors=False,
    ).points

    filtered_results = []

    for result in search_results:
        payload = result.payload or {}

        if is_real_corpus_record(payload):
            filtered_results.append(result)

    return filtered_results[:k]


# ============================================================================
# RESULT HELPERS
# ============================================================================


def get_source(
    result: Any,
) -> str:
    """
    Get source document from a Qdrant result.
    """

    payload = result.payload or {}

    metadata = payload.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        return ""

    return str(
        metadata.get(
            "source",
            "",
        )
    )


def get_chunk_id(
    result: Any,
) -> str:
    """
    Get original chunk ID from a Qdrant result.
    """

    payload = result.payload or {}

    return str(
        payload.get(
            "original_chunk_id",
            "",
        )
    )


def get_text(
    result: Any,
) -> str:
    """
    Get stored text.
    """

    payload = result.payload or {}

    return str(
        payload.get(
            "text",
            "",
        )
    )


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_setting(
    qdrant_client: QdrantClient,
    embedding_client: OpenAI,
    embedding_model: str,
    collection_name: str,
    setting: dict[str, Any],
    query_cache: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """
    Evaluate one retrieval configuration against all test queries.
    """

    rows = []

    k = int(setting["k"])
    min_score = float(setting["min_score"])

    logger.info("")
    logger.info(
        "SETTING: %s",
        setting["name"],
    )
    logger.info(
        "k = %s",
        k,
    )
    logger.info(
        "minimum score = %.2f",
        min_score,
    )

    for test_case in TEST_QUERIES:

        query = test_case["query"]
        expected_source = test_case["expected_source"]
        expected_chunk = test_case["expected_chunk"]

        logger.info("")
        logger.info(
            "QUERY: %s",
            query,
        )

        logger.info(
            "EXPECTED SOURCE: %s",
            expected_source,
        )

        logger.info(
            "EXPECTED CHUNK: %s",
            expected_chunk,
        )

        query_vector = get_query_embedding(
            client=embedding_client,
            model=embedding_model,
            query=query,
            cache=query_cache,
        )

        results = retrieve(
            client=qdrant_client,
            collection_name=collection_name,
            query_vector=query_vector,
            k=k,
        )

        kept_results = [
            result
            for result in results
            if float(result.score) >= min_score
        ]

        returned_sources = [
            get_source(result)
            for result in kept_results
        ]

        returned_chunks = [
            get_chunk_id(result)
            for result in kept_results
        ]

        top1_chunk = (
            returned_chunks[0]
            if returned_chunks
            else None
        )

        source_hit = expected_source in returned_sources

        chunk_hit = expected_chunk in returned_chunks

        top1_hit = (
            top1_chunk == expected_chunk
        )

        logger.info(
            "Returned results: %s",
            len(kept_results),
        )

        for rank, result in enumerate(
            kept_results,
            start=1,
        ):
            logger.info(
                "  %s. score=%.4f | chunk=%s | source=%s",
                rank,
                float(result.score),
                get_chunk_id(result),
                get_source(result),
            )

            logger.info(
                "     text=%s",
                get_text(result)[:150],
            )

        logger.info(
            "Source hit: %s",
            "PASS" if source_hit else "FAIL",
        )

        logger.info(
            "Chunk hit: %s",
            "PASS" if chunk_hit else "FAIL",
        )

        logger.info(
            "Top-1 hit: %s",
            "PASS" if top1_hit else "FAIL",
        )

        rows.append(
            {
                "query": query,
                "expected_source": expected_source,
                "expected_chunk": expected_chunk,
                "returned_sources": returned_sources,
                "returned_chunks": returned_chunks,
                "top1_chunk": top1_chunk,
                "source_hit": source_hit,
                "chunk_hit": chunk_hit,
                "top1_hit": top1_hit,
            }
        )

    return rows


# ============================================================================
# SUMMARY
# ============================================================================


def summarize_setting(
    setting_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate relevance metrics.
    """

    total = len(rows)

    source_hits = sum(
        1
        for row in rows
        if row["source_hit"]
    )

    chunk_hits = sum(
        1
        for row in rows
        if row["chunk_hit"]
    )

    top1_hits = sum(
        1
        for row in rows
        if row["top1_hit"]
    )

    source_hit_rate = (
        source_hits / total
        if total
        else 0.0
    )

    chunk_hit_rate = (
        chunk_hits / total
        if total
        else 0.0
    )

    top1_hit_rate = (
        top1_hits / total
        if total
        else 0.0
    )

    return {
        "setting": setting_name,
        "queries": total,
        "source_hits": source_hits,
        "source_hit_rate": source_hit_rate,
        "chunk_hits": chunk_hits,
        "chunk_hit_rate": chunk_hit_rate,
        "top1_hits": top1_hits,
        "top1_hit_rate": top1_hit_rate,
        "details": rows,
    }


def choose_best_setting(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Choose a setting using chunk hit rate first, then top-1 hit rate.

    If multiple settings have the same relevance, prefer the setting
    with the smaller k because it retrieves less context.
    """

    ranked = sorted(
        summaries,
        key=lambda item: (
            item["chunk_hit_rate"],
            item["top1_hit_rate"],
            -int(
                next(
                    setting["k"]
                    for setting in RETRIEVAL_SETTINGS
                    if setting["name"] == item["setting"]
                )
            ),
        ),
        reverse=True,
    )

    return ranked[0]


# ============================================================================
# SAVE JSON RESULTS
# ============================================================================


def save_results_json(
    summaries: list[dict[str, Any]],
    best: dict[str, Any],
) -> Path:
    """
    Save machine-readable tuning results.
    """

    output_file = OUTPUT_DIR / "retrieval_tuning_results.json"

    data = {
        "test_queries": TEST_QUERIES,
        "settings": RETRIEVAL_SETTINGS,
        "results": summaries,
        "best_setting": best["setting"],
    }

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )

    return output_file


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    configure_logging()

    print()
    print("=" * 70)
    print("PHARMALENS RETRIEVAL RELEVANCE TUNING")
    print("=" * 70)
    print()

    logger.info("=" * 70)
    logger.info(
        "PHARMALENS - RETRIEVAL RELEVANCE TUNING"
    )
    logger.info("=" * 70)

    # ------------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------------

    settings = load_settings()

    qdrant_url = settings["qdrant_url"]
    collection_name = settings["qdrant_collection"]
    embedding_model = settings["embed_model"]

    vector_dimension = int(
        settings["vector_dimension"]
    )

    print(
        f"Qdrant URL: {qdrant_url}"
    )

    print(
        f"Collection: {collection_name}"
    )

    print(
        f"Embedding model: {embedding_model}"
    )

    print(
        f"Vector dimension: {vector_dimension}"
    )

    logger.info(
        "Qdrant URL: %s",
        qdrant_url,
    )

    logger.info(
        "Collection: %s",
        collection_name,
    )

    logger.info(
        "Embedding model: %s",
        embedding_model,
    )

    # ------------------------------------------------------------------------
    # Task 1 - Test queries
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TASK 1 - TEST QUERIES")
    print("=" * 70)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TASK 1 - TEST QUERIES")
    logger.info("=" * 70)

    for index, test_case in enumerate(
        TEST_QUERIES,
        start=1,
    ):
        print(
            f"{index}. {test_case['query']}"
        )

        print(
            f"   Expected source: "
            f"{test_case['expected_source']}"
        )

        print(
            f"   Expected chunk: "
            f"{test_case['expected_chunk']}"
        )

        logger.info(
            "%s. %s",
            index,
            test_case["query"],
        )

        logger.info(
            "   Expected source: %s",
            test_case["expected_source"],
        )

        logger.info(
            "   Expected chunk: %s",
            test_case["expected_chunk"],
        )

    # ------------------------------------------------------------------------
    # Create clients
    # ------------------------------------------------------------------------

    embedding_client = OpenAI(
        api_key=settings["openai_api_key"],
        base_url=settings["openai_base_url"] or None,
    )

    qdrant_client = create_qdrant_client(
        qdrant_url=qdrant_url,
    )

    print()
    print(
        "Qdrant connection: SUCCESS"
    )

    logger.info(
        "Qdrant connection: SUCCESS"
    )

    # ------------------------------------------------------------------------
    # Verify collection
    # ------------------------------------------------------------------------

    verify_collection(
        client=qdrant_client,
        collection_name=collection_name,
        expected_dimension=vector_dimension,
    )

    # ------------------------------------------------------------------------
    # Corpus record count
    # ------------------------------------------------------------------------

    corpus_records = get_corpus_records(
        client=qdrant_client,
        collection_name=collection_name,
    )

    print(
        f"Corpus records used for tuning: "
        f"{len(corpus_records)}"
    )

    logger.info(
        "Corpus records used for tuning: %s",
        len(corpus_records),
    )

    # ------------------------------------------------------------------------
    # Load query cache
    # ------------------------------------------------------------------------

    query_cache = load_query_cache()

    print()
    print(
        f"Cached query embeddings: "
        f"{len(query_cache)}"
    )

    logger.info(
        "Cached query embeddings: %s",
        len(query_cache),
    )

    # ------------------------------------------------------------------------
    # Evaluate settings
    # ------------------------------------------------------------------------

    summaries = []

    for setting in RETRIEVAL_SETTINGS:

        print()
        print("=" * 70)
        print(
            f"SETTING: {setting['name']}"
        )
        print("=" * 70)

        try:
            rows = evaluate_setting(
                qdrant_client=qdrant_client,
                embedding_client=embedding_client,
                embedding_model=embedding_model,
                collection_name=collection_name,
                setting=setting,
                query_cache=query_cache,
            )

        except RuntimeError as error:

            print()
            print(
                "RETRIEVAL EVALUATION STOPPED:"
            )

            print(
                str(error)
            )

            logger.error(
                "Retrieval evaluation stopped: %s",
                error,
            )

            print()
            print(
                "If this is a Gemini embedding quota error, "
                "wait for the quota to reset and run the command again."
            )

            print(
                f"Query cache file: {QUERY_CACHE_FILE}"
            )

            return

        summary = summarize_setting(
            setting_name=setting["name"],
            rows=rows,
        )

        summaries.append(summary)

        print()
        print(
            f"Source hit rate: "
            f"{summary['source_hit_rate']:.2%}"
        )

        print(
            f"Chunk hit rate: "
            f"{summary['chunk_hit_rate']:.2%}"
        )

        print(
            f"Top-1 hit rate: "
            f"{summary['top1_hit_rate']:.2%}"
        )

    # ------------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TASK 3 - RELEVANCE RESULTS")
    print("=" * 70)

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "TASK 3 - RELEVANCE RESULTS"
    )
    logger.info("=" * 70)

    print()

    print(
        f"{'SETTING':<20}"
        f"{'SOURCE HIT':<15}"
        f"{'CHUNK HIT':<15}"
        f"{'TOP-1 HIT':<15}"
    )

    print("-" * 65)

    for summary in summaries:

        print(
            f"{summary['setting']:<20}"
            f"{summary['source_hit_rate']:<15.2%}"
            f"{summary['chunk_hit_rate']:<15.2%}"
            f"{summary['top1_hit_rate']:<15.2%}"
        )

        logger.info(
            "%s | source hit rate=%.2f%% | "
            "chunk hit rate=%.2f%% | "
            "top-1 hit rate=%.2f%%",
            summary["setting"],
            summary["source_hit_rate"] * 100,
            summary["chunk_hit_rate"] * 100,
            summary["top1_hit_rate"] * 100,
        )

    # ------------------------------------------------------------------------
    # Choose best
    # ------------------------------------------------------------------------

    best = choose_best_setting(
        summaries
    )

    best_setting = next(
        setting
        for setting in RETRIEVAL_SETTINGS
        if setting["name"] == best["setting"]
    )

    print()
    print("=" * 70)
    print("TASK 4 - BEST SETTINGS")
    print("=" * 70)

    print()
    print(
        f"Selected setting: {best['setting']}"
    )

    print(
        f"k: {best_setting['k']}"
    )

    print(
        f"Minimum score: "
        f"{best_setting['min_score']}"
    )

    print(
        f"Chunk hit rate: "
        f"{best['chunk_hit_rate']:.2%}"
    )

    print(
        f"Top-1 hit rate: "
        f"{best['top1_hit_rate']:.2%}"
    )

    # ------------------------------------------------------------------------
    # Justification
    # ------------------------------------------------------------------------

    if best_setting["name"] == "baseline_k3":

        justification = (
            "baseline_k3 was selected because it achieved the best "
            "relevance result while retrieving only three chunks. "
            "Using a smaller k reduces unnecessary context and keeps "
            "retrieval efficient when relevance is already sufficient."
        )

    elif best_setting["name"] == "expanded_k5":

        justification = (
            "expanded_k5 was selected because retrieving five chunks "
            "provided better coverage of the expected chunks than the "
            "smaller baseline configuration."
        )

    else:

        justification = (
            "strict_k3 was selected because the score threshold removed "
            "low-confidence results while preserving the strongest "
            "relevant chunks."
        )

    print()
    print(
        "Justification:"
    )

    print(
        justification
    )

    logger.info("")
    logger.info(
        "SELECTED SETTING: %s",
        best["setting"],
    )

    logger.info(
        "JUSTIFICATION: %s",
        justification,
    )

    # ------------------------------------------------------------------------
    # Save machine-readable results
    # ------------------------------------------------------------------------

    results_file = save_results_json(
        summaries=summaries,
        best=best,
    )

    # ------------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVAL TUNING VALIDATION SUCCESS")
    print("=" * 70)

    print()
    print(
        f"Test queries: {len(TEST_QUERIES)}"
    )

    print(
        f"Settings compared: "
        f"{len(RETRIEVAL_SETTINGS)}"
    )

    print(
        f"Corpus records evaluated: "
        f"{len(corpus_records)}"
    )

    print(
        f"Best setting: {best['setting']}"
    )

    print(
        f"Results JSON: {results_file}"
    )

    print(
        f"Summary log: {SUMMARY_FILE}"
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "RETRIEVAL TUNING VALIDATION SUCCESS"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()