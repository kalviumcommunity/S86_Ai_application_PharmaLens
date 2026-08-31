from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.batch_embedding import create_sample_chunks
from src.config import load_settings


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"

EMBEDDING_STORE = OUTPUT_DIR / "batch_embeddings.json"

SUMMARY_FILE = OUTPUT_DIR / "indexing_summary.log"


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 100


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> logging.Logger:
    """
    Configure logging to both terminal and output file.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        "pharmalens_indexing"
    )

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        formatter
    )

    file_handler = logging.FileHandler(
        SUMMARY_FILE,
        mode="w",
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    return logger


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings() -> dict[str, dict[str, Any]]:
    """
    Load embeddings from batch_embeddings.json.

    Actual structure:

    {
        "study-001-chunk-001": {
            "chunk_id": "...",
            "embedding": [...],
            "metadata": {...}
        }
    }
    """

    if not EMBEDDING_STORE.exists():
        raise FileNotFoundError(
            f"Embedding store not found: {EMBEDDING_STORE}"
        )

    with EMBEDDING_STORE.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected embedding store to contain a dictionary."
        )

    return data


# ============================================================
# LOAD SOURCE TEXT
# ============================================================

def load_source_chunks() -> dict[str, dict[str, Any]]:
    """
    Load the source chunks from the existing
    batch embedding sample corpus.

    batch_embedding.py contains the original text,
    while batch_embeddings.json contains the vectors.
    """

    chunks = create_sample_chunks()

    source_chunks = {}

    for chunk in chunks:

        source_chunks[chunk["id"]] = {
            "text": chunk["text"],
            "metadata": chunk["metadata"],
        }

    return source_chunks


# ============================================================
# VALIDATION
# ============================================================

def validate_embedding_record(
    chunk_id: str,
    record: dict[str, Any],
) -> None:
    """
    Validate one embedding record.
    """

    required_fields = [
        "chunk_id",
        "embedding",
        "metadata",
    ]

    missing = [
        field
        for field in required_fields
        if field not in record
    ]

    if missing:
        raise ValueError(
            f"{chunk_id}: missing fields {missing}"
        )

    if not isinstance(
        record["embedding"],
        list,
    ):
        raise ValueError(
            f"{chunk_id}: embedding must be a list."
        )

    if len(record["embedding"]) == 0:
        raise ValueError(
            f"{chunk_id}: embedding is empty."
        )

    if not isinstance(
        record["metadata"],
        dict,
    ):
        raise ValueError(
            f"{chunk_id}: metadata must be a dictionary."
        )


# ============================================================
# DETERMINISTIC QDRANT ID
# ============================================================

def make_qdrant_id(
    chunk_id: str,
) -> str:
    """
    Convert the application chunk ID into a deterministic UUID.

    Qdrant accepts unsigned integers or UUIDs as point IDs.

    UUID5 guarantees that the same chunk ID always produces
    the same Qdrant point ID.
    """

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            chunk_id,
        )
    )


# ============================================================
# BUILD QDRANT RECORD
# ============================================================

def build_point(
    chunk_id: str,
    embedding_record: dict[str, Any],
    source_record: dict[str, Any],
) -> PointStruct:
    """
    Combine embedding + source text + metadata
    into a Qdrant PointStruct.
    """

    metadata = source_record["metadata"]

    payload = {
        "original_chunk_id": chunk_id,

        "text": source_record["text"],

        "metadata": {
            "source": metadata.get(
                "source"
            ),

            "study_id": metadata.get(
                "study_id"
            ),

            "chunk_index": metadata.get(
                "chunk_index"
            ),

            "section": metadata.get(
                "section"
            ),

            "page": metadata.get(
                "page"
            ),
        },
    }

    return PointStruct(
        id=make_qdrant_id(
            chunk_id
        ),

        vector=embedding_record[
            "embedding"
        ],

        payload=payload,
    )


# ============================================================
# BATCHING
# ============================================================

def batches(
    items: list[PointStruct],
    size: int,
):
    """
    Yield Qdrant points in batches.
    """

    if size <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )

    for start in range(
        0,
        len(items),
        size,
    ):
        yield items[
            start:start + size
        ]


# ============================================================
# CONNECT TO QDRANT
# ============================================================

def connect_qdrant(
    qdrant_url: str,
) -> QdrantClient:

    client = QdrantClient(
        url=qdrant_url
    )

    # Connection test
    client.get_collections()

    return client


# ============================================================
# VERIFY COLLECTION
# ============================================================

def verify_collection(
    client: QdrantClient,
    collection_name: str,
    expected_dimension: int,
) -> None:

    collections = client.get_collections()

    names = {
        collection.name
        for collection in collections.collections
    }

    if collection_name not in names:
        raise ValueError(
            f"Collection '{collection_name}' does not exist."
        )

    info = client.get_collection(
        collection_name=collection_name
    )

    actual_dimension = (
        info.config.params.vectors.size
    )

    if actual_dimension != expected_dimension:

        raise ValueError(
            "Vector dimension mismatch: "
            f"Qdrant={actual_dimension}, "
            f"expected={expected_dimension}"
        )


# ============================================================
# INDEX RECORDS
# ============================================================

def index_records(
    client: QdrantClient,
    collection_name: str,
    points: list[PointStruct],
    logger: logging.Logger,
) -> tuple[int, list[dict[str, str]]]:

    inserted = 0

    failures = []

    total_batches = (
        (len(points) + BATCH_SIZE - 1)
        // BATCH_SIZE
    )

    for batch_number, batch in enumerate(
        batches(points, BATCH_SIZE),
        start=1,
    ):

        logger.info(
            "Indexing batch %d/%d",
            batch_number,
            total_batches,
        )

        logger.info(
            "Records in batch: %d",
            len(batch),
        )

        try:

            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )

            inserted += len(batch)

            logger.info(
                "Batch %d: SUCCESS",
                batch_number,
            )

        except Exception as error:

            logger.exception(
                "Batch %d failed",
                batch_number,
            )

            failures.append(
                {
                    "batch": str(
                        batch_number
                    ),
                    "error": str(
                        error
                    ),
                }
            )

    return inserted, failures


# ============================================================
# VERIFY ALL EXPECTED RECORDS
# ============================================================

def verify_indexed_records(
    client: QdrantClient,
    collection_name: str,
    chunk_ids: list[str],
) -> int:

    point_ids = [
        make_qdrant_id(chunk_id)
        for chunk_id in chunk_ids
    ]

    stored = client.retrieve(
        collection_name=collection_name,
        ids=point_ids,
        with_vectors=True,
        with_payload=True,
    )

    stored_ids = {
        str(point.id)
        for point in stored
    }

    missing = [
        point_id
        for point_id in point_ids
        if point_id not in stored_ids
    ]

    if missing:

        raise AssertionError(
            "Missing indexed records: "
            + ", ".join(missing)
        )

    return len(stored)


# ============================================================
# SPOT CHECK
# ============================================================

def spot_check(
    client: QdrantClient,
    collection_name: str,
    chunk_id: str,
    embedding_record: dict[str, Any],
    source_record: dict[str, Any],
) -> None:

    point_id = make_qdrant_id(
        chunk_id
    )

    stored = client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_vectors=True,
        with_payload=True,
    )

    if not stored:
        raise AssertionError(
            "Spot-check failed: record not found."
        )

    point = stored[0]

    payload = point.payload or {}

    stored_text = payload.get(
        "text"
    )

    stored_metadata = payload.get(
        "metadata",
        {},
    )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if payload.get(
        "original_chunk_id"
    ) != chunk_id:

        raise AssertionError(
            "Spot-check failed: ID mismatch."
        )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    if stored_text != source_record["text"]:

        raise AssertionError(
            "Spot-check failed: text mismatch."
        )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    expected_metadata = (
        source_record["metadata"]
    )

    for key in [
        "source",
        "study_id",
        "chunk_index",
    ]:

        if (
            stored_metadata.get(key)
            != expected_metadata.get(key)
        ):

            raise AssertionError(
                f"Spot-check failed: "
                f"{key} mismatch."
            )

    # --------------------------------------------------------
    # VECTOR
    # --------------------------------------------------------

    if point.vector is None:

        raise AssertionError(
            "Spot-check failed: vector missing."
        )

    stored_dimension = len(
        point.vector
    )

    expected_dimension = len(
        embedding_record["embedding"]
    )

    if (
        stored_dimension
        != expected_dimension
    ):

        raise AssertionError(
            "Spot-check failed: "
            "vector dimension mismatch."
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print(
        "Spot-check verification:"
    )

    print("-" * 60)

    print(
        f"Original chunk ID: {chunk_id}"
    )

    print(
        f"Qdrant point ID: {point_id}"
    )

    print(
        f"Vector length: {stored_dimension}"
    )

    print(
        f"Source: "
        f"{stored_metadata.get('source')}"
    )

    print(
        f"Study ID: "
        f"{stored_metadata.get('study_id')}"
    )

    print(
        f"Chunk index: "
        f"{stored_metadata.get('chunk_index')}"
    )

    print(
        f"Text: {stored_text}"
    )

    print("-" * 60)

    print(
        "ID match: PASS"
    )

    print(
        "Text match: PASS"
    )

    print(
        "Metadata match: PASS"
    )

    print(
        "Vector length match: PASS"
    )

    print(
        "Spot-check: PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "========== PHARMALENS EMBEDDING INDEXING =========="
    )
    print()

    logger = setup_logging()

    logger.info(
        "======================================================================"
    )

    logger.info(
        "PHARMALENS - INDEXING EMBEDDINGS & METADATA"
    )

    logger.info(
        "======================================================================"
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    settings = load_settings(
        require_chat=False,
        require_embedding=True,
        require_vector_db=True,
    )

    qdrant_url = settings[
        "qdrant_url"
    ]

    collection_name = settings[
        "qdrant_collection"
    ]

    vector_dimension = int(
        settings[
            "vector_dimension"
        ]
    )

    print(
        f"Qdrant URL: {qdrant_url}"
    )

    print(
        f"Collection: {collection_name}"
    )

    print(
        f"Vector dimension: "
        f"{vector_dimension}"
    )

    print(
        f"Embedding model: "
        f"{settings['embed_model']}"
    )

    # --------------------------------------------------------
    # TASK 1
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TASK 1 - LOAD CORPUS EMBEDDINGS"
    )
    print("=" * 70)

    embeddings = load_embeddings()

    source_chunks = (
        load_source_chunks()
    )

    print(
        f"Embedding records: "
        f"{len(embeddings)}"
    )

    print(
        f"Source chunks: "
        f"{len(source_chunks)}"
    )

    logger.info(
        "Embedding records loaded: %d",
        len(embeddings),
    )

    # --------------------------------------------------------
    # Validate that every embedding has source text
    # --------------------------------------------------------

    if set(embeddings.keys()) != set(
        source_chunks.keys()
    ):

        missing_text = (
            set(embeddings.keys())
            - set(source_chunks.keys())
        )

        missing_embeddings = (
            set(source_chunks.keys())
            - set(embeddings.keys())
        )

        raise AssertionError(
            "Embedding/source chunk IDs do not match.\n"
            f"Missing source text: {missing_text}\n"
            f"Missing embeddings: {missing_embeddings}"
        )

    print(
        "Embedding/source ID reconciliation: PASSED"
    )

    # --------------------------------------------------------
    # TASK 2
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TASK 2 - CONNECT TO QDRANT"
    )
    print("=" * 70)

    client = connect_qdrant(
        qdrant_url
    )

    verify_collection(
        client=client,
        collection_name=collection_name,
        expected_dimension=vector_dimension,
    )

    print(
        "Qdrant connection: SUCCESS"
    )

    print(
        "Collection verification: SUCCESS"
    )

    # --------------------------------------------------------
    # TASK 3
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TASK 3 - BUILD AND INDEX RECORDS"
    )
    print("=" * 70)

    points = []

    for chunk_id, embedding_record in (
        embeddings.items()
    ):

        validate_embedding_record(
            chunk_id,
            embedding_record,
        )

        source_record = source_chunks[
            chunk_id
        ]

        point = build_point(
            chunk_id=chunk_id,
            embedding_record=embedding_record,
            source_record=source_record,
        )

        points.append(point)

    print(
        f"Records prepared: {len(points)}"
    )

    print(
        "Record schema:"
    )

    print(
        "  - Qdrant point ID"
    )

    print(
        "  - embedding vector"
    )

    print(
        "  - source text"
    )

    print(
        "  - source document metadata"
    )

    print(
        "  - study ID"
    )

    print(
        "  - chunk index"
    )

    inserted, failures = index_records(
        client=client,
        collection_name=collection_name,
        points=points,
        logger=logger,
    )

    print()
    print(
        f"Inserted/upserted this run: "
        f"{inserted}"
    )

    print(
        f"Failed batches: "
        f"{len(failures)}"
    )

    # --------------------------------------------------------
    # TASK 4
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TASK 4 - INDEXED COUNT VALIDATION"
    )
    print("=" * 70)

    chunk_ids = list(
        embeddings.keys()
    )

    indexed_count = (
        verify_indexed_records(
            client=client,
            collection_name=collection_name,
            chunk_ids=chunk_ids,
        )
    )

    expected_count = len(
        embeddings
    )

    print(
        f"Expected corpus chunks: "
        f"{expected_count}"
    )

    print(
        f"Indexed corpus records: "
        f"{indexed_count}"
    )

    if indexed_count != expected_count:

        raise AssertionError(
            "Indexed count does not match "
            "expected corpus chunk count."
        )

    print(
        "Count validation: PASSED"
    )

    # --------------------------------------------------------
    # TASK 5
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TASK 5 - SPOT CHECK"
    )
    print("=" * 70)

    sample_id = chunk_ids[0]

    spot_check(
        client=client,
        collection_name=collection_name,
        chunk_id=sample_id,
        embedding_record=embeddings[
            sample_id
        ],
        source_record=source_chunks[
            sample_id
        ],
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "INDEXING VALIDATION SUCCESS"
    )
    print("=" * 70)

    print(
        f"Expected chunks: {expected_count}"
    )

    print(
        f"Indexed corpus records: "
        f"{indexed_count}"
    )

    print(
        f"Inserted/upserted this run: "
        f"{inserted}"
    )

    print(
        f"Failures: {len(failures)}"
    )

    print(
        "Count validation: PASSED"
    )

    print(
        "Spot-check: PASSED"
    )

    print(
        "Schema: vector + text + metadata"
    )

    print()

    print(
        f"Summary log: "
        f"{SUMMARY_FILE.resolve()}"
    )

    logger.info(
        "======================================================================"
    )

    logger.info(
        "INDEXING VALIDATION SUCCESS"
    )

    logger.info(
        "Expected chunks: %d",
        expected_count,
    )

    logger.info(
        "Indexed corpus records: %d",
        indexed_count,
    )

    logger.info(
        "Inserted/upserted this run: %d",
        inserted,
    )

    logger.info(
        "Failures: %d",
        len(failures),
    )

    logger.info(
        "Count validation: PASSED"
    )

    logger.info(
        "Spot-check: PASSED"
    )


if __name__ == "__main__":
    main()