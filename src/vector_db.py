from __future__ import annotations

import logging
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import load_settings
from src.llm_client import create_client


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "vector_db_setup.log"


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

def setup_logging() -> None:
    """Configure logging to terminal and output file."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers.
    if logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        OUTPUT_FILE,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# -------------------------------------------------------------------
# Qdrant connection
# -------------------------------------------------------------------

def create_qdrant_client(settings: dict[str, str]) -> QdrantClient:
    """
    Create a Qdrant client using the configured URL.
    """

    qdrant_url = settings["qdrant_url"]

    logging.info("Connecting to Qdrant: %s", qdrant_url)

    return QdrantClient(
        url=qdrant_url,
    )


# -------------------------------------------------------------------
# Connection test
# -------------------------------------------------------------------

def check_connection(qdrant_client: QdrantClient) -> bool:
    """
    Verify that Qdrant is reachable.
    """

    try:
        qdrant_client.get_collections()

        print("Qdrant connection: SUCCESS")
        logging.info("Qdrant connection successful.")

        return True

    except Exception as error:
        print("Qdrant connection: FAILED")
        print(f"Error: {error}")

        logging.exception(
            "Qdrant connection failed: %s",
            error,
        )

        return False


# -------------------------------------------------------------------
# Collection setup
# -------------------------------------------------------------------

def collection_exists(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> bool:
    """
    Check whether a collection already exists.
    """

    try:
        response = qdrant_client.collection_exists(
            collection_name=collection_name
        )

        return bool(response)

    except Exception:
        # Compatibility fallback for older Qdrant client versions.
        collections = qdrant_client.get_collections()

        return any(
            collection.name == collection_name
            for collection in collections.collections
        )


def create_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
    vector_dimension: int,
) -> None:
    """
    Create the Qdrant collection if it does not already exist.

    Vectors use cosine similarity because the PharmaLens
    embedding demo compares semantic similarity using cosine distance.
    """

    if collection_exists(
        qdrant_client,
        collection_name,
    ):
        print(
            f"Collection '{collection_name}' already exists."
        )

        logging.info(
            "Collection already exists: %s",
            collection_name,
        )

        return

    logging.info(
        "Creating collection: %s",
        collection_name,
    )

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dimension,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Collection '{collection_name}' created successfully."
    )

    logging.info(
        "Collection created successfully."
    )


# -------------------------------------------------------------------
# Collection information
# -------------------------------------------------------------------

def print_collection_info(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> None:
    """
    Display collection configuration.
    """

    collection_info = qdrant_client.get_collection(
        collection_name=collection_name
    )

    print("\nCollection configuration:")

    print(
        f"Status: {collection_info.status}"
    )

    print(
        f"Vectors configuration: "
        f"{collection_info.config.params.vectors}"
    )

    logging.info(
        "Collection status: %s",
        collection_info.status,
    )


# -------------------------------------------------------------------
# Embedding generation
# -------------------------------------------------------------------

def generate_test_embedding(
    client,
    model: str,
    text: str,
) -> list[float]:
    """
    Generate one embedding for the test record.

    The Gemini OpenAI-compatible endpoint may return
    index=None, so we directly read the first embedding
    instead of sorting by response.data[index].
    """

    logging.info(
        "Generating test embedding..."
    )

    response = client.embeddings.create(
        model=model,
        input=text,
    )

    if not response.data:
        raise ValueError(
            "Embedding API returned no embedding data."
        )

    embedding = response.data[0].embedding

    logging.info(
        "Generated embedding dimension: %d",
        len(embedding),
    )

    return embedding


# -------------------------------------------------------------------
# Test record
# -------------------------------------------------------------------

def create_test_record(
    embedding: list[float],
    text: str,
) -> tuple[str, dict]:
    """
    Create a test record containing:

    - valid Qdrant point ID
    - embedding vector
    - original text
    - source metadata
    - chunk position
    - section
    - page
    """

    # Qdrant accepts an unsigned integer or UUID.
    # UUID is better for demonstrating a realistic stable ID.
    point_id = str(uuid.uuid4())

    metadata = {
        "source": "clinical-trial-demo.txt",
        "chunk_index": 0,
        "section": "Introduction",
        "page": 1,
    }

    payload = {
        "text": text,
        "metadata": metadata,
    }

    return point_id, payload


# -------------------------------------------------------------------
# Insert record
# -------------------------------------------------------------------

def insert_test_record(
    qdrant_client: QdrantClient,
    collection_name: str,
    point_id: str,
    embedding: list[float],
    payload: dict,
) -> None:
    """
    Insert one test vector into Qdrant.

    The point ID is a UUID because Qdrant does not accept
    arbitrary strings such as 'clinical-trial-demo:0'.
    """

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        ],
        wait=True,
    )

    print("Record inserted successfully.")

    logging.info(
        "Test record inserted successfully. ID=%s",
        point_id,
    )


# -------------------------------------------------------------------
# Read record
# -------------------------------------------------------------------

def read_test_record(
    qdrant_client: QdrantClient,
    collection_name: str,
    point_id: str,
):
    """
    Read the inserted record back from Qdrant.
    """

    records = qdrant_client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_vectors=True,
        with_payload=True,
    )

    if not records:
        raise ValueError(
            "Inserted record could not be found."
        )

    return records[0]


# -------------------------------------------------------------------
# Display readback
# -------------------------------------------------------------------

def print_readback(record) -> None:
    """
    Display the important fields returned by Qdrant.
    """

    print("\nReadback verification:")
    print("-" * 60)

    print(
        f"ID: {record.id}"
    )

    if record.vector is None:
        print(
            "Vector length: NOT RETURNED"
        )
    else:
        print(
            f"Vector length: {len(record.vector)}"
        )

    payload = record.payload or {}

    print(
        f"Text: {payload.get('text', '')}"
    )

    print(
        f"Metadata: {payload.get('metadata', {})}"
    )

    print("-" * 60)

    logging.info(
        "Readback ID: %s",
        record.id,
    )

    if record.vector is not None:
        logging.info(
            "Readback vector length: %d",
            len(record.vector),
        )

    logging.info(
        "Readback payload: %s",
        payload,
    )


# -------------------------------------------------------------------
# Validate record
# -------------------------------------------------------------------

def validate_readback(
    record,
    expected_dimension: int,
) -> None:
    """
    Validate that the stored record contains
    the expected vector, text and metadata.
    """

    payload = record.payload or {}

    vector = record.vector

    if vector is None:
        raise ValueError(
            "Readback validation failed: vector missing."
        )

    if len(vector) != expected_dimension:
        raise ValueError(
            "Readback validation failed: "
            f"expected vector dimension {expected_dimension}, "
            f"got {len(vector)}."
        )

    if not payload.get("text"):
        raise ValueError(
            "Readback validation failed: text missing."
        )

    metadata = payload.get("metadata")

    if not isinstance(metadata, dict):
        raise ValueError(
            "Readback validation failed: metadata missing."
        )

    required_metadata = [
        "source",
        "chunk_index",
        "section",
        "page",
    ]

    missing_metadata = [
        field
        for field in required_metadata
        if field not in metadata
    ]

    if missing_metadata:
        raise ValueError(
            "Readback validation failed: "
            f"missing metadata fields: {missing_metadata}"
        )

    print("\nReadback validation: SUCCESS")

    logging.info(
        "Readback validation successful."
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """
    Run the complete Vector Database Setup assignment.
    """

    settings = load_settings(
        require_chat=False,
        require_embedding=True,
        require_vector_db=True,
    )

    qdrant_url = settings["qdrant_url"]
    collection_name = settings["qdrant_collection"]
    vector_dimension = int(
        settings["vector_dimension"]
    )
    embedding_model = settings["embed_model"]

    print(
        "\n========== PHARMALENS VECTOR DATABASE SETUP =========="
    )

    logging.info("=" * 70)
    logging.info(
        "PHARMALENS - VECTOR DATABASE SETUP"
    )
    logging.info("=" * 70)

    logging.info(
        "Qdrant URL: %s",
        qdrant_url,
    )

    logging.info(
        "Collection: %s",
        collection_name,
    )

    logging.info(
        "Embedding model: %s",
        embedding_model,
    )

    logging.info(
        "Expected vector dimension: %d",
        vector_dimension,
    )

    # ================================================================
    # TASK 1
    # ================================================================

    print("\n" + "=" * 70)
    print("TASK 1 - VECTOR DATABASE CONNECTION")
    print("=" * 70)

    qdrant_client = create_qdrant_client(settings)

    if not check_connection(qdrant_client):
        raise RuntimeError(
            "Cannot continue because Qdrant is unreachable."
        )

    # ================================================================
    # TASK 2
    # ================================================================

    print("\n" + "=" * 70)
    print("TASK 2 - COLLECTION SETUP")
    print("=" * 70)

    print(
        f"Collection: {collection_name}"
    )

    print(
        f"Vector dimension: {vector_dimension}"
    )

    print(
        "Distance metric: COSINE"
    )

    create_collection(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        vector_dimension=vector_dimension,
    )

    print("Collection setup: SUCCESS")

    # ================================================================
    # TASK 3
    # ================================================================

    print("\n" + "=" * 70)
    print("TASK 3 - RECORD SCHEMA")
    print("=" * 70)

    print(
        """
Each record contains:

- id
- vector
- text
- metadata.source
- metadata.chunk_index
- metadata.section
- metadata.page
"""
    )

    logging.info(
        "Record schema validated conceptually."
    )

    # ================================================================
    # TEST EMBEDDING
    # ================================================================

    print("\n" + "=" * 70)
    print("GENERATING TEST EMBEDDING")
    print("=" * 70)

    llm_client = create_client(settings)

    test_text = (
        "Clinical trials evaluate medical treatments "
        "for safety and effectiveness."
    )

    embedding = generate_test_embedding(
        client=llm_client,
        model=embedding_model,
        text=test_text,
    )

    print(
        "Embedding generated successfully."
    )

    print(
        f"Vector dimension: {len(embedding)}"
    )

    if len(embedding) != vector_dimension:
        raise ValueError(
            "Embedding dimension mismatch! "
            f"Expected {vector_dimension}, "
            f"got {len(embedding)}."
        )

    # ================================================================
    # TASK 4 - INSERT
    # ================================================================

    print("\n" + "=" * 70)
    print("TASK 4 - INSERT TEST RECORD")
    print("=" * 70)

    point_id, payload = create_test_record(
        embedding=embedding,
        text=test_text,
    )

    print(
        f"Generated valid Qdrant point ID: {point_id}"
    )

    insert_test_record(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        point_id=point_id,
        embedding=embedding,
        payload=payload,
    )

    # ================================================================
    # TASK 4 - READBACK
    # ================================================================

    print("\n" + "=" * 70)
    print("TASK 4 - READ BACK TEST RECORD")
    print("=" * 70)

    record = read_test_record(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        point_id=point_id,
    )

    print_readback(record)

    validate_readback(
        record=record,
        expected_dimension=vector_dimension,
    )

    # ================================================================
    # FINAL SUMMARY
    # ================================================================

    print("\n" + "=" * 70)
    print("VECTOR DATABASE VALIDATION SUCCESS")
    print("=" * 70)

    print(
        f"Qdrant URL: {qdrant_url}"
    )

    print(
        f"Collection: {collection_name}"
    )

    print(
        f"Vector dimension: {vector_dimension}"
    )

    print(
        "Distance metric: COSINE"
    )

    print(
        "Test record: INSERTED + READ BACK"
    )

    print(
        "Schema: vector + text + metadata"
    )

    print(
        f"\nLog saved to: {OUTPUT_FILE}"
    )

    logging.info("=" * 70)
    logging.info(
        "VECTOR DATABASE VALIDATION SUCCESS"
    )
    logging.info("=" * 70)


if __name__ == "__main__":
    setup_logging()
    main()