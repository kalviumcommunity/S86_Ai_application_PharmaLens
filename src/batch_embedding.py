from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import tiktoken
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError

from src.config import load_settings


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EMBEDDING_STORE = OUTPUT_DIR / "batch_embeddings.json"
SUMMARY_FILE = OUTPUT_DIR / "batch_embedding_summary.log"


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 4
MAX_RETRIES = 3

# Gemini embedding-001 standard price:
# $0.15 per 1 million input tokens.
#
# Keep this configurable because pricing can change.
PRICE_PER_1M_TOKENS = 0.15


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> None:
    """Configure logging to terminal and output file."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                SUMMARY_FILE,
                mode="w",
                encoding="utf-8",
            ),
        ],
    )


# ============================================================
# SAMPLE CHUNKS
# ============================================================

def create_sample_chunks() -> list[dict]:
    """
    Create representative PharmaLens chunks.

    In the future these objects will come directly from
    the corpus ingestion/chunking pipeline.
    """

    return [
        {
            "id": "study-001-chunk-001",
            "text": (
                "Study 001 evaluated the safety and efficacy of Drug X "
                "in adult patients with moderate disease."
            ),
            "metadata": {
                "source": "Study_001_Clinical_Report.pdf",
                "study_id": "STUDY-001",
                "chunk_index": 1,
            },
        },
        {
            "id": "study-001-chunk-002",
            "text": (
                "The primary endpoint was the change in disease severity "
                "from baseline after twelve weeks of treatment."
            ),
            "metadata": {
                "source": "Study_001_Clinical_Report.pdf",
                "study_id": "STUDY-001",
                "chunk_index": 2,
            },
        },
        {
            "id": "study-001-chunk-003",
            "text": (
                "Common adverse events included headache, nausea, "
                "and fatigue during the treatment period."
            ),
            "metadata": {
                "source": "Study_001_Clinical_Report.pdf",
                "study_id": "STUDY-001",
                "chunk_index": 3,
            },
        },
        {
            "id": "study-002-chunk-001",
            "text": (
                "Study 002 compared Drug Y with placebo to evaluate "
                "its effectiveness and safety."
            ),
            "metadata": {
                "source": "Study_002_Clinical_Report.pdf",
                "study_id": "STUDY-002",
                "chunk_index": 1,
            },
        },
        {
            "id": "study-002-chunk-002",
            "text": (
                "The treatment group showed improvement in the primary "
                "clinical outcome compared with the placebo group."
            ),
            "metadata": {
                "source": "Study_002_Clinical_Report.pdf",
                "study_id": "STUDY-002",
                "chunk_index": 2,
            },
        },
        {
            "id": "study-002-chunk-003",
            "text": (
                "The safety analysis identified no new major safety "
                "signals during the study period."
            ),
            "metadata": {
                "source": "Study_002_Clinical_Report.pdf",
                "study_id": "STUDY-002",
                "chunk_index": 3,
            },
        },
        {
            "id": "drug-x-label-001",
            "text": (
                "Drug X should be administered according to the dosage "
                "and administration instructions in the approved label."
            ),
            "metadata": {
                "source": "Drug_X_Label.pdf",
                "study_id": None,
                "chunk_index": 1,
            },
        },
        {
            "id": "safety-bulletin-001",
            "text": (
                "The safety bulletin summarizes important adverse "
                "event information associated with Drug X."
            ),
            "metadata": {
                "source": "Drug_X_Safety_Bulletin.pdf",
                "study_id": None,
                "chunk_index": 1,
            },
        },
    ]


# ============================================================
# TOKEN COUNTING
# ============================================================

def count_tokens(text: str) -> int:
    """
    Estimate token count using cl100k_base.

    This is an estimate for cost tracking. The provider's
    actual tokenizer may differ.
    """

    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


# ============================================================
# BATCHING
# ============================================================

def batches(items: list[dict], size: int):
    """Yield items in configurable batches."""

    if size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    for start in range(0, len(items), size):
        yield items[start:start + size]


# ============================================================
# EMBEDDING STORE
# ============================================================

def load_existing_embeddings() -> dict:
    """
    Load previously generated embeddings.

    The dictionary key is the chunk ID.
    """

    if not EMBEDDING_STORE.exists():
        return {}

    try:
        return json.loads(
            EMBEDDING_STORE.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        logging.warning(
            "Existing embedding store is invalid. "
            "Starting with an empty store."
        )
        return {}


def save_embedding_store(store: dict) -> None:
    """Persist embeddings after every successful batch."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    EMBEDDING_STORE.write_text(
        json.dumps(store, indent=2),
        encoding="utf-8",
    )


# ============================================================
# RETRY LOGIC
# ============================================================

def embed_batch_with_retry(
    client: OpenAI,
    model: str,
    texts: list[str],
):
    """
    Generate embeddings with exponential backoff.

    Retryable errors:
    - rate limits
    - connection failures
    - timeouts
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            logging.info(
                "Sending embedding batch with %d chunks.",
                len(texts),
            )

            return client.embeddings.create(
                model=model,
                input=texts,
            )

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
        ) as error:

            if attempt == MAX_RETRIES:
                logging.error(
                    "Batch failed after %d attempts: %s",
                    MAX_RETRIES,
                    error,
                )
                raise

            wait_seconds = 2 ** (attempt - 1)

            logging.warning(
                "Temporary embedding error: %s",
                error,
            )

            logging.warning(
                "Retrying attempt %d/%d after %d seconds.",
                attempt + 1,
                MAX_RETRIES,
                wait_seconds,
            )

            time.sleep(wait_seconds)


# ============================================================
# COST
# ============================================================

def estimate_cost(input_tokens: int) -> float:
    """
    Estimate embedding cost in USD.
    """

    return (input_tokens / 1_000_000) * PRICE_PER_1M_TOKENS


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_batch_embedding() -> None:

    settings = load_settings()

    model = settings["embed_model"]

    client = OpenAI(
        base_url=settings["openai_base_url"],
        api_key=settings["openai_api_key"],
    )

    chunks = create_sample_chunks()

    existing_embeddings = load_existing_embeddings()

    logging.info("=" * 70)
    logging.info("PHARMALENS - BATCH EMBEDDING PIPELINE")
    logging.info("=" * 70)

    logging.info("Embedding model: %s", model)
    logging.info("Batch size: %d", BATCH_SIZE)
    logging.info("Total chunks: %d", len(chunks))

    # --------------------------------------------------------
    # TASK 4: SKIP ALREADY EMBEDDED
    # --------------------------------------------------------

    pending_chunks = [
        chunk
        for chunk in chunks
        if chunk["id"] not in existing_embeddings
    ]

    skipped_chunks = len(chunks) - len(pending_chunks)

    logging.info(
        "Already embedded: %d",
        skipped_chunks,
    )

    logging.info(
        "Pending chunks: %d",
        len(pending_chunks),
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_input_tokens = 0
    embedded_count = 0
    failed_count = 0
    batch_count = 0
    failed_chunks = []

    # --------------------------------------------------------
    # TASK 1: PROCESS IN BATCHES
    # --------------------------------------------------------

    for batch in batches(pending_chunks, BATCH_SIZE):

        batch_count += 1

        logging.info("")
        logging.info(
            "========== BATCH %d ==========",
            batch_count,
        )

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        batch_tokens = sum(
            count_tokens(text)
            for text in texts
        )

        total_input_tokens += batch_tokens

        logging.info(
            "Batch chunks: %d",
            len(batch),
        )

        logging.info(
            "Estimated batch tokens: %d",
            batch_tokens,
        )

        try:

            response = embed_batch_with_retry(
                client=client,
                model=model,
                texts=texts,
            )

            # Preserve response order.
            embeddings = [
                item.embedding
                for item in response.data
            ]

            if len(embeddings) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} embeddings, "
                    f"received {len(embeddings)}."
                )

            # ------------------------------------------------
            # SAVE IMMEDIATELY
            # ------------------------------------------------

            for chunk, embedding in zip(
                batch,
                embeddings,
            ):

                existing_embeddings[chunk["id"]] = {
                    "chunk_id": chunk["id"],
                    "embedding": embedding,
                    "metadata": chunk["metadata"],
                }

                embedded_count += 1

            save_embedding_store(existing_embeddings)

            logging.info(
                "Batch %d completed successfully.",
                batch_count,
            )

        except Exception as error:

            failed_count += len(batch)

            logging.error(
                "Batch %d failed: %s",
                batch_count,
                error,
            )

            for chunk in batch:
                failed_chunks.append({
                    "id": chunk["id"],
                    "error": str(error),
                })

    # --------------------------------------------------------
    # COST
    # --------------------------------------------------------

    estimated_cost = estimate_cost(
        total_input_tokens
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    logging.info("")
    logging.info("=" * 70)
    logging.info("BATCH EMBEDDING RUN SUMMARY")
    logging.info("=" * 70)

    logging.info(
        "Total chunks: %d",
        len(chunks),
    )

    logging.info(
        "Embeddings generated: %d",
        embedded_count,
    )

    logging.info(
        "Skipped existing: %d",
        skipped_chunks,
    )

    logging.info(
        "Failed chunks: %d",
        failed_count,
    )

    logging.info(
        "Batches processed: %d",
        batch_count,
    )

    logging.info(
        "Estimated input tokens: %d",
        total_input_tokens,
    )

    logging.info(
        "Estimated cost: $%.8f",
        estimated_cost,
    )

    if failed_chunks:

        logging.info("")
        logging.info("FAILED CHUNKS:")

        for failure in failed_chunks:
            logging.info(
                "%s -> %s",
                failure["id"],
                failure["error"],
            )

    logging.info("")
    logging.info(
        "Embedding store: %s",
        EMBEDDING_STORE,
    )

    logging.info(
        "Summary log: %s",
        SUMMARY_FILE,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    setup_logging()

    run_batch_embedding()