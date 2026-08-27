from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from src.config import load_settings


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_FILE = OUTPUT_DIR / "embedding_demo.log"


# ============================================================
# SAMPLE TEXTS
# ============================================================

@dataclass(frozen=True)
class PreparedChunk:
    source_text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class StoredEmbedding:
    source_text: str
    metadata: dict[str, str | int]
    vector: list[float]


TEXTS = [
    PreparedChunk(
        source_text=(
            "Patients receiving Drug X experienced mild headache "
            "and nausea during the clinical trial."
        ),
        metadata={"source_document": "clinical_trial_overview.txt", "chunk_index": 1, "section": "Adverse Events"},
    ),
    PreparedChunk(
        source_text=(
            "The clinical study reported headache and nausea as "
            "commonly observed adverse events in participants "
            "treated with Drug X."
        ),
        metadata={"source_document": "clinical_trial_overview.txt", "chunk_index": 2, "section": "Adverse Events"},
    ),
    PreparedChunk(
        source_text=(
            "The hospital cafeteria serves pasta, rice, vegetables, "
            "and sandwiches during lunch."
        ),
        metadata={"source_document": "hospital_facilities.txt", "chunk_index": 1, "section": "Cafeteria"},
    ),
    PreparedChunk(
        source_text=(
            "The weather forecast predicts heavy rainfall and "
            "strong winds this weekend."
        ),
        metadata={"source_document": "weather_report.txt", "chunk_index": 1, "section": "Forecast"},
    ),
]


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> None:
    """
    Configure logging to the terminal and output file.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
# EMBEDDING GENERATION
# ============================================================

def generate_embeddings(
    client: OpenAI,
    model: str,
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    """

    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    # Sort by index so the returned embeddings match
    # the original input order.
    data = sorted(response.data, key=lambda item: item.index)

    if len(data) != len(texts):
        raise ValueError(
            f"Expected {len(texts)} embeddings, but received {len(data)}."
        )

    return [
        item.embedding
        for item in data
    ]


def store_embeddings(
    chunks: list[PreparedChunk],
    embeddings: list[list[float]],
) -> list[StoredEmbedding]:
    """Pair each API vector with its source chunk and retrieval metadata."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Expected one embedding per chunk, got {len(embeddings)} for "
            f"{len(chunks)} chunks."
        )

    return [
        StoredEmbedding(
            source_text=chunk.source_text,
            metadata=chunk.metadata,
            vector=vector,
        )
        for chunk, vector in zip(chunks, embeddings)
    ]


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Vectors must have the same dimension."
        )

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(
            a * a
            for a in vector_a
        )
    )

    magnitude_b = math.sqrt(
        sum(
            b * b
            for b in vector_b
        )
    )

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError(
            "Cannot calculate similarity for a zero vector."
        )

    return dot_product / (
        magnitude_a * magnitude_b
    )


# ============================================================
# DIMENSION VALIDATION
# ============================================================

def validate_dimensions(
    embeddings: list[list[float]],
) -> int:
    """
    Confirm that all embeddings have the same dimension.
    """

    if not embeddings:
        raise ValueError(
            "No embeddings were generated."
        )

    dimensions = {
        len(vector)
        for vector in embeddings
    }

    if len(dimensions) != 1:
        raise ValueError(
            f"Embedding dimensions are inconsistent: "
            f"{dimensions}"
        )

    return len(embeddings[0])


# ============================================================
# BUILD REPORT
# ============================================================

def build_report(
    model: str,
    stored_embeddings: list[StoredEmbedding],
) -> str:
    """
    Build the sample output report.
    """

    lines: list[str] = []

    lines.append(
        "PHARMALENS - EMBEDDINGS FUNDAMENTALS & VECTOR REPRESENTATION"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        f"Embedding model: {model}"
    )

    lines.append("")

    # --------------------------------------------------------
    # Task 1
    # --------------------------------------------------------

    lines.append(
        "TASK 1 - EMBEDDING GENERATION"
    )

    lines.append(
        "-" * 70
    )

    lines.append(f"Embedded chunks: {len(stored_embeddings)}")

    lines.append("")

    # --------------------------------------------------------
    # Task 2
    # --------------------------------------------------------

    dimension = validate_dimensions(
        [record.vector for record in stored_embeddings]
    )

    lines.append(
        "TASK 2 - VECTOR DIMENSION"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Vector dimension: {dimension}"
    )

    lines.append(
        "All sample embeddings have the same dimension: YES"
    )

    lines.append("")

    # --------------------------------------------------------
    # Sample vector output
    # --------------------------------------------------------

    lines.append(
        "SAMPLE VECTOR OUTPUT"
    )

    lines.append(
        "-" * 70
    )

    for index, record in enumerate(
        stored_embeddings,
        start=1,
    ):

        lines.append(
            f"Sample {index}: {record.metadata['source_document']}"
        )

        lines.append(
            f"Text: {record.source_text}"
        )

        lines.append(
            f"Metadata: {record.metadata}"
        )

        lines.append(
            f"Vector length: {len(record.vector)}"
        )

        # Only show a small portion of the vector.
        lines.append(
            "First 8 vector values: "
            + str(record.vector[:8])
        )

        lines.append("")

    # --------------------------------------------------------
    # Task 3
    # --------------------------------------------------------

    similar_score = cosine_similarity(
        stored_embeddings[0].vector,
        stored_embeddings[1].vector,
    )

    unrelated_score = cosine_similarity(
        stored_embeddings[0].vector,
        stored_embeddings[2].vector,
    )

    different_topic_score = cosine_similarity(
        stored_embeddings[0].vector,
        stored_embeddings[3].vector,
    )

    lines.append(
        "TASK 3 - COSINE SIMILARITY"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "Similar pair:"
    )

    lines.append(
        "Clinical trial adverse events vs "
        "Drug X adverse events"
    )

    lines.append(
        f"Cosine similarity: {similar_score:.6f}"
    )

    lines.append("")

    lines.append(
        "Dissimilar pair:"
    )

    lines.append(
        "Clinical trial adverse events vs cafeteria menu"
    )

    lines.append(
        f"Cosine similarity: {unrelated_score:.6f}"
    )

    lines.append("")

    lines.append(
        "Different-topic pair:"
    )

    lines.append(
        "Clinical trial adverse events vs weather"
    )

    lines.append(
        f"Cosine similarity: {different_topic_score:.6f}"
    )

    lines.append("")

    if similar_score > unrelated_score:

        lines.append(
            "RESULT: PASS"
        )

        lines.append(
            "The semantically similar pair has a higher "
            "similarity score than the unrelated pair."
        )

    else:

        lines.append(
            "RESULT: CHECK"
        )

        lines.append(
            "The expected similarity ranking was not observed."
        )

    lines.append("")

    # --------------------------------------------------------
    # Task 4
    # --------------------------------------------------------

    lines.append(
        "TASK 4 - WHAT EMBEDDING VECTORS REPRESENT"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "An embedding vector is a numerical representation "
        "of the semantic meaning of text."
    )

    lines.append(
        "It is not a random ID and it is not simply a count "
        "of the keywords in the text."
    )

    lines.append(
        "Texts with similar meanings tend to have vectors "
        "that are closer together in vector space."
    )

    lines.append(
        "This allows PharmaLens to retrieve relevant clinical "
        "document chunks even when the user's question uses "
        "different wording."
    )

    lines.append("")

    # --------------------------------------------------------
    # Task 5
    # --------------------------------------------------------

    lines.append(
        "TASK 5 - DEMONSTRATION SUMMARY"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "Embedding generation: IMPLEMENTED"
    )

    lines.append(
        "Vector dimension reporting: IMPLEMENTED"
    )

    lines.append(
        "Dimension consistency validation: IMPLEMENTED"
    )

    lines.append(
        "Cosine similarity comparison: IMPLEMENTED"
    )

    lines.append(
        "Sample vector values: INCLUDED"
    )

    lines.append(
        "Embedding explanation: INCLUDED"
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    setup_logging()

    print(
        "\n========== PHARMALENS EMBEDDINGS DEMO ==========\n"
    )

    settings = load_settings(
        require_chat=False,
        require_embedding=True,
    )

    model = settings["embed_model"]

    if settings["openai_base_url"]:
        client = OpenAI(
            base_url=settings["openai_base_url"],
            api_key=settings["openai_api_key"],
        )
    else:
        client = OpenAI(api_key=settings["openai_api_key"])

    texts = [
        chunk.source_text
        for chunk in TEXTS
    ]

    logging.info(
        "Embedding model: %s",
        model,
    )

    logging.info(
        "Generating embeddings for %d texts.",
        len(texts),
    )

    embeddings = generate_embeddings(
        client=client,
        model=model,
        texts=texts,
    )

    stored_embeddings = store_embeddings(
        chunks=TEXTS,
        embeddings=embeddings,
    )

    report = build_report(
        model=model,
        stored_embeddings=stored_embeddings,
    )

    print(
        report
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"\nSample output saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()