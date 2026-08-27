from __future__ import annotations

import logging
from pathlib import Path

from src.token_chunker import (
    count_tokens,
    token_chunks,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_FILE = OUTPUT_DIR / "corpus_ingestion.log"


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".html",
}


CHUNK_SIZE = 400

CHUNK_OVERLAP = 40


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
# DOCUMENT DISCOVERY
# ============================================================

def discover_documents(
    folder: Path,
) -> list[Path]:
    """
    Find all supported source documents recursively.
    """

    files = []

    for path in folder.rglob("*"):

        if not path.is_file():
            continue

        if path.name == ".gitkeep":
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(path)

    return sorted(files)


# ============================================================
# TEXT LOADING
# ============================================================

def load_text(
    path: Path,
) -> str:
    """
    Load a text-based document using UTF-8 encoding.
    """

    return path.read_text(
        encoding="utf-8",
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    text: str,
) -> str:
    """
    Clean extracted text before chunking.

    This performs the basic cleaning steps from the
    previous text-cleaning assignment.
    """

    import re
    import unicodedata

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # Remove page footer patterns.
    text = re.sub(
        r"Page\s+\d+\s+of\s+\d+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Collapse spaces and tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Collapse excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# METADATA
# ============================================================

def create_metadata(
    path: Path,
    chunk_index: int,
    total_chunks: int,
) -> dict[str, object]:
    """
    Create metadata for one chunk.
    """

    return {
        "source": path.name,
        "source_path": str(path.relative_to(PROJECT_ROOT)),
        "chunk_index": chunk_index,
        "chunk_number": chunk_index + 1,
        "total_chunks": total_chunks,
    }


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(
    path: Path,
) -> tuple[list[dict[str, object]], int]:
    """
    Load, clean, chunk, and tag one document.

    Returns:
        chunks, original_token_count
    """

    # -------------------------------
    # Load
    # -------------------------------

    raw_text = load_text(
        path,
    )

    if not raw_text.strip():
        raise ValueError(
            "Document is empty."
        )

    # -------------------------------
    # Clean
    # -------------------------------

    cleaned_text = clean_text(
        raw_text,
    )

    if not cleaned_text:
        raise ValueError(
            "Document became empty after cleaning."
        )

    # -------------------------------
    # Token-aware chunking
    # -------------------------------

    chunks = token_chunks(
        cleaned_text,
        size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise ValueError(
            "No chunks were produced."
        )

    # -------------------------------
    # Metadata
    # -------------------------------

    total_chunks = len(chunks)

    tagged_chunks = []

    for index, chunk in enumerate(chunks):

        metadata = create_metadata(
            path,
            index,
            total_chunks,
        )

        metadata["token_count"] = count_tokens(
            chunk,
        )

        metadata["character_count"] = len(
            chunk,
        )

        tagged_chunks.append(
            {
                "text": chunk,
                "metadata": metadata,
            }
        )

    return tagged_chunks, count_tokens(
        cleaned_text,
    )


# ============================================================
# FULL CORPUS INGESTION
# ============================================================

def ingest_corpus(
    folder: Path,
) -> dict[str, object]:
    """
    Run the complete ingestion pipeline.

    Every discovered document is recorded as either
    successfully ingested or failed.
    """

    files = discover_documents(
        folder,
    )

    successful_documents = []

    failures = []

    all_chunks = []

    logging.info(
        "Discovered %d source documents.",
        len(files),
    )

    # --------------------------------------------------------
    # Process every document
    # --------------------------------------------------------

    for number, path in enumerate(
        files,
        start=1,
    ):

        logging.info(
            "Processing %d/%d: %s",
            number,
            len(files),
            path.name,
        )

        try:

            chunks, original_tokens = process_document(
                path,
            )

            successful_documents.append(
                {
                    "source": path.name,
                    "path": str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "original_tokens": original_tokens,
                    "chunks": len(chunks),
                }
            )

            all_chunks.extend(
                chunks
            )

            logging.info(
                "SUCCESS: %s -> %d chunks",
                path.name,
                len(chunks),
            )

        except Exception as error:

            failures.append(
                {
                    "source": path.name,
                    "path": str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "error": str(error),
                }
            )

            logging.error(
                "FAILED: %s -> %s",
                path.name,
                error,
            )

    return {
        "files": files,
        "successful_documents": successful_documents,
        "failures": failures,
        "chunks": all_chunks,
    }


# ============================================================
# COMPLETENESS VALIDATION
# ============================================================

def validate_completeness(
    total_files: int,
    successful_count: int,
    failure_count: int,
) -> bool:
    """
    Verify that every discovered source document has
    exactly one outcome: success or recorded failure.
    """

    accounted_for = (
        successful_count
        + failure_count
    )

    print(
        "\nCOMPLETENESS VALIDATION"
    )

    print(
        "-" * 70
    )

    print(
        f"Source documents          : {total_files}"
    )

    print(
        f"Successfully ingested     : {successful_count}"
    )

    print(
        f"Recorded failures         : {failure_count}"
    )

    print(
        f"Accounted documents       : {accounted_for}"
    )

    print()

    if accounted_for != total_files:

        print(
            "FAIL - A document was silently dropped."
        )

        raise AssertionError(
            "Source count does not equal "
            "successful documents + failures."
        )

    print(
        "PASS - No documents were silently dropped."
    )

    return True


# ============================================================
# REPORT
# ============================================================

def build_report(
    result: dict[str, object],
) -> str:
    """
    Build the final ingestion report.
    """

    files = result["files"]

    successful_documents = (
        result["successful_documents"]
    )

    failures = result["failures"]

    chunks = result["chunks"]

    successful_count = len(
        successful_documents
    )

    failure_count = len(
        failures
    )

    total_files = len(
        files
    )

    lines = []

    lines.append(
        "PHARMALENS - CORPUS PREPARATION & INGESTION VALIDATION"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        "INGESTION SUMMARY"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Total source documents : {total_files}"
    )

    lines.append(
        f"Successfully ingested  : {successful_count}"
    )

    lines.append(
        f"Failed documents       : {failure_count}"
    )

    lines.append(
        f"Total chunks created   : {len(chunks)}"
    )

    lines.append("")

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    accounted_for = (
        successful_count
        + failure_count
    )

    lines.append(
        "COMPLETENESS VALIDATION"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"{total_files} source files = "
        f"{successful_count} successful + "
        f"{failure_count} failures"
    )

    if accounted_for == total_files:

        lines.append(
            "PASS - No documents were silently dropped."
        )

    else:

        lines.append(
            "FAIL - Document accounting mismatch."
        )

    lines.append("")

    # --------------------------------------------------------
    # Successful documents
    # --------------------------------------------------------

    lines.append(
        "SUCCESSFUL DOCUMENTS"
    )

    lines.append(
        "-" * 70
    )

    for document in successful_documents:

        lines.append(
            f"{document['source']} | "
            f"{document['original_tokens']} tokens | "
            f"{document['chunks']} chunks"
        )

    lines.append("")

    # --------------------------------------------------------
    # Failures
    # --------------------------------------------------------

    lines.append(
        "FAILURES / SKIPPED FILES"
    )

    lines.append(
        "-" * 70
    )

    if failures:

        for failure in failures:

            lines.append(
                f"FAILED: {failure['source']}"
            )

            lines.append(
                f"Reason: {failure['error']}"
            )

    else:

        lines.append(
            "None"
        )

    lines.append("")

    # --------------------------------------------------------
    # Sample chunks
    # --------------------------------------------------------

    lines.append(
        "SAMPLE CHUNKS WITH METADATA"
    )

    lines.append(
        "-" * 70
    )

    sample_chunks = chunks[:5]

    for index, chunk in enumerate(
        sample_chunks,
        start=1,
    ):

        text = chunk["text"]

        metadata = chunk["metadata"]

        preview = " ".join(
            str(text).split()
        )

        lines.append(
            f"Sample {index}"
        )

        lines.append(
            f"Text: {preview[:300]}"
        )

        lines.append(
            f"Metadata: {metadata}"
        )

        lines.append("")

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    lines.append(
        "PIPELINE"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "Load → Clean → Token Chunk → Metadata → Validation"
    )

    lines.append("")

    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    setup_logging()

    print(
        "\n========== PHARMALENS CORPUS INGESTION ==========\n"
    )

    print(
        f"Corpus directory : {DATA_DIR}"
    )

    print(
        f"Chunk size       : {CHUNK_SIZE} tokens"
    )

    print(
        f"Chunk overlap    : {CHUNK_OVERLAP} tokens"
    )

    print()

    # --------------------------------------------------------
    # Run ingestion
    # --------------------------------------------------------

    result = ingest_corpus(
        DATA_DIR
    )

    files = result["files"]

    successful_documents = (
        result["successful_documents"]
    )

    failures = result["failures"]

    chunks = result["chunks"]

    # --------------------------------------------------------
    # Validate completeness
    # --------------------------------------------------------

    validate_completeness(
        total_files=len(files),
        successful_count=len(
            successful_documents
        ),
        failure_count=len(
            failures
        ),
    )

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = build_report(
        result
    )

    print(
        "\n" + report
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"Report saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()