from __future__ import annotations

import logging
from pathlib import Path

import tiktoken

from src.corpus_loader_demo import load_documents


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

SAMPLE_CORPUS_DIR = DATA_DIR / "sample_corpus"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_FILE = OUTPUT_DIR / "token_chunking.log"


# ============================================================
# CHUNK CONFIGURATION
# ============================================================

CHUNK_SIZE = 400

CHUNK_OVERLAP = 40

TOKENIZER_NAME = "cl100k_base"


# ============================================================
# TOKENIZER
# ============================================================

ENCODER = tiktoken.get_encoding(TOKENIZER_NAME)


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> None:
    """
    Configure logging for the chunking demonstration.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# ============================================================
# TOKEN COUNT
# ============================================================

def count_tokens(text: str) -> int:
    """
    Return the number of tokens in a text string.
    """

    return len(
        ENCODER.encode(text)
    )


# ============================================================
# TOKEN-AWARE CHUNKING
# ============================================================

def token_chunks(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into chunks based on token count.

    Each chunk contains at most `size` tokens.

    The next chunk starts `overlap` tokens before the
    previous chunk ended.
    """

    if size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "Overlap cannot be negative."
        )

    if overlap >= size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    tokens = ENCODER.encode(text)

    chunks: list[str] = []

    step = size - overlap

    start = 0

    while start < len(tokens):

        end = start + size

        chunk_tokens = tokens[start:end]

        if not chunk_tokens:
            break

        chunk_text = ENCODER.decode(
            chunk_tokens
        )

        chunks.append(
            chunk_text
        )

        start += step

    return chunks


# ============================================================
# CHUNK INFORMATION
# ============================================================

def describe_chunks(
    chunks: list[str],
) -> list[dict[str, object]]:
    """
    Return token-count information for each chunk.
    """

    results = []

    for index, chunk in enumerate(chunks, start=1):

        results.append(
            {
                "chunk_number": index,
                "token_count": count_tokens(chunk),
                "character_count": len(chunk),
                "text": chunk,
            }
        )

    return results


# ============================================================
# OVERLAP DEMONSTRATION
# ============================================================

def demonstrate_overlap() -> str:
    """
    Demonstrate how overlap preserves boundary context.

    The important clinical statement is intentionally placed
    near a chunk boundary.
    """

    boundary_text = (
        "The study enrolled adult participants with confirmed "
        "disease. Participants were randomly assigned to receive "
        "Drug X or placebo. The primary endpoint was the change "
        "in disease severity after twelve weeks. "
        "Safety monitoring included adverse events, laboratory "
        "tests, vital signs, and treatment discontinuation. "
        "The study reported headache and nausea as commonly "
        "observed adverse events. Serious adverse events were "
        "reviewed separately by the study investigators."
    )

    # Use a small chunk size here so the boundary is easy to see.
    demo_size = 35

    without_overlap = token_chunks(
        boundary_text,
        size=demo_size,
        overlap=0,
    )

    with_overlap = token_chunks(
        boundary_text,
        size=demo_size,
        overlap=10,
    )

    lines: list[str] = []

    lines.append(
        "BOUNDARY CONTEXT DEMONSTRATION"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "The same text is chunked twice:"
    )

    lines.append(
        "1. Without overlap"
    )

    lines.append(
        "2. With 10-token overlap"
    )

    lines.append("")

    lines.append(
        "WITHOUT OVERLAP"
    )

    lines.append(
        "-" * 70
    )

    for index, chunk in enumerate(
        without_overlap,
        start=1,
    ):

        lines.append(
            f"Chunk {index} "
            f"({count_tokens(chunk)} tokens):"
        )

        lines.append(
            chunk
        )

        lines.append("")

    lines.append(
        "WITH OVERLAP"
    )

    lines.append(
        "-" * 70
    )

    for index, chunk in enumerate(
        with_overlap,
        start=1,
    ):

        lines.append(
            f"Chunk {index} "
            f"({count_tokens(chunk)} tokens):"
        )

        lines.append(
            chunk
        )

        lines.append("")

    lines.append(
        "OBSERVATION"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        "With overlap, the end of one chunk is repeated "
        "at the beginning of the next chunk."
    )

    lines.append(
        "This gives retrieval another opportunity to capture "
        "an idea that lies near a chunk boundary."
    )

    return "\n".join(lines)


# ============================================================
# CORPUS PATHS
# ============================================================

def collect_sample_paths() -> list[Path]:
    """
    Collect supported sample documents.

    If your previous text-cleaning assignment added
    noisy_clinical_report.txt, it will also be included.
    """

    paths = [
        SAMPLE_CORPUS_DIR / "clinical_trial_overview.txt",
        SAMPLE_CORPUS_DIR / "eligibility_criteria.md",
        SAMPLE_CORPUS_DIR / "study_export.html",
    ]

    noisy_report = (
        SAMPLE_CORPUS_DIR
        / "noisy_clinical_report.txt"
    )

    if noisy_report.exists():
        paths.append(noisy_report)

    return [
        path
        for path in paths
        if path.exists()
    ]


# ============================================================
# MAIN DEMONSTRATION
# ============================================================

def main() -> None:

    setup_logging()

    print(
        "\n========== PHARMALENS TOKEN-AWARE CHUNKING ==========\n"
    )

    print(
        f"Tokenizer     : {TOKENIZER_NAME}"
    )

    print(
        f"Chunk size    : {CHUNK_SIZE} tokens"
    )

    print(
        f"Chunk overlap : {CHUNK_OVERLAP} tokens"
    )

    print(
        f"Overlap ratio : "
        f"{CHUNK_OVERLAP / CHUNK_SIZE:.0%}"
    )

    print()

    # --------------------------------------------------------
    # Load documents using the existing corpus loader.
    # --------------------------------------------------------

    paths = collect_sample_paths()

    if not paths:

        raise RuntimeError(
            "No sample documents were found."
        )

    documents, skipped = load_documents(
        paths
    )

    # --------------------------------------------------------
    # Create report.
    # --------------------------------------------------------

    report: list[str] = []

    report.append(
        "PHARMALENS - TOKEN-AWARE CHUNK SIZING & OVERLAP"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    report.append(
        f"Tokenizer: {TOKENIZER_NAME}"
    )

    report.append(
        f"Chunk size: {CHUNK_SIZE} tokens"
    )

    report.append(
        f"Chunk overlap: {CHUNK_OVERLAP} tokens"
    )

    report.append(
        f"Overlap ratio: "
        f"{CHUNK_OVERLAP / CHUNK_SIZE:.0%}"
    )

    report.append("")

    # --------------------------------------------------------
    # Task 1 + Task 2
    # --------------------------------------------------------

    report.append(
        "TASK 1 + TASK 2 - TOKEN SIZING AND OVERLAP"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    for document in documents:

        total_tokens = count_tokens(
            document.text
        )

        chunks = token_chunks(
            document.text,
            size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        report.append(
            f"SOURCE: {document.source_id}"
        )

        report.append(
            f"Original tokens: {total_tokens}"
        )

        report.append(
            f"Number of chunks: {len(chunks)}"
        )

        report.append("")

        chunk_info = describe_chunks(
            chunks
        )

        for info in chunk_info[:3]:

            report.append(
                f"Chunk {info['chunk_number']}: "
                f"{info['token_count']} tokens, "
                f"{info['character_count']} characters"
            )

            preview = str(
                info["text"]
            ).replace(
                "\n",
                " ",
            )

            report.append(
                f"Preview: {preview[:250]}"
            )

            report.append("")

        report.append(
            "-" * 70
        )

        report.append("")

    # --------------------------------------------------------
    # Task 3
    # --------------------------------------------------------

    report.append(
        "TASK 3 - OVERLAP EFFECT"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    report.append(
        demonstrate_overlap()
    )

    report.append("")

    # --------------------------------------------------------
    # Task 4
    # --------------------------------------------------------

    report.append(
        "TASK 4 - CHUNK SIZE AND OVERLAP JUSTIFICATION"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    report.append(
        "Chosen configuration:"
    )

    report.append(
        f"- Chunk size: {CHUNK_SIZE} tokens"
    )

    report.append(
        f"- Overlap: {CHUNK_OVERLAP} tokens"
    )

    report.append("")

    report.append(
        "Why 400 tokens?"
    )

    report.append(
        "A 400-token chunk is large enough to preserve "
        "a meaningful piece of a clinical document while "
        "remaining small enough for efficient retrieval."
    )

    report.append("")

    report.append(
        "Why 40-token overlap?"
    )

    report.append(
        "A 40-token overlap represents 10% of the chunk. "
        "It provides boundary context without duplicating "
        "too much text."
    )

    report.append("")

    report.append(
        "Cost and retrieval trade-off:"
    )

    report.append(
        "Increasing chunk size reduces the number of chunks "
        "but places more text into each retrieval result."
    )

    report.append(
        "Increasing overlap preserves more boundary context "
        "but creates duplicated tokens, increasing embedding "
        "and storage costs."
    )

    report.append("")

    report.append(
        "These values are a starting point for PharmaLens. "
        "They should later be evaluated using retrieval "
        "precision, recall, answer correctness, and "
        "context-window constraints."
    )

    report.append("")

    # --------------------------------------------------------
    # Task 5
    # --------------------------------------------------------

    report.append(
        "TASK 5 - SUMMARY"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    report.append(
        "Token-based chunking: IMPLEMENTED"
    )

    report.append(
        "Controlled overlap: IMPLEMENTED"
    )

    report.append(
        "Boundary demonstration: IMPLEMENTED"
    )

    report.append(
        "Chunk size justification: DOCUMENTED"
    )

    report.append(
        "Sample output: GENERATED"
    )

    report_text = "\n".join(
        report
    )

    print(
        report_text
    )

    # --------------------------------------------------------
    # Save output.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        report_text,
        encoding="utf-8",
    )

    print(
        f"\nSample output saved to: {OUTPUT_FILE}"
    )

    if skipped:

        print("\nSkipped documents:")

        for item in skipped:
            print(item)


if __name__ == "__main__":
    main()