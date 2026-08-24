from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

from src.corpus_loader_demo import (
    LoadedDocument,
    load_documents,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "text_cleaning.log"
SAMPLE_CORPUS_DIR = PROJECT_ROOT / "data" / "sample_corpus"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    text: str,
    repeated_boilerplate: set[str] | None = None,
) -> str:
    """
    Clean extracted document text for downstream RAG processing.

    Cleaning steps:
    1. Normalize Unicode using NFKC.
    2. Normalize line endings.
    3. Remove page-number boilerplate.
    4. Remove known repeated headers/boilerplate.
    5. Normalize spaces and tabs.
    6. Remove spaces around line breaks.
    7. Collapse excessive blank lines.
    8. Strip leading/trailing whitespace.
    """

    # --------------------------------------------------------
    # 1. Unicode normalization
    # --------------------------------------------------------

    cleaned = unicodedata.normalize("NFKC", text)

    # --------------------------------------------------------
    # 2. Normalize line endings
    # --------------------------------------------------------

    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = cleaned.replace("\r", "\n")

    # --------------------------------------------------------
    # 3. Remove page footer/header patterns
    # --------------------------------------------------------

    cleaned = re.sub(
        r"(?im)^\s*Page\s+\d+\s+of\s+\d+\s*$",
        "",
        cleaned,
    )

    # Also handle common "Page 3" style markers.
    cleaned = re.sub(
        r"(?im)^\s*Page\s+\d+\s*$",
        "",
        cleaned,
    )

    # --------------------------------------------------------
    # 4. Remove repeated boilerplate
    # --------------------------------------------------------

    if repeated_boilerplate:
        lines = cleaned.split("\n")

        filtered_lines = []

        for line in lines:
            normalized_line = line.strip()

            if normalized_line in repeated_boilerplate:
                continue

            filtered_lines.append(line)

        cleaned = "\n".join(filtered_lines)

    # --------------------------------------------------------
    # 5. Normalize spaces and tabs
    # --------------------------------------------------------

    cleaned = re.sub(
        r"[ \t]+",
        " ",
        cleaned,
    )

    # --------------------------------------------------------
    # 6. Remove trailing spaces
    # --------------------------------------------------------

    cleaned = re.sub(
        r"[ \t]+\n",
        "\n",
        cleaned,
    )

    # --------------------------------------------------------
    # 7. Collapse excessive blank lines
    # --------------------------------------------------------

    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    # --------------------------------------------------------
    # 8. Final cleanup
    # --------------------------------------------------------

    return cleaned.strip()


# ============================================================
# DETECT REPEATED LINES
# ============================================================

def find_repeated_lines(
    documents: list[LoadedDocument],
    minimum_occurrences: int = 2,
) -> set[str]:
    """
    Detect repeated non-empty lines across the corpus.

    These lines can be candidates for boilerplate removal.
    """

    counter: Counter[str] = Counter()

    for document in documents:

        for line in document.text.splitlines():

            normalized = line.strip()

            if not normalized:
                continue

            counter[normalized] += 1

    return {
        line
        for line, count in counter.items()
        if count >= minimum_occurrences
    }


# ============================================================
# APPLY CLEANING TO CORPUS
# ============================================================

def clean_documents(
    documents: list[LoadedDocument],
) -> list[LoadedDocument]:
    """
    Apply the same cleaning pipeline to every document.
    """

    repeated_lines = find_repeated_lines(
        documents,
        minimum_occurrences=2,
    )

    cleaned_documents: list[LoadedDocument] = []

    for document in documents:

        cleaned_text = clean_text(
            document.text,
            repeated_boilerplate=repeated_lines,
        )

        cleaned_documents.append(
            LoadedDocument(
                source_id=document.source_id,
                text=cleaned_text,
            )
        )

    return cleaned_documents


# ============================================================
# SAMPLE CORPUS
# ============================================================

def collect_cleaning_paths() -> list[Path]:
    """
    Select supported sample documents for the cleaning demo.
    """

    return [
        SAMPLE_CORPUS_DIR / "clinical_trial_overview.txt",
        SAMPLE_CORPUS_DIR / "eligibility_criteria.md",
        SAMPLE_CORPUS_DIR / "study_export.html",
        SAMPLE_CORPUS_DIR / "noisy_clinical_report.txt",
    ]


# ============================================================
# REPORT
# ============================================================

def build_report(
    raw_documents: list[LoadedDocument],
    cleaned_documents: list[LoadedDocument],
) -> str:

    lines: list[str] = []

    lines.append(
        "PHARMALENS TEXT EXTRACTION & CLEANING PIPELINE"
    )
    lines.append("=" * 70)
    lines.append("")

    lines.append(
        f"Documents processed: {len(raw_documents)}"
    )
    lines.append("")

    lines.append(
        "Cleaning steps:"
    )
    lines.append(
        "1. Unicode NFKC normalization"
    )
    lines.append(
        "2. Line-ending normalization"
    )
    lines.append(
        "3. Page-number boilerplate removal"
    )
    lines.append(
        "4. Repeated-header/boilerplate removal"
    )
    lines.append(
        "5. Space and tab normalization"
    )
    lines.append(
        "6. Excessive blank-line removal"
    )
    lines.append("")

    lines.append("=" * 70)
    lines.append("BEFORE / AFTER COMPARISON")
    lines.append("=" * 70)
    lines.append("")

    for raw, cleaned in zip(
        raw_documents,
        cleaned_documents,
    ):

        lines.append(
            f"SOURCE: {raw.source_id}"
        )
        lines.append(
            "-" * 70
        )

        lines.append(
            f"BEFORE LENGTH: {len(raw.text)} characters"
        )
        lines.append(
            f"AFTER LENGTH : {len(cleaned.text)} characters"
        )

        lines.append("")

        lines.append(
            "BEFORE:"
        )
        lines.append(
            raw.text[:600]
        )

        lines.append("")

        lines.append(
            "AFTER:"
        )
        lines.append(
            cleaned.text[:600]
        )

        lines.append("")
        lines.append("=" * 70)
        lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    paths = collect_cleaning_paths()

    # --------------------------------------------------------
    # Extract/load raw documents using the existing loader.
    # --------------------------------------------------------

    raw_documents, skipped = load_documents(paths)

    if skipped:

        print("\nSkipped files:")

        for message in skipped:
            print(message)

    if not raw_documents:

        raise RuntimeError(
            "No documents were loaded for cleaning."
        )

    # --------------------------------------------------------
    # Apply the SAME cleaning pipeline to every document.
    # --------------------------------------------------------

    cleaned_documents = clean_documents(
        raw_documents
    )

    # --------------------------------------------------------
    # Generate before/after report.
    # --------------------------------------------------------

    report = build_report(
        raw_documents,
        cleaned_documents,
    )

    print("\n")
    print(report)

    # --------------------------------------------------------
    # Save sample output.
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
        f"Sample output saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()