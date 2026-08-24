from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from src.corpus_loader_demo import LoadedDocument, load_documents
from src.text_cleaning import clean_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "chunking_comparison.md"
SAMPLE_CORPUS_DIR = PROJECT_ROOT / "data" / "sample_corpus"


@dataclass(frozen=True)
class Chunk:
    """A retrievable text unit whose metadata can identify its source."""

    source_id: str
    chunk_id: int
    text: str
    metadata: dict[str, str | int | None]


def tag_chunks(
    source: str,
    chunks: list[tuple[str, int]],
    *,
    section: str | None = None,
    page: int | None = None,
) -> list[Chunk]:
    """Attach consistent source metadata to text and its document offset."""
    tagged: list[Chunk] = []

    for chunk_index, (text, char_start) in enumerate(chunks, start=1):
        tagged.append(
            Chunk(
                source_id=source,
                chunk_id=chunk_index,
                text=text,
                metadata={
                    "source": source,
                    "chunk_index": chunk_index,
                    "char_start": char_start,
                    "char_end": char_start + len(text),
                    "section": section,
                    "page": page,
                },
            )
        )

    return tagged


def paragraph_chunks(
    document: LoadedDocument,
    max_characters: int = 220,
) -> list[Chunk]:
    """Pack complete paragraphs together without crossing the size limit."""
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", document.text)
        if paragraph.strip()
    ]

    chunks: list[tuple[str, int]] = []
    current = ""
    current_start = 0
    paragraph_start = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_characters:
            if current:
                chunks.append((current, current_start))
                current = ""
            chunks.extend(
                (paragraph[index : index + max_characters], paragraph_start + index)
                for index in range(0, len(paragraph), max_characters)
            )
        elif not current:
            current = paragraph
            current_start = paragraph_start
        elif len(current) + 2 + len(paragraph) <= max_characters:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append((current, current_start))
            current = paragraph
            current_start = paragraph_start

        paragraph_start += len(paragraph) + 2

    if current:
        chunks.append((current, current_start))

    return tag_chunks(document.source_id, chunks)


def fixed_size_chunks(
    document: LoadedDocument,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    """Split text into fixed character windows with deterministic overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = re.sub(r"\s+", " ", document.text).strip()
    step = chunk_size - overlap
    chunks = [
        (text[index : index + chunk_size], index)
        for index in range(0, len(text), step)
        if text[index : index + chunk_size]
    ]
    return tag_chunks(document.source_id, chunks)


def average_chunk_size(chunks: list[Chunk]) -> float:
    """Return the mean chunk length in characters."""
    return sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0.0


def build_report(document: LoadedDocument) -> str:
    """Compare both strategies on one cleaned document and show boundaries."""
    strategies = {
        "Paragraph-aware (max 220 characters)": paragraph_chunks(document),
        "Fixed-size (180 characters, 30-character overlap)": fixed_size_chunks(
            document
        ),
    }

    lines = [
        "# Chunking Strategy Comparison",
        "",
        f"Comparison document: `{document.source_id}` (cleaned text)",
        "",
        "| Strategy | Chunk count | Average size (characters) |",
        "| --- | ---: | ---: |",
    ]

    for name, chunks in strategies.items():
        lines.append(f"| {name} | {len(chunks)} | {average_chunk_size(chunks):.1f} |")

    lines.extend(
        [
            "",
            "## Choice",
            "",
            "Paragraph-aware chunks are the recommended strategy for this corpus. The source files contain short, semantically complete clinical paragraphs and eligibility bullets, so keeping those units intact gives retrieval useful context and avoids splitting a finding or criterion mid-sentence. The fixed-size baseline provides predictable capacity and overlap, but its samples split sentences and can duplicate fragments across retrieval results. The 220-character limit keeps chunks small enough for economical embedding while allowing related sentences to stay together.",
            "",
            "## Sample Chunks",
            "",
            "Samples use the first three chunks from each strategy so reviewers can inspect boundaries.",
        ]
    )

    for name, chunks in strategies.items():
        lines.extend(["", f"### {name}", ""])
        for chunk in chunks[:3]:
            lines.extend(
                [
                    f"**{chunk.source_id} / chunk {chunk.chunk_id}** ({len(chunk.text)} characters)",
                    "",
                    f"> {chunk.text.replace(chr(10), ' ')}",
                    "",
                    f"Metadata: `{chunk.metadata}`",
                    "",
                ]
            )

    traced_chunk = strategies["Paragraph-aware (max 220 characters)"][0]
    source_link = f"../data/sample_corpus/{traced_chunk.metadata['source']}"
    lines.extend(
        [
            "## Traceback Example",
            "",
            "A retrieved chunk keeps its source metadata, so the answer layer can cite the exact document and character range:",
            "",
            f"Retrieved chunk: `{traced_chunk.metadata['chunk_index']}`",
            f"Source: [{traced_chunk.metadata['source']}]({source_link})",
            f"Location: characters {traced_chunk.metadata['char_start']}-{traced_chunk.metadata['char_end']}",
            f"Text: > {traced_chunk.text}",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    paths = [SAMPLE_CORPUS_DIR / "noisy_clinical_report.txt"]
    raw_documents, skipped = load_documents(paths)
    if skipped or not raw_documents:
        raise RuntimeError("Unable to load the comparison document: " + "; ".join(skipped))

    cleaned_document = clean_documents(raw_documents)[0]
    report = build_report(cleaned_document)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved comparison report to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()