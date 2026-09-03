from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from src.config import load_settings
from src.llm_client import create_client


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

settings = load_settings(
    require_chat=True,
    require_embedding=True,
    require_vector_db=True,
)

client = create_client(settings)

qdrant_client = QdrantClient(
    url=settings["qdrant_url"]
)

COLLECTION_NAME = settings["qdrant_collection"]
EMBED_MODEL = settings["embed_model"]
CHAT_MODEL = settings["chat_model"]


# ---------------------------------------------------------
# 1. QUERY EMBEDDING
# ---------------------------------------------------------

def embed_query(query: str) -> list[float]:
    """
    Convert the user's query into an embedding vector.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
    )

    return response.data[0].embedding


# ---------------------------------------------------------
# 2. RETRIEVAL
# ---------------------------------------------------------

def retrieve_chunks(
    query_vector: list[float],
    k: int = 3,
) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant corpus chunks from Qdrant.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    search_limit = max(k * 3, k)

    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=search_limit,
        with_payload=True,
        with_vectors=False,
    )

    chunks = []

    for point in response.points:

        payload = point.payload or {}

        # Ignore temporary test records.
        original_chunk_id = payload.get(
            "original_chunk_id"
        )

        if not original_chunk_id:
            continue

        metadata = payload.get(
            "metadata",
            {}
        )

        text = payload.get(
            "text",
            ""
        )

        chunks.append(
            {
                "id": original_chunk_id,
                "score": point.score,
                "text": text,
                "metadata": metadata,
            }
        )

        if len(chunks) >= k:
            break

    return chunks


# ---------------------------------------------------------
# 3. CONTEXT ASSEMBLY
# ---------------------------------------------------------

def assemble_context(
    chunks: list[dict[str, Any]],
) -> tuple[str, dict[str, dict[str, Any]]]:
    """
    Build context for the LLM and create a citation map.

    Example:

    [1] Source: Study_001_Clinical_Report.pdf
    Chunk ID: study-001-chunk-001
    ...

    Returns:
        context
        citation_map
    """

    if not chunks:
        return "", {}

    parts = []
    citation_map = {}

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        marker = f"[{index}]"

        metadata = chunk.get(
            "metadata",
            {}
        )

        source = metadata.get(
            "source",
            "Unknown source"
        )

        chunk_id = metadata.get(
            "chunk_id",
            chunk.get("id")
        )

        chunk_index = metadata.get(
            "chunk_index"
        )

        section = metadata.get(
            "section"
        )

        page = metadata.get(
            "page"
        )

        text = chunk.get(
            "text",
            ""
        )

        # Context sent to the model
        parts.append(
            f"{marker} Source: {source}\n"
            f"Chunk ID: {chunk_id}\n"
            f"{text}"
        )

        # Citation information returned to user
        citation_map[marker] = {
            "source": source,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "section": section,
            "page": page,
            "text": text,
        }

    context = "\n\n".join(parts)

    return context, citation_map


# ---------------------------------------------------------
# 4. CITATION MAP
# ---------------------------------------------------------

def build_citation_map(
    chunks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Create a stable citation marker for every retrieved chunk.

    Example:

    {
        "[1]": {
            "source": "...",
            "chunk_id": "...",
            "chunk_index": 1,
            "section": "...",
            "page": 1,
            "text": "..."
        }
    }
    """

    citation_map = {}

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        marker = f"[{index}]"

        metadata = chunk.get(
            "metadata",
            {}
        )

        citation_map[marker] = {
            "source": metadata.get(
                "source",
                "Unknown source"
            ),

            "chunk_id": metadata.get(
                "chunk_id",
                chunk.get("id")
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

            "text": chunk.get(
                "text",
                ""
            ),
        }

    return citation_map


# ---------------------------------------------------------
# 5. CITED PROMPT
# ---------------------------------------------------------

def build_cited_prompt(
    question: str,
    chunks: list[dict[str, Any]],
) -> str:
    """
    Build a prompt that forces the model to use only
    retrieved sources and valid citation markers.
    """

    context, citation_map = assemble_context(
        chunks
    )

    if not citation_map:
        return ""

    available_markers = ", ".join(
        citation_map.keys()
    )

    return f"""
You are PharmaLens, a clinical research intelligence assistant.

Answer the user's question using ONLY the provided context.

Citation rules:
1. Cite every factual claim using a citation marker such as [1] or [2].
2. Only use citation markers that are provided in the context.
3. Available citation markers are: {available_markers}
4. Never create or invent citation markers.
5. Never cite a source that is not provided in the context.
6. If the context does not contain enough information to answer,
   say that you do not have enough information.
7. Do not use outside knowledge.
8. Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""


# ---------------------------------------------------------
# 6. GROUNDED GENERATION WITH CITATIONS
# ---------------------------------------------------------

def generate_cited_answer(
    question: str,
    chunks: list[dict[str, Any]],
) -> str:
    """
    Generate an answer containing citations.
    """

    prompt = build_cited_prompt(
        question,
        chunks,
    )

    if not prompt:
        return (
            "I don't have enough information "
            "in the provided context."
        )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        return (
            "The model did not return an answer."
        )

    return answer.strip()


# ---------------------------------------------------------
# 7. VERIFY CITATION
# ---------------------------------------------------------

def verify_citation(
    citation_marker: str,
    citation_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Verify that a citation exists and maps to real
    retrieved source text.
    """

    citation = citation_map.get(
        citation_marker
    )

    if not citation:
        return {
            "valid": False,
            "reason": "Citation marker does not exist.",
        }

    source = citation.get(
        "source"
    )

    text = citation.get(
        "text"
    )

    if not source:
        return {
            "valid": False,
            "reason": "Citation has no source.",
        }

    if not text:
        return {
            "valid": False,
            "reason": "Citation has no source text.",
        }

    return {
        "valid": True,
        "source": source,
        "chunk_id": citation.get(
            "chunk_id"
        ),
        "chunk_index": citation.get(
            "chunk_index"
        ),
        "section": citation.get(
            "section"
        ),
        "page": citation.get(
            "page"
        ),
        "text": text,
    }


# ---------------------------------------------------------
# 8. EXTRACT CITATION MARKERS
# ---------------------------------------------------------

def extract_citation_markers(
    answer: str,
) -> list[str]:
    """
    Extract citation markers such as [1], [2], [3]
    from the generated answer.
    """

    import re

    markers = re.findall(
        r"\[\d+\]",
        answer,
    )

    # Remove duplicates while preserving order.
    return list(
        dict.fromkeys(markers)
    )


# ---------------------------------------------------------
# 9. COMPLETE CITATION PIPELINE
# ---------------------------------------------------------

def answer_with_citations(
    question: str,
    k: int = 4,
) -> dict[str, Any]:
    """
    Complete citation-aware RAG pipeline.

    Flow:

    Question
        ↓
    Query Embedding
        ↓
    Retrieval
        ↓
    Citation Mapping
        ↓
    Grounded Generation
        ↓
    Citation Verification
        ↓
    Answer + Citations
    """

    # Stage 1
    query_vector = embed_query(
        question
    )

    # Stage 2
    chunks = retrieve_chunks(
        query_vector,
        k=k,
    )

    # No-source fallback
    if not chunks:
        return {
            "answer": (
                "I don't have enough information "
                "in the provided context."
            ),
            "citations": {},
            "verified_citations": {},
            "retrieved_chunks": [],
        }

    # Stage 3
    citation_map = build_citation_map(
        chunks
    )

    # Stage 4
    answer = generate_cited_answer(
        question,
        chunks,
    )

    # Stage 5
    markers = extract_citation_markers(
        answer
    )

    verified_citations = {}

    for marker in markers:

        verification = verify_citation(
            marker,
            citation_map,
        )

        if verification["valid"]:
            verified_citations[
                marker
            ] = verification

    return {
        "question": question,
        "answer": answer,
        "citations": citation_map,
        "verified_citations": verified_citations,
        "retrieved_chunks": chunks,
    }


# ---------------------------------------------------------
# 10. DISPLAY RESULT
# ---------------------------------------------------------

def print_citation_result(
    result: dict[str, Any],
) -> None:
    """
    Display the citation-aware answer and
    citation-to-source mappings.
    """

    print()
    print("=" * 70)
    print("PHARMALENS SOURCE CITATION PIPELINE")
    print("=" * 70)

    print()
    print("Question:")
    print(result.get("question"))

    print()
    print("Generated Answer:")
    print(result["answer"])

    print()
    print("Citation Mapping:")
    print("-" * 70)

    citations = result.get(
        "citations",
        {}
    )

    for marker, citation in citations.items():

        print()
        print(marker)

        print(
            f"Source: "
            f"{citation['source']}"
        )

        print(
            f"Chunk ID: "
            f"{citation['chunk_id']}"
        )

        print(
            f"Chunk Index: "
            f"{citation['chunk_index']}"
        )

        print(
            f"Section: "
            f"{citation['section']}"
        )

        print(
            f"Page: "
            f"{citation['page']}"
        )

        print(
            f"Original Text: "
            f"{citation['text']}"
        )

    print()
    print("Citation Verification:")
    print("-" * 70)

    verified = result.get(
        "verified_citations",
        {}
    )

    if not verified:
        print(
            "No valid citations were found "
            "in the generated answer."
        )
    else:
        for marker, citation in verified.items():

            print(
                f"{marker} -> "
                f"{citation['source']} "
                f"-> {citation['chunk_id']} "
                f"-> VERIFIED"
            )

    print()
    print("=" * 70)


# ---------------------------------------------------------
# 11. DEMO
# ---------------------------------------------------------

def main() -> None:

    question = (
        "What did Study 001 evaluate?"
    )

    print()
    print(
        "Running PharmaLens citation pipeline..."
    )

    result = answer_with_citations(
        question,
        k=4,
    )

    print_citation_result(
        result
    )


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()