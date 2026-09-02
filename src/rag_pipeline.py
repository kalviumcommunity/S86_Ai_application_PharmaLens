from __future__ import annotations

from typing import Any

import tiktoken

try:
    from qdrant_client import QdrantClient
except ImportError:  # pragma: no cover - optional dependency for local vector retrieval
    QdrantClient = None

from src.config import load_settings
from src.llm_client import create_client


# ---------------------------------------------------------
# 1. LOAD SETTINGS
# ---------------------------------------------------------

settings = load_settings(
    require_chat=True,
    require_embedding=True,
    require_vector_db=True,
)

client = create_client(settings)

qdrant_client = (
    QdrantClient(url=settings["qdrant_url"]) if QdrantClient is not None else None
)

COLLECTION_NAME = settings["qdrant_collection"]
EMBED_MODEL = settings["embed_model"]
CHAT_MODEL = settings["chat_model"]


# ---------------------------------------------------------
# 2. QUERY EMBEDDING
# ---------------------------------------------------------

def embed_query(query: str) -> list[float]:
    """
    Convert the user's query into a vector embedding.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
    )

    # Gemini may return index=None, so we directly
    # use the first embedding instead of sorting by index.
    vector = response.data[0].embedding

    return vector


# ---------------------------------------------------------
# 3. RETRIEVAL
# ---------------------------------------------------------

def retrieve_chunks(
    query_vector: list[float],
    k: int = 3,
) -> list[dict[str, Any]]:
    """
    Search Qdrant for chunks similar to the query.

    We retrieve extra records because the current Qdrant
    collection also contains a test record from the
    vector database setup. Corpus records contain
    'original_chunk_id'.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    # Retrieve extra results so we can ignore non-corpus
    # test records if they appear in the results.
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

        # Corpus records contain original_chunk_id.
        # Ignore the temporary test record.
        original_chunk_id = payload.get("original_chunk_id")

        if not original_chunk_id:
            continue

        metadata = payload.get("metadata", {})
        text = payload.get("text", "")

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
# 4. CONTEXT ASSEMBLY
# ---------------------------------------------------------

MAX_CONTEXT_TOKENS = 5000


def count_tokens(text: str) -> int:
    """Estimate token usage using the cl100k_base tokenizer."""
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def format_chunk(index: int, chunk: dict[str, Any]) -> str:
    """Format a retrieved chunk with a source marker for citation."""
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "Unknown source")
    chunk_index = metadata.get("chunk_index")
    marker = f"[{index}] {source}#{chunk_index}"
    return f"{marker}\n{chunk.get('text', '')}"


def assemble_context(
    chunks: list[dict[str, Any]],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> tuple[str, int]:
    """
    Convert retrieved chunks into a structured context
    that can be passed to the language model while
    staying within a token budget.
    """
    if not chunks:
        return "", 0

    selected: list[str] = []
    used_tokens = 0

    for index, chunk in enumerate(chunks, start=1):
        formatted = format_chunk(index, chunk)
        token_count = count_tokens(formatted)

        if used_tokens + token_count > max_tokens:
            break

        selected.append(formatted)
        used_tokens += token_count

    context = "\n\n---\n\n".join(selected)
    return context, used_tokens


def build_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> dict[str, Any]:
    """Build a grounded prompt with token-budgeted context and source markers."""
    context, context_tokens = assemble_context(retrieved_chunks, max_tokens=max_tokens)

    prompt = f"""
You are a grounded assistant.
Answer the question using only the provided context.
If the answer is not in the context, say: "I don't have enough information in the provided context."
When possible, cite sources using the markers like [1] or [2].

Context:
{context}

Question:
{question}
"""

    return {
        "prompt": prompt,
        "context_tokens": context_tokens,
        "sources_used": [chunk.get("metadata", {}) for chunk in retrieved_chunks],
    }


# ---------------------------------------------------------
# 5. GROUNDED GENERATION
# ---------------------------------------------------------

def generate_answer(
    query: str,
    context: str,
) -> str:
    """
    Generate an answer using only the retrieved context.
    """

    if not context.strip():
        return (
            "I could not find relevant information in the "
            "available clinical research documents."
        )

    prompt = f"""
You are PharmaLens, a clinical research intelligence assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the context does not contain enough information,
   clearly say that the available context is insufficient.
4. Keep the answer concise and factual.
5. Mention the source document used when possible.

Context:
{context}

Question:
{query}

Answer:
"""

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
        return "The model did not return an answer."

    return answer.strip()


# ---------------------------------------------------------
# 6. COMPLETE RAG PIPELINE
# ---------------------------------------------------------

def generate_ungrounded_answer(query: str) -> str:
    """
    Generate an answer using only the model's internal knowledge,
    without any retrieved context. Used for comparison with grounded answers.
    """
    prompt = f"""
Answer this question based on your general knowledge:

{query}
"""

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
        return "The model did not return an answer."

    return answer.strip()


def answer_query(
    query: str,
    k: int = 3,
) -> dict[str, Any]:
    """
    Run the complete RAG pipeline:

    Query
       ↓
    Query Embedding
       ↓
    Retrieval
       ↓
    Context Assembly
       ↓
    Grounded Generation
       ↓
    Answer + Sources
    """

    # Stage 1: Query Embedding
    query_vector = embed_query(query)

    # Stage 2: Retrieval
    chunks = retrieve_chunks(
        query_vector=query_vector,
        k=k,
    )

    # Stage 3: Context Assembly
    prompt_payload = build_prompt(query, chunks)
    context = prompt_payload["prompt"]

    # Stage 4: Grounded Generation
    answer = generate_answer(
        query=query,
        context=context,
    )

    # Collect unique source documents
    sources = []

    for chunk in chunks:
        source = chunk.get("metadata", {}).get(
            "source",
            "Unknown source",
        )

        if source not in sources:
            sources.append(source)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunks,
        "context": context,
        "embedding_dimension": len(query_vector),
    }


# ---------------------------------------------------------
# 7. DISPLAY PIPELINE RESULT
# ---------------------------------------------------------

def print_pipeline_result(
    result: dict[str, Any],
) -> None:
    """
    Print the complete RAG pipeline execution result.
    """

    print()
    print("=" * 70)
    print("PHARMALENS RAG PIPELINE")
    print("=" * 70)

    # Stage 1
    print()
    print("Stage 1: Query Embedding")
    print("-" * 70)

    print(
        f"Embedding model: {EMBED_MODEL}"
    )

    print(
        f"Vector dimension: "
        f"{result['embedding_dimension']}"
    )

    print("Status: SUCCESS")

    # Stage 2
    print()
    print("Stage 2: Retrieval")
    print("-" * 70)

    chunks = result["retrieved_chunks"]

    print(
        f"Retrieved chunks: {len(chunks)}"
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        source = chunk.get(
            "metadata",
            {},
        ).get(
            "source",
            "Unknown source",
        )

        print()
        print(f"Result {index}")
        print(f"Chunk ID: {chunk['id']}")
        print(f"Score: {chunk['score']:.4f}")
        print(f"Source: {source}")

    # Stage 3
    print()
    print("Stage 3: Context Assembly")
    print("-" * 70)

    print(
        f"Context chunks: {len(chunks)}"
    )

    print("Status: SUCCESS")

    # Stage 4
    print()
    print("Stage 4: Grounded Generation")
    print("-" * 70)

    print(
        f"Chat model: {CHAT_MODEL}"
    )

    print("Status: SUCCESS")

    # Final answer
    print()
    print("Final RAG Result")
    print("-" * 70)

    print()
    print("Question:")
    print(result["query"])

    print()
    print("Answer:")
    print(result["answer"])

    print()
    print("Sources:")

    for source in result["sources"]:
        print(f"- {source}")

    print()
    print("=" * 70)


# ---------------------------------------------------------
# 8. END-TO-END DEMO
# ---------------------------------------------------------

def main() -> None:
    """
    Run a sample end-to-end RAG query.
    """

    query = "What did Study 001 evaluate?"

    print()
    print("Starting PharmaLens RAG pipeline...")
    print(f"Query: {query}")

    result = answer_query(
        query=query,
        k=3,
    )

    print_pipeline_result(result)


# ---------------------------------------------------------
# 9. PROGRAM ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()