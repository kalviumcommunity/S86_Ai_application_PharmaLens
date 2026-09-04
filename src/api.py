"""FastAPI backend for the PharmaLens RAG service.

Exposes two endpoints:
  POST /query  — accepts a question, returns a grounded answer with sources
  GET  /health — liveness check for infrastructure / load-balancers

Run with:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_settings
from src.rag_pipeline import answer_with_citations


# ---------------------------------------------------------------------------
# Lifespan — validate config before accepting traffic
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail fast at startup if required environment variables are missing."""
    load_settings(
        require_chat=True,
        require_embedding=True,
        require_vector_db=True,
    )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PharmaLens RAG API",
    description=(
        "Retrieval-augmented generation over pharmaceutical research documents. "
        "Submit a question; receive a grounded answer with source citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming question payload."""

    question: str = Field(
        min_length=3,
        max_length=1000,
        description="The question to answer using the RAG pipeline.",
    )


class Source(BaseModel):
    """A single cited source returned with the answer."""

    source: str = Field(description="File or document name the chunk came from.")
    chunk_id: str | None = Field(default=None, description="Unique chunk identifier.")
    score: float | None = Field(default=None, description="Relevance score (0–1).")


class QueryResponse(BaseModel):
    """Structured response from the RAG pipeline."""

    answer: str = Field(description="Grounded answer to the question.")
    sources: list[Source] = Field(description="Cited sources used to build the answer.")
    status: str = Field(
        description=(
            "Pipeline status: 'answered', 'no_context', or 'insufficient_relevance'."
        )
    )


class HealthResponse(BaseModel):
    """Liveness response."""

    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_status(retrieval_quality: dict[str, Any]) -> str:
    """Convert the pipeline's retrieval_quality reason to a response status."""
    reason = retrieval_quality.get("reason", "")
    if reason == "sufficient_relevance":
        return "answered"
    # Pass through 'no_context' and 'insufficient_relevance' as-is.
    return reason or "unknown"


def _build_sources(citations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert the citation map from the pipeline into the Source list shape.

    Only verified / grounded citations are surfaced; each entry already
    carries a source name, chunk_id, and (optionally) a relevance score.
    """
    sources = []
    for citation in citations.values():
        sources.append(
            {
                "source": citation.get("source", ""),
                "chunk_id": citation.get("chunk_id"),
                # score lives on retrieved_chunks, not on citation_map;
                # include it if present, otherwise leave as None.
                "score": citation.get("score"),
            }
        )
    return sources


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check. Returns 200 immediately with no external I/O."""
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Submit a question to the PharmaLens RAG pipeline.

    The pipeline:
    1. Embeds the question.
    2. Retrieves the most relevant chunks from Qdrant.
    3. Assesses retrieval quality (hallucination guard).
    4. Generates a grounded, citation-marked answer via the LLM.
    5. Verifies that every citation maps to a real source.

    When retrieval quality is insufficient the pipeline returns a safe
    fallback answer; this endpoint surfaces that as status 'no_context'
    or 'insufficient_relevance' rather than an error.
    """
    try:
        result = answer_with_citations(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="RAG service failed") from exc

    retrieval_quality: dict[str, Any] = result.get("retrieval_quality", {})
    status = _map_status(retrieval_quality)

    # Use verified_citations when available; fall back to the full citation map
    # so sources are always populated on a successful answer.
    citations: dict[str, dict[str, Any]] = (
        result.get("verified_citations")
        or result.get("citations")
        or {}
    )

    # Enrich citations with scores from retrieved_chunks where possible.
    chunk_score_map: dict[str, float] = {
        chunk.get("id", ""): chunk.get("score", 0.0)
        for chunk in result.get("retrieved_chunks", [])
    }
    for citation in citations.values():
        chunk_id = citation.get("chunk_id", "")
        if chunk_id and "score" not in citation:
            score = chunk_score_map.get(chunk_id)
            if score is not None:
                citation["score"] = score

    sources = _build_sources(citations)

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        status=status,
    )
