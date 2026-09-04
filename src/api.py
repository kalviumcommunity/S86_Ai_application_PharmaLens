"""FastAPI backend for the PharmaLens RAG service.

Exposes three endpoints:
  POST /query      — accepts a question, returns a grounded answer with sources
  POST /documents  — uploads a document, ingests + embeds + indexes it at runtime
  GET  /health     — liveness check for infrastructure / load-balancers

Run with:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.config import load_settings
from src.corpus_ingestion import process_document
from src.rag_pipeline import answer_with_citations, client, embed_query, COLLECTION_NAME
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


# ---------------------------------------------------------------------------
# Upload config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"

# Must match what corpus_loader_demo.to_plain_text() supports.
SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm"}

# 10 MB hard limit per upload.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


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


class UploadSummary(BaseModel):
    """Per-document ingestion statistics."""

    document: str = Field(description="Path where the file was stored.")
    chunks_produced: int = Field(description="Number of chunks created from the document.")
    chunks_indexed: int = Field(description="Number of chunks successfully indexed into Qdrant.")


class UploadResponse(BaseModel):
    """Response from the document upload endpoint."""

    status: str = Field(description="'indexed' on success, 'failed' on error.")
    filename: str = Field(description="Original uploaded filename.")
    summary: UploadSummary


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


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def _safe_filename(original: str) -> str:
    """
    Strip directory components and append a short unique suffix so
    two uploads of the same filename never clobber each other.
    """
    stem = Path(original).stem
    suffix = Path(original).suffix.lower()
    uid = uuid.uuid4().hex[:8]
    # Keep only alphanumeric, dash, and underscore in the stem.
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return f"{safe_stem}_{uid}{suffix}"


def _embed_and_index_chunks(
    tagged_chunks: list[dict[str, Any]],
    settings: dict[str, Any],
) -> int:
    """
    Embed each chunk and upsert into Qdrant.

    Returns the number of successfully indexed chunks.
    """
    from src.indexing import make_qdrant_id

    qdrant = QdrantClient(url=settings["qdrant_url"])
    indexed = 0

    for chunk in tagged_chunks:
        text = chunk["text"]
        metadata = chunk["metadata"]

        # Build a stable chunk ID from source name + index.
        chunk_id = (
            f"{metadata.get('source', 'upload')}"
            f":{metadata.get('chunk_index', 0)}"
        )

        try:
            vector = embed_query(text)
        except Exception as exc:
            # Log and skip individual chunks that fail to embed.
            print(f"[upload] embed failed for {chunk_id}: {exc}")
            continue

        point = PointStruct(
            id=make_qdrant_id(chunk_id),
            vector=vector,
            payload={
                "original_chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "source": metadata.get("source"),
                    "chunk_index": metadata.get("chunk_index"),
                    "chunk_number": metadata.get("chunk_number"),
                    "total_chunks": metadata.get("total_chunks"),
                    "token_count": metadata.get("token_count"),
                },
            },
        )

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[point],
            wait=True,
        )
        indexed += 1

    return indexed


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@app.post("/documents", response_model=UploadResponse, tags=["rag"], status_code=201)
async def upload_document(
    file: UploadFile = File(..., description="Document to ingest (.txt, .md, .html)"),
) -> UploadResponse:
    """
    Upload a document and make it searchable immediately.

    Pipeline:
    1. Validate file type and size.
    2. Store the file under uploads/.
    3. Load, clean, and chunk using the same corpus ingestion pipeline.
    4. Embed each chunk and upsert into Qdrant.

    After a successful upload, POST /query can retrieve content from
    the new document without restarting the server.
    """
    # -- Validate extension --------------------------------------------------
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # -- Read bytes (enforces size limit) ------------------------------------
    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB limit.",
        )

    # -- Store file ----------------------------------------------------------
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(file.filename or "upload.txt")
    dest_path = UPLOAD_DIR / safe_name
    dest_path.write_bytes(raw_bytes)

    # -- Ingest: load → clean → chunk → metadata ----------------------------
    try:
        tagged_chunks, _token_count = process_document(dest_path)
    except ValueError as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc

    # -- Embed + index -------------------------------------------------------
    try:
        settings = load_settings(
            require_chat=False,
            require_embedding=True,
            require_vector_db=True,
        )
        indexed = _embed_and_index_chunks(tagged_chunks, settings)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document indexing failed.",
        ) from exc

    return UploadResponse(
        status="indexed",
        filename=file.filename or safe_name,
        summary=UploadSummary(
            document=str(dest_path.relative_to(PROJECT_ROOT)),
            chunks_produced=len(tagged_chunks),
            chunks_indexed=indexed,
        ),
    )
