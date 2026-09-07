"""
PharmaLens RAG Pipeline

Pipeline Flow:

User Question
    ->
Cache Check
    ->
Query Embedding
    ->
Qdrant Retrieval
    ->
Context Assembly
    ->
Grounded Answer Generation
    ->
Citation Mapping
    ->
Logging and Usage Monitoring
    ->
Return Answer
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from qdrant_client import QdrantClient

from src.config import load_settings


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

CACHE_FILE = OUTPUT_DIR / "rag_cache.json"

REQUEST_LOG_FILE = LOG_DIR / "rag_requests.jsonl"

USAGE_REPORT_FILE = OUTPUT_DIR / "usage_report.json"


# ============================================================
# CACHE CONFIGURATION
# ============================================================

CACHE_TTL_SECONDS = 15 * 60


# ============================================================
# APPROXIMATE COST CONFIGURATION
# ============================================================

MODEL_INPUT_COST_PER_1K = 0.00015

MODEL_OUTPUT_COST_PER_1K = 0.00060


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger("pharmalens_rag")

logger.setLevel(logging.INFO)

if not logger.handlers:

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)


# ============================================================
# SETTINGS
# ============================================================

def get_settings() -> dict[str, str | int]:
    """
    Load all settings required for the RAG pipeline.
    """

    return load_settings(
        require_chat=True,
        require_embedding=True,
        require_vector_db=True,
    )


# ============================================================
# CLIENTS
# ============================================================

def create_embedding_client(
    settings: dict[str, str | int],
) -> OpenAI:
    """
    Create OpenAI-compatible embedding client.
    """

    return OpenAI(
        api_key=str(settings["openai_api_key"]),
        base_url=str(settings["openai_base_url"]),
    )


def create_chat_client(
    settings: dict[str, str | int],
) -> OpenAI:
    """
    Create OpenAI-compatible chat client.
    """

    return OpenAI(
        api_key=str(settings["openai_api_key"]),
        base_url=str(settings["openai_base_url"]),
    )


def create_qdrant_client(
    settings: dict[str, str | int],
) -> QdrantClient:
    """
    Create Qdrant client.
    """

    return QdrantClient(
        url=str(settings["qdrant_url"])
    )


# ============================================================
# CACHE
# ============================================================

def load_cache() -> dict[str, Any]:
    """
    Load persistent query cache.
    """

    if not CACHE_FILE.exists():
        return {}

    try:

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception as error:

        logger.warning(
            "Could not load cache: %s",
            error,
        )

        return {}


def save_cache(
    cache: dict[str, Any],
) -> None:
    """
    Save persistent query cache.
    """

    try:

        with open(
            CACHE_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                cache,
                file,
                indent=2,
                ensure_ascii=False,
            )

    except Exception as error:

        logger.error(
            "Could not save cache: %s",
            error,
        )


def cache_key(
    question: str,
    k: int = 4,
    filters: Optional[dict[str, Any]] = None,
) -> str:
    """
    Create stable cache key.

    The question, retrieval count, and filters
    are included so different settings do not
    incorrectly share cached responses.
    """

    data = {
        "question": question.strip().lower(),
        "k": k,
        "filters": filters or {},
    }

    raw = json.dumps(
        data,
        sort_keys=True,
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def get_cached_answer(
    question: str,
    k: int = 4,
    filters: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """
    Return cached response if available
    and not expired.
    """

    cache = load_cache()

    key = cache_key(
        question,
        k,
        filters,
    )

    cached = cache.get(key)

    if not cached:
        return None

    created_at = cached.get(
        "created_at",
        0,
    )

    if (
        time.time() - created_at
        > CACHE_TTL_SECONDS
    ):

        logger.info(
            "Cache entry expired."
        )

        cache.pop(
            key,
            None,
        )

        save_cache(cache)

        return None

    logger.info(
        "Cache HIT."
    )

    return cached.get(
        "response"
    )


def save_cached_answer(
    question: str,
    response: dict[str, Any],
    k: int = 4,
    filters: Optional[dict[str, Any]] = None,
) -> None:
    """
    Save response to cache.
    """

    cache = load_cache()

    key = cache_key(
        question,
        k,
        filters,
    )

    cache[key] = {
        "created_at": time.time(),
        "response": response,
    }

    save_cache(cache)

    logger.info(
        "Response saved to cache."
    )


# ============================================================
# TOKEN ESTIMATION
# ============================================================

def estimate_tokens(
    text: str,
) -> int:
    """
    Approximate token count.

    Rough estimate:
    1 token ~= 4 characters.
    """

    if not text:
        return 0

    return max(
        1,
        len(text) // 4,
    )


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate approximate request cost.
    """

    input_cost = (
        input_tokens / 1000
    ) * MODEL_INPUT_COST_PER_1K

    output_cost = (
        output_tokens / 1000
    ) * MODEL_OUTPUT_COST_PER_1K

    return round(
        input_cost + output_cost,
        8,
    )


# ============================================================
# STRUCTURED LOGGING
# ============================================================

def write_request_log(
    record: dict[str, Any],
) -> None:
    """
    Write request record as JSONL.
    """

    try:

        with open(
            REQUEST_LOG_FILE,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    except Exception as error:

        logger.error(
            "Failed to write request log: %s",
            error,
        )


def log_rag_request(
    request_id: str,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    cache_hit: bool,
    input_tokens: int,
    output_tokens: int,
    estimated_cost: float,
    latency_ms: float,
    error: Optional[str] = None,
) -> None:
    """
    Create structured log record.
    """

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "request_id": request_id,

        "question": question,

        "answer_preview": answer[:300],

        "sources": sources,

        "cache_hit": cache_hit,

        "input_tokens": input_tokens,

        "output_tokens": output_tokens,

        "estimated_cost": estimated_cost,

        "latency_ms": round(
            latency_ms,
            2,
        ),

        "error": error,
    }

    write_request_log(record)

    logger.info(
        json.dumps(
            record,
            ensure_ascii=False,
        )
    )


# ============================================================
# USAGE REPORT
# ============================================================

def load_request_logs() -> list[dict[str, Any]]:
    """
    Load structured request logs.
    """

    if not REQUEST_LOG_FILE.exists():
        return []

    records: list[dict[str, Any]] = []

    try:

        with open(
            REQUEST_LOG_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    records.append(
                        json.loads(line)
                    )

                except json.JSONDecodeError:

                    continue

    except Exception as error:

        logger.error(
            "Could not read logs: %s",
            error,
        )

    return records


def summarize_usage(
    log_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate usage summary.
    """

    total_requests = len(log_records)

    if total_requests == 0:

        return {
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "total_requests": 0,

            "cache_hits": 0,

            "cache_misses": 0,

            "cache_hit_rate": 0.0,

            "total_input_tokens": 0,

            "total_output_tokens": 0,

            "total_estimated_cost": 0.0,

            "average_latency_ms": 0.0,

            "errors": 0,
        }

    cache_hits = sum(
        1
        for item in log_records
        if item.get("cache_hit")
    )

    total_input_tokens = sum(
        item.get(
            "input_tokens",
            0,
        )
        for item in log_records
    )

    total_output_tokens = sum(
        item.get(
            "output_tokens",
            0,
        )
        for item in log_records
    )

    total_cost = sum(
        item.get(
            "estimated_cost",
            0.0,
        )
        for item in log_records
    )

    total_latency = sum(
        item.get(
            "latency_ms",
            0.0,
        )
        for item in log_records
    )

    errors = sum(
        1
        for item in log_records
        if item.get("error")
    )

    return {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "total_requests": total_requests,

        "cache_hits": cache_hits,

        "cache_misses": (
            total_requests
            - cache_hits
        ),

        "cache_hit_rate": round(
            cache_hits / total_requests,
            2,
        ),

        "total_input_tokens": (
            total_input_tokens
        ),

        "total_output_tokens": (
            total_output_tokens
        ),

        "total_estimated_cost": round(
            total_cost,
            8,
        ),

        "average_latency_ms": round(
            total_latency
            / total_requests,
            2,
        ),

        "errors": errors,
    }


def generate_usage_report() -> dict[str, Any]:
    """
    Generate and save usage report.
    """

    logs = load_request_logs()

    report = summarize_usage(logs)

    with open(
        USAGE_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    return report


# ============================================================
# STAGE 1 - QUERY EMBEDDING
# ============================================================

def embed_query(
    query: str,
    embedding_client: OpenAI,
    embedding_model: str,
) -> list[float]:
    """
    Convert query into embedding vector.
    """

    logger.info(
        "Generating query embedding."
    )

    response = (
        embedding_client
        .embeddings
        .create(
            model=embedding_model,
            input=query,
        )
    )

    return (
        response
        .data[0]
        .embedding
    )


# ============================================================
# STAGE 2 - RETRIEVAL
# ============================================================

def retrieve_context(
    query_vector: list[float],
    qdrant_client: QdrantClient,
    collection_name: str,
    k: int = 4,
) -> list[Any]:
    """
    Retrieve top-k chunks from Qdrant.
    """

    logger.info(
        "Retrieving top %s chunks.",
        k,
    )

    results = (
        qdrant_client
        .query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
    )

    return results.points


# ============================================================
# CHUNK NORMALIZATION
# ============================================================

def normalize_chunk(
    point: Any,
) -> dict[str, Any]:
    """
    Convert Qdrant result into
    consistent chunk format.
    """

    payload = point.payload or {}

    metadata = payload.get(
        "metadata",
        {},
    )

    chunk_id = payload.get(
        "original_chunk_id",
        str(point.id),
    )

    return {
        "id": str(point.id),

        "chunk_id": chunk_id,

        "score": point.score,

        "text": payload.get(
            "text",
            "",
        ),

        "metadata": metadata,
    }


# ============================================================
# STAGE 3 - CONTEXT ASSEMBLY
# ============================================================

def assemble_context(
    chunks: list[dict[str, Any]],
) -> str:
    """
    Build context with citation markers.
    """

    parts: list[str] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        metadata = chunk.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "Unknown Source",
        )

        chunk_id = chunk.get(
            "chunk_id",
            chunk.get("id"),
        )

        text = chunk.get(
            "text",
            "",
        )

        parts.append(
            f"[{index}]\n"
            f"Source: {source}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Text: {text}"
        )

    return "\n\n".join(parts)


# ============================================================
# CITATION MAP
# ============================================================

def build_citation_map(
    chunks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Map citation markers to real metadata.
    """

    citation_map: dict[
        str,
        dict[str, Any]
    ] = {}

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        metadata = chunk.get(
            "metadata",
            {},
        )

        citation_map[
            f"[{index}]"
        ] = {

            "source": metadata.get(
                "source"
            ),

            "chunk_id": chunk.get(
                "chunk_id",
                chunk.get("id"),
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
                "text"
            ),
        }

    return citation_map


# ============================================================
# STAGE 4 - ANSWER GENERATION
# ============================================================

def generate_cited_answer(
    question: str,
    chunks: list[dict[str, Any]],
    chat_client: OpenAI,
    chat_model: str,
) -> str:
    """
    Generate grounded answer using
    only retrieved context.
    """

    context = assemble_context(chunks)

    prompt = f"""
You are PharmaLens, a clinical research intelligence assistant.

Answer the user's question using ONLY the retrieved context.

Rules:

1. Use only the information in the retrieved context.
2. Do not add outside knowledge.
3. Cite factual claims using [1], [2], etc.
4. Only use citation markers provided in the context.
5. Do not invent citations.
6. If the context is insufficient, say exactly:

"I don't have enough information in the retrieved context to answer this question."

Retrieved Context:

{context}

Question:

{question}

Answer:
"""

    logger.info(
        "Generating grounded answer."
    )

    response = (
        chat_client
        .chat
        .completions
        .create(
            model=chat_model,

            messages=[
                {
                    "role": "system",

                    "content": (
                        "You are a grounded RAG assistant. "
                        "Use only the provided context."
                    ),
                },

                {
                    "role": "user",

                    "content": prompt,
                },
            ],

            temperature=0,
        )
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    if not answer:

        return (
            "I don't have enough information "
            "in the retrieved context to answer "
            "this question."
        )

    return answer.strip()


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def extract_sources(
    citation_map: dict[
        str,
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    """
    Create simple source list.
    """

    sources: list[
        dict[str, Any]
    ] = []

    for marker, citation in (
        citation_map.items()
    ):

        sources.append(
            {
                "citation": marker,

                "source": citation.get(
                    "source"
                ),

                "chunk_id": citation.get(
                    "chunk_id"
                ),

                "chunk_index": citation.get(
                    "chunk_index"
                ),
            }
        )

    return sources


# ============================================================
# CITATION VERIFICATION
# ============================================================

def verify_citations(
    answer: str,
    citation_map: dict[
        str,
        dict[str, Any]
    ],
) -> dict[str, dict[str, Any]]:
    """
    Verify citations used in the answer.

    A citation is considered verified when:
    - The marker appears in the answer.
    - The marker exists in the citation map.
    - The mapped chunk contains original text.
    """

    verification: dict[
        str,
        dict[str, Any]
    ] = {}

    for marker, citation in (
        citation_map.items()
    ):

        used = marker in answer

        verification[marker] = {

            "used_in_answer": used,

            "source": citation.get(
                "source"
            ),

            "chunk_id": citation.get(
                "chunk_id"
            ),

            "verified": (
                used
                and bool(
                    citation.get("text")
                )
            ),
        }

    return verification


# ============================================================
# FULL RAG PIPELINE
# ============================================================

def answer_with_citations(
    question: str,
    k: int = 4,
    filters: Optional[
        dict[str, Any]
    ] = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Complete RAG pipeline.

    Flow:

    Question
        ->
    Cache
        ->
    Embed
        ->
    Retrieve
        ->
    Generate
        ->
    Citations
        ->
    Logging
        ->
    Return
    """

    request_id = str(
        uuid.uuid4()
    )

    start_time = time.perf_counter()

    try:

        # ====================================================
        # CACHE CHECK
        # ====================================================

        if use_cache:

            cached_response = (
                get_cached_answer(
                    question=question,
                    k=k,
                    filters=filters,
                )
            )

            if cached_response:

                latency_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000

                # Create a copy so the
                # stored cached response
                # is not permanently modified.
                result = json.loads(
                    json.dumps(
                        cached_response
                    )
                )

                result["request_id"] = (
                    request_id
                )

                result.setdefault(
                    "usage",
                    {}
                )

                result["usage"][
                    "cache_hit"
                ] = True

                result["usage"][
                    "latency_ms"
                ] = round(
                    latency_ms,
                    2,
                )

                log_rag_request(
                    request_id=request_id,

                    question=question,

                    answer=result.get(
                        "answer",
                        "",
                    ),

                    sources=result.get(
                        "sources",
                        [],
                    ),

                    cache_hit=True,

                    input_tokens=0,

                    output_tokens=0,

                    estimated_cost=0.0,

                    latency_ms=latency_ms,
                )

                return result


        # ====================================================
        # SETTINGS
        # ====================================================

        settings = get_settings()


        # ====================================================
        # CLIENTS
        # ====================================================

        embedding_client = (
            create_embedding_client(
                settings
            )
        )

        chat_client = (
            create_chat_client(
                settings
            )
        )

        qdrant_client = (
            create_qdrant_client(
                settings
            )
        )


        # ====================================================
        # EMBEDDING
        # ====================================================

        query_vector = (
            embed_query(
                query=question,

                embedding_client=(
                    embedding_client
                ),

                embedding_model=str(
                    settings["embed_model"]
                ),
            )
        )


        # ====================================================
        # RETRIEVAL
        # ====================================================

        retrieved_points = (
            retrieve_context(
                query_vector=query_vector,

                qdrant_client=(
                    qdrant_client
                ),

                collection_name=str(
                    settings[
                        "qdrant_collection"
                    ]
                ),

                k=k,
            )
        )


        # ====================================================
        # EMPTY RETRIEVAL FALLBACK
        # ====================================================

        if not retrieved_points:

            answer = (
                "I don't have enough information "
                "in the retrieved context to answer "
                "this question."
            )

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            result = {
                "request_id": request_id,

                "answer": answer,

                "citations": {},

                "citation_verification": {},

                "sources": [],

                "retrieved_chunks": [],

                "usage": {
                    "input_tokens": 0,

                    "output_tokens": 0,

                    "estimated_cost": 0.0,

                    "cache_hit": False,

                    "latency_ms": round(
                        latency_ms,
                        2,
                    ),
                },
            }

            log_rag_request(
                request_id=request_id,

                question=question,

                answer=answer,

                sources=[],

                cache_hit=False,

                input_tokens=0,

                output_tokens=0,

                estimated_cost=0.0,

                latency_ms=latency_ms,
            )

            return result


        # ====================================================
        # NORMALIZE CHUNKS
        # ====================================================

        chunks = [
            normalize_chunk(point)
            for point in retrieved_points
        ]


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        answer = (
            generate_cited_answer(
                question=question,

                chunks=chunks,

                chat_client=chat_client,

                chat_model=str(
                    settings["chat_model"]
                ),
            )
        )


        # ====================================================
        # CITATIONS
        # ====================================================

        citation_map = (
            build_citation_map(
                chunks
            )
        )

        sources = (
            extract_sources(
                citation_map
            )
        )

        citation_verification = (
            verify_citations(
                answer,
                citation_map,
            )
        )


        # ====================================================
        # TOKEN MONITORING
        # ====================================================

        context = assemble_context(
            chunks
        )

        input_tokens = (
            estimate_tokens(
                question
                + "\n"
                + context
            )
        )

        output_tokens = (
            estimate_tokens(
                answer
            )
        )

        estimated_cost = (
            estimate_cost(
                input_tokens,
                output_tokens,
            )
        )


        # ====================================================
        # LATENCY
        # ====================================================

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


        # ====================================================
        # FINAL RESULT
        # ====================================================

        result = {
            "request_id": request_id,

            "answer": answer,

            "citations": citation_map,

            "citation_verification": (
                citation_verification
            ),

            "sources": sources,

            "retrieved_chunks": chunks,

            "usage": {
                "input_tokens": (
                    input_tokens
                ),

                "output_tokens": (
                    output_tokens
                ),

                "estimated_cost": (
                    estimated_cost
                ),

                "cache_hit": False,

                "latency_ms": round(
                    latency_ms,
                    2,
                ),
            },
        }


        # ====================================================
        # SAVE CACHE
        # ====================================================

        if use_cache:

            save_cached_answer(
                question=question,

                response=result,

                k=k,

                filters=filters,
            )


        # ====================================================
        # LOG REQUEST
        # ====================================================

        log_rag_request(
            request_id=request_id,

            question=question,

            answer=answer,

            sources=sources,

            cache_hit=False,

            input_tokens=input_tokens,

            output_tokens=output_tokens,

            estimated_cost=estimated_cost,

            latency_ms=latency_ms,
        )


        return result


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.exception(
            "RAG pipeline failed."
        )

        fallback_answer = (
            "The RAG pipeline encountered an error "
            "while processing the request."
        )

        log_rag_request(
            request_id=request_id,

            question=question,

            answer=fallback_answer,

            sources=[],

            cache_hit=False,

            input_tokens=0,

            output_tokens=0,

            estimated_cost=0.0,

            latency_ms=latency_ms,

            error=str(error),
        )

        return {
            "request_id": request_id,

            "answer": fallback_answer,

            "citations": {},

            "citation_verification": {},

            "sources": [],

            "retrieved_chunks": [],

            "error": str(error),

            "usage": {
                "input_tokens": 0,

                "output_tokens": 0,

                "estimated_cost": 0.0,

                "cache_hit": False,

                "latency_ms": round(
                    latency_ms,
                    2,
                ),
            },
        }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result: dict[str, Any],
) -> None:
    """
    Print pipeline result.
    """

    print()

    print("=" * 70)

    print(
        "PHARMALENS RAG RESPONSE"
    )

    print("=" * 70)

    print()

    print("Request ID:")

    print(
        result.get(
            "request_id"
        )
    )

    print()

    print("Answer:")

    print(
        result.get(
            "answer"
        )
    )

    print()

    print("Sources:")

    sources = result.get(
        "sources",
        [],
    )

    if not sources:

        print(
            "No sources retrieved."
        )

    else:

        for source in sources:

            print(
                f"{source.get('citation')} "
                f"-> "
                f"{source.get('source')} "
                f"-> "
                f"{source.get('chunk_id')}"
            )

    print()

    print("Usage:")

    usage = result.get(
        "usage",
        {},
    )

    print(
        f"Cache Hit: "
        f"{usage.get('cache_hit')}"
    )

    print(
        f"Input Tokens: "
        f"{usage.get('input_tokens')}"
    )

    print(
        f"Output Tokens: "
        f"{usage.get('output_tokens')}"
    )

    print(
        f"Estimated Cost: "
        f"{usage.get('estimated_cost')}"
    )

    print(
        f"Latency: "
        f"{usage.get('latency_ms')} ms"
    )

    print()

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Demonstrate:

    1. Normal RAG request.
    2. Same request served from cache.
    3. Usage report generation.
    """

    print()

    print(
        "Running PharmaLens RAG pipeline..."
    )

    question = (
        "What did Study 001 evaluate?"
    )


    # ========================================================
    # FIRST REQUEST
    # ========================================================

    print()

    print(
        "FIRST REQUEST"
    )

    first_result = (
        answer_with_citations(
            question=question,

            k=4,

            use_cache=True,
        )
    )

    print_result(
        first_result
    )


    # ========================================================
    # SECOND REQUEST - CACHE TEST
    # ========================================================

    print()

    print(
        "SECOND REQUEST (CACHE TEST)"
    )

    second_result = (
        answer_with_citations(
            question=question,

            k=4,

            use_cache=True,
        )
    )

    print_result(
        second_result
    )


    # ========================================================
    # USAGE REPORT
    # ========================================================

    print()

    print(
        "GENERATING USAGE REPORT..."
    )

    report = (
        generate_usage_report()
    )

    print()

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()

    print(
        "Usage report saved to:"
    )

    print(
        USAGE_REPORT_FILE
    )


if __name__ == "__main__":
    main()