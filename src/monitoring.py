"""
PharmaLens RAG Caching, Logging and Usage Monitoring
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import uuid

from src.monitoring import (
    get_cached_response,
    save_cached_response,
    log_rag_request,
    log_rag_error,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

REQUEST_LOG_FILE = OUTPUT_DIR / "rag_requests.jsonl"

CACHE_TTL_SECONDS = 15 * 60

query_cache = {}


# Approximate cost configuration.
# Update these values based on the model/provider pricing if needed.
MODEL_INPUT_COST_PER_1K = 0.00015
MODEL_OUTPUT_COST_PER_1K = 0.00060


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger("pharmalens_monitoring")

if not logger.handlers:

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)


# ============================================================
# CACHE FUNCTIONS
# ============================================================

def create_cache_key(question, filters=None, k=4):
    """
    Create a stable cache key.

    The key includes:
    - normalized question
    - metadata filters
    - retrieval k value

    This ensures that different retrieval settings
    do not incorrectly share cached responses.
    """

    raw_data = {
        "question": question.strip().lower(),
        "filters": filters or {},
        "k": k,
    }

    raw_string = json.dumps(
        raw_data,
        sort_keys=True
    )

    return hashlib.sha256(
        raw_string.encode("utf-8")
    ).hexdigest()


def get_cached_response(question, filters=None, k=4):
    """
    Return a cached response if available and not expired.
    """

    key = create_cache_key(
        question,
        filters,
        k,
    )

    cached = query_cache.get(key)

    if cached is None:
        return None

    created_at = cached["created_at"]

    age = time.time() - created_at

    if age > CACHE_TTL_SECONDS:

        query_cache.pop(key, None)

        return None

    return cached["response"]


def save_cached_response(question, response, filters=None, k=4):
    """
    Save a response in the cache.
    """

    key = create_cache_key(
        question,
        filters,
        k,
    )

    query_cache[key] = {
        "created_at": time.time(),
        "response": response,
    }


def clear_cache():
    """
    Clear all cached responses.
    """

    query_cache.clear()


def cache_size():
    """
    Return the number of cached entries.
    """

    return len(query_cache)


# ============================================================
# TOKEN ESTIMATION
# ============================================================

def estimate_tokens(text):
    """
    Rough token estimation.

    Approximation:
    1 token ≈ 4 characters.

    This is useful when exact provider token usage
    is unavailable.
    """

    if not text:
        return 0

    return max(
        1,
        len(text) // 4
    )


# ============================================================
# COST ESTIMATION
# ============================================================

def estimate_cost(
    input_tokens,
    output_tokens,
):
    """
    Estimate generation cost.
    """

    input_cost = (
        input_tokens / 1000
    ) * MODEL_INPUT_COST_PER_1K

    output_cost = (
        output_tokens / 1000
    ) * MODEL_OUTPUT_COST_PER_1K

    total_cost = (
        input_cost +
        output_cost
    )

    return round(
        total_cost,
        8
    )


# ============================================================
# SOURCE FORMATTING
# ============================================================

def extract_sources(response):
    """
    Extract simplified source information
    from the RAG response.
    """

    sources = []

    citations = response.get(
        "citations",
        {}
    )

    for marker, citation in citations.items():

        source_data = {
            "citation": marker,
            "source": citation.get("source"),
            "chunk_id": citation.get("chunk_id"),
            "chunk_index": citation.get("chunk_index"),
            "section": citation.get("section"),
            "page": citation.get("page"),
        }

        sources.append(
            source_data
        )

    return sources


# ============================================================
# STRUCTURED REQUEST LOGGING
# ============================================================

def write_request_log(record):
    """
    Write a structured JSON log record.

    JSONL format:
    one JSON object per line.
    """

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


def log_rag_request(
    request_id,
    question,
    response,
    cache_hit,
    latency_ms,
    error=None,
):
    """
    Log a complete RAG request.
    """

    answer = response.get(
        "answer",
        ""
    ) if response else ""

    sources = (
        extract_sources(response)
        if response
        else []
    )

    input_tokens = estimate_tokens(
        question
    )

    output_tokens = estimate_tokens(
        answer
    )

    estimated_cost = estimate_cost(
        input_tokens,
        output_tokens,
    )

    record = {
        "timestamp": datetime.now().isoformat(),
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

    write_request_log(
        record
    )

    logger.info(
        json.dumps(
            record,
            ensure_ascii=False,
        )
    )

    return record


# ============================================================
# ERROR LOGGING
# ============================================================

def log_rag_error(
    request_id,
    question,
    error_message,
    latency_ms,
):
    """
    Log a failed RAG request.
    """

    record = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "question": question,
        "answer_preview": "",
        "sources": [],
        "cache_hit": False,
        "input_tokens": estimate_tokens(
            question
        ),
        "output_tokens": 0,
        "estimated_cost": 0,
        "latency_ms": round(
            latency_ms,
            2,
        ),
        "error": error_message,
    }

    write_request_log(
        record
    )

    logger.error(
        json.dumps(
            record,
            ensure_ascii=False,
        )
    )

    return record