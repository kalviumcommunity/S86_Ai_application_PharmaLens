from __future__ import annotations

import os

from dotenv import load_dotenv


def load_settings(
    require_chat: bool = True,
    require_embedding: bool = False,
    require_vector_db: bool = False,
) -> dict[str, str | int]:
    """Load application settings from the .env file."""

    """
    Load application settings from the .env file.

    Args:
        require_chat: Require CHAT_MODEL.
        require_embedding: Require EMBED_MODEL.
        require_vector_db: Require Qdrant configuration.

    Returns:
        A dictionary containing LLM, embedding, and optional vector database
        configuration.
    """
 

    load_dotenv()

    vector_dimension_raw = os.getenv("VECTOR_DIMENSION", "3072").strip()

    try:
        vector_dimension = int(vector_dimension_raw)
    except ValueError as exc:
        raise ValueError("VECTOR_DIMENSION must be a valid integer.") from exc

    settings: dict[str, str | int] = {
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "").strip(),
        "openai_api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        "chat_model": os.getenv("CHAT_MODEL", "").strip(),
        "embed_model": os.getenv("EMBED_MODEL", "").strip(),
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333").strip(),
        "qdrant_collection": os.getenv("QDRANT_COLLECTION", "rag_chunks").strip(),
        "vector_dimension": vector_dimension,
    }

    required_settings: dict[str, str | int] = {
        "OPENAI_API_KEY": settings["openai_api_key"],
    }

    if require_chat:
        required_settings["CHAT_MODEL"] = settings["chat_model"]

    if require_embedding:
        required_settings["EMBED_MODEL"] = settings["embed_model"]

    if require_vector_db:
        required_settings["QDRANT_URL"] = settings["qdrant_url"]
        required_settings["QDRANT_COLLECTION"] = settings["qdrant_collection"]
        required_settings["VECTOR_DIMENSION"] = settings["vector_dimension"]

    missing_settings = [
        name for name, value in required_settings.items() if not value
    ]

    if missing_settings:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing_settings)
        )

    if settings["vector_dimension"] <= 0:
        raise ValueError("VECTOR_DIMENSION must be greater than 0.")

    return settings