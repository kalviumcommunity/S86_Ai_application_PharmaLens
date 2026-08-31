from __future__ import annotations

import os

from dotenv import load_dotenv


def load_settings(
    require_chat: bool = True,
    require_embedding: bool = False,
    require_vector_db: bool = False,
 feature/indexing-embeddings
) -> dict[str, str]
) -> dict:
  main
    """
    Load application settings from the .env file.

    Args:
        require_chat: Require CHAT_MODEL.
        require_embedding: Require EMBED_MODEL.
        require_vector_db: Require Qdrant configuration.

    Returns:
 feature/indexing-embeddings
        A dictionary containing LLM, embedding, and
        optional vector database configuration.

        A dictionary containing application configuration.
 main
    """

    load_dotenv()

    vector_dimension_raw = os.getenv(
        "VECTOR_DIMENSION",
        "1536",
    ).strip()

    try:
        vector_dimension = int(vector_dimension_raw)
    except ValueError:
        raise ValueError(
            "VECTOR_DIMENSION must be a valid integer."
        )

    settings = {
        "openai_base_url": os.getenv(
            "OPENAI_BASE_URL",
 feature/indexing-embeddings
            ""

            "",
 main
        ).strip(),

        "openai_api_key": os.getenv(
            "OPENAI_API_KEY",
 feature/indexing-embeddings
            ""

            "",
 main
        ).strip(),

        "chat_model": os.getenv(
            "CHAT_MODEL",
 feature/indexing-embeddings
            ""

            "",
 main
        ).strip(),

        "embed_model": os.getenv(
            "EMBED_MODEL",
 feature/indexing-embeddings
            ""

            "",
 main
        ).strip(),

        "qdrant_url": os.getenv(
            "QDRANT_URL",
 feature/indexing-embeddings
            "http://localhost:6333"

            "http://localhost:6333",
 main
        ).strip(),

        "qdrant_collection": os.getenv(
            "QDRANT_COLLECTION",
 feature/indexing-embeddings
            "rag_chunks"
        ).strip(),

        "vector_dimension": os.getenv(
            "VECTOR_DIMENSION",
            "3072"
        ).strip(),

            "rag_chunks",
        ).strip(),

        "vector_dimension": vector_dimension,
 main
    }

    required_settings = {
        "OPENAI_API_KEY": settings[
            "openai_api_key"
        ],
    }

    if require_chat:
        required_settings[
            "CHAT_MODEL"
        ] = settings["chat_model"]

    if require_embedding:
        required_settings[
            "EMBED_MODEL"
        ] = settings["embed_model"]

    if require_vector_db:
        required_settings[
            "QDRANT_URL"
        ] = settings["qdrant_url"]

        required_settings[
            "QDRANT_COLLECTION"
        ] = settings["qdrant_collection"]

        required_settings[
            "VECTOR_DIMENSION"
        ] = settings["vector_dimension"]

    if require_vector_db:
        required_settings["QDRANT_URL"] = settings["qdrant_url"]
        required_settings["QDRANT_COLLECTION"] = (
            settings["qdrant_collection"]
        )

    missing_settings = [
        name
        for name, value in required_settings.items()
        if not value
    ]

    if missing_settings:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing_settings)
        )

    if settings["vector_dimension"] <= 0:
        raise ValueError(
            "VECTOR_DIMENSION must be greater than 0."
        )

    return settings