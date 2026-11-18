"""Chroma persistent vector store helpers."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from costume_mcp.config import get_settings, resolve_path

logger = logging.getLogger(__name__)

# Cosine distance: similarity = 1 - distance. Embeddings are normalized.
_LEGACY_COLLECTION_METADATA = {"hnsw:space": "cosine"}
_COLLECTION_CONFIGURATION = {"hnsw": {"space": "cosine"}}


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    path = resolve_path(get_settings().chroma_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger.info("Opening Chroma at %s", path)
    return chromadb.PersistentClient(
        path=str(path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(name: str | None = None) -> Any:
    """Return (creating if needed) the named collection with cosine distance."""
    name = name or get_settings().collection
    client = get_client()
    try:
        return client.get_or_create_collection(
            name=name,
            configuration=_COLLECTION_CONFIGURATION,
        )
    except (TypeError, ValueError):
        # Older Chroma releases only accept the legacy metadata form.
        return client.get_or_create_collection(
            name=name,
            metadata=_LEGACY_COLLECTION_METADATA,
        )


def reset_collection(name: str | None = None) -> None:
    """Drop a collection if it exists; safe to call before re-indexing."""
    name = name or get_settings().collection
    try:
        get_client().delete_collection(name)
        logger.info("Deleted collection %s", name)
    except Exception as exc:  # noqa: BLE001 - Chroma raises different types per version
        logger.info("Collection %s was not deleted (%s)", name, exc)


def collection_stats(name: str | None = None) -> dict[str, Any]:
    name = name or get_settings().collection
    return {"collection": name, "chunks": get_collection(name).count()}
