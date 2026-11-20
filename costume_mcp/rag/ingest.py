"""Ingestion pipeline: load -> chunk -> embed -> upsert into Chroma."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from costume_mcp.config import get_settings
from costume_mcp.rag.embedding import get_embedder
from costume_mcp.rag.loader import (
    MARKDOWN_SUFFIXES,
    Chunk,
    chunk_markdown,
    chunk_units,
    iter_source_files,
    load_units,
    make_splitter,
    source_label,
)
from costume_mcp.rag.store import get_collection

logger = logging.getLogger(__name__)


def _chunk_id(chunk: Chunk) -> str:
    key = f"{chunk.source}:{chunk.chunk_index}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _metadata(chunk: Chunk) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
    }
    if chunk.page is not None:
        metadata["page"] = chunk.page
    if chunk.heading_path:
        metadata["heading_path"] = chunk.heading_path
    return metadata


def ingest_paths(
    paths: list[Path],
    collection_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, int]:
    """Ingest files/directories into a Chroma collection.

    Each document is re-indexed atomically from the caller's perspective:
    previous chunks for the same source are deleted before the new ones are
    upserted, so edits and shrinking documents do not leave stale chunks.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    splitter = make_splitter(chunk_size, chunk_overlap)

    embedder = get_embedder()
    collection = get_collection(collection_name)

    stats = {"files": 0, "chunks": 0}
    for path in iter_source_files(paths):
        if path.suffix.lower() in MARKDOWN_SUFFIXES:
            source = source_label(path)
            chunks = chunk_markdown(
                path.read_text(encoding="utf-8", errors="ignore"),
                source,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=settings.min_chunk_size,
                max_chunk_chars=settings.max_chunk_chars,
            )
        else:
            units = load_units(path)
            chunks = chunk_units(units, splitter)
        if not chunks:
            logger.warning("No text extracted from %s", path)
            continue

        source = chunks[0].source
        # Remove the previous version of this document (also stale chunks).
        collection.delete(where={"source": source})

        texts = [chunk.text for chunk in chunks]
        # The heading breadcrumb is embedded together with the body so that
        # section titles contribute to retrieval, but only the body is stored.
        embed_texts = [
            f"{chunk.heading_path}\n\n{chunk.text}" if chunk.heading_path else chunk.text
            for chunk in chunks
        ]
        vectors = embedder.embed_documents(embed_texts)
        collection.upsert(
            ids=[_chunk_id(chunk) for chunk in chunks],
            documents=texts,
            embeddings=vectors,
            metadatas=[_metadata(chunk) for chunk in chunks],
        )

        stats["files"] += 1
        stats["chunks"] += len(chunks)
        logger.info("Ingested %s (%d chunks)", source, len(chunks))

    return stats
