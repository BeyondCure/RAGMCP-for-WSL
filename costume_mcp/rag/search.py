"""Semantic retrieval over the Chroma index."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from costume_mcp.config import get_settings
from costume_mcp.rag.embedding import get_embedder
from costume_mcp.rag.store import get_collection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    text: str
    source: str
    score: float
    page: int | None = None
    chunk_index: int | None = None
    heading_path: str | None = None


def search(
    query: str,
    top_k: int | None = None,
    collection_name: str | None = None,
) -> list[SearchHit]:
    """Return the most relevant chunks for *query*.

    Uses dense retrieval with cosine distance (similarity = 1 - distance).
    If reranking is enabled, more candidates are fetched and reordered by a
    cross-encoder before truncating to ``top_k``.
    """
    settings = get_settings()
    top_k = top_k or settings.top_k
    fetch_n = max(top_k, settings.recall_top_n) if settings.rerank_enabled else top_k

    collection = get_collection(collection_name)
    query_vector = get_embedder().embed_query(query)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[SearchHit] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        hits.append(
            SearchHit(
                text=document or "",
                source=str(metadata.get("source", "unknown")),
                page=metadata.get("page"),
                chunk_index=metadata.get("chunk_index"),
                heading_path=str(metadata["heading_path"]) if metadata.get("heading_path") else None,
                score=1.0 - float(distance),
            )
        )

    if settings.rerank_enabled:
        from costume_mcp.rag.rerank import get_reranker

        hits = get_reranker().rerank(query, hits)

    return hits[:top_k]


def format_hits(hits: list[SearchHit]) -> str:
    """Render hits as compact markdown for an MCP tool response."""
    if not hits:
        return "未找到相关内容。"

    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        location = hit.source
        if hit.heading_path:
            location += f" › {hit.heading_path}"
        if hit.page is not None:
            location += f" · p.{hit.page}"
        blocks.append(
            f"[{index}] score={hit.score:.3f}  source={location}\n{hit.text.strip()}"
        )
    return "\n\n---\n\n".join(blocks)
