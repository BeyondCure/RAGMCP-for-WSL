"""Semantic retrieval over the Chroma index."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from costume_mcp.config import get_settings
from costume_mcp.rag.embedding import get_embedder
from costume_mcp.rag.store import get_collection

logger = logging.getLogger(__name__)

_SCORE_LABELS = {
    "dense": "score",
    "bm25": "bm25_score",
    "rrf": "fusion_score",
    "rerank": "rerank_score",
}


@dataclass(frozen=True)
class SearchHit:
    text: str
    source: str
    score: float
    page: int | None = None
    chunk_index: int | None = None
    heading_path: str | None = None
    chunk_id: str | None = None
    score_kind: str = "dense"


def _dense_search(collection: Any, query: str, n_results: int) -> list[SearchHit]:
    query_vector = get_embedder().embed_query(query)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[SearchHit] = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        hits.append(
            SearchHit(
                text=document or "",
                source=str(metadata.get("source", "unknown")),
                page=metadata.get("page"),
                chunk_index=metadata.get("chunk_index"),
                heading_path=str(metadata["heading_path"]) if metadata.get("heading_path") else None,
                chunk_id=chunk_id,
                score_kind="dense",
                score=1.0 - float(distance),
            )
        )
    return hits


def search(
    query: str,
    top_k: int | None = None,
    collection_name: str | None = None,
) -> list[SearchHit]:
    """Return the most relevant chunks for *query*.

    Dense retrieval uses cosine similarity (score = 1 - distance). When hybrid
    retrieval is enabled, dense and BM25 results are fused with RRF before the
    optional cross-encoder rerank.
    """
    settings = get_settings()
    top_k = top_k or settings.top_k
    collection_name = collection_name or settings.collection
    collection = get_collection(collection_name)

    if settings.hybrid_enabled:
        dense_hits = _dense_search(
            collection,
            query,
            max(top_k, settings.hybrid_dense_n),
        )
        try:
            from costume_mcp.rag.hybrid import get_bm25_index, rrf_fuse

            bm25_hits = get_bm25_index(collection_name).search(
                query,
                top_n=max(top_k, settings.hybrid_bm25_n),
            )
            hits = rrf_fuse([dense_hits, bm25_hits], k=settings.hybrid_rrf_k)
            hits = hits[: max(top_k, settings.hybrid_rerank_n)]
        except Exception:  # noqa: BLE001 - hybrid retrieval is optional
            logger.warning("Hybrid retrieval failed; using dense results", exc_info=True)
            hits = dense_hits[: max(top_k, settings.hybrid_rerank_n)]
    else:
        fetch_n = max(top_k, settings.recall_top_n) if settings.rerank_enabled else top_k
        hits = _dense_search(collection, query, fetch_n)

    if settings.rerank_enabled:
        from costume_mcp.rag.rerank import get_reranker

        try:
            hits = get_reranker().rerank(query, hits)
        except Exception:  # noqa: BLE001 - reranking must not break retrieval
            logger.warning("Reranking failed; returning retrieval results", exc_info=True)

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
        label = _SCORE_LABELS.get(hit.score_kind, "score")
        blocks.append(
            f"[{index}] {label}={hit.score:.3f}  source={location}\n{hit.text.strip()}"
        )
    return "\n\n---\n\n".join(blocks)
