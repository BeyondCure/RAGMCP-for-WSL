"""Optional cross-encoder reranking (disabled by default)."""

from __future__ import annotations

import logging
from functools import lru_cache

from costume_mcp.config import configure_huggingface, get_settings
from costume_mcp.rag.search import SearchHit

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512) -> None:
        configure_huggingface()
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker %s (device=%s)", model_name, device)
        try:
            self._model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=device,
                trust_remote_code=True,
            )
        except TypeError:
            # Older sentence-transformers releases lack trust_remote_code.
            self._model = CrossEncoder(model_name, max_length=max_length, device=device)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return hits
        scores = self._model.predict([(query, hit.text) for hit in hits])
        ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [
            SearchHit(
                text=hit.text,
                source=hit.source,
                page=hit.page,
                chunk_index=hit.chunk_index,
                score=float(score),
            )
            for hit, score in ranked
        ]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    settings = get_settings()
    return Reranker(
        settings.rerank_model,
        device=settings.embed_device,
        max_length=settings.rerank_max_length,
    )
