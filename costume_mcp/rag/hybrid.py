"""Lexical BM25 retrieval and reciprocal-rank fusion."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from functools import lru_cache

from rank_bm25 import BM25Okapi

from costume_mcp.rag.search import SearchHit
from costume_mcp.rag.store import get_collection

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_#.]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATOR_RE = re.compile(r"[_#.]+")


def tokenize(text: str) -> list[str]:
    """Code-aware tokenizer: keeps identifiers and also emits camelCase/subwords."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        for chunk in _SEPARATOR_RE.split(raw):
            if not chunk:
                continue
            tokens.append(chunk.lower())
            parts = _CAMEL_RE.sub(" ", chunk).split()
            if len(parts) > 1:
                tokens.extend(part.lower() for part in parts if part)
    return tokens


class BM25Index:
    """In-memory BM25 index built from one Chroma collection."""

    def __init__(self, collection_name: str) -> None:
        collection = get_collection(collection_name)
        data = collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        self.collection_name = collection_name
        self._entries: list[SearchHit] = []
        tokenized_docs: list[list[str]] = []

        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            self._entries.append(
                SearchHit(
                    text=document or "",
                    source=str(metadata.get("source", "unknown")),
                    score=0.0,
                    page=metadata.get("page"),
                    chunk_index=metadata.get("chunk_index"),
                    heading_path=(
                        str(metadata["heading_path"]) if metadata.get("heading_path") else None
                    ),
                    chunk_id=chunk_id,
                    score_kind="bm25",
                )
            )
            tokenized_docs.append(tokenize(document or "") or ["__empty__"])

        self._bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None

    def search(self, query: str, top_n: int) -> list[SearchHit]:
        if self._bm25 is None or top_n <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            range(len(scores)), key=lambda index: float(scores[index]), reverse=True
        )[:top_n]
        return [
            replace(self._entries[index], score=float(scores[index]))
            for index in ranked
            if float(scores[index]) > 0
        ]


@lru_cache(maxsize=8)
def get_bm25_index(collection_name: str) -> BM25Index:
    logger.info("Building BM25 index for collection %s", collection_name)
    return BM25Index(collection_name)


def rrf_fuse(hit_lists: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    """Reciprocal Rank Fusion, deduplicated by chunk id."""
    fused: dict[str, tuple[SearchHit, float]] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            key = hit.chunk_id or f"{hit.source}:{hit.chunk_index}:{hash(hit.text)}"
            if key not in fused:
                fused[key] = (hit, 0.0)
            base, score = fused[key]
            fused[key] = (base, score + 1.0 / (k + rank))

    ranked = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)
    return [
        replace(hit, score=score, score_kind="rrf")
        for hit, score in ranked
    ]
