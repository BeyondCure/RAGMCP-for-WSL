"""Text embeddings backed by a sentence-transformers model."""

from __future__ import annotations

import logging
from functools import lru_cache

from costume_mcp.config import configure_huggingface, get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """Thin wrapper around a sentence-transformers model.

    Loading is lazy: constructing this class pulls the model into memory
    (downloading it on first use). Vectors are always L2-normalized so the
    cosine distance configured on the Chroma collection stays consistent.
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 8) -> None:
        configure_huggingface()
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s (device=%s)", model_name, device)
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    settings = get_settings()
    return Embedder(
        settings.embed_model,
        device=settings.embed_device,
        batch_size=settings.embed_batch_size,
    )
