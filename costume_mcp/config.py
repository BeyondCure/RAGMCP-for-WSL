"""Central configuration, loaded from environment variables / .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    chroma_dir: Path = PROJECT_ROOT / "data" / "index"
    collection: str = "default"

    # Embedding
    embed_model: str = "BAAI/bge-m3"
    embed_device: str = "cpu"
    embed_batch_size: int = 8
    
    # Use the local HuggingFace cache only, skipping Hub metadata requests.
    # Set to false when you need to download a model for the first time.
    hf_hub_offline: bool = True

    # Chunking (characters)
    chunk_size: int = 700
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 5
    recall_top_n: int = 20
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_length: int = 512

    # MCP server transport
    mcp_transport: str = "stdio"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def resolve_path(path: Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_huggingface() -> None:
    """Put huggingface_hub into offline mode before it is imported.

    Loading a cached model by repo ID still triggers Hub metadata requests
    (commit checks and per-module configs). When the network is unreachable
    those requests retry with exponential backoff. ``HF_HUB_OFFLINE``
    short-circuits all of them and uses the local cache immediately.
    """
    offline = "1" if get_settings().hf_hub_offline else "0"
    os.environ["HF_HUB_OFFLINE"] = offline
    os.environ["TRANSFORMERS_OFFLINE"] = offline
