"""MCP server: exposes the local RAG index over the Model Context Protocol."""

from __future__ import annotations

import logging
import threading
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from costume_mcp import __version__
from costume_mcp.config import get_settings
from costume_mcp.rag.search import format_hits, search
from costume_mcp.rag.store import get_client

logger = logging.getLogger(__name__)

# sentence-transformers models are not designed for concurrent encode calls.
_SEARCH_LOCK = threading.Lock()

SERVER_INSTRUCTIONS = (
    "Local retrieval-augmented generation service over NeoForge 1.21.1 documentation. "
    "Use search_knowledge for questions about NeoForge APIs, registries, resources, and mod "
    "development. Results are markdown passages with a source file and heading path; cite both "
    "when answering. Use list_collections to discover available collections. The default "
    "collection is configured by MCP_DEFAULT_COLLECTION (neoforge). Re-indexing is performed "
    "offline with ingest.py and requires a server restart."
)


def _collection_names() -> list[str]:
    return [getattr(c, "name", str(c)) for c in get_client().list_collections()]


def _require_collection(name: str) -> None:
    names = _collection_names()
    if name not in names:
        available = ", ".join(sorted(names)) or "<none>"
        raise ToolError(
            f"Collection {name!r} not found. Available collections: {available}"
        )


def _collection_stats(collection: Any) -> tuple[int, int, int]:
    """Return (chunks, documents, chunks_with_heading)."""
    chunks = collection.count()
    if not chunks:
        return 0, 0, 0
    metadatas = collection.get(include=["metadatas"]).get("metadatas") or []
    sources: set[str] = set()
    with_heading = 0
    for metadata in metadatas:
        if not metadata:
            continue
        sources.add(str(metadata.get("source", "<unknown>")))
        if metadata.get("heading_path"):
            with_heading += 1
    return chunks, len(sources), with_heading


def create_server() -> MCPServer:
    settings = get_settings()
    default_collection = settings.mcp_default_collection
    default_top_k = settings.top_k
    max_top_k = settings.mcp_max_top_k
    max_result_chars = settings.mcp_max_result_chars

    server = MCPServer(
        name="costume-rag",
        title="CostumeMCP RAG",
        description="Local RAG over NeoForge 1.21.1 documentation",
        instructions=SERVER_INSTRUCTIONS,
        version=__version__,
        log_level="INFO",
    )

    @server.tool(
        name="search_knowledge",
        description=(
            "Search the local NeoForge 1.21.1 documentation index. Returns the most relevant "
            "documentation chunks with source file, heading path, and similarity score. Use this "
            "for questions about NeoForge APIs, blocks, items, registries, networking, resources, "
            "or mod development. Cite the returned source and heading path in your answer."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
        structured_output=False,
    )
    def search_knowledge(
        query: str,
        top_k: int = default_top_k,
        collection: str = default_collection,
    ) -> str:
        query = (query or "").strip()
        if not query:
            raise ToolError("query must be a non-empty string")

        collection = (collection or default_collection).strip()
        _require_collection(collection)

        requested_top_k = default_top_k if top_k is None else int(top_k)
        bounded_top_k = max(1, min(requested_top_k, max_top_k))

        try:
            with _SEARCH_LOCK:
                hits = search(query, top_k=bounded_top_k, collection_name=collection)
        except Exception as exc:  # noqa: BLE001 - surfaced to the MCP client
            logger.exception("Search failed for collection %s", collection)
            raise ToolError(f"Search failed: {exc}") from exc

        if not hits:
            return (
                f"No results in collection {collection!r} for query: {query}\n"
                f"The collection may be empty; run: python ingest.py <path> "
                f"--collection {collection}"
            )

        body = format_hits(hits)
        if max_result_chars > 0 and len(body) > max_result_chars:
            body = (
                body[:max_result_chars].rstrip()
                + f"\n\n[... truncated at {max_result_chars} characters ...]"
            )
        return f"collection: {collection} | {len(hits)} hit(s)\n\n{body}"

    @server.tool(
        name="list_collections",
        description=(
            "List the available local RAG collections with chunk counts, document counts, and "
            "heading-path coverage. Use this to discover which collection to pass to "
            "search_knowledge."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
        structured_output=False,
    )
    def list_collections() -> str:
        try:
            collections = get_client().list_collections()
        except Exception as exc:  # noqa: BLE001 - surfaced to the MCP client
            logger.exception("Failed to list collections")
            raise ToolError(f"Failed to list collections: {exc}") from exc

        if not collections:
            return "No collections found. Run `python ingest.py <path>` to build an index."

        lines = ["Collections:"]
        for collection in collections:
            name = getattr(collection, "name", str(collection))
            marker = " (default)" if name == default_collection else ""
            try:
                chunks, documents, with_heading = _collection_stats(collection)
            except Exception as exc:  # noqa: BLE001 - keep listing other collections
                lines.append(f"- {name}{marker}: unavailable ({exc})")
                continue
            lines.append(
                f"- {name}{marker}: {chunks} chunks, "
                f"{documents} document{'s' if documents != 1 else ''}, "
                f"{with_heading} with heading paths"
            )
        return "\n".join(lines)

    return server
