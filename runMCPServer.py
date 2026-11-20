#!/usr/bin/env python3
"""Start the CostumeMCP MCP server.

Examples:
    python runMCPServer.py              # HTTP on 127.0.0.1:8000/mcp (resident)
    python runMCPServer.py --stdio      # stdio transport (integration testing)
    python runMCPServer.py --no-preload # skip embedding-model warmup
"""

from __future__ import annotations

import argparse
import logging
import sys

from costume_mcp.config import Settings, get_settings
from costume_mcp.mcp_server import create_server
from costume_mcp.rag.embedding import get_embedder
from costume_mcp.rag.store import get_client

logger = logging.getLogger("costume_mcp.server")

VALID_TRANSPORTS = {"stdio", "streamable-http"}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    # Keep noisy third-party INFO/WARNING logs out of the way.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def preload(settings: Settings) -> None:
    """Fail fast if the index is unusable, then warm up the embedding model."""
    names = [getattr(c, "name", str(c)) for c in get_client().list_collections()]
    if settings.mcp_default_collection not in names:
        available = ", ".join(sorted(names)) or "<none>"
        raise RuntimeError(
            f"Default collection {settings.mcp_default_collection!r} not found. "
            f"Available collections: {available}"
        )

    logger.info("Loading embedding model %s ...", settings.embed_model)
    get_embedder().embed_query("warmup")
    logger.info("Embedding model ready (default collection: %s)", settings.mcp_default_collection)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run over stdio instead of HTTP (testing only)",
    )
    parser.add_argument(
        "--transport",
        choices=sorted(VALID_TRANSPORTS),
        default=None,
        help=f"Transport (default: {settings.mcp_transport})",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip embedding-model warmup",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    args = build_parser().parse_args(argv)
    configure_logging()

    if args.stdio and args.transport and args.transport != "stdio":
        logger.error("--stdio conflicts with --transport %s", args.transport)
        return 2

    transport = "stdio" if args.stdio else (args.transport or settings.mcp_transport)
    if transport not in VALID_TRANSPORTS:
        logger.error(
            "Unsupported transport %r; use one of: %s",
            transport,
            ", ".join(sorted(VALID_TRANSPORTS)),
        )
        return 2

    if transport == "streamable-http" and settings.mcp_preload and not args.no_preload:
        try:
            preload(settings)
        except Exception:  # noqa: BLE001 - startup must fail fast
            logger.exception("Preload failed; refusing to start")
            return 1
    elif transport == "stdio":
        logger.info("stdio mode: embedding model will load on the first search")
    else:
        logger.info("Preload disabled: embedding model will load on the first search")

    server = create_server()

    if transport == "stdio":
        logger.info("Starting MCP server over stdio")
        server.run(transport="stdio")
    else:
        logger.info(
            "Starting MCP server at http://%s:%s/mcp",
            settings.mcp_host,
            settings.mcp_port,
        )
        server.run(
            transport="streamable-http",
            host=settings.mcp_host,
            port=settings.mcp_port,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
