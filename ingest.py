#!/usr/bin/env python3
"""Ingest local documents into the Chroma index.

Examples:
    python ingest.py data/raw
    python ingest.py data/raw/handbook.pdf --collection costume
    python ingest.py data/raw --reset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from costume_mcp.config import get_settings
from costume_mcp.rag.ingest import ingest_paths
from costume_mcp.rag.store import collection_stats, reset_collection


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to ingest",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=f"Collection name (default: {settings.collection})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Chunk size in characters (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help=f"Chunk overlap in characters (default: {settings.chunk_overlap})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the collection before ingesting",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = build_parser().parse_args(argv)

    if args.reset:
        reset_collection(args.collection)

    stats = ingest_paths(
        args.paths,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    summary = collection_stats(args.collection)
    logging.getLogger(__name__).info(
        "Done: %d files, %d chunks this run; collection %r now has %d chunks",
        stats["files"],
        stats["chunks"],
        summary["collection"],
        summary["chunks"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
