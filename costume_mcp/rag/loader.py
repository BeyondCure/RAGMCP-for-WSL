"""Document loading and chunking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from costume_mcp.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx"}

# Chinese-aware separators: prefer paragraph breaks, then sentence punctuation.
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


@dataclass(frozen=True)
class LoadedUnit:
    """A loadable slice of a document (e.g. one PDF page)."""

    text: str
    source: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_index: int
    page: int | None = None


def iter_source_files(paths: list[Path]) -> Iterator[Path]:
    """Yield supported files from files/directories, de-duplicated."""
    seen: set[Path] = set()
    for path in paths:
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(p for p in path.rglob("*") if p.is_file())
        else:
            logger.warning("Skipping missing path: %s", path)
            continue

        for candidate in candidates:
            if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def source_label(path: Path) -> str:
    """Stable, human-readable source identifier used in chunk metadata."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_units(path: Path) -> list[LoadedUnit]:
    suffix = path.suffix.lower()
    source = source_label(path)
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [LoadedUnit(text=text, source=source)] if text.strip() else []
    if suffix == ".pdf":
        return _load_pdf(path, source)
    if suffix == ".docx":
        return _load_docx(path, source)
    logger.warning("Unsupported file type: %s", path)
    return []


def _load_pdf(path: Path, source: str) -> list[LoadedUnit]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    units: list[LoadedUnit] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            units.append(LoadedUnit(text=text, source=source, page=page_number))
    return units


def _load_docx(path: Path, source: str) -> list[LoadedUnit]:
    from docx import Document

    document = Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [LoadedUnit(text=text, source=source)] if text.strip() else []


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
        length_function=len,
    )


def chunk_units(
    units: list[LoadedUnit],
    splitter: RecursiveCharacterTextSplitter,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for unit in units:
        pieces = [piece.strip() for piece in splitter.split_text(unit.text) if piece.strip()]
        for piece in pieces:
            chunks.append(
                Chunk(
                    text=piece,
                    source=unit.source,
                    chunk_index=len(chunks),
                    page=unit.page,
                )
            )
    return chunks
