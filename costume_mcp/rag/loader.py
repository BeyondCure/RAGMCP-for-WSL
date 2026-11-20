"""Document loading and chunking.

Markdown files are chunked structurally:

* sections follow H1-H3 headings and carry a breadcrumb path;
* fenced code blocks, Docusaurus ``:::`` containers, JSX ``<Tabs>`` blocks
  and tables are kept atomic and never split in the middle;
* small sibling sections are merged when they share a parent and fit.

PDF/DOCX/TXT fall back to recursive character splitting.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from costume_mcp.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

MARKDOWN_SUFFIXES = {".md", ".markdown"}
TEXT_SUFFIXES = {".txt"} | MARKDOWN_SUFFIXES
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx"}

# Used for non-markdown formats and for oversized prose blocks.
TEXT_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
]

MD_MAX_HEADING_LEVEL = 3
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_CONTAINER_OPEN_RE = re.compile(r"^:::[A-Za-z][\w-]*")
_FENCE_PREFIXES = ("```", "~~~")


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
    heading_path: str | None = None


@dataclass
class _Section:
    path: tuple[str, ...]
    text: str


@dataclass
class _Piece:
    path: tuple[str, ...]
    text: str


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
    """Load a non-markdown document as page/paragraph units."""
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
        separators=TEXT_SEPARATORS,
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


# ---------------------------------------------------------------------------
# Markdown-aware chunking
# ---------------------------------------------------------------------------


def _strip_front_matter(text: str) -> str:
    if text.startswith("﻿"):
        text = text[1:]
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return text
    for index in range(1, min(len(lines), 80)):
        if lines[index].strip() == "---":
            return "".join(lines[index + 1 :]).lstrip("\n")
    return text


def _has_body(text: str) -> bool:
    """True if the text contains anything besides headings and blank lines."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADING_RE.match(stripped):
            continue
        return True
    return False


def _split_sections(text: str) -> list[_Section]:
    """Split markdown into H1-H3 sections, ignoring headings in protected regions."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    path: tuple[str, ...] = ()
    buffer: list[str] = []

    in_code = False
    fence = ""
    container_depth = 0
    in_tabs = False

    def flush() -> None:
        nonlocal buffer
        body = "".join(buffer).strip("\n")
        buffer = []
        if body.strip() and _has_body(body):
            sections.append(_Section(path=path, text=body))

    for line in text.splitlines(keepends=True):
        stripped = line.strip()

        if in_code:
            buffer.append(line)
            if stripped.startswith(fence):
                in_code = False
                fence = ""
            continue

        if in_tabs:
            buffer.append(line)
            if "</Tabs>" in line:
                in_tabs = False
            continue

        if container_depth > 0:
            buffer.append(line)
            if stripped.startswith(_FENCE_PREFIXES):
                in_code = True
                fence = stripped[:3]
            elif _CONTAINER_OPEN_RE.match(stripped):
                container_depth += 1
            elif stripped == ":::":
                container_depth -= 1
            continue

        if stripped.startswith(_FENCE_PREFIXES):
            in_code = True
            fence = stripped[:3]
            buffer.append(line)
            continue

        if _CONTAINER_OPEN_RE.match(stripped):
            container_depth = 1
            buffer.append(line)
            continue

        if "<Tabs" in line:
            in_tabs = True
            buffer.append(line)
            continue

        match = _HEADING_RE.match(stripped)
        if match and len(match.group(1)) <= MD_MAX_HEADING_LEVEL:
            if stack:
                flush()
            # Content before the first heading belongs to that first section,
            # so preamble (e.g. imports) is intentionally not flushed here.
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = [(lvl, name) for lvl, name in stack if lvl < level]
            stack.append((level, title))
            path = tuple(name for _, name in stack)
            buffer.append(line)
            continue

        buffer.append(line)

    flush()
    return sections


def _split_blocks(text: str) -> list[tuple[str, bool]]:
    """Split section text into markdown blocks.

    The boolean marks protected blocks (code fences, containers, Tabs,
    tables) which must stay atomic.
    """
    lines = text.splitlines(keepends=True)
    blocks: list[tuple[str, bool]] = []
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            block = "".join(buffer)
            if block.strip():
                blocks.append((block, False))
            buffer = []

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith(_FENCE_PREFIXES):
            flush()
            marker = stripped[:3]
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if lines[index].strip().startswith(marker):
                    index += 1
                    break
                index += 1
            blocks.append(("".join(block), True))
            continue

        if _CONTAINER_OPEN_RE.match(stripped):
            flush()
            block = [line]
            depth = 1
            index += 1
            while index < len(lines) and depth > 0:
                candidate = lines[index]
                candidate_stripped = candidate.strip()
                block.append(candidate)
                if _CONTAINER_OPEN_RE.match(candidate_stripped):
                    depth += 1
                elif candidate_stripped == ":::":
                    depth -= 1
                index += 1
            blocks.append(("".join(block), True))
            continue

        if "<Tabs" in line:
            flush()
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if "</Tabs>" in lines[index]:
                    index += 1
                    break
                index += 1
            blocks.append(("".join(block), True))
            continue

        if stripped.startswith("|"):
            flush()
            block = [line]
            index += 1
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index])
                index += 1
            blocks.append(("".join(block), True))
            continue

        if not stripped:
            flush()
            index += 1
            continue

        buffer.append(line)
        index += 1

    flush()
    return blocks


def _pack_blocks(
    blocks: list[tuple[str, bool]],
    splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> list[str]:
    pieces: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            pieces.append(current.strip())
        current = ""

    for block, protected in blocks:
        if len(block) > chunk_size:
            flush()
            if protected:
                # Code/containers/tables are never cut: allow controlled overflow.
                pieces.append(block.strip())
            else:
                pieces.extend(
                    piece.strip() for piece in splitter.split_text(block) if piece.strip()
                )
            continue

        if not current:
            current = block
        elif len(current) + 2 + len(block) <= chunk_size:
            current = current.rstrip("\n") + "\n\n" + block.lstrip("\n")
        else:
            flush()
            current = block

    flush()
    return pieces


def _section_pieces(
    section: _Section,
    splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> list[_Piece]:
    if len(section.text) <= chunk_size:
        return [_Piece(path=section.path, text=section.text)]
    blocks = _split_blocks(section.text)
    return [
        _Piece(path=section.path, text=text)
        for text in _pack_blocks(blocks, splitter, chunk_size)
    ]


def _merge_parent(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...] | None:
    """Return the breadcrumb for a merged piece, or None if merging is not allowed.

    H3 siblings under the same H2 may merge. H2 siblings under the same H1
    may not: they are separate topics and merging would mix their content.
    Parent/child pieces (one path is a prefix of the other) may merge.
    """
    common: list[str] = []
    for left, right in zip(a, b):
        if left != right:
            break
        common.append(left)
    if not common:
        return () if a == b else None
    parent = tuple(common)
    if len(parent) >= 2:
        return parent
    if a == parent or b == parent:
        return parent
    return None


def _merge_small(
    pieces: list[_Piece],
    chunk_size: int,
    min_chunk_size: int,
) -> list[_Piece]:
    """Merge adjacent small pieces that share a heading ancestor."""
    merged: list[_Piece] = []
    for piece in pieces:
        if merged:
            previous = merged[-1]
            parent = _merge_parent(previous.path, piece.path)
            combined_length = len(previous.text) + 2 + len(piece.text)
            if (
                parent is not None
                and (len(previous.text) < min_chunk_size or len(piece.text) < min_chunk_size)
                and combined_length <= chunk_size
            ):
                merged[-1] = _Piece(
                    path=parent,
                    text=previous.text.rstrip("\n") + "\n\n" + piece.text.lstrip("\n"),
                )
                continue
        merged.append(piece)
    return merged


def chunk_markdown(
    text: str,
    source: str,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 100,
    min_chunk_size: int = 300,
    max_chunk_chars: int = 6000,
) -> list[Chunk]:
    """Structure-aware markdown chunking."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    text = _strip_front_matter(text)
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=TEXT_SEPARATORS,
        length_function=len,
    )

    pieces: list[_Piece] = []
    for section in _split_sections(text):
        pieces.extend(_section_pieces(section, splitter, chunk_size))
    pieces = _merge_small(pieces, chunk_size, min_chunk_size)

    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        if len(piece.text) > max_chunk_chars:
            logger.warning(
                "Large chunk (%d chars) in %s [%s]; code blocks are kept intact",
                len(piece.text),
                source,
                " › ".join(piece.path) or "-",
            )
        chunks.append(
            Chunk(
                text=piece.text,
                source=source,
                chunk_index=index,
                heading_path=" › ".join(piece.path) if piece.path else None,
            )
        )
    return chunks
