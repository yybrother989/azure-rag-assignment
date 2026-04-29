"""
Structure-aware, token-bounded chunking from Markdown.

Pipeline:
  1. Strip `<!-- PageNumber=N -->` markers (DI emits one per page boundary)
     while remembering where each page starts. We use char-offset bookkeeping
     to map every later chunk back to the page it came from.
  2. MarkdownHeaderTextSplitter: split by `#`/`##`/`###` so each chunk
     carries its heading_path metadata.
  3. RecursiveCharacterTextSplitter (token-aware via tiktoken): sub-split
     any heading section that exceeds CHUNK_MAX_TOKENS.
  4. Emit `Chunk` dataclasses with idempotent content-hash IDs.
"""

from __future__ import annotations

import hashlib
import os
import re
from bisect import bisect_right
from dataclasses import dataclass, field

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .extract import ExtractedDocument

DEFAULT_TOKENIZER_ENCODING = "cl100k_base"   # matches text-embedding-3-small
HEADER_LEVELS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]
# DI emits page boundaries as either:
#   <!-- PageNumber="2" -->   (digital PDFs, ~88% of pages in our test corpus)
#   <!-- PageBreak -->        (always emitted; we use it to bump the counter
#                              when an explicit PageNumber marker is missing)
PAGE_NUMBER_RE = re.compile(r"<!--\s*PageNumber\s*=\s*[\"']?(\d+)[\"']?\s*-->")
PAGE_BREAK_RE = re.compile(r"<!--\s*PageBreak\s*-->")
# Combined matcher used to walk the markdown in document order.
PAGE_BOUNDARY_RE = re.compile(
    r"<!--\s*(?:PageNumber\s*=\s*[\"']?(\d+)[\"']?|PageBreak)\s*-->"
)


@dataclass
class Chunk:
    chunk_id: str
    chunk_index: int
    content: str
    doc_id: str
    source_path: str
    file_name: str
    file_type: str
    category: str               # legacy: "manual" | "troubleshooting" | "policy" | "other"
    page_number: int | None
    page_end: int | None
    heading_path: str | None
    # Structured metadata derived from the blob path (see extract._parse_path_metadata).
    scope: str = "device"       # "device" | "shared"
    device_family: str | None = None
    device: str | None = None   # e.g. "deviceA"; None for shared or legacy paths
    doc_type: str | None = None # "manual" | "troubleshooting" | "policy" | "other"
    topic: str | None = None    # filename-derived tag, e.g. "error101"
    version: str | None = None  # e.g. "1.2"
    is_shared: bool = False
    tables: list[dict] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)


def make_doc_id(source_path: str) -> str:
    return hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]


def make_chunk_id(source_path: str, chunk_index: int, content: str) -> str:
    h = hashlib.sha256()
    h.update(source_path.encode("utf-8"))
    h.update(b"::")
    h.update(str(chunk_index).encode())
    h.update(b"::")
    h.update(content.encode("utf-8"))
    return h.hexdigest()


def _strip_page_markers(markdown: str) -> tuple[str, list[tuple[int, int]]]:
    """Remove DI's page-boundary HTML comments (PageNumber, PageBreak) AND
    PageHeader/PageFooter chrome, then build a sorted list of
    `(char_offset_in_clean_markdown, page_number)` so chunks can be
    attributed back to their source page.

    DI doesn't always emit a PageNumber for page 1; we default to 1 and
    bump the counter on every PageBreak that lacks an adjacent PageNumber."""
    clean_parts: list[str] = []
    page_starts: list[tuple[int, int]] = [(0, 1)]
    current_page = 1
    cursor = 0
    output_offset = 0
    for match in PAGE_BOUNDARY_RE.finditer(markdown):
        # Append everything up to the marker (preserves content)
        clean_parts.append(markdown[cursor:match.start()])
        output_offset += (match.start() - cursor)
        page_no_str = match.group(1)   # None for PageBreak, digit-string for PageNumber
        if page_no_str is not None:
            current_page = int(page_no_str)
        else:
            current_page += 1
        page_starts.append((output_offset, current_page))
        cursor = match.end()
    clean_parts.append(markdown[cursor:])

    clean_md = "".join(clean_parts)

    # Strip leftover PageHeader/PageFooter chrome (they're decorative, would
    # bloat chunks and may even appear inside chunk bodies).
    clean_md = re.sub(r"<!--\s*Page(Header|Footer)[^>]*-->\s*", "", clean_md)
    return clean_md, page_starts


def _page_for_offset(page_starts: list[tuple[int, int]], offset: int) -> int | None:
    """Binary-search the page mapping for the largest start <= offset."""
    if not page_starts:
        return None
    keys = [start for start, _ in page_starts]
    idx = bisect_right(keys, offset) - 1
    if idx < 0:
        return page_starts[0][1]
    return page_starts[idx][1]


def _heading_path_from_metadata(meta: dict) -> str | None:
    """LangChain's MarkdownHeaderTextSplitter puts each header's text under
    `h1`/`h2`/`h3`/`h4` keys in chunk.metadata. Join into a path."""
    parts = []
    for _, key in HEADER_LEVELS_TO_SPLIT:
        v = meta.get(key)
        if v:
            parts.append(str(v).strip())
    return " > ".join(parts) if parts else None


def _token_count(text: str) -> int:
    enc = tiktoken.get_encoding(DEFAULT_TOKENIZER_ENCODING)
    return len(enc.encode(text))


HEADING_RE = re.compile(r"(?m)^(#{1,4})[ \t]+(.+?)\s*$")


def _split_by_headers_with_offsets(clean_md: str) -> list[dict]:
    """In-house markdown-header splitter that retains EXACT offsets.

    LangChain's MarkdownHeaderTextSplitter rewrites whitespace and discards
    positions, which made the page-attribution lookup brittle — sections
    drifted to the wrong offsets, so chunks for whole page ranges got pinned
    to the same page and figure attachment broke.

    Returns a list of `{start, end, text, heading_path}` dicts in document
    order. `heading_path` is the H1>H2>H3>H4 trail for that section."""
    matches = list(HEADING_RE.finditer(clean_md))
    if not matches:
        return [{"start": 0, "end": len(clean_md), "text": clean_md, "heading_path": None}]
    sections: list[dict] = []
    crumbs: list[str | None] = [None, None, None, None]   # h1, h2, h3, h4
    boundaries = [(m.start(), len(m.group(1)), m.group(2).strip()) for m in matches]
    boundaries.append((len(clean_md), 0, ""))   # sentinel for the last section's end
    # Lead-in (anything before the first heading) — keep if non-trivial.
    if matches[0].start() > 0:
        lead = clean_md[:matches[0].start()]
        if lead.strip():
            sections.append({
                "start": 0, "end": matches[0].start(),
                "text": lead, "heading_path": None,
            })
    for i in range(len(matches)):
        start, level, heading = boundaries[i]
        next_start = boundaries[i + 1][0]
        crumbs[level - 1] = heading
        for deeper in range(level, 4):
            crumbs[deeper] = None
        path_parts = [c for c in crumbs if c]
        sections.append({
            "start": start,
            "end": next_start,
            "text": clean_md[start:next_start],
            "heading_path": " > ".join(path_parts) if path_parts else None,
        })
    return sections


def _table_anchors(extracted: ExtractedDocument) -> list[tuple[str, dict]]:
    """For each TableRef, take the first ~120 chars of its HTML span as a
    unique anchor. Used to detect whether a chunk piece contains the table.

    DI gives us `<table>...` then the actual cell content — that prefix is
    long enough to be unique across the document but short enough to survive
    LangChain's recursive splitting (it preserves contiguous text below the
    chunk_size threshold). Page-marker / header / footer HTML comments may
    sit inside the raw span but get stripped from clean_md, so we strip them
    from the anchor too before matching."""
    anchors: list[tuple[str, dict]] = []
    for tref in extracted.tables:
        if tref.markdown_length <= 0:
            continue
        # Pull more than 120 chars to absorb any HTML comments we'll strip.
        end = tref.markdown_offset + min(tref.markdown_length, 240)
        raw = extracted.markdown[tref.markdown_offset:end]
        cleaned = PAGE_BOUNDARY_RE.sub("", raw)
        cleaned = re.sub(r"<!--\s*Page(Header|Footer)[^>]*-->\s*", "", cleaned)
        anchor = cleaned[:120].strip()
        if anchor:
            anchors.append((anchor, tref.to_dict()))
    return anchors


def chunk_document(extracted: ExtractedDocument) -> list[Chunk]:
    """Two-stage split: by Markdown header, then token-bounded sub-split."""
    max_tokens = int(os.environ.get("CHUNK_MAX_TOKENS", "512"))
    overlap_tokens = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "50"))

    clean_md, page_starts = _strip_page_markers(extracted.markdown)
    if not clean_md.strip():
        return []

    table_anchors = _table_anchors(extracted)
    figures_by_page: dict[int, list[dict]] = {}
    for fig in extracted.figures:
        if fig.blob_url:    # only carry figures we actually rendered + uploaded
            figures_by_page.setdefault(fig.page, []).append(fig.to_dict())

    # Stage 1 — split by markdown headers, retaining EXACT offsets.
    sections = _split_by_headers_with_offsets(clean_md)

    # Stage 2 — token-bounded sub-split when a header section exceeds the budget.
    sub_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=DEFAULT_TOKENIZER_ENCODING,
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
    )

    doc_id = make_doc_id(extracted.source_path)
    chunks: list[Chunk] = []
    chunk_idx = 0

    for section in sections:
        text = (section["text"] or "").strip()
        if not text:
            continue
        section_offset = section["start"]
        page_no = _page_for_offset(page_starts, section_offset)
        heading_path = section["heading_path"]

        # If the section fits in budget, emit as one chunk; else sub-split.
        if _token_count(text) <= max_tokens:
            pieces = [text]
        else:
            pieces = sub_splitter.split_text(text)

        # Walk pieces in order, tracking each one's offset within the section
        # so a long section that spans pages doesn't pin every sub-piece to
        # the section's start page (this was the figure-attribution gap).
        piece_cursor = 0
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            piece_offset_in_section = text.find(piece, piece_cursor)
            if piece_offset_in_section == -1:
                piece_offset_in_section = piece_cursor
            piece_cursor = piece_offset_in_section + len(piece)
            piece_global_offset = section_offset + piece_offset_in_section
            piece_page = _page_for_offset(page_starts, piece_global_offset) or page_no
            piece_end_global = piece_global_offset + len(piece)
            piece_page_end = _page_for_offset(page_starts, piece_end_global) or piece_page
            piece_tables = [td for anchor, td in table_anchors if anchor in piece]
            piece_figures = figures_by_page.get(piece_page, []) if piece_page else []
            chunks.append(Chunk(
                chunk_id=make_chunk_id(extracted.source_path, chunk_idx, piece),
                chunk_index=chunk_idx,
                content=piece,
                doc_id=doc_id,
                source_path=extracted.source_path,
                file_name=extracted.file_name,
                file_type=extracted.file_type,
                category=extracted.category,
                page_number=piece_page,
                page_end=piece_page_end,
                heading_path=heading_path,
                scope=extracted.scope,
                device_family=extracted.device_family,
                device=extracted.device,
                doc_type=extracted.doc_type,
                topic=extracted.topic,
                version=extracted.version,
                is_shared=extracted.is_shared,
                tables=piece_tables,
                figures=piece_figures,
            ))
            chunk_idx += 1

    return chunks
