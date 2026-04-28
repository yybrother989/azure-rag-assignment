"""
Structure-aware, token-bounded chunking via Docling's HybridChunker.

Each chunk carries the metadata fields the AI Search index expects:
chunk_id (deterministic for idempotent upserts), doc_id, source_path, file_name,
file_type, category, page_number, heading_path, chunk_index.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from docling.chunking import HybridChunker

from .extract import ExtractedDocument

DEFAULT_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Chunk:
    chunk_id: str
    chunk_index: int
    content: str
    doc_id: str
    source_path: str
    file_name: str
    file_type: str
    category: str
    page_number: int | None
    heading_path: str | None


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


_chunker: HybridChunker | None = None


def _chunker_singleton() -> HybridChunker:
    global _chunker
    if _chunker is None:
        _chunker = HybridChunker(
            tokenizer=DEFAULT_TOKENIZER,
            max_tokens=int(os.environ.get("CHUNK_MAX_TOKENS", "512")),
            merge_peers=True,
        )
    return _chunker


def _heading_path(meta) -> str | None:
    headings = getattr(meta, "headings", None)
    if not headings:
        return None
    return " > ".join(str(h) for h in headings)


def _page_number(meta) -> int | None:
    doc_items = getattr(meta, "doc_items", None) or []
    for item in doc_items:
        prov = getattr(item, "prov", None) or []
        for p in prov:
            page_no = getattr(p, "page_no", None)
            if page_no is not None:
                return int(page_no)
    return None


def chunk_document(extracted: ExtractedDocument) -> list[Chunk]:
    """Run Docling's HybridChunker and project each chunk into our Chunk dataclass."""
    chunker = _chunker_singleton()
    raw_chunks = list(chunker.chunk(extracted.docling_document))

    doc_id = make_doc_id(extracted.source_path)
    chunks: list[Chunk] = []
    for idx, rc in enumerate(raw_chunks):
        # Use the chunker's contextualized text — it prepends headings to each
        # chunk's body for retrieval quality, matching what HybridChunker exposes
        # via .contextualize(); fall back to the raw text if unavailable.
        text = chunker.contextualize(chunk=rc) if hasattr(chunker, "contextualize") else rc.text
        if not text or not text.strip():
            continue
        meta = rc.meta
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(extracted.source_path, idx, text),
                chunk_index=idx,
                content=text,
                doc_id=doc_id,
                source_path=extracted.source_path,
                file_name=extracted.file_name,
                file_type=extracted.file_type,
                category=extracted.category,
                page_number=_page_number(meta),
                heading_path=_heading_path(meta),
            )
        )
    return chunks
