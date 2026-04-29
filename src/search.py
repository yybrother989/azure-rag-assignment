"""
Retrieval ONLY.

`hybrid_search` runs an Azure AI Search query in one of four modes
(bm25 / vector / hybrid / hybrid_semantic) and returns a list of
`RetrievalResult` records carrying every field a downstream evaluator,
trace renderer, or generator might want — including the semantic caption
and reranker score when present.

No LLM calls. No generation. The function is fully deterministic given
the same index state and query embedding.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)

from .embed import embed_query
from .index import get_search_client

PREVIEW_CHARS = 240
SUPPORTED_MODES = {"bm25", "vector", "hybrid", "hybrid_semantic"}


@dataclass
class RetrievalResult:
    rank: int
    score: float
    reranker_score: float | None
    semantic_caption: str | None
    content_preview: str
    content: str
    chunk_id: str
    source_path: str
    file_name: str
    file_type: str
    category: str
    page_number: int | None
    page_end: int | None
    heading_path: str | None
    scope: str = "device"
    device_family: str | None = None
    device: str | None = None
    doc_type: str | None = None
    topic: str | None = None
    version: str | None = None
    is_shared: bool = False
    tables: list[dict] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _semantic_config_name() -> str:
    return os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIG", "kb-semantic")


def _build_filter(filters: dict | None) -> str | None:
    """Translate a small {field: value | list[value]} dict into an OData filter."""
    if not filters:
        return None
    parts: list[str] = []
    for k, v in filters.items():
        if isinstance(v, list):
            inner = " or ".join(_eq(k, x) for x in v)
            parts.append(f"({inner})")
        else:
            parts.append(_eq(k, v))
    return " and ".join(parts) if parts else None


def _eq(field: str, value) -> str:
    if isinstance(value, bool):
        return f"{field} eq {'true' if value else 'false'}"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"{field} eq '{escaped}'"
    return f"{field} eq {value}"


def _extract_caption(raw_doc: dict) -> str | None:
    captions = raw_doc.get("@search.captions")
    if not captions:
        return None
    first = captions[0]
    text = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
    return text


def hybrid_search(
    query: str,
    *,
    filters: dict | None = None,
    top_k: int = 5,
    search_mode: str = "hybrid_semantic",
) -> list[RetrievalResult]:
    if search_mode not in SUPPORTED_MODES:
        raise ValueError(f"search_mode must be one of {SUPPORTED_MODES}, got {search_mode!r}")

    client = get_search_client()
    odata_filter = _build_filter(filters)

    select_fields = [
        "chunk_id", "content", "doc_id", "source_path", "file_name",
        "file_type", "category", "page_number", "page_end", "heading_path",
        "tables_json", "figures_json",
        "scope", "device_family", "device", "doc_type", "topic", "version", "is_shared",
    ]

    kwargs: dict = {
        "select": select_fields,
        "top": top_k,
        "filter": odata_filter,
    }

    if search_mode in {"vector", "hybrid", "hybrid_semantic"}:
        kwargs["vector_queries"] = [
            VectorizedQuery(
                vector=embed_query(query),
                k_nearest_neighbors=max(top_k, 50),
                fields="content_vector",
            )
        ]

    if search_mode == "vector":
        kwargs["search_text"] = None
    else:
        kwargs["search_text"] = query

    if search_mode == "hybrid_semantic":
        kwargs.update(
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name=_semantic_config_name(),
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
        )

    raw = client.search(**kwargs)

    results: list[RetrievalResult] = []
    for rank, doc in enumerate(raw, start=1):
        content = doc.get("content", "") or ""
        tables_json = doc.get("tables_json") or ""
        figures_json = doc.get("figures_json") or ""
        try:
            tables = json.loads(tables_json) if tables_json else []
        except json.JSONDecodeError:
            tables = []
        try:
            figures = json.loads(figures_json) if figures_json else []
        except json.JSONDecodeError:
            figures = []
        results.append(
            RetrievalResult(
                rank=rank,
                score=float(doc.get("@search.score", 0.0)),
                reranker_score=(
                    float(doc["@search.reranker_score"])
                    if doc.get("@search.reranker_score") is not None
                    else None
                ),
                semantic_caption=_extract_caption(doc),
                content_preview=content[:PREVIEW_CHARS] + ("…" if len(content) > PREVIEW_CHARS else ""),
                content=content,
                chunk_id=doc.get("chunk_id", ""),
                source_path=doc.get("source_path", ""),
                file_name=doc.get("file_name", ""),
                file_type=doc.get("file_type", ""),
                category=doc.get("category", ""),
                page_number=doc.get("page_number"),
                page_end=doc.get("page_end"),
                heading_path=doc.get("heading_path"),
                scope=doc.get("scope", "device"),
                device_family=doc.get("device_family"),
                device=doc.get("device"),
                doc_type=doc.get("doc_type"),
                topic=doc.get("topic"),
                version=doc.get("version"),
                is_shared=bool(doc.get("is_shared", False)),
                tables=tables,
                figures=figures,
            )
        )
    return results
