"""
Structured execution traces — one type per LangGraph node + a top-level
`FinalRagTrace` that bundles them.

Every trace is JSON-serialisable so the same payload drives:
  - the Chainlit step panels (live UI),
  - the CLI Rich panels and `--json` mode,
  - the eval harness (recall@k, groundedness, citation correctness),
  - the JSONL audit log on disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .generate import Citation
from .search import RetrievalResult


@dataclass
class QueryPlanTrace:
    original_query: str
    # Two-route classification ------------------------------------------------
    route: str                      # "NO_RETRIEVAL" | "SIMPLE_RAG"
    router_reason: str              # one-sentence rationale from the router LLM
    query_type: str                 # "troubleshoot" | "manual_lookup" | "policy_check"
                                    # | "general_kb" | "conceptual"
    detected_device_family: str | None
    detected_device: str | None     # e.g. "deviceA"; None if not mentioned
    detected_doc_type: str | None   # "manual" | "troubleshooting" | "policy" | None
    detected_topic: str | None      # key topic from query, e.g. "error101"
    detected_error_code: str | None # explicit error code, e.g. "error101"
    allow_shared_fallback: bool     # whether shared docs are a valid fallback
    # Retrieval scope (populated by build_retrieval_scope node) ---------------
    primary_filter: dict[str, Any]  # OData-ready filter dict for primary search
    fallback_filter: dict[str, Any] | None  # filter for shared-doc fallback search
    search_mode: str
    top_k: int
    notes: str | None = None
    # Legacy aliases kept so existing eval harness code that reads `intent`,
    # `rewritten_query`, and `filters` from FinalRagTrace dicts still works.
    intent: str = ""                # set to query_type for backward compat
    rewritten_query: str = ""       # set to original_query for backward compat
    filters: dict[str, Any] = field(default_factory=dict)  # set to primary_filter


@dataclass
class RetrievalTrace:
    query: str
    filters: dict[str, Any]
    search_mode: str
    results: list[RetrievalResult]
    latency_ms: int
    fallback_triggered: bool = False    # True when fallback filter was used


@dataclass
class EvidenceSelectionTrace:
    candidate_count: int
    selected_chunk_ids: list[str]
    selection_strategy: str           # top_k | mmr | llm_rerank
    rationale: str | None = None


@dataclass
class GenerationTrace:
    model: str
    context_chunk_ids: list[str]
    prompt_token_estimate: int
    completion_token_estimate: int
    answer: str
    citations: list[Citation]
    latency_ms: int


@dataclass
class FinalRagTrace:
    user_query: str
    query_plan: QueryPlanTrace
    retrieval: RetrievalTrace
    evidence_selection: EvidenceSelectionTrace
    generation: GenerationTrace
    total_latency_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def emit_trace_jsonl(trace: FinalRagTrace, path: str | None = None) -> None:
    """Append the trace to the configured JSONL log; safe to call from any frontend."""
    target = Path(path or os.environ.get("LOG_TRACE_JSONL", "traces.jsonl"))
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace.to_dict(), ensure_ascii=False, default=str))
        f.write("\n")
