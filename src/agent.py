"""
Two-route LangGraph orchestration — metadata-first, prototype-friendly.

Routes
------
NO_RETRIEVAL
    System capability, usage, and conceptual questions that do not require
    searching the knowledge base. Answered directly by the LLM without any
    retrieval calls, embedding, or Azure AI Search cost.

SIMPLE_RAG
    Direct retrieval questions. Uses metadata extracted by the router to scope
    the search as narrowly as possible before broadening to shared docs.

    Flow: build_retrieval_scope → simple_rag_search → evidence_selector
          → generator_node → response_formatter

Graph
-----
    START
      → intent_router
          ├─ blocked_input  ──────────────────────────── response_formatter → END
          ├─ NO_RETRIEVAL → no_retrieval_response ────── response_formatter → END
          └─ SIMPLE_RAG   → build_retrieval_scope
                            → simple_rag_search
                            → evidence_selector
                            → generator_node
                            → response_formatter → END

Design principles
-----------------
- Deterministic retrieval policy: metadata filters are built without LLM calls.
- Device-first search: narrowest scope (device + doc_type) tried first; shared
  docs are a deterministic fallback, not the default search space.
- Evidence sufficiency guard: generator is not called with empty results.
- Safety gates: pre-router (input) and post-generator (output) via Azure CS.
- Every decision is captured in FinalRagTrace for auditability.
"""

from __future__ import annotations

import json
import operator
import os
import time
import uuid
from typing import Annotated, Any, Callable, Required, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .embed import get_azure_openai_client
from .generate import GenerationOutput, generate_grounded_answer
from .safety import SafetyResult, check_input, check_output
from .search import RetrievalResult, hybrid_search
from .telemetry import get_tracer
from .tracing import (
    EvidenceSelectionTrace,
    FinalRagTrace,
    GenerationTrace,
    QueryPlanTrace,
    RetrievalTrace,
    emit_trace_jsonl,
)

_tracer = get_tracer()

# ---------- canned replies ----------

SAFETY_INPUT_REPLY = (
    "Your message was flagged by Azure AI Content Safety and can't be processed. "
    "Please rephrase your question."
)
SAFETY_OUTPUT_REPLY = (
    "The generated answer was flagged by Azure AI Content Safety and has been "
    "withheld. Please rephrase your question."
)
NO_EVIDENCE_REPLY = (
    "I could not find relevant documents in the knowledge base for your query. "
    "Please check that the relevant documents have been ingested, or try "
    "rephrasing your question."
)

# ---------- router prompt ----------

ROUTER_SYSTEM = """\
You are a query router for a retail IT support assistant covering in-store hardware
(receipt printers, payment terminals, check scanners, network appliances) and IT policies.
Classify the user query and reply as JSON with EXACTLY this schema.

ROUTE VALUES:
  "NO_RETRIEVAL"  – Questions about what this assistant can do, which devices it covers,
                    or how to use it. Does NOT require searching the knowledge base.
                    Examples:
                      "What devices do you support?"
                      "What can you help me with?"
                      "How do I search for a document?"
                      "What is this tool?"

  "SIMPLE_RAG"    – Any question requiring actual device or policy information from
                    the knowledge base — troubleshooting, setup, error codes, manuals, policies.
                    Examples:
                      "How do I reset a Cisco Meraki MX67?"
                      "What does the warranty policy say?"
                      "Show me the paper loading steps for the Epson TM-m30II."
                      "The Ingenico terminal is showing error E015."

QUERY_TYPE values: "troubleshoot" | "manual_lookup" | "policy_check" | "general_kb" | "conceptual"

JSON schema (all fields required, use null when not applicable):
{
  "route": "NO_RETRIEVAL" | "SIMPLE_RAG",
  "reason": "<one sentence>",
  "query_type": "<query_type>",
  "detected_device_family": "<device family e.g. network_access> | null",
  "detected_device": "<device model e.g. meraki_mx67> | null",
  "detected_doc_type": "manual" | "troubleshooting" | "policy" | null,
  "detected_topic": "<key topic from query> | null",
  "detected_error_code": "<error code e.g. error101> | null",
  "allow_shared_fallback": true | false
}

Reply with VALID JSON ONLY. No markdown fences, no prose."""

# ---------- direct-response prompt for NO_RETRIEVAL ----------

NO_RETRIEVAL_SYSTEM = """\
You are a Retail IT Support Assistant for in-store hardware and IT policy questions.
You cover four device families:
  - Receipt printers (Epson TM-M30II)
  - Payment terminals (Ingenico Desk5000)
  - Check scanners (Canon CR-120)
  - Network appliances (Meraki MX67)

Answer questions about:
  - What devices and topics this assistant covers
  - How to use this assistant (e.g. how to search, how to browse documents)
  - General guidance on navigating the support system

Be concise and friendly. Do NOT attempt to answer specific troubleshooting questions,
error codes, setup steps, or policy details from memory — those must be looked up
in the knowledge base and will be handled by a separate search step."""

# ---------- state ----------

HISTORY_TURNS_FOR_PROMPT = 3
HISTORY_CONTENT_TRUNCATE = 400


class RagState(TypedDict, total=False):
    user_query: Required[str]
    # conversation_history uses operator.add so each turn appends without replacing.
    conversation_history: Annotated[list[dict], operator.add]

    # Router outputs — populated by intent_router ─────────────────────────────
    route: str              # "NO_RETRIEVAL" | "SIMPLE_RAG" | "blocked_input"
    router_reason: str
    query_type: str
    detected_device_family: str | None
    detected_device: str | None
    detected_doc_type: str | None
    detected_topic: str | None
    detected_error_code: str | None
    allow_shared_fallback: bool
    query_plan: dict        # QueryPlanTrace as dict (used by response_formatter)

    # Retrieval scope — populated by build_retrieval_scope ────────────────────
    primary_filter: dict
    fallback_filter: dict | None
    fallback_triggered: bool

    # Trace payloads — populated by their respective nodes ───────────────────
    retrieval: dict         # RetrievalTrace as dict
    evidence: dict          # EvidenceSelectionTrace as dict
    generation: dict        # GenerationTrace as dict
    safety_input: dict      # SafetyResult as dict
    safety_output: dict     # SafetyResult as dict
    safety_blocked: bool

    final_message: str
    final_trace: dict       # FinalRagTrace as dict
    started_at: float
    retrieval_results: Required[list]  # list[RetrievalResult as dict]


# ---------- shared helpers ----------

def _llm_json(system: str, user: str) -> dict:
    client = get_azure_openai_client()
    deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)  # type: ignore[arg-type]


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    recent = history[-HISTORY_TURNS_FOR_PROMPT * 2:]
    lines = []
    for entry in recent:
        role = entry.get("role", "?").upper()
        content = entry.get("content", "")
        if len(content) > HISTORY_CONTENT_TRUNCATE:
            content = content[:HISTORY_CONTENT_TRUNCATE] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _with_history(query: str, history: list[dict]) -> str:
    block = _format_history(history)
    if not block:
        return query
    return (
        "<recent_conversation>\n"
        f"{block}\n"
        "</recent_conversation>\n\n"
        f"Current user input: {query}"
    )


def _dataclass_to_dict(obj: Any) -> dict:
    from dataclasses import asdict, is_dataclass
    if is_dataclass(obj):
        return asdict(cast(Any, obj))
    if isinstance(obj, dict):
        return obj
    raise TypeError(type(obj))


def _unwrap_retrieval(d: dict) -> dict:
    out = dict(d)
    out["results"] = [RetrievalResult(**r) for r in d["results"]]
    return out


def _unwrap_generation(d: dict) -> dict:
    from .generate import Citation
    out = dict(d)
    out["citations"] = [Citation(**c) for c in d["citations"]]
    return out


# ---------- graph nodes ----------

def intent_router(state: RagState) -> dict:
    """Classify query into NO_RETRIEVAL or SIMPLE_RAG; gate unsafe input early."""
    with _tracer.start_as_current_span("intent_router") as span:
        query = state["user_query"]
        history = state.get("conversation_history", [])
        span.set_attribute("rag.query_length", len(query))

        # Pre-LLM safety gate — avoids any model cost on unsafe inputs.
        safety_in = check_input(query)
        span.set_attribute("rag.safety_input_passed", safety_in.passed)
        if not safety_in.passed:
            plan = _make_plan_dict(
                query=query, route="blocked_input",
                reason=f"blocked by content safety: {safety_in.blocked_category}",
                query_type="", device_family=None, device=None, doc_type=None, topic=None,
                error_code=None, allow_fallback=False,
                primary_filter={}, fallback_filter=None,
            )
            return {
                "route": "blocked_input",
                "router_reason": plan["router_reason"],
                "query_plan": plan,
                "started_at": time.perf_counter(),
                "safety_input": safety_in.to_dict(),
                "safety_blocked": True,
            }

        # LLM-based routing.
        raw = _llm_json(ROUTER_SYSTEM, _with_history(query, history))
        route = raw.get("route", "SIMPLE_RAG")
        if route not in ("NO_RETRIEVAL", "SIMPLE_RAG"):
            route = "SIMPLE_RAG"

        device_family = raw.get("detected_device_family") or None
        device = raw.get("detected_device") or None
        doc_type = raw.get("detected_doc_type") or None
        topic = raw.get("detected_topic") or None
        error_code = raw.get("detected_error_code") or None
        allow_fallback = bool(raw.get("allow_shared_fallback", True))
        query_type = raw.get("query_type", "general_kb")
        reason = raw.get("reason", "")

        span.set_attribute("rag.route", route)
        span.set_attribute("rag.query_type", query_type)
        span.set_attribute("rag.detected_device_family", device_family or "")
        span.set_attribute("rag.detected_device", device or "")

        # Build a partial plan dict — primary/fallback filters filled in by
        # build_retrieval_scope for SIMPLE_RAG.
        plan = _make_plan_dict(
            query=query, route=route, reason=reason, query_type=query_type,
            device_family=device_family,
            device=device, doc_type=doc_type, topic=topic, error_code=error_code,
            allow_fallback=allow_fallback,
            primary_filter={}, fallback_filter=None,
        )
        return {
            "route": route,
            "router_reason": reason,
            "query_type": query_type,
            "detected_device_family": device_family,
            "detected_device": device,
            "detected_doc_type": doc_type,
            "detected_topic": topic,
            "detected_error_code": error_code,
            "allow_shared_fallback": allow_fallback,
            "query_plan": plan,
            "started_at": time.perf_counter(),
            "safety_input": safety_in.to_dict(),
            "safety_blocked": False,
        }


def no_retrieval_response(state: RagState) -> dict:
    """Answer system capability / conceptual questions without any retrieval."""
    with _tracer.start_as_current_span("no_retrieval_response") as span:
        query = state["user_query"]
        history = state.get("conversation_history", [])
        user_content = _with_history(query, history) if history else query

        client = get_azure_openai_client()
        deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": NO_RETRIEVAL_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        answer = resp.choices[0].message.content or ""
        span.set_attribute("rag.latency_ms", latency_ms)

        empty_ret = RetrievalTrace(
            query=query, filters={}, search_mode="none", results=[], latency_ms=0
        )
        empty_ev = EvidenceSelectionTrace(
            candidate_count=0, selected_chunk_ids=[],
            selection_strategy="skipped", rationale="NO_RETRIEVAL route",
        )
        gen = GenerationTrace(
            model=deployment, context_chunk_ids=[],
            prompt_token_estimate=0, completion_token_estimate=0,
            answer=answer, citations=[], latency_ms=latency_ms,
        )
        return {
            "retrieval": _dataclass_to_dict(empty_ret),
            "retrieval_results": [],
            "evidence": _dataclass_to_dict(empty_ev),
            "generation": _dataclass_to_dict(gen),
        }


def build_retrieval_scope(state: RagState) -> dict:
    """Convert router output into deterministic OData filter dicts.

    Retrieval priority (narrowest → broadest):
      1. device + doc_type  → device-specific doc_type docs
      2. device only        → all docs for that device
      3. doc_type only      → that doc_type across all scopes
      4. shared policy hint → shared docs directly
      5. no signal          → no filter (full corpus)

    When allow_shared_fallback=True and a primary filter was applied,
    a fallback filter for shared docs of the same doc_type is added.
    """
    with _tracer.start_as_current_span("build_retrieval_scope") as span:
        device_family = state.get("detected_device_family")
        device = state.get("detected_device")
        doc_type = state.get("detected_doc_type")
        allow_fallback = state.get("allow_shared_fallback", True)

        primary: dict[str, Any] = {}
        fallback: dict[str, Any] | None = None

        if device_family and device and doc_type:
            primary = {
                "scope": "device",
                "device_family": device_family,
                "device": device,
                "doc_type": doc_type,
            }
            if allow_fallback:
                fallback = {"scope": "shared", "doc_type": doc_type}
        elif device_family and device:
            primary = {"scope": "device", "device_family": device_family, "device": device}
            if allow_fallback:
                fallback = {"scope": "shared"}
        elif device_family:
            primary = {"scope": "device", "device_family": device_family}
            if allow_fallback:
                fallback = {"scope": "shared"}
        elif device and doc_type:
            primary = {"scope": "device", "device": device, "doc_type": doc_type}
            if allow_fallback:
                fallback = {"scope": "shared", "doc_type": doc_type}
        elif device:
            primary = {"scope": "device", "device": device}
            if allow_fallback:
                fallback = {"scope": "shared"}
        elif doc_type:
            if doc_type == "policy":
                # Policy questions without a device usually target shared policies.
                primary = {"scope": "shared", "doc_type": "policy"}
            else:
                primary = {"doc_type": doc_type}
        # else: no filter — search full corpus

        span.set_attribute("rag.primary_filter", json.dumps(primary))
        span.set_attribute("rag.has_fallback", fallback is not None)

        # Update query_plan with the now-known filters.
        plan = dict(state.get("query_plan") or {})
        plan["primary_filter"] = primary
        plan["fallback_filter"] = fallback
        plan["filters"] = primary  # legacy alias

        return {
            "primary_filter": primary,
            "fallback_filter": fallback,
            "query_plan": plan,
        }


def simple_rag_search(state: RagState) -> dict:
    """Hybrid search with device-first scope and deterministic shared-doc fallback.

    Steps:
      1. Run primary search (narrowest scope).
      2. If fewer than MIN_RESULTS found AND a fallback filter exists,
         run fallback search against shared documents.
      3. Merge + deduplicate by chunk_id; return best results.
    """
    with _tracer.start_as_current_span("simple_rag_search") as span:
        query = state["user_query"]
        top_k = int(os.environ.get("RETRIEVAL_TOP_K", "5"))
        primary_filter: dict = state.get("primary_filter") or {}
        fallback_filter: dict | None = state.get("fallback_filter")
        MIN_RESULTS = 2

        t0 = time.perf_counter()

        # Step 1: primary search.
        results = hybrid_search(
            query=query,
            filters=primary_filter or None,
            top_k=top_k,
            search_mode="hybrid_semantic",
        )
        span.set_attribute("rag.primary_result_count", len(results))

        fallback_triggered = False

        # Step 2: fallback to shared docs if primary returned too few results.
        if len(results) < MIN_RESULTS and fallback_filter:
            fallback_results = hybrid_search(
                query=query,
                filters=fallback_filter,
                top_k=top_k,
                search_mode="hybrid_semantic",
            )
            # Merge: keep primary results first (higher priority), append new ones.
            seen_ids = {r.chunk_id for r in results}
            unique_fallback = [r for r in fallback_results if r.chunk_id not in seen_ids]
            results = results + unique_fallback
            fallback_triggered = bool(unique_fallback)
            span.set_attribute("rag.fallback_result_count", len(unique_fallback))

        # Step 3: if still empty, run unfiltered search so legacy data (without
        # scope/device fields) is not silently excluded.
        if not results and primary_filter:
            results = hybrid_search(
                query=query, filters=None, top_k=top_k, search_mode="hybrid_semantic"
            )
            fallback_triggered = bool(results)
            span.set_attribute("rag.unfiltered_fallback", fallback_triggered)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        span.set_attribute("rag.total_result_count", len(results))
        span.set_attribute("rag.latency_ms", latency_ms)

        trace = RetrievalTrace(
            query=query,
            filters=primary_filter,
            search_mode="hybrid_semantic",
            results=results,
            latency_ms=latency_ms,
            fallback_triggered=fallback_triggered,
        )
        return {
            "retrieval": _dataclass_to_dict(trace),
            "retrieval_results": [r.to_dict() for r in results],
            "fallback_triggered": fallback_triggered,
        }


def evidence_selector(state: RagState) -> dict:
    """Select top-N evidence chunks ranked by reranker_score (or score)."""
    with _tracer.start_as_current_span("evidence_selector") as span:
        results: list[dict] = state["retrieval_results"]
        n = int(os.environ.get("EVIDENCE_TOP_N", "4"))

        def sort_key(r: dict):
            rs = r.get("reranker_score")
            return rs if rs is not None else r.get("score", 0.0)

        ranked = sorted(results, key=sort_key, reverse=True)[:n]
        span.set_attribute("rag.candidate_count", len(results))
        span.set_attribute("rag.selected_count", len(ranked))
        trace = EvidenceSelectionTrace(
            candidate_count=len(results),
            selected_chunk_ids=[r["chunk_id"] for r in ranked],
            selection_strategy="top_n_by_reranker_then_score",
            rationale=f"Picked top {len(ranked)} of {len(results)} by reranker_score, falling back to score",
        )
        return {"evidence": _dataclass_to_dict(trace), "retrieval_results": ranked}


def generator_node(state: RagState, config: RunnableConfig) -> dict:
    """Generate a grounded answer. Guards against empty evidence."""
    with _tracer.start_as_current_span("generator_node") as span:
        selected: list[RetrievalResult] = [RetrievalResult(**r) for r in state["retrieval_results"]]
        span.set_attribute("rag.context_chunks", len(selected))

        # Evidence sufficiency guard — do not call the LLM with empty context.
        if not selected:
            stub = GenerationTrace(
                model="(skipped)", context_chunk_ids=[],
                prompt_token_estimate=0, completion_token_estimate=0,
                answer=NO_EVIDENCE_REPLY, citations=[], latency_ms=0,
            )
            return {
                "generation": _dataclass_to_dict(stub),
                "safety_output": SafetyResult(passed=True, blocked_category=None, severities={}, skipped=True).to_dict(),
                "safety_blocked": False,
            }

        handler = (config.get("configurable") or {}).get("stream_handler")
        history = state.get("conversation_history", [])
        query = state["user_query"]
        contextual_query = _with_history(query, history) if history else query

        t0 = time.perf_counter()
        out: GenerationOutput = generate_grounded_answer(
            query=contextual_query,
            selected_chunks=selected,
            stream_handler=handler,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        span.set_attribute("rag.model", out.model)
        span.set_attribute("rag.prompt_tokens", out.prompt_token_estimate)
        span.set_attribute("rag.latency_ms", latency_ms)

        # Post-generation safety gate.
        safety_out = check_output(out.answer)
        if not safety_out.passed:
            answer, citations = SAFETY_OUTPUT_REPLY, []
        else:
            answer, citations = out.answer, out.citations

        trace = GenerationTrace(
            model=out.model,
            context_chunk_ids=out.context_chunk_ids,
            prompt_token_estimate=out.prompt_token_estimate,
            completion_token_estimate=out.completion_token_estimate,
            answer=answer,
            citations=citations,
            latency_ms=latency_ms,
        )
        return {
            "generation": _dataclass_to_dict(trace),
            "safety_output": safety_out.to_dict(),
            "safety_blocked": not safety_out.passed,
        }


def response_formatter(state: RagState) -> dict:
    """Assemble FinalRagTrace, emit JSONL, build conversation history."""
    with _tracer.start_as_current_span("response_formatter") as span:
        plan_dict = state.get("query_plan") or {}
        route = state.get("route", "SIMPLE_RAG")
        started = state.get("started_at", time.perf_counter())
        total_ms = int((time.perf_counter() - started) * 1000)
        span.set_attribute("rag.route", route)
        span.set_attribute("rag.total_latency_ms", total_ms)

        user_entry = {"role": "user", "content": state["user_query"], "chunk_ids": []}

        if route == "blocked_input":
            reply = SAFETY_INPUT_REPLY
            final = _build_stub_trace(plan_dict, reply, total_ms, "blocked_input")
            emit_trace_jsonl(final)
            return {
                "final_message": reply,
                "final_trace": final.to_dict(),
                "conversation_history": [user_entry, {"role": "assistant", "content": reply, "chunk_ids": []}],
            }

        # Both NO_RETRIEVAL and SIMPLE_RAG have generation populated in state.
        gen = state.get("generation") or {}
        answer = gen.get("answer", "")

        final = FinalRagTrace(
            user_query=state["user_query"],
            query_plan=QueryPlanTrace(**_complete_plan_dict(plan_dict, state)),
            retrieval=RetrievalTrace(**_unwrap_retrieval(state.get("retrieval") or _empty_retrieval_dict(state["user_query"]))),
            evidence_selection=EvidenceSelectionTrace(**(state.get("evidence") or _empty_evidence_dict(route))),
            generation=GenerationTrace(**_unwrap_generation(gen)),
            total_latency_ms=total_ms,
        )
        emit_trace_jsonl(final)
        assistant_entry = {
            "role": "assistant",
            "content": answer,
            "chunk_ids": [c["chunk_id"] for c in gen.get("citations", [])],
        }
        return {
            "final_message": answer,
            "final_trace": final.to_dict(),
            "conversation_history": [user_entry, assistant_entry],
        }


# ---------- response_formatter helpers ----------

def _make_plan_dict(
    *, query: str, route: str, reason: str, query_type: str,
    device_family: str | None, device: str | None, doc_type: str | None, topic: str | None,
    error_code: str | None, allow_fallback: bool,
    primary_filter: dict, fallback_filter: dict | None,
) -> dict:
    top_k = int(os.environ.get("RETRIEVAL_TOP_K", "5"))
    return {
        "original_query": query,
        "route": route,
        "router_reason": reason,
        "query_type": query_type,
        "detected_device_family": device_family,
        "detected_device": device,
        "detected_doc_type": doc_type,
        "detected_topic": topic,
        "detected_error_code": error_code,
        "allow_shared_fallback": allow_fallback,
        "primary_filter": primary_filter,
        "fallback_filter": fallback_filter,
        "search_mode": "hybrid_semantic",
        "top_k": top_k,
        "notes": reason,
        # Legacy aliases.
        "intent": query_type,
        "rewritten_query": query,
        "filters": primary_filter,
    }


def _complete_plan_dict(plan_dict: dict, state: RagState) -> dict:
    """Merge any state fields missing from the stored plan_dict (e.g. filters
    computed by build_retrieval_scope after the router ran)."""
    merged = dict(plan_dict)
    # Fill in defaults for any required QueryPlanTrace fields that might be absent
    # (e.g. when NO_RETRIEVAL skipped build_retrieval_scope).
    merged.setdefault("route", state.get("route", "SIMPLE_RAG"))
    merged.setdefault("router_reason", state.get("router_reason", ""))
    merged.setdefault("query_type", state.get("query_type", ""))
    merged.setdefault("detected_device_family", state.get("detected_device_family"))
    merged.setdefault("detected_device", state.get("detected_device"))
    merged.setdefault("detected_doc_type", state.get("detected_doc_type"))
    merged.setdefault("detected_topic", state.get("detected_topic"))
    merged.setdefault("detected_error_code", state.get("detected_error_code"))
    merged.setdefault("allow_shared_fallback", state.get("allow_shared_fallback", True))
    pf = state.get("primary_filter") or {}
    merged.setdefault("primary_filter", pf)
    merged.setdefault("fallback_filter", state.get("fallback_filter"))
    merged.setdefault("search_mode", "hybrid_semantic")
    merged.setdefault("top_k", int(os.environ.get("RETRIEVAL_TOP_K", "5")))
    merged.setdefault("intent", merged.get("query_type", ""))
    merged.setdefault("rewritten_query", merged.get("original_query", ""))
    merged.setdefault("filters", pf)
    return merged


def _empty_retrieval_dict(query: str) -> dict:
    return {
        "query": query, "filters": {}, "search_mode": "none",
        "results": [], "latency_ms": 0, "fallback_triggered": False,
    }


def _empty_evidence_dict(route: str) -> dict:
    return {
        "candidate_count": 0, "selected_chunk_ids": [],
        "selection_strategy": "skipped",
        "rationale": f"{route} route — no evidence selection",
    }


def _build_stub_trace(plan_dict: dict, reply: str, total_ms: int, route: str) -> FinalRagTrace:
    query = plan_dict.get("original_query", "")
    empty_ret = RetrievalTrace(
        query=query, filters={}, search_mode="none", results=[], latency_ms=0
    )
    empty_ev = EvidenceSelectionTrace(
        candidate_count=0, selected_chunk_ids=[],
        selection_strategy="skipped", rationale=f"route={route}",
    )
    empty_gen = GenerationTrace(
        model="(skipped)", context_chunk_ids=[],
        prompt_token_estimate=0, completion_token_estimate=0,
        answer=reply, citations=[], latency_ms=0,
    )
    return FinalRagTrace(
        user_query=query,
        query_plan=QueryPlanTrace(**_complete_plan_dict(plan_dict, {})),  # type: ignore[arg-type]
        retrieval=empty_ret,
        evidence_selection=empty_ev,
        generation=empty_gen,
        total_latency_ms=total_ms,
    )


# ---------- graph wiring ----------

def _route_after_router(state: RagState) -> str:
    route = state.get("route", "SIMPLE_RAG")
    if route == "blocked_input":
        return "blocked"
    if route == "NO_RETRIEVAL":
        return "no_retrieval"
    return "simple_rag"


def build_graph():
    builder = StateGraph(RagState)

    builder.add_node("intent_router", intent_router)
    builder.add_node("no_retrieval_response", no_retrieval_response)
    builder.add_node("build_retrieval_scope", build_retrieval_scope)
    builder.add_node("simple_rag_search", simple_rag_search)
    builder.add_node("evidence_selector", evidence_selector)
    builder.add_node("generator_node", generator_node)
    builder.add_node("response_formatter", response_formatter)

    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        _route_after_router,
        {
            "blocked":      "response_formatter",
            "no_retrieval": "no_retrieval_response",
            "simple_rag":   "build_retrieval_scope",
        },
    )
    builder.add_edge("no_retrieval_response", "response_formatter")
    builder.add_edge("build_retrieval_scope", "simple_rag_search")
    builder.add_edge("simple_rag_search", "evidence_selector")
    builder.add_edge("evidence_selector", "generator_node")
    builder.add_edge("generator_node", "response_formatter")
    builder.add_edge("response_formatter", END)

    return builder.compile(checkpointer=MemorySaver())


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run(
    user_query: str,
    *,
    thread_id: str | None = None,
    stream_handler: Callable[[str], None] | None = None,
) -> dict:
    """Convenience wrapper for the CLI / notebooks. Returns the FinalRagTrace dict."""
    graph = get_graph()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id or str(uuid.uuid4()),
            "stream_handler": stream_handler,
        }
    }
    final_state = graph.invoke({"user_query": user_query}, config=config)
    return final_state["final_trace"]


if __name__ == "__main__":  # python -m src.agent "<query>"
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    q = " ".join(sys.argv[1:]) or "How do I reset a Cisco Meraki MX67?"
    trace = run(q)
    print(json.dumps(trace, indent=2, default=str))
