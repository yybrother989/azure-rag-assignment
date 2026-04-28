"""
LangGraph workflow — six explicit nodes, one conditional edge.

The whole point of the graph is to make the RAG decisions auditable: every node
emits a typed trace object that downstream frontends (Chainlit, CLI) and the
eval harness consume directly. No hidden chain-of-thought is exposed.

Flow:
    intent_router ──┐
                    ├──▶ out_of_scope ─▶ response_formatter ─▶ END
                    └──▶ query_planner ─▶ retriever_node ─▶
                         evidence_selector ─▶ generator_node ─▶
                         response_formatter ─▶ END
"""

from __future__ import annotations

import json
import operator
import os
import time
import uuid
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .embed import get_azure_openai_client
from .generate import GenerationOutput, generate_grounded_answer
from .search import RetrievalResult, hybrid_search
from .tracing import (
    EvidenceSelectionTrace,
    FinalRagTrace,
    GenerationTrace,
    QueryPlanTrace,
    RetrievalTrace,
    emit_trace_jsonl,
)

INTENT_TO_CATEGORY = {
    "manual_lookup": "manual",
    "troubleshoot": "troubleshooting",
    "policy_check": "policy",
    "general_kb": None,        # no filter
    "out_of_scope": None,
}

ROUTER_SYSTEM = """You are a query router for a product knowledge base.
Classify the user query into exactly ONE intent and reply as JSON:
{"intent": "manual_lookup" | "troubleshoot" | "policy_check" | "general_kb" | "out_of_scope",
 "rationale": "<one short sentence>"}

Intents:
- manual_lookup: how-to / setup / configuration / feature questions about a product
- troubleshoot:  errors, failures, "X is broken", error codes
- policy_check:  rules, compliance, "are we allowed to", organisational policy
- general_kb:    knowledge-base question that crosses categories
- out_of_scope:  smalltalk, opinion, or unrelated to the KB

Reply with VALID JSON ONLY."""

PLANNER_SYSTEM = """You rewrite a user question into a search-friendly query.
Expand abbreviations, drop filler ("please", "could you"), keep it under 20 words.
Reply as JSON: {"rewritten_query": "..."}
Reply with VALID JSON ONLY."""

OUT_OF_SCOPE_REPLY = (
    "I can only answer questions about the indexed knowledge base "
    "(product manuals, troubleshooting guides, and policies). "
    "Please ask a question that fits one of those categories."
)


class RagState(TypedDict, total=False):
    user_query: str
    # conversation_history: list of {"role": "user"|"assistant", "content": str, "chunk_ids": list[str]}.
    # The operator.add reducer appends new entries instead of replacing the list, so each turn's
    # response_formatter can yield only the latest pair and LangGraph's checkpointer accumulates
    # the full session history per thread_id.
    conversation_history: Annotated[list[dict], operator.add]
    query_plan: dict          # QueryPlanTrace as dict
    retrieval: dict           # RetrievalTrace as dict
    evidence: dict            # EvidenceSelectionTrace as dict
    generation: dict          # GenerationTrace as dict
    final_message: str
    final_trace: dict         # FinalRagTrace as dict
    started_at: float
    retrieval_results: list   # RetrievalResult objects, kept for evidence_selector & generator


HISTORY_TURNS_FOR_PROMPT = 3      # how many recent turns to inject into router/planner/generator prompts
HISTORY_CONTENT_TRUNCATE = 400    # per-entry char cap so prompts don't bloat


def _format_history(history: list[dict]) -> str:
    """Render the last few turns for prompt injection. Empty string if no history."""
    if not history:
        return ""
    # Each turn = 2 entries (user + assistant), so take 2 * HISTORY_TURNS_FOR_PROMPT.
    recent = history[-HISTORY_TURNS_FOR_PROMPT * 2 :]
    lines = []
    for entry in recent:
        role = entry.get("role", "?").upper()
        content = entry.get("content", "")
        if len(content) > HISTORY_CONTENT_TRUNCATE:
            content = content[:HISTORY_CONTENT_TRUNCATE] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------- node implementations ----------

def _llm_json(system: str, user: str) -> dict:
    client = get_azure_openai_client()
    deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def _with_history(query: str, history: list[dict]) -> str:
    """Wrap the user query with a recent-conversation block when history exists."""
    block = _format_history(history)
    if not block:
        return query
    return (
        "<recent_conversation>\n"
        f"{block}\n"
        "</recent_conversation>\n\n"
        f"Current user input: {query}"
    )


def intent_router(state: RagState) -> dict:
    query = state["user_query"]
    history = state.get("conversation_history", [])
    parsed = _llm_json(ROUTER_SYSTEM, _with_history(query, history))
    intent = parsed.get("intent", "general_kb")
    if intent not in INTENT_TO_CATEGORY:
        intent = "general_kb"
    category = INTENT_TO_CATEGORY[intent]
    filters: dict[str, Any] = {"category": category} if category else {}
    plan = QueryPlanTrace(
        original_query=query,
        intent=intent,
        rewritten_query=query,                  # planner refines this next
        filters=filters,
        search_mode="hybrid_semantic",
        top_k=int(os.environ.get("RETRIEVAL_TOP_K", "5")),
        notes=parsed.get("rationale"),
    )
    return {"query_plan": plan.__dict__, "started_at": time.perf_counter()}


def query_planner(state: RagState) -> dict:
    plan = state["query_plan"]
    history = state.get("conversation_history", [])
    parsed = _llm_json(PLANNER_SYSTEM, _with_history(plan["original_query"], history))
    rewritten = parsed.get("rewritten_query", plan["original_query"]).strip() or plan["original_query"]
    plan = {**plan, "rewritten_query": rewritten}
    return {"query_plan": plan}


def retriever_node(state: RagState) -> dict:
    plan = state["query_plan"]
    t0 = time.perf_counter()
    results = hybrid_search(
        query=plan["rewritten_query"],
        filters=plan["filters"] or None,
        top_k=plan["top_k"],
        search_mode=plan["search_mode"],
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    trace = RetrievalTrace(
        query=plan["rewritten_query"],
        filters=plan["filters"],
        search_mode=plan["search_mode"],
        results=results,
        latency_ms=latency_ms,
    )
    # Store as dicts (not RetrievalResult dataclass instances) so LangGraph's
    # msgpack-based checkpointer can serialize the state without registering custom types.
    return {"retrieval": _dataclass_to_dict(trace), "retrieval_results": [r.to_dict() for r in results]}


def evidence_selector(state: RagState) -> dict:
    results: list[dict] = state["retrieval_results"]   # dicts, not dataclasses
    n = int(os.environ.get("EVIDENCE_TOP_N", "4"))

    # Prefer reranker_score where available (semantic mode); fall back to score.
    def sort_key(r: dict):
        return r["reranker_score"] if r.get("reranker_score") is not None else r["score"]

    ranked = sorted(results, key=sort_key, reverse=True)[:n]
    trace = EvidenceSelectionTrace(
        candidate_count=len(results),
        selected_chunk_ids=[r["chunk_id"] for r in ranked],
        selection_strategy="top_n_by_reranker_then_score",
        rationale=f"Picked top {len(ranked)} of {len(results)} by reranker_score, falling back to score",
    )
    return {"evidence": _dataclass_to_dict(trace), "retrieval_results": ranked}


def generator_node(state: RagState, config: RunnableConfig) -> dict:
    # Re-hydrate from dicts (see retriever_node note about checkpoint serialization).
    selected: list[RetrievalResult] = [RetrievalResult(**r) for r in state["retrieval_results"]]
    handler = (config.get("configurable") or {}).get("stream_handler")
    history = state.get("conversation_history", [])
    rewritten = state["query_plan"]["rewritten_query"]
    # If we have prior turns, prefix them so the generator can resolve pronouns
    # ("it", "that one", "for the other device") against the actual prior exchange.
    contextual_query = _with_history(rewritten, history) if history else rewritten
    t0 = time.perf_counter()
    out: GenerationOutput = generate_grounded_answer(
        query=contextual_query,
        selected_chunks=selected,
        stream_handler=handler,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    trace = GenerationTrace(
        model=out.model,
        context_chunk_ids=out.context_chunk_ids,
        prompt_token_estimate=out.prompt_token_estimate,
        completion_token_estimate=out.completion_token_estimate,
        answer=out.answer,
        citations=out.citations,
        latency_ms=latency_ms,
    )
    return {"generation": _dataclass_to_dict(trace)}


def response_formatter(state: RagState) -> dict:
    plan = state["query_plan"]
    started = state.get("started_at", time.perf_counter())
    total_ms = int((time.perf_counter() - started) * 1000)

    # Build the user-side history entry once; assistant entry differs by branch below.
    user_entry = {"role": "user", "content": plan["original_query"], "chunk_ids": []}

    if plan["intent"] == "out_of_scope":
        # Build a minimal trace — retrieval/evidence/generation are absent.
        empty_retrieval = RetrievalTrace(
            query=plan["original_query"], filters={}, search_mode="none", results=[], latency_ms=0
        )
        empty_evidence = EvidenceSelectionTrace(
            candidate_count=0, selected_chunk_ids=[], selection_strategy="skipped",
            rationale="intent=out_of_scope",
        )
        empty_gen = GenerationTrace(
            model="(skipped)", context_chunk_ids=[], prompt_token_estimate=0,
            completion_token_estimate=0, answer=OUT_OF_SCOPE_REPLY, citations=[], latency_ms=0,
        )
        final = FinalRagTrace(
            user_query=plan["original_query"],
            query_plan=QueryPlanTrace(**plan),
            retrieval=empty_retrieval,
            evidence_selection=empty_evidence,
            generation=empty_gen,
            total_latency_ms=total_ms,
        )
        emit_trace_jsonl(final)
        assistant_entry = {"role": "assistant", "content": OUT_OF_SCOPE_REPLY, "chunk_ids": []}
        return {
            "final_message": OUT_OF_SCOPE_REPLY,
            "final_trace": final.to_dict(),
            "conversation_history": [user_entry, assistant_entry],
        }

    final = FinalRagTrace(
        user_query=plan["original_query"],
        query_plan=QueryPlanTrace(**plan),
        retrieval=RetrievalTrace(**_unwrap_retrieval(state["retrieval"])),
        evidence_selection=EvidenceSelectionTrace(**state["evidence"]),
        generation=GenerationTrace(**_unwrap_generation(state["generation"])),
        total_latency_ms=total_ms,
    )
    emit_trace_jsonl(final)
    gen = state["generation"]
    assistant_entry = {
        "role": "assistant",
        "content": gen["answer"],
        "chunk_ids": [c["chunk_id"] for c in gen.get("citations", [])],
    }
    return {
        "final_message": gen["answer"],
        "final_trace": final.to_dict(),
        "conversation_history": [user_entry, assistant_entry],
    }


# ---------- helpers ----------

def _dataclass_to_dict(obj: Any) -> dict:
    """asdict-equivalent that also walks list[dataclass]."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(type(obj))


def _unwrap_retrieval(d: dict) -> dict:
    """Re-hydrate the RetrievalResult dataclasses from their dict form."""
    out = dict(d)
    out["results"] = [RetrievalResult(**r) for r in d["results"]]
    return out


def _unwrap_generation(d: dict) -> dict:
    """Re-hydrate the Citation dataclasses from their dict form."""
    from .generate import Citation

    out = dict(d)
    out["citations"] = [Citation(**c) for c in d["citations"]]
    return out


# ---------- graph wiring ----------

def _route_after_intent(state: RagState) -> str:
    return "out_of_scope" if state["query_plan"]["intent"] == "out_of_scope" else "in_scope"


def build_graph():
    builder = StateGraph(RagState)
    builder.add_node("intent_router", intent_router)
    builder.add_node("query_planner", query_planner)
    builder.add_node("retriever_node", retriever_node)
    builder.add_node("evidence_selector", evidence_selector)
    builder.add_node("generator_node", generator_node)
    builder.add_node("response_formatter", response_formatter)

    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        _route_after_intent,
        {"in_scope": "query_planner", "out_of_scope": "response_formatter"},
    )
    builder.add_edge("query_planner", "retriever_node")
    builder.add_edge("retriever_node", "evidence_selector")
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
    q = " ".join(sys.argv[1:]) or "How do I factory-reset Device A?"
    trace = run(q)
    print(json.dumps(trace, indent=2, default=str))
