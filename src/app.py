"""
Chainlit step-by-step UI.

Each LangGraph node renders as its own Chainlit step so the reviewer can watch
the RAG path execute live: intent → planning → retrieval → evidence → generation
→ final answer with citations.

The trace shown in the Chain-of-Thought panel is the same `FinalRagTrace` the
CLI prints and the eval harness consumes — the UI is a renderer, not a fork.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Chainlit loads app.py as a standalone module file (not as `src.app`), so the
# repo root must be on sys.path before the `src.` imports below resolve.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chainlit as cl  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.agent import get_graph  # noqa: E402  — env must load before graph init

NODE_LABELS = {
    "intent_router":     "① Intent Detection",
    "query_planner":     "② Query Planning",
    "retriever_node":    "③ Retrieval Results",
    "evidence_selector": "④ Evidence Selection",
    "generator_node":    "⑤ Generation",
    "response_formatter": "⑥ Final Answer",
}


def _render_intent(qp: dict) -> str:
    return (
        f"**intent:** `{qp['intent']}`  \n"
        f"**filters:** `{json.dumps(qp['filters'])}`  \n"
        f"**rationale:** {qp.get('notes') or '—'}"
    )


def _render_planning(qp: dict) -> str:
    return (
        f"**original:**  {qp['original_query']}  \n"
        f"**rewritten:** _{qp['rewritten_query']}_  \n"
        f"**mode:** `{qp['search_mode']}` &nbsp; **top_k:** `{qp['top_k']}`"
    )


def _render_retrieval(retrieval: dict) -> str:
    rows = ["| # | score | rerank | file | p. | heading | caption / preview |",
            "|---|------:|-------:|------|---:|---------|-------------------|"]
    for r in retrieval["results"]:
        cap = (r.get("semantic_caption") or r.get("content_preview", ""))[:140]
        rerank = r.get("reranker_score")
        rerank_cell = f"{rerank:.3f}" if rerank is not None else "—"
        rows.append(
            f"| {r['rank']} "
            f"| {r['score']:.3f} "
            f"| {rerank_cell} "
            f"| `{r.get('file_name','')}` "
            f"| {r.get('page_number') or '—'} "
            f"| {r.get('heading_path') or '—'} "
            f"| {cap} |"
        )
    rows.append(f"\n_latency: {retrieval['latency_ms']} ms_")
    return "\n".join(rows)


def _render_evidence(ev: dict) -> str:
    ids = ", ".join(f"`{c[:8]}…`" for c in ev["selected_chunk_ids"])
    return (
        f"**strategy:** `{ev['selection_strategy']}`  \n"
        f"**candidates:** {ev['candidate_count']} → **selected:** {len(ev['selected_chunk_ids'])}  \n"
        f"**rationale:** {ev.get('rationale') or '—'}  \n"
        f"**ids:** {ids}"
    )


def _render_generation(g: dict) -> str:
    return (
        f"**model:** `{g['model']}`  \n"
        f"**prompt~tokens:** {g['prompt_token_estimate']} &nbsp;"
        f" **completion~tokens:** {g['completion_token_estimate']}  \n"
        f"**latency:** {g['latency_ms']} ms"
    )


async def _run_node_step(node_name: str, payload: dict) -> None:
    label = NODE_LABELS.get(node_name, node_name)
    if node_name == "intent_router":
        body = _render_intent(payload["query_plan"])
    elif node_name == "query_planner":
        body = _render_planning(payload["query_plan"])
    elif node_name == "retriever_node":
        body = _render_retrieval(payload["retrieval"])
    elif node_name == "evidence_selector":
        body = _render_evidence(payload["evidence"])
    elif node_name == "generator_node":
        body = _render_generation(payload["generation"])
    else:
        return  # response_formatter is rendered as the final user-visible message
    async with cl.Step(name=label, type="tool") as step:
        step.output = body


@cl.on_chat_start
async def on_start():
    await cl.Message(
        content=(
            "Welcome to **Azure Observable RAG**. Each query streams through "
            "six explicit steps; the trace from each appears in the "
            "Chain-of-Thought panel above the final answer.\n\n"
            "_Try:_ *How do I factory-reset Device A?* — *What's the storage policy?*"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    graph = get_graph()
    config = {"configurable": {"thread_id": cl.context.session.id}}
    node_outputs: dict[str, dict] = {}

    async for event in graph.astream(
        {"user_query": message.content},
        config=config,
        stream_mode="updates",
    ):
        for node_name, payload in event.items():
            node_outputs[node_name] = payload
            await _run_node_step(node_name, payload)

    formatter = node_outputs.get("response_formatter", {})
    final_message = formatter.get("final_message", "(no answer)")
    trace = formatter.get("final_trace", {})
    citations = (trace.get("generation") or {}).get("citations", [])

    sources_md = ""
    if citations:
        sources_md = "\n\n---\n**Sources**\n"
        for i, c in enumerate(citations, start=1):
            page = f" p.{c['page_number']}" if c.get("page_number") is not None else ""
            heading = f" — _{c['heading_path']}_" if c.get("heading_path") else ""
            sources_md += f"{i}. `{c['file_name']}`{page}{heading}\n"

    await cl.Message(content=final_message + sources_md).send()
