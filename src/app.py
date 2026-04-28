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
import os
import sys
import tempfile
from collections import defaultdict
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


# ============================================================================
# File inventory + cited-source attachment
# ============================================================================

CATEGORY_ICONS = {"manual": "📘", "troubleshooting": "🔧", "policy": "📋", "other": "📄"}
CATEGORY_ORDER = ["manual", "troubleshooting", "policy", "other"]
INVENTORY_TRIGGERS = {"files", "/files", "list", "/list", "index", "/index"}
TEXT_EXTENSIONS = {".md", ".txt", ".html", ".htm"}
_BLOB_TOP_TO_CATEGORY = {"manuals": "manual", "troubleshooting": "troubleshooting", "policies": "policy"}


def _resolve_file(source_path: str) -> Path | None:
    """Return a local Path for the given blob source_path. Try data/ first,
    fall back to a temp-dir cache populated from blob storage on first miss."""
    local = _REPO_ROOT / "data" / source_path
    if local.exists():
        return local

    cache_dir = Path(tempfile.gettempdir()) / "azure-rag-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / source_path.replace("/", "_")
    if cached.exists():
        return cached

    try:
        from src.ingest import _blob_service_client
        container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "kb-docs")
        container = _blob_service_client().get_container_client(container_name)
        with cached.open("wb") as f:
            f.write(container.download_blob(source_path).readall())
        return cached
    except Exception as e:
        print(f"[app] failed to fetch {source_path} from blob: {type(e).__name__}: {e}")
        return None


def _build_element(source_path: str, file_name: str):
    """Construct the right Chainlit element for a file path. None if unsupported / missing."""
    local = _resolve_file(source_path)
    if local is None:
        return None
    suffix = local.suffix.lower()
    if suffix == ".pdf":
        return cl.Pdf(name=file_name, path=str(local), display="side")
    if suffix in TEXT_EXTENSIONS:
        try:
            content = local.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        language = "markdown" if suffix == ".md" else None
        return cl.Text(name=file_name, content=content, display="side", language=language)
    return None


def _attach_cited_files_to_message(citations: list[dict]) -> list:
    """Build a deduped list of side-panel elements, one per unique cited source_path."""
    elements = []
    seen = set()
    for c in citations:
        sp = c.get("source_path")
        if not sp or sp in seen:
            continue
        seen.add(sp)
        elem = _build_element(sp, c.get("file_name", sp))
        if elem is not None:
            elements.append(elem)
    return elements


def _build_inventory() -> dict:
    """Return {category: [file_meta, ...]} where file_meta has chunk_count + status.

    Status is 'indexed' if the file appears in AI Search, 'pending' if it's only
    in the blob container (uploaded but not yet ingested)."""
    from src.index import get_search_client

    indexed: dict[str, dict] = {}
    try:
        client = get_search_client()
        # Walk all docs (small index — we have <30 chunks). For large indexes,
        # switch to the facets API: facets=["file_name,count:200"].
        results = client.search(
            search_text="*",
            select=["file_name", "source_path", "category", "file_type"],
            top=1000,
        )
        for doc in results:
            fn = doc.get("file_name")
            if not fn:
                continue
            if fn not in indexed:
                indexed[fn] = {
                    "file_name": fn,
                    "source_path": doc.get("source_path", ""),
                    "category": doc.get("category", "other"),
                    "file_type": doc.get("file_type", ""),
                    "chunk_count": 0,
                    "status": "indexed",
                }
            indexed[fn]["chunk_count"] += 1
    except Exception as e:
        print(f"[app] inventory: index query failed: {type(e).__name__}: {e}")

    # Blob diff — anything in container but not in AI Search is pending ingestion.
    try:
        from src.ingest import _blob_service_client
        container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "kb-docs")
        container = _blob_service_client().get_container_client(container_name)
        for blob in container.list_blobs():
            blob_name = Path(blob.name).name
            if blob_name in indexed:
                continue
            top = blob.name.split("/", 1)[0] if "/" in blob.name else "other"
            indexed[blob_name] = {
                "file_name": blob_name,
                "source_path": blob.name,
                "category": _BLOB_TOP_TO_CATEGORY.get(top, "other"),
                "file_type": Path(blob.name).suffix.lstrip(".").lower(),
                "chunk_count": 0,
                "status": "pending",
            }
    except Exception as e:
        print(f"[app] inventory: blob list failed: {type(e).__name__}: {e}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for meta in indexed.values():
        grouped[meta["category"]].append(meta)
    for cat in grouped:
        grouped[cat].sort(key=lambda m: m["file_name"])
    return dict(grouped)


async def _send_inventory() -> None:
    inv = _build_inventory()
    total_files = sum(len(v) for v in inv.values())
    total_chunks = sum(m["chunk_count"] for v in inv.values() for m in v)
    pending = sum(1 for v in inv.values() for m in v if m["status"] == "pending")

    body_lines = [
        f"📚 **Knowledge base** — {total_files} files · {total_chunks} chunks"
        + (f" · ⚠️ {pending} pending ingestion" if pending else ""),
        "",
    ]
    actions = []
    for cat in CATEGORY_ORDER:
        if cat not in inv:
            continue
        files = inv[cat]
        icon = CATEGORY_ICONS.get(cat, "📄")
        body_lines.append(f"### {icon} {cat} ({len(files)})")
        for m in files:
            badge = "" if m["status"] == "indexed" else "  ⚠️ pending ingestion"
            body_lines.append(f"- `{m['file_name']}` · {m['chunk_count']} chunks{badge}")
            actions.append(cl.Action(
                name="open_file",
                payload={"source_path": m["source_path"], "file_name": m["file_name"]},
                label=f"📂 {m['file_name']}",
            ))
        body_lines.append("")

    body_lines.append("_Click any 📂 button to open the source file in the side panel._")
    await cl.Message(content="\n".join(body_lines), actions=actions).send()


@cl.action_callback("open_file")
async def on_open_file(action: cl.Action) -> None:
    sp = action.payload["source_path"]
    fn = action.payload["file_name"]
    elem = _build_element(sp, fn)
    if elem is None:
        await cl.Message(
            content=f"❌ Couldn't load `{fn}` (not in `data/` and not in blob container, or unsupported file type)."
        ).send()
        return
    await cl.Message(content=f"📖 Opened `{fn}` in the side panel →", elements=[elem]).send()


# ============================================================================


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
    text = message.content.strip()

    # Inventory command — exact, case-insensitive match. Anything else (incl.
    # "files about X") falls through to the LangGraph as a normal query.
    if text.lower() in INVENTORY_TRIGGERS:
        await _send_inventory()
        return

    graph = get_graph()
    config = {"configurable": {"thread_id": cl.context.session.id}}
    node_outputs: dict[str, dict] = {}

    async for event in graph.astream(
        {"user_query": text},
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

    # Attach cited source files as side-panel elements — one per unique source_path.
    elements = _attach_cited_files_to_message(citations) if citations else []

    await cl.Message(content=final_message + sources_md, elements=elements).send()
