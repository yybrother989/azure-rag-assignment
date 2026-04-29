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
import re
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
from langchain_core.runnables import RunnableConfig  # noqa: E402

load_dotenv()

from src.agent import get_graph  # noqa: E402  — env must load before graph init

NODE_LABELS = {
    "intent_router":          "① Route Classification",
    "no_retrieval_response":  "② Direct Response (no search)",
    "build_retrieval_scope":  "② Retrieval Scope",
    "simple_rag_search":      "③ Hybrid Search",
    "evidence_selector":      "④ Evidence Selection",
    "generator_node":         "⑤ Generation",
    "response_formatter":     "⑥ Final Answer",
}


def _render_intent(qp: dict) -> str:
    route = qp.get("route") or qp.get("intent", "—")
    device_family = qp.get("detected_device_family") or "—"
    device = qp.get("detected_device") or "—"
    doc_type = qp.get("detected_doc_type") or "—"
    topic = qp.get("detected_topic") or "—"
    error_code = qp.get("detected_error_code") or "—"
    fallback = "yes" if qp.get("allow_shared_fallback") else "no"
    return (
        f"**route:** `{route}`  \n"
        f"**query_type:** `{qp.get('query_type') or qp.get('intent', '—')}`  \n"
        f"**device_family:** `{device_family}` &nbsp; **device:** `{device}`  \n"
        f"**doc_type:** `{doc_type}`  \n"
        f"**topic:** `{topic}` &nbsp; **error_code:** `{error_code}`  \n"
        f"**shared_fallback:** `{fallback}`  \n"
        f"**reason:** {qp.get('router_reason') or qp.get('notes') or '—'}"
    )


def _render_scope(qp: dict) -> str:
    primary = qp.get("primary_filter") or {}
    fallback = qp.get("fallback_filter")
    return (
        f"**primary filter:** `{json.dumps(primary) if primary else '(none — full corpus)'}`  \n"
        f"**fallback filter:** `{json.dumps(fallback) if fallback else '(none)'}`  \n"
        f"**search mode:** `{qp.get('search_mode', 'hybrid_semantic')}` "
        f"&nbsp; **top_k:** `{qp.get('top_k', 5)}`"
    )


_MD_STRIP_PATTERNS = [
    (re.compile(r"<!--.*?-->", re.DOTALL), ""),                  # HTML comments (page markers, etc.)
    (re.compile(r"</?(?:figure|table|tr|td|th|tbody|thead|figcaption)[^>]*>"), ""),  # block HTML tags
    (re.compile(r"^\s*#{1,6}\s*", re.MULTILINE), ""),            # heading marks
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),                   # images
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),               # links → label only
    (re.compile(r"`([^`]+)`"), r"\1"),                           # inline code
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),                     # **bold**
    (re.compile(r"__([^_]+)__"), r"\1"),                         # __bold__
    (re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)"), r"\1"),            # *em*
    (re.compile(r"(?<!_)_([^_]+)_(?!_)"), r"\1"),                # _em_
    (re.compile(r"\|"), " "),                                    # table pipes
    (re.compile(r"\s+"), " "),                                   # collapse whitespace
]


def _humanize_markdown(text: str) -> str:
    """Best-effort markdown→prose for display in trace tables. Not a full
    parser — handles the noise common in DI's output (headings, table pipes,
    HTML comments, inline emphasis) so retrieval previews read like text."""
    if not text:
        return ""
    out = text
    for pattern, repl in _MD_STRIP_PATTERNS:
        out = pattern.sub(repl, out)
    return out.strip()


def _render_retrieval(retrieval: dict) -> str:
    fallback_note = " _(shared-doc fallback triggered)_" if retrieval.get("fallback_triggered") else ""
    rows = [
        f"| # | Lv1 Rank | Lv2 Rank | Scope | Device | File | Page | Section | Caption / Preview |",
        "|---|--------:|---------:|-------|--------|------|-----:|---------|-------------------|",
    ]
    for r in retrieval["results"]:
        raw_cap = r.get("semantic_caption") or r.get("content_preview", "")
        cap = _humanize_markdown(raw_cap)[:120]
        rerank = r.get("reranker_score")
        rerank_cell = f"{rerank:.3f}" if rerank is not None else "—"
        scope = r.get("scope") or "—"
        device = r.get("device") or "—"
        rows.append(
            f"| {r['rank']} "
            f"| {r['score']:.3f} "
            f"| {rerank_cell} "
            f"| {scope} "
            f"| {device} "
            f"| `{r.get('file_name','')}` "
            f"| {r.get('page_number') or '—'} "
            f"| {r.get('heading_path') or '—'} "
            f"| {cap} |"
        )
    rows.append(f"\n_latency: {retrieval['latency_ms']} ms{fallback_note}_")
    return "\n".join(rows)


def _render_evidence(ev: dict, ranked: list[dict] | None = None) -> str:
    """Show selected sources as `[N] file p.M` (matches the [N] tokens the
    LLM emits in its answer). `ranked` is the same selected-order list the
    generator received; if absent (e.g. older trace replay), fall back to
    the truncated chunk_ids."""
    selected = ev.get("selected_chunk_ids", [])
    if ranked:
        # ranked is in the same order the generator's [N] indices use.
        labels = []
        for i, r in enumerate(ranked, start=1):
            page = r.get("page_number")
            page_part = f" p.{page}" if page else ""
            labels.append(f"`[{i}] {r.get('file_name','?')}{page_part}`")
        sources_md = ", ".join(labels)
    else:
        sources_md = ", ".join(f"`{c[:8]}…`" for c in selected)
    return (
        f"**strategy:** `{ev['selection_strategy']}`  \n"
        f"**candidates:** {ev['candidate_count']} → **selected:** {len(selected)}  \n"
        f"**rationale:** {ev.get('rationale') or '—'}  \n"
        f"**sources:** {sources_md}"
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
_DOC_TYPE_TO_CATEGORY = {"manuals": "manual", "troubleshooting": "troubleshooting", "policies": "policy"}


def _category_from_blob_path(blob_path: str) -> str:
    """Extract category from both legacy (manuals/file) and new (devices/.../doc_type/file
    or shared/doc_type/file) path layouts."""
    parts = blob_path.split("/")
    if parts[0] == "devices" and len(parts) >= 5:
        return _DOC_TYPE_TO_CATEGORY.get(parts[3], "other")
    if parts[0] == "shared" and len(parts) >= 3:
        return _DOC_TYPE_TO_CATEGORY.get(parts[1], "other")
    return _BLOB_TOP_TO_CATEGORY.get(parts[0], "other")


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


FIGURES_PAGE_WINDOW = 2     # ± pages around each cited page to harvest figures from
FIGURES_MAX_PER_ANSWER = 6  # cap so a hit on a figure-heavy page doesn't flood the chat


def _fetch_nearby_figures(source_path: str, pages: set[int]) -> list[dict]:
    """For a cited document + the set of cited pages, query AI Search for any
    chunks within ±FIGURES_PAGE_WINDOW pages that carry figures. Returns the
    deduped figure dicts (`{figure_id, page, blob_url, caption}`).

    Why this exists: DI often tags a figure to the page where its bounding box
    starts, while the surrounding paragraph (which the LLM cites) gets tagged
    to the previous page. Exact-page attribution misses these. Widening to
    ±2 pages catches the common off-by-one without flooding the answer."""
    if not pages:
        return []
    from src.index import get_search_client

    min_p, max_p = min(pages) - FIGURES_PAGE_WINDOW, max(pages) + FIGURES_PAGE_WINDOW
    escaped = source_path.replace("'", "''")
    odata = (
        f"source_path eq '{escaped}' "
        f"and page_number ge {min_p} and page_number le {max_p}"
    )
    try:
        results = get_search_client().search(
            search_text="*",
            filter=odata,
            select=["page_number", "figures_json"],
            top=50,
        )
    except Exception as e:
        print(f"[app] nearby-figures fetch failed for {source_path}: {type(e).__name__}: {e}")
        return []

    seen_urls: set[str] = set()
    figures: list[dict] = []
    for doc in results:
        fj = doc.get("figures_json") or ""
        if not fj:
            continue
        try:
            for fig in json.loads(fj):
                url = fig.get("blob_url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                figures.append(fig)
        except json.JSONDecodeError:
            continue
    return figures


def _attach_cited_files_to_message(citations: list[dict]) -> list:
    """Build the answer message's elements:
      - one side-panel viewer per unique cited source_path (cl.Pdf / cl.Text)
      - one inline cl.Image per figure found on or near the cited pages

    Figures from cited chunks themselves go first (most relevant); then we
    fan out to nearby pages of the same source to catch DI's page off-by-one.
    Capped at FIGURES_MAX_PER_ANSWER to avoid flooding."""
    elements = []
    seen_files: set[str] = set()
    seen_figs: set[str] = set()
    pages_by_source: dict[str, set[int]] = defaultdict(set)

    # Pass 1: side-panel viewers + figures already on the cited chunks themselves.
    for c in citations:
        sp = c.get("source_path")
        if sp and sp not in seen_files:
            seen_files.add(sp)
            elem = _build_element(sp, c.get("file_name", sp))
            if elem is not None:
                elements.append(elem)
        page = c.get("page_number")
        if sp and page is not None:
            pages_by_source[sp].add(page)
        for fig in c.get("figures") or []:
            url = fig.get("blob_url")
            if not url or url in seen_figs:
                continue
            seen_figs.add(url)
            elements.append(_figure_element(fig))

    # Pass 2: expand to nearby pages of the same cited PDFs.
    for sp, pages in pages_by_source.items():
        for fig in _fetch_nearby_figures(sp, pages):
            url = fig.get("blob_url")
            if not url or url in seen_figs:
                continue
            if sum(1 for e in elements if isinstance(e, cl.Image)) >= FIGURES_MAX_PER_ANSWER:
                break
            seen_figs.add(url)
            elements.append(_figure_element(fig))
    return elements


def _figure_element(fig: dict):
    """Build one inline `cl.Image` from a figure dict. Caption (when present)
    becomes part of the visible label so the user knows what they're looking at."""
    page = fig.get("page")
    caption = (fig.get("caption") or "").strip()
    label = f"page {page}" if page else "figure"
    if caption:
        label = f"{label} — {caption[:80]}"
    return cl.Image(name=label, url=fig["blob_url"], display="inline")


def _build_inventory() -> dict:
    """Return a nested folder tree mirroring the blob container path structure.

    Each internal node is a dict of {folder_name: subtree}.
    Leaf nodes store files under the special key "_files": [file_meta, ...].
    file_meta keys: file_name, source_path, file_type, chunk_count, status.

    Status is 'indexed' if the file appears in AI Search, 'pending' if it's only
    in the blob container (uploaded but not yet ingested)."""
    from src.index import get_search_client

    # source_path → file_meta (deduped by source_path, chunk_count accumulated)
    by_path: dict[str, dict] = {}
    try:
        client = get_search_client()
        results = client.search(
            search_text="*",
            select=["file_name", "source_path", "file_type"],
            top=1000,
        )
        for doc in results:
            sp = doc.get("source_path", "")
            fn = doc.get("file_name")
            if not sp or not fn:
                continue
            if sp not in by_path:
                by_path[sp] = {
                    "file_name": fn,
                    "source_path": sp,
                    "file_type": doc.get("file_type", ""),
                    "chunk_count": 0,
                    "status": "indexed",
                }
            by_path[sp]["chunk_count"] += 1
    except Exception as e:
        print(f"[app] inventory: index query failed: {type(e).__name__}: {e}")

    # Blob diff — anything in container but not in AI Search is pending ingestion.
    try:
        from src.ingest import _blob_service_client
        container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "kb-docs")
        container = _blob_service_client().get_container_client(container_name)
        for blob in container.list_blobs():
            if blob.name not in by_path:
                by_path[blob.name] = {
                    "file_name": Path(blob.name).name,
                    "source_path": blob.name,
                    "file_type": Path(blob.name).suffix.lstrip(".").lower(),
                    "chunk_count": 0,
                    "status": "pending",
                }
    except Exception as e:
        print(f"[app] inventory: blob list failed: {type(e).__name__}: {e}")

    # Build nested tree from flat source_paths.
    tree: dict = {}
    for meta in by_path.values():
        parts = meta["source_path"].split("/")
        node = tree
        for part in parts[:-1]:          # walk/create folder nodes
            node = node.setdefault(part, {})
        node.setdefault("_files", []).append(meta)

    # Sort files within each leaf alphabetically.
    def _sort_tree(node: dict) -> None:
        if "_files" in node:
            node["_files"].sort(key=lambda m: m["file_name"])
        for k, v in node.items():
            if k != "_files":
                _sort_tree(v)

    _sort_tree(tree)
    return tree


async def _send_inventory() -> None:
    inv = _build_inventory()
    elem = cl.CustomElement(name="FileTree", props={"tree": inv}, display="inline")
    await cl.Message(content="", elements=[elem]).send()


@cl.action_callback("open_file")
async def on_open_file(action: cl.Action) -> None:
    sp = action.payload["source_path"]  # type: ignore[attr-defined]
    fn = action.payload["file_name"]  # type: ignore[attr-defined]
    elem = _build_element(sp, fn)
    if elem is None:
        await cl.Message(
            content=f"❌ Couldn't load `{fn}` (not in `data/` and not in blob container, or unsupported file type)."
        ).send()
        return
    await cl.ElementSidebar.set_elements([elem])
    await cl.ElementSidebar.set_title(f"📖 {fn}")


@cl.action_callback("refresh_file_tree")
async def on_refresh_file_tree(action: cl.Action) -> None:
    await _send_inventory()


# ============================================================================


async def _run_node_step(node_name: str, payload: dict) -> None:
    label = NODE_LABELS.get(node_name, node_name)
    if node_name == "intent_router":
        body = _render_intent(payload.get("query_plan") or {})
    elif node_name == "build_retrieval_scope":
        body = _render_scope(payload.get("query_plan") or {})
    elif node_name == "no_retrieval_response":
        gen = payload.get("generation") or {}
        body = f"**route:** `NO_RETRIEVAL` — answered directly without search  \n\n{gen.get('answer', '')}"
    elif node_name == "simple_rag_search":
        body = _render_retrieval(payload.get("retrieval") or {"results": [], "latency_ms": 0})
    elif node_name == "evidence_selector":
        body = _render_evidence(payload.get("evidence") or {}, payload.get("retrieval_results"))
    elif node_name == "generator_node":
        body = _render_generation(payload.get("generation") or {})
    else:
        return  # response_formatter is rendered as the final user-visible message
    async with cl.Step(name=label, type="tool") as step:
        step.output = body


@cl.on_chat_start
async def on_start():
    await cl.Message(
        content=(
            "Hi! I'm your **Retail IT Support Assistant**. I can help with hardware "
            "troubleshooting, setup guides, and company IT policies for your in-store devices.\n\n"
            "**Devices I know about:**\n"
            "- 🖨️ Epson TM-M30II receipt printer\n"
            "- 💳 Ingenico Desk5000 payment terminal\n"
            "- 📄 Canon CR-120 check scanner\n"
            "- 🌐 Meraki MX67 network appliance\n\n"
            "**Try asking:**\n"
            "- *The receipt printer is showing a red LED — what does it mean?*\n"
            "- *How do I load paper in the Epson TM-M30II?*\n"
            "- *The Ingenico terminal won't read cards — how do I fix it?*\n"
            "- *What's our policy on third-party software on store devices?*\n\n"
            "Each answer shows its reasoning steps live above the response, "
            "and every claim links back to the source document."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    await cl.ElementSidebar.set_elements([])
    text = message.content.strip()

    # Inventory command — exact, case-insensitive match. Anything else (incl.
    # "files about X") falls through to the LangGraph as a normal query.
    if text.lower() in INVENTORY_TRIGGERS:
        await _send_inventory()
        return

    graph = get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": cl.context.session.id}}
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
        for c in citations:
            # Use the original [N] index the LLM emitted — falls back to a
            # positional counter only if an older trace lacks the field.
            n = c.get("index") or (citations.index(c) + 1)
            page = f" p.{c['page_number']}" if c.get("page_number") is not None else ""
            heading = f" — _{c['heading_path']}_" if c.get("heading_path") else ""
            sources_md += f"{n}. `{c['file_name']}`{page}{heading}\n"

    # Attach cited source files as side-panel elements — one per unique source_path.
    elements = _attach_cited_files_to_message(citations) if citations else []

    await cl.Message(content=final_message + sources_md, elements=elements).send()
