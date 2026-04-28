"""
Terminal frontend.

Same LangGraph + same trace objects as the Chainlit UI; different renderer.

  python -m src.cli ask "how do I reset device A"          # pretty panels (default)
  python -m src.cli ask "..." --json                        # raw FinalRagTrace JSON
  python -m src.cli ask "..." --filter category=manual --top-k 5
  python -m src.cli search "reset device A" --top-k 5 --mode hybrid_semantic
  python -m src.cli ingest
  python -m src.cli eval
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

load_dotenv()

app = typer.Typer(add_completion=False, help="Azure Observable RAG — terminal frontend.")
console = Console()


# ---------- shared helpers ----------

def _parse_filters(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"--filter must be field=value (got {item!r})")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _retrieval_table(results: list[dict], *, title: str = "Retrieval Results") -> Table:
    t = Table(title=title, show_lines=False, header_style="bold cyan")
    t.add_column("#", justify="right", width=3)
    t.add_column("score", justify="right", width=7)
    t.add_column("rerank", justify="right", width=7)
    t.add_column("file", overflow="fold")
    t.add_column("p.", justify="right", width=4)
    t.add_column("heading", overflow="fold")
    t.add_column("caption / preview", overflow="fold", ratio=2)
    for r in results:
        cap = r.get("semantic_caption") or r.get("content_preview", "")
        t.add_row(
            str(r["rank"]),
            f"{r['score']:.3f}",
            f"{r['reranker_score']:.3f}" if r.get("reranker_score") is not None else "—",
            r.get("file_name", ""),
            str(r.get("page_number") or "—"),
            r.get("heading_path") or "—",
            cap[:160] + ("…" if len(cap) > 160 else ""),
        )
    return t


# ---------- pretty trace renderer ----------

def _render_trace(trace: dict) -> None:
    qp = trace["query_plan"]
    console.print(Panel(
        Text.assemble(
            ("intent: ", "bold"), (qp["intent"], "cyan"), "\n",
            ("filters: ", "bold"), (json.dumps(qp["filters"]), "white"), "\n",
            ("rationale: ", "bold"), (qp.get("notes") or "—", "italic dim"),
        ),
        title="① Intent Detection", border_style="cyan",
    ))

    console.print(Panel(
        Text.assemble(
            ("original:  ", "bold"), (qp["original_query"], "white"), "\n",
            ("rewritten: ", "bold"), (qp["rewritten_query"], "green"), "\n",
            ("mode: ", "bold"), (qp["search_mode"], "white"),
            ("    top_k: ", "bold"), (str(qp["top_k"]), "white"),
        ),
        title="② Query Planning", border_style="cyan",
    ))

    ret = trace["retrieval"]
    console.print(Panel(
        _retrieval_table(ret["results"], title=f"③ Retrieval Results  ({ret['latency_ms']} ms)"),
        border_style="cyan",
    ))

    ev = trace["evidence_selection"]
    console.print(Panel(
        Text.assemble(
            ("strategy: ", "bold"), (ev["selection_strategy"], "white"), "\n",
            ("candidates: ", "bold"), (str(ev["candidate_count"]), "white"),
            ("    selected: ", "bold"), (str(len(ev["selected_chunk_ids"])), "green"), "\n",
            ("rationale: ", "bold"), (ev.get("rationale") or "—", "italic dim"), "\n",
            ("ids: ", "bold"),
            (", ".join(c[:8] for c in ev["selected_chunk_ids"]), "white"),
        ),
        title="④ Evidence Selection", border_style="cyan",
    ))

    g = trace["generation"]
    console.print(Panel(
        Text.assemble(
            ("model: ", "bold"), (g["model"], "white"),
            ("    latency: ", "bold"), (f"{g['latency_ms']} ms", "white"), "\n",
            ("prompt~tok: ", "bold"), (str(g["prompt_token_estimate"]), "white"),
            ("    completion~tok: ", "bold"), (str(g["completion_token_estimate"]), "white"),
        ),
        title="⑤ Generation", border_style="cyan",
    ))

    answer_lines = [g["answer"]]
    if g["citations"]:
        answer_lines.append("")
        answer_lines.append("[bold]Sources:[/bold]")
        for i, c in enumerate(g["citations"], start=1):
            page = f" p.{c['page_number']}" if c.get("page_number") is not None else ""
            heading = f' — {c["heading_path"]}' if c.get("heading_path") else ""
            answer_lines.append(
                f"  [{i}] {c['file_name']}{page}{heading}  [dim]({c['chunk_id'][:12]}…)[/dim]"
            )
    console.print(Panel(
        "\n".join(answer_lines),
        title=f"⑥ Final Answer  (total {trace['total_latency_ms']} ms)",
        border_style="green",
    ))


# ---------- commands ----------

@app.command("ask")
def cmd_ask(
    query: str = typer.Argument(..., help="Question to ask the knowledge base."),
    json_out: bool = typer.Option(False, "--json", help="Emit raw FinalRagTrace JSON."),
    filter_: list[str] = typer.Option(
        [], "--filter", help="Override router filters, e.g. category=manual. Repeatable."
    ),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Override RETRIEVAL_TOP_K."),
):
    """Run the full LangGraph: intent → plan → retrieve → select → generate → final answer."""
    from .agent import run

    if filter_:
        os.environ.setdefault("_CLI_FILTER_OVERRIDE", "1")  # informational only
        # We don't have a clean override path inside the graph yet; rely on env-driven
        # defaults. For now, log a warning if user passed --filter (stubbed for future).
        console.print(
            f"[yellow]warning:[/yellow] --filter is informational only in this build; "
            f"the intent_router decides filters. Got: {_parse_filters(filter_)}"
        )
    if top_k is not None:
        os.environ["RETRIEVAL_TOP_K"] = str(top_k)

    trace = run(query)
    if json_out:
        sys.stdout.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
    else:
        _render_trace(trace)


@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(5, "--top-k"),
    mode: str = typer.Option(
        "hybrid_semantic", "--mode",
        help="bm25 | vector | hybrid | hybrid_semantic",
    ),
    filter_: list[str] = typer.Option([], "--filter", help="field=value, repeatable"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Retrieval ONLY — no LLM calls. Useful for auditing the index in isolation."""
    from .search import hybrid_search

    filters = _parse_filters(filter_)
    results = hybrid_search(query, filters=filters or None, top_k=top_k, search_mode=mode)
    payload = [r.to_dict() for r in results]
    if json_out:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    else:
        console.print(_retrieval_table(payload, title=f"Retrieval Results — mode={mode}, top_k={top_k}"))


@app.command("ingest")
def cmd_ingest(
    data_dir: str = typer.Option("data", "--data-dir"),
):
    """Walk /data, upload to blob, extract+chunk+embed, upsert into AI Search."""
    from .ingest import ingest

    summary = ingest(data_dir)
    console.print(Panel(
        Syntax(json.dumps(summary, indent=2), "json", theme="ansi_dark"),
        title="Ingestion summary", border_style="green",
    ))


@app.command("eval")
def cmd_eval(
    gold: str = typer.Option("notebooks/gold_qa.json", "--gold", help="Gold Q&A JSON file."),
):
    """Run the retrieval + generation eval harness in scripted mode."""
    try:
        from .eval_harness import run_eval
    except ImportError:
        console.print(
            "[yellow]eval harness not available as a script — open notebooks/eval.ipynb instead.[/yellow]"
        )
        raise typer.Exit(code=2)
    summary = run_eval(gold_path=gold)
    console.print(Panel(
        Syntax(json.dumps(summary, indent=2), "json", theme="ansi_dark"),
        title="Eval summary", border_style="green",
    ))


if __name__ == "__main__":
    app()
