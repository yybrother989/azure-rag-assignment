"""
Terminal frontend.

Same LangGraph + same trace objects as the Chainlit UI; different renderer.

  python -m src.cli chat                                    # interactive REPL (Claude-Code style)
  python -m src.cli ask "how do I reset device A"          # one-shot, pretty panels
  python -m src.cli ask "..." --json                        # raw FinalRagTrace JSON
  python -m src.cli search "reset device A" --top-k 5      # retrieval only, no LLM
  python -m src.cli ingest
  python -m src.cli eval
"""

from __future__ import annotations

import json
import os
import sys
import time
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


# ============================================================================
# `cli chat` — interactive REPL, Claude-Code-style
# ============================================================================

# gpt-4o pricing as of 2025-11 (USD per 1M tokens). Update if model changes.
PRICE_PER_M = {
    "gpt-4o":               {"in": 5.00,  "out": 20.00},
    "gpt-4o-2024-11-20":    {"in": 5.00,  "out": 20.00},
    "gpt-4.1-mini":         {"in": 0.40,  "out": 1.60},
    "gpt-4o-mini":          {"in": 0.15,  "out": 0.60},
    "gpt-35-turbo":         {"in": 0.50,  "out": 1.50},
}
EMBED_PRICE_PER_M_IN = 0.02

# Compact node labels rendered as the graph streams updates.
NODE_LABELS = {
    "intent_router":     "① intent",
    "query_planner":     "② plan",
    "retriever_node":    "③ retrieve",
    "evidence_selector": "④ select",
    "generator_node":    "⑤ generate",
    "response_formatter": "⑥ done",
}

SLASH_COMMANDS = [
    "/help", "/exit", "/quit", "/clear", "/cost", "/trace", "/sources",
    "/mode", "/category", "/topk", "/model", "/save",
]


def _price_per_m(model: str) -> dict:
    # gpt-4o-2024-11-20, gpt-4o-mini-2024-07-18 etc. — strip version suffix
    base = model
    for known in PRICE_PER_M:
        if model == known or model.startswith(known + "-"):
            base = known
            break
    return PRICE_PER_M.get(base, PRICE_PER_M["gpt-4o"])


class ChatSession:
    """Persistent REPL with multi-turn memory, slash commands, and live token streaming."""

    def __init__(self) -> None:
        import uuid
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory
        from pathlib import Path

        self.console = Console()
        self.thread_id = str(uuid.uuid4())
        self.turn = 0
        self.session_in_tokens = 0
        self.session_out_tokens = 0
        self.session_cost_usd = 0.0
        self.last_trace: dict | None = None
        self.last_results: list = []          # selected RetrievalResult objects from last turn
        self.transcript: list[dict] = []      # raw turn log for /save
        # Per-next-query overrides (consumed and reset after one query)
        self.override_mode: str | None = None
        self.override_category: str | None = None
        self.override_topk: int | None = None
        # Streaming state for Ctrl-C
        self._streaming = False

        history_file = Path.home() / ".azure_rag_history"
        self.prompt_session = PromptSession(
            history=FileHistory(str(history_file)),
            completer=WordCompleter(SLASH_COMMANDS, ignore_case=True, sentence=True),
            bottom_toolbar=self._bottom_toolbar,
            complete_while_typing=False,
        )

    # ---------- public entrypoint ----------

    def run(self) -> None:
        self._show_banner()
        while True:
            try:
                line = self.prompt_session.prompt("›  ")
            except EOFError:
                self.console.print("\n[dim]bye.[/dim]")
                break
            except KeyboardInterrupt:
                self.console.print("[dim](press Ctrl-D or type /exit to leave)[/dim]")
                continue

            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if self._handle_slash(line) is False:
                    break
            else:
                self.turn += 1
                self._handle_query(line)

    # ---------- query path ----------

    def _handle_query(self, query: str) -> None:
        from .agent import get_graph
        from langchain_core.runnables import RunnableConfig

        graph = get_graph()
        config: RunnableConfig = {
            "configurable": {
                "thread_id": self.thread_id,
                "stream_handler": self._stream_token,
            }
        }

        # Apply pending overrides — set in env so existing nodes pick them up.
        env_snapshot = {}
        if self.override_topk is not None:
            env_snapshot["RETRIEVAL_TOP_K"] = os.environ.get("RETRIEVAL_TOP_K")
            os.environ["RETRIEVAL_TOP_K"] = str(self.override_topk)

        self.console.print()
        try:
            self._streaming = False
            t0 = time.perf_counter()
            for event in graph.stream({"user_query": query}, config=config, stream_mode="updates"):
                for node_name, payload in event.items():
                    self._render_node_status(node_name, payload, query)
            elapsed = time.perf_counter() - t0
            self._streaming = False
        except KeyboardInterrupt:
            self._streaming = False
            self.console.print("\n[yellow](interrupted)[/yellow]")
            return
        finally:
            # Restore env
            for k, v in env_snapshot.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            # One-shot overrides clear after each query
            self.override_mode = None
            self.override_category = None
            self.override_topk = None

        # Final answer + sources rendered after the stream completes.
        self._render_final()
        self.console.print(f"  [dim]⏱ {elapsed:.1f}s   "
                           f"{self.last_trace['generation']['prompt_token_estimate']} in + "
                           f"{self.last_trace['generation']['completion_token_estimate']} out tok   "
                           f"${self._turn_cost():.4f}[/dim]\n")

    def _stream_token(self, token: str) -> None:
        """Called from generate.py for each delta token. Print live with no newline."""
        if not self._streaming:
            # First token of this generation — print the ⑤ streaming status line + leading blank
            self.console.print(f"  [yellow]⠋[/yellow] [cyan]⑤ generate  [/cyan] [dim]streaming…[/dim]\n")
            self._streaming = True
            sys.stdout.write("  ")  # indent for the streamed paragraph
        # Use built-in print to bypass Rich markup escaping for streamed content
        # Re-indent after newlines so the answer body lines up under the status indent
        if "\n" in token:
            token = token.replace("\n", "\n  ")
        sys.stdout.write(token)
        sys.stdout.flush()

    def _render_node_status(self, node_name: str, payload: dict, original_query: str) -> None:
        """Print a one-line status as each LangGraph node finishes."""
        label = NODE_LABELS.get(node_name)
        if not label:
            return

        if node_name == "intent_router":
            qp = payload["query_plan"]
            cat = qp["filters"].get("category", "all")
            self.console.print(f"  [green]✓[/green] [cyan]{label:<12}[/cyan] {qp['intent']}  [dim](category={cat})[/dim]")
        elif node_name == "query_planner":
            qp = payload["query_plan"]
            self.console.print(f"  [green]✓[/green] [cyan]{label:<12}[/cyan] [italic]\"{qp['rewritten_query']}\"[/italic]")
        elif node_name == "retriever_node":
            ret = payload["retrieval"]
            results = ret["results"]
            top = results[0] if results else None
            top_str = (
                f"top: {top['file_name']}" + (f" p.{top['page_number']}" if top.get("page_number") else "")
                if top else "no results"
            )
            self.console.print(
                f"  [green]✓[/green] [cyan]{label:<12}[/cyan] {len(results)} chunks · {ret['latency_ms']}ms · {top_str}"
            )
        elif node_name == "evidence_selector":
            ev = payload["evidence"]
            self.console.print(
                f"  [green]✓[/green] [cyan]{label:<12}[/cyan] {len(ev['selected_chunk_ids'])} of {ev['candidate_count']} by reranker_score"
            )
        elif node_name == "generator_node":
            # End the streamed token output line
            if self._streaming:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._streaming = False
            g = payload["generation"]
            self.console.print(f"  [green]✓[/green] [cyan]{label:<12}[/cyan] [dim]{g['model']} · {g['latency_ms']}ms[/dim]")
        elif node_name == "response_formatter":
            self.last_trace = payload.get("final_trace")
            # last_results: lift from the last retrieval trace's selected chunks
            if self.last_trace:
                ret_results = self.last_trace["retrieval"]["results"]
                selected_ids = set(self.last_trace["evidence_selection"]["selected_chunk_ids"])
                self.last_results = [r for r in ret_results if r["chunk_id"] in selected_ids]
            # Update session cost trackers
            if self.last_trace:
                gen = self.last_trace["generation"]
                self.session_in_tokens += gen["prompt_token_estimate"]
                self.session_out_tokens += gen["completion_token_estimate"]
                self.session_cost_usd += self._turn_cost()
                self.transcript.append({
                    "turn": self.turn,
                    "query": self.last_trace["user_query"],
                    "answer": self.last_trace["generation"]["answer"],
                    "citations": self.last_trace["generation"]["citations"],
                })

    def _render_final(self) -> None:
        """Print sources footer. The answer body itself was already printed live as it
        streamed — re-rendering it as Markdown here would duplicate it on screen."""
        if not self.last_trace:
            return
        gen = self.last_trace["generation"]
        if self.last_trace["query_plan"]["intent"] == "out_of_scope":
            # No streaming happened for out-of-scope; print the canned reply.
            self.console.print(f"\n  [yellow]{gen['answer']}[/yellow]\n")
            return
        if gen["citations"]:
            self.console.print()
            self.console.print("  [bold]Sources[/bold]")
            for i, c in enumerate(gen["citations"], start=1):
                page = f" p.{c['page_number']}" if c.get("page_number") else ""
                heading = f" — {c['heading_path']}" if c.get("heading_path") else ""
                self.console.print(f"    [{i}] {c['file_name']}{page}{heading}")

    # ---------- slash commands ----------

    def _handle_slash(self, line: str) -> bool:
        """Return False to exit the session; True to continue."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        match cmd:
            case "/exit" | "/quit":
                self.console.print("[dim]bye.[/dim]")
                return False
            case "/help":           self._cmd_help()
            case "/clear":          self._cmd_clear()
            case "/cost":           self._cmd_cost()
            case "/trace":          self._cmd_trace()
            case "/sources":        self._cmd_sources()
            case "/mode":           self._cmd_mode(arg)
            case "/category":       self._cmd_category(arg)
            case "/topk":           self._cmd_topk(arg)
            case "/model":          self._cmd_model()
            case "/save":           self._cmd_save(arg)
            case _:
                self.console.print(f"  [yellow]unknown command: {cmd}[/yellow]  (try /help)")
        return True

    def _cmd_help(self) -> None:
        rows = [
            ("/help",                "show this list"),
            ("/exit, /quit, Ctrl-D", "leave the session"),
            ("/clear",               "start a fresh thread (forget conversation memory)"),
            ("/cost",                "show session token + USD totals"),
            ("/trace",               "pretty-print the last query's full FinalRagTrace JSON"),
            ("/sources",             "show full content of last query's selected chunks"),
            ("/mode <m>",            "next query: bm25 | vector | hybrid | hybrid_semantic"),
            ("/category <c>",        "next query: manual | troubleshooting | policy | all"),
            ("/topk <N>",            "next query: change RETRIEVAL_TOP_K"),
            ("/model",               "show current AOAI deployment + Foundry endpoint"),
            ("/save <path>",         "write session transcript to a file"),
        ]
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style="cyan", no_wrap=True)
        t.add_column(style="white")
        for cmd, desc in rows:
            t.add_row(cmd, desc)
        self.console.print(Panel(t, title="commands", border_style="cyan"))

    def _cmd_clear(self) -> None:
        import uuid
        self.thread_id = str(uuid.uuid4())
        self.console.print(f"  [green]new thread[/green] {self.thread_id[:8]}…  [dim](memory cleared)[/dim]")

    def _cmd_cost(self) -> None:
        self.console.print(Panel(
            f"  turns:        {self.turn}\n"
            f"  prompt tok:   {self.session_in_tokens:,}\n"
            f"  output tok:   {self.session_out_tokens:,}\n"
            f"  est. cost:    ${self.session_cost_usd:.4f}  [dim](gpt-4o $5/$20 per 1M)[/dim]",
            title="session cost", border_style="green",
        ))

    def _cmd_trace(self) -> None:
        if not self.last_trace:
            self.console.print("  [yellow]no trace yet — ask a question first[/yellow]")
            return
        self.console.print(Panel(
            Syntax(json.dumps(self.last_trace, indent=2, default=str), "json", theme="ansi_dark"),
            title="last FinalRagTrace", border_style="cyan",
        ))

    def _cmd_sources(self) -> None:
        if not self.last_results:
            self.console.print("  [yellow]no sources yet[/yellow]")
            return
        for i, r in enumerate(self.last_results, start=1):
            page = f" p.{r['page_number']}" if r.get("page_number") else ""
            heading = f" — {r['heading_path']}" if r.get("heading_path") else ""
            header = f"[{i}] {r['file_name']}{page}{heading}  [dim]({r['chunk_id'][:12]}…)[/dim]"
            self.console.print(Panel(r["content"], title=header, border_style="dim"))

    def _cmd_mode(self, arg: str) -> None:
        valid = {"bm25", "vector", "hybrid", "hybrid_semantic"}
        if arg not in valid:
            self.console.print(f"  [yellow]/mode requires one of:[/yellow] {', '.join(sorted(valid))}")
            return
        # NOTE: mode override needs node-level wiring. For now we only show the intent;
        # the existing graph hard-codes hybrid_semantic in intent_router.
        self.override_mode = arg
        self.console.print(f"  [yellow]/mode override is informational in this build[/yellow] "
                           f"[dim](next query intent_router will still pick hybrid_semantic; see roadmap)[/dim]")

    def _cmd_category(self, arg: str) -> None:
        valid = {"manual", "troubleshooting", "policy", "all"}
        if arg not in valid:
            self.console.print(f"  [yellow]/category requires one of:[/yellow] {', '.join(sorted(valid))}")
            return
        self.override_category = None if arg == "all" else arg
        self.console.print(f"  [yellow]/category override is informational in this build[/yellow] "
                           f"[dim](intent_router decides filters; see roadmap)[/dim]")

    def _cmd_topk(self, arg: str) -> None:
        try:
            n = int(arg)
            if n < 1 or n > 50:
                raise ValueError
        except ValueError:
            self.console.print("  [yellow]/topk requires an integer 1-50[/yellow]")
            return
        self.override_topk = n
        self.console.print(f"  [green]top_k for next query → {n}[/green]")

    def _cmd_model(self) -> None:
        chat = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "?")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "?")
        foundry = os.environ.get("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT") or "(direct AOAI, no Foundry routing)"
        self.console.print(Panel(
            f"  chat deployment:  [cyan]{chat}[/cyan]\n"
            f"  AOAI endpoint:    [dim]{endpoint}[/dim]\n"
            f"  Foundry project:  [dim]{foundry}[/dim]",
            title="model", border_style="cyan",
        ))

    def _cmd_save(self, arg: str) -> None:
        if not arg:
            self.console.print("  [yellow]/save requires a path, e.g. /save chat.md[/yellow]")
            return
        from pathlib import Path
        p = Path(arg).expanduser()
        with p.open("w", encoding="utf-8") as f:
            f.write(f"# Azure Observable RAG — chat transcript\n\nthread: `{self.thread_id}`\n\n")
            for entry in self.transcript:
                f.write(f"## Turn {entry['turn']}\n\n**Q:** {entry['query']}\n\n**A:** {entry['answer']}\n\n")
                if entry["citations"]:
                    f.write("**Sources:**\n")
                    for c in entry["citations"]:
                        page = f" p.{c['page_number']}" if c.get("page_number") else ""
                        f.write(f"- `{c['file_name']}`{page}\n")
                f.write("\n")
        self.console.print(f"  [green]wrote[/green] {p}  [dim]({len(self.transcript)} turns)[/dim]")

    # ---------- presentation ----------

    def _show_banner(self) -> None:
        chat = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "?")
        index = os.environ.get("AZURE_SEARCH_INDEX", "?")
        body = (
            f"  model    [cyan]{chat}[/cyan]  [dim](swedencentral · Foundry-AAD)[/dim]\n"
            f"  index    [cyan]{index}[/cyan]\n"
            f"  thread   [dim]{self.thread_id[:8]}…[/dim]\n"
            f"  help     [dim]/help    exit  /exit   ctrl-c interrupt mid-stream[/dim]"
        )
        self.console.print(Panel(body, title="Azure Observable RAG · chat", border_style="cyan"))

    def _bottom_toolbar(self):
        from prompt_toolkit.formatted_text import HTML
        chat = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "?")
        return HTML(
            f" <b>{chat}</b> · turn {self.turn} · "
            f"session: {self.session_in_tokens:,} in + {self.session_out_tokens:,} out tok · "
            f"${self.session_cost_usd:.4f} · /help "
        )

    def _turn_cost(self) -> float:
        if not self.last_trace:
            return 0.0
        gen = self.last_trace["generation"]
        p = _price_per_m(gen["model"])
        return (
            gen["prompt_token_estimate"] / 1_000_000 * p["in"]
            + gen["completion_token_estimate"] / 1_000_000 * p["out"]
        )


@app.command("chat")
def cmd_chat() -> None:
    """Interactive REPL — multi-turn, streamed, slash-commanded. Claude-Code-style."""
    ChatSession().run()


if __name__ == "__main__":
    app()
