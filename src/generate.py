"""
Generation ONLY.

Takes a query plus the chunks the evidence-selector picked and returns a
grounded answer with explicit `[N]` citations (1-indexed against the supplied
context). The system prompt forbids the model from drawing on anything
outside the supplied context.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

import tiktoken
from openai.types.chat import ChatCompletionMessageParam

from .embed import get_azure_openai_client
from .search import RetrievalResult

CITATION_PATTERN = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You are a knowledge-base assistant for product manuals, troubleshooting "
    "guides, and policies.\n"
    "RULES:\n"
    "1. Use ONLY the provided context chunks. Never use outside knowledge.\n"
    "2. If the context does not contain enough information to answer, say so "
    "plainly and stop. Do not guess.\n"
    "3. Cite every factual claim with the source's bracketed index, e.g. [1] "
    "or [2]. The indices are the numbers shown before each chunk in the "
    "context. Use ONLY indices that appear in the supplied context — do not "
    "invent or extrapolate higher numbers.\n"
    "4. Be concise and concrete. Prefer numbered steps for procedures.\n"
    "5. Some chunks include rendered Markdown tables and 'Figure description' "
    "lines from a vision model — treat these as authoritative source content. "
    "When a figure description names callouts (① ② ③ etc.), reference them by "
    "the same numbers in your answer so the user can match text to diagram.\n"
)


@dataclass
class Citation:
    chunk_id: str
    source_path: str
    file_name: str
    page_number: int | None
    heading_path: str | None
    figures: list[dict] = field(default_factory=list)
    # The 1-based index the LLM used in its answer (e.g. [4] → index=4).
    # Preserved so the Sources footer can label "4." instead of renumbering
    # in citation-mention order, which would mismatch the in-text [N] tokens.
    index: int = 0


@dataclass
class GenerationOutput:
    answer: str
    citations: list[Citation]
    model: str
    context_chunk_ids: list[str]
    prompt_token_estimate: int
    completion_token_estimate: int

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _render_table_as_markdown(table: dict) -> str:
    """Convert a structured table {headers, rows} into GitHub-flavored Markdown.

    Done at prompt-construction time (not at index time) so the model sees a
    clean table even when DI's inline HTML has been chopped by chunking. Empty
    headers fall back to col1/col2/... so the separator row stays valid."""
    headers = [h or f"col{i+1}" for i, h in enumerate(table.get("headers") or [])]
    rows = table.get("rows") or []
    if not headers and rows:
        headers = [f"col{i+1}" for i in range(len(rows[0]))]
    if not headers:
        return ""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [(str(c) if c is not None else "").replace("\n", " ").replace("|", "\\|") for c in row]
        # Pad/truncate to header width so Markdown stays well-formed.
        if len(cells) < len(headers):
            cells = cells + [""] * (len(headers) - len(cells))
        else:
            cells = cells[: len(headers)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_context(chunks: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        loc = c.file_name
        if c.page_number is not None:
            loc += f" p.{c.page_number}"
        heading = f' — "{c.heading_path}"' if c.heading_path else ""
        body = c.content
        # If this chunk carries structured tables, append clean Markdown
        # renderings so the model can reason over rows even if the inline HTML
        # in `content` is fragmented.
        if c.tables:
            rendered: list[str] = []
            for t in c.tables:
                md = _render_table_as_markdown(t)
                if not md:
                    continue
                page_tag = f" (p.{t.get('page')})" if t.get("page") else ""
                rendered.append(f"Table{page_tag}:\n{md}")
            if rendered:
                body = body + "\n\n" + "\n\n".join(rendered)
        blocks.append(f"[{i}] ({loc}{heading})\n{body}")
    return "\n\n".join(blocks)


def _count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _parse_citations(answer: str, chunks_by_index: dict[int, RetrievalResult]) -> list[Citation]:
    """Walk `[N]` references in `answer` in first-mention order and resolve
    each to the chunk at that 1-indexed position. Indices outside the supplied
    set are silently dropped — model hallucinated a higher number."""
    seen: set[int] = set()
    citations: list[Citation] = []
    for match in CITATION_PATTERN.finditer(answer):
        try:
            idx = int(match.group(1))
        except ValueError:
            continue
        if idx in seen or idx not in chunks_by_index:
            continue
        seen.add(idx)
        c = chunks_by_index[idx]
        citations.append(
            Citation(
                chunk_id=c.chunk_id,
                source_path=c.source_path,
                file_name=c.file_name,
                page_number=c.page_number,
                heading_path=c.heading_path,
                figures=c.figures,
                index=idx,
            )
        )
    # Sort by the original index so the Sources footer reads 1, 2, 4 (in
    # ascending order) instead of citation-mention order. Aligns with how
    # users expect numbered references to flow.
    citations.sort(key=lambda c: c.index)
    return citations


def generate_grounded_answer(
    query: str,
    selected_chunks: list[RetrievalResult],
    *,
    model: str | None = None,
    stream_handler: Callable[[str], None] | None = None,
) -> GenerationOutput:
    deployment = model or os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    context = _format_context(selected_chunks)

    user_msg = (
        f"Question: {query}\n\n"
        f"Context (each chunk preceded by its [N] index):\n{context}\n\n"
        "Answer with [N] citations:"
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    client = get_azure_openai_client()
    stream = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.1,
        stream=True,
    )
    parts: list[str] = []
    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        token = getattr(delta, "content", None)
        if token:
            parts.append(token)
            if stream_handler is not None:
                stream_handler(token)
    answer = "".join(parts)

    chunks_by_index = {i: c for i, c in enumerate(selected_chunks, start=1)}
    citations = _parse_citations(answer, chunks_by_index)

    return GenerationOutput(
        answer=answer,
        citations=citations,
        model=deployment,
        context_chunk_ids=[c.chunk_id for c in selected_chunks],
        prompt_token_estimate=_count_tokens(SYSTEM_PROMPT) + _count_tokens(user_msg),
        completion_token_estimate=_count_tokens(answer),
    )
