"""
Generation ONLY.

Takes a query plus the chunks the evidence-selector picked and returns a
grounded answer with explicit `[chunk_id]` citations. The system prompt
forbids the model from drawing on anything outside the supplied context.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Callable

import tiktoken

from .embed import get_azure_openai_client
from .search import RetrievalResult

CITATION_PATTERN = re.compile(r"\[([a-f0-9]{16,})\]")

SYSTEM_PROMPT = (
    "You are a knowledge-base assistant for product manuals, troubleshooting "
    "guides, and policies.\n"
    "RULES:\n"
    "1. Use ONLY the provided context chunks. Never use outside knowledge.\n"
    "2. If the context does not contain enough information to answer, say so "
    "plainly and stop. Do not guess.\n"
    "3. Cite every factual claim with the chunk id in square brackets, e.g. "
    "[a1b2c3...]. Use only chunk ids that appear in the supplied context.\n"
    "4. Be concise and concrete. Prefer numbered steps for procedures.\n"
)


@dataclass
class Citation:
    chunk_id: str
    source_path: str
    file_name: str
    page_number: int | None
    heading_path: str | None


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


def _format_context(chunks: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for c in chunks:
        loc = c.file_name
        if c.page_number is not None:
            loc += f" p.{c.page_number}"
        heading = f' — "{c.heading_path}"' if c.heading_path else ""
        blocks.append(f"[{c.chunk_id}] ({loc}{heading})\n{c.content}")
    return "\n\n".join(blocks)


def _count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _parse_citations(answer: str, chunks_by_id: dict[str, RetrievalResult]) -> list[Citation]:
    seen: set[str] = set()
    citations: list[Citation] = []
    for match in CITATION_PATTERN.finditer(answer):
        cid = match.group(1)
        if cid in seen or cid not in chunks_by_id:
            continue
        seen.add(cid)
        c = chunks_by_id[cid]
        citations.append(
            Citation(
                chunk_id=cid,
                source_path=c.source_path,
                file_name=c.file_name,
                page_number=c.page_number,
                heading_path=c.heading_path,
            )
        )
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
        f"Context (each chunk preceded by its [chunk_id]):\n{context}\n\n"
        "Answer with citations:"
    )
    messages = [
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

    chunks_by_id = {c.chunk_id: c for c in selected_chunks}
    citations = _parse_citations(answer, chunks_by_id)

    return GenerationOutput(
        answer=answer,
        citations=citations,
        model=deployment,
        context_chunk_ids=[c.chunk_id for c in selected_chunks],
        prompt_token_estimate=_count_tokens(SYSTEM_PROMPT) + _count_tokens(user_msg),
        completion_token_estimate=_count_tokens(answer),
    )
