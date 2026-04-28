"""
Evaluation harness for the observable RAG pipeline.

Reads a gold Q&A file (JSON list, see notebooks/gold_qa.json), runs each query
through the full LangGraph, and computes:

  Retrieval (programmatic)
    - recall@k          (k = RETRIEVAL_TOP_K, default 5)
    - MRR               (mean reciprocal rank of the first gold chunk hit)

  Generation (programmatic)
    - citation_correctness  (every cited chunk_id appears in context — boolean)

  Generation (LLM-as-judge, gpt-4o-mini, JSON-mode, scale 1-5)
    - groundedness      (every factual claim is supported by context)
    - answer_relevance  (the answer actually addresses the question)

Per-question results are appended to eval_results.jsonl. Aggregate scores are
returned and summarised in notebooks/eval.ipynb.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

from .agent import run
from .embed import get_azure_openai_client

JUDGE_GROUNDEDNESS = """You judge whether an answer is grounded in given context.
Score 1-5: 5 = every claim has direct support in context, 1 = answer largely fabricated.
Reply JSON: {"score": <int 1-5>, "reason": "<one short sentence>"}"""

JUDGE_RELEVANCE = """You judge whether an answer addresses the user's question.
Score 1-5: 5 = directly and fully addresses, 1 = does not address at all.
Reply JSON: {"score": <int 1-5>, "reason": "<one short sentence>"}"""


def _judge(system: str, user: str) -> dict:
    client = get_azure_openai_client()
    deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    parsed = json.loads(resp.choices[0].message.content)
    score = int(parsed.get("score", 0))
    return {"score": max(1, min(5, score)), "reason": parsed.get("reason", "")}


def _recall_at_k(gold_ids: list[str], retrieved: list[dict], k: int) -> float:
    if not gold_ids:
        return float("nan")
    top = {r["chunk_id"] for r in retrieved[:k]}
    hits = sum(1 for g in gold_ids if g in top)
    return hits / len(gold_ids)


def _mrr(gold_ids: list[str], retrieved: list[dict]) -> float:
    if not gold_ids:
        return float("nan")
    gold = set(gold_ids)
    for i, r in enumerate(retrieved, start=1):
        if r["chunk_id"] in gold:
            return 1.0 / i
    return 0.0


def _citation_correctness(generation: dict) -> bool:
    context_ids = set(generation["context_chunk_ids"])
    cited_ids = {c["chunk_id"] for c in generation["citations"]}
    return cited_ids.issubset(context_ids)


def evaluate_one(item: dict) -> dict:
    """Run one gold Q&A item through the pipeline and score it."""
    query = item["question"]
    gold_chunk_ids = item.get("gold_chunk_ids", [])
    k = int(os.environ.get("RETRIEVAL_TOP_K", "5"))

    trace = run(query)
    retrieved = trace["retrieval"]["results"]
    generation = trace["generation"]
    answer = generation["answer"]

    judge_g = _judge(
        JUDGE_GROUNDEDNESS,
        f"Question: {query}\n\nContext chunk_ids: {generation['context_chunk_ids']}\n\nAnswer: {answer}",
    )
    judge_r = _judge(JUDGE_RELEVANCE, f"Question: {query}\n\nAnswer: {answer}")

    return {
        "question": query,
        "intent": trace["query_plan"]["intent"],
        "recall_at_k": _recall_at_k(gold_chunk_ids, retrieved, k),
        "mrr": _mrr(gold_chunk_ids, retrieved),
        "citation_correctness": _citation_correctness(generation),
        "groundedness": judge_g["score"],
        "groundedness_reason": judge_g["reason"],
        "answer_relevance": judge_r["score"],
        "answer_relevance_reason": judge_r["reason"],
        "answer": answer,
        "retrieved_chunk_ids": [r["chunk_id"] for r in retrieved],
        "gold_chunk_ids": gold_chunk_ids,
    }


def _aggregate(rows: list[dict]) -> dict:
    def mean_clean(vals):
        clean = [v for v in vals if isinstance(v, (int, float)) and v == v]  # filter NaN
        return statistics.mean(clean) if clean else None

    return {
        "n": len(rows),
        "recall_at_k_mean":          mean_clean([r["recall_at_k"] for r in rows]),
        "mrr_mean":                   mean_clean([r["mrr"] for r in rows]),
        "citation_correctness_rate":  mean_clean([1.0 if r["citation_correctness"] else 0.0 for r in rows]),
        "groundedness_mean":          mean_clean([r["groundedness"] for r in rows]),
        "answer_relevance_mean":      mean_clean([r["answer_relevance"] for r in rows]),
    }


def run_eval(gold_path: str = "notebooks/gold_qa.json") -> dict:
    gold_file = Path(gold_path)
    if not gold_file.exists():
        raise FileNotFoundError(f"Gold Q&A file not found: {gold_file}. See notebooks/gold_qa.json template.")
    items = json.loads(gold_file.read_text())
    if not items:
        raise ValueError(f"{gold_file} is empty — fill in some Q&A entries first.")

    out_path = Path(os.environ.get("EVAL_RESULTS", "eval_results.jsonl"))
    out_path.unlink(missing_ok=True)

    rows: list[dict] = []
    for item in items:
        row = evaluate_one(item)
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{len(rows)}/{len(items)}] {row['question'][:60]} → "
              f"recall@k={row['recall_at_k']:.2f} MRR={row['mrr']:.2f} "
              f"ground={row['groundedness']} rel={row['answer_relevance']}")

    summary = _aggregate(rows)
    print("\nSummary:", json.dumps(summary, indent=2))
    return {"summary": summary, "rows": rows, "results_file": str(out_path)}


if __name__ == "__main__":  # python -m src.eval_harness
    from dotenv import load_dotenv
    load_dotenv()
    run_eval()
