"""Measure retrieval quality for BM25, embeddings and hybrid. No LLM calls.

For each hand-written question in ``eval/questions.jsonl`` we know which docs
page(s) answer it. A retrieved chunk is relevant if it comes from one of those
pages. Metrics (page-level, over the ranked chunk list):

- hit@1: the top chunk is relevant
- hit@5: at least one of the top 5 chunks is relevant
- MRR@10: mean of 1/rank of the first relevant chunk (0 if none in the top 10)

Usage:
    python -m eval.run_retrieval_eval [--index-dir data/index] [--markdown]
        [--json eval/results.json] [--min-hit5 0.8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from retrieval.embeddings import FastEmbedEmbedder
from retrieval.index import MODES, Retriever

QUESTIONS = Path(__file__).with_name("questions.jsonl")
DEPTH = 10


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_relevant_rank(hits: list[dict], relevant: set[str]) -> int | None:
    for hit in hits:
        if hit["page"] in relevant:
            return hit["rank"]
    return None


def evaluate(retriever: Retriever, questions: list[dict]) -> dict:
    known_pages = {chunk["page"] for chunk in retriever.chunks}
    unknown = sorted({p for q in questions for p in q["relevant_pages"]} - known_pages)
    if unknown:
        raise SystemExit(f"questions reference pages missing from the index: {unknown}")

    results: dict = {"questions": len(questions), "modes": {}, "per_question": []}
    ranks: dict[str, list[int | None]] = {mode: [] for mode in MODES}
    elapsed = 0.0

    for question in questions:
        relevant = set(question["relevant_pages"])
        row = {"id": question["id"], "style": question["style"]}
        for mode in MODES:
            started = time.perf_counter()
            hits = retriever.search(question["question"], mode=mode, top_k=DEPTH)
            elapsed += time.perf_counter() - started
            found = first_relevant_rank(hits, relevant)
            ranks[mode].append(found)
            row[mode] = found
        results["per_question"].append(row)

    def summarise(values: list[int | None]) -> dict:
        n = len(values) or 1
        return {
            "hit@1": sum(1 for r in values if r == 1) / n,
            "hit@5": sum(1 for r in values if r is not None and r <= 5) / n,
            "mrr@10": sum(1 / r for r in values if r is not None) / n,
        }

    # Every search computes all three signals, so one latency figure covers them.
    results["ms_per_search"] = 1000 * elapsed / max(len(questions) * len(MODES), 1)
    for mode in MODES:
        summary = summarise(ranks[mode])
        by_style = {}
        for style in sorted({q["style"] for q in questions}):
            subset = [r for r, q in zip(ranks[mode], questions, strict=True) if q["style"] == style]
            by_style[style] = {"n": len(subset), **summarise(subset)}
        summary["by_style"] = by_style
        results["modes"][mode] = summary
    return results


def markdown_table(results: dict) -> str:
    lines = [
        "| Mode | hit@1 | hit@5 | MRR@10 | hit@5 keyword | hit@5 paraphrase |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"bm25": "BM25", "vector": "Embeddings (bge-small)", "hybrid": "Hybrid (RRF)"}
    for mode in MODES:
        m = results["modes"][mode]
        styles = m["by_style"]
        lines.append(
            f"| {labels[mode]} | {m['hit@1']:.2f} | {m['hit@5']:.2f} | {m['mrr@10']:.3f} "
            f"| {styles.get('keyword', {}).get('hit@5', 0):.2f} | {styles.get('paraphrase', {}).get('hit@5', 0):.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--json", type=Path, help="write full results here")
    parser.add_argument("--markdown", action="store_true", help="print only the Markdown table")
    parser.add_argument("--stemming", action="store_true", help="Porter-stem BM25 tokens (off by default)")
    parser.add_argument("--min-hit5", type=float, default=None, help="fail if hybrid hit@5 is below this")
    args = parser.parse_args()

    if not Retriever.exists(args.index_dir):
        raise SystemExit(f"no index at {args.index_dir}; run `python -m ingest.build_index` first")
    retriever = Retriever.load(args.index_dir, FastEmbedEmbedder(), stemming=args.stemming)
    questions = load_questions(args.questions)
    results = evaluate(retriever, questions)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    if args.markdown:
        print(markdown_table(results))
    else:
        print(f"{results['questions']} questions, {len(retriever.chunks)} chunks\n")
        print(markdown_table(results))
        print()
        print(f"{results['ms_per_search']:.1f} ms per search (BM25 + embedding + fusion, CPU)")
        misses = [r for r in results["per_question"] if r["hybrid"] is None or r["hybrid"] > 5]
        if misses:
            print("\nhybrid misses (not in top 5):", ", ".join(r["id"] for r in misses))

    if args.min_hit5 is not None and results["modes"]["hybrid"]["hit@5"] < args.min_hit5:
        print(f"FAIL: hybrid hit@5 below {args.min_hit5}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
