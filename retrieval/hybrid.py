"""Reciprocal rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

RRF combines ranked lists using ranks only: score(d) = sum_i 1 / (k + rank_i(d)).
That sidesteps the fact that BM25 scores (unbounded, query-dependent) and
cosine similarities (roughly 0.3-0.9 for this model) live on different scales,
so no score normalisation or per-corpus weight tuning is needed.
"""

from __future__ import annotations

RRF_K = 60


def rank(scores: dict[int, float], depth: int | None = None) -> list[int]:
    """Document ids by descending score, ties broken by id for determinism."""
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return ordered[:depth] if depth else ordered


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + position)
    return fused
