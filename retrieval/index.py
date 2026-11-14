"""One retriever, three modes: ``bm25``, ``vector`` and ``hybrid``.

Every search computes all three signals so the UI's "How was this answered?"
panel can show each chunk's BM25 score, cosine similarity and fused RRF score
side by side, whichever mode chose the ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .bm25 import BM25Index
from .embeddings import Embedder, VectorIndex
from .hybrid import rank, reciprocal_rank_fusion

Mode = Literal["bm25", "vector", "hybrid"]
MODES: tuple[Mode, ...] = ("bm25", "vector", "hybrid")
FUSION_DEPTH = 50

CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.npy"
MANIFEST_FILE = "manifest.json"


def embedding_text(chunk: dict) -> str:
    return f"{chunk['title']}\n\n{chunk['text']}"


class Retriever:
    def __init__(self, chunks: list[dict], vectors: VectorIndex, embedder: Embedder, stemming: bool = False):
        self.chunks = chunks
        self.by_id = {chunk["id"]: chunk for chunk in chunks}
        self.embedder = embedder
        self.vectors = vectors
        self.stemming = stemming
        self.bm25 = BM25Index(
            [{"id": c["id"], "title": c["title"], "text": c["text"]} for c in chunks],
            stemming=stemming,
        )

    @classmethod
    def build(cls, chunks: list[dict], embedder: Embedder, stemming: bool = False) -> Retriever:
        ids = [chunk["id"] for chunk in chunks]
        vectors = VectorIndex.build(embedder, [embedding_text(c) for c in chunks], ids)
        return cls(chunks, vectors, embedder, stemming)

    def save(self, directory: Path, manifest: dict | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        self.vectors.save(directory / VECTORS_FILE)
        info = {"chunks": len(self.chunks), "embedding_model": self.embedder.name, "stemming": self.stemming}
        info.update(manifest or {})
        (directory / MANIFEST_FILE).write_text(json.dumps(info, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder, stemming: bool = False) -> Retriever:
        with (directory / CHUNKS_FILE).open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        vectors = VectorIndex.load(directory / VECTORS_FILE, [c["id"] for c in chunks])
        return cls(chunks, vectors, embedder, stemming)

    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / CHUNKS_FILE).exists() and (directory / VECTORS_FILE).exists()

    def search(self, query: str, mode: Mode = "hybrid", top_k: int = 5) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not self.chunks:
            return []

        bm25_scores = self.bm25.scores(query)
        vector_scores = self.vectors.scores(self.embedder.embed_query(query))
        bm25_ranking = rank(bm25_scores, FUSION_DEPTH)
        vector_ranking = rank(vector_scores, FUSION_DEPTH)
        fused_scores = reciprocal_rank_fusion([bm25_ranking, vector_ranking])
        fused_ranking = rank(fused_scores)

        chosen = {"bm25": bm25_ranking, "vector": vector_ranking, "hybrid": fused_ranking}[mode][:top_k]
        bm25_rank = {doc_id: i for i, doc_id in enumerate(bm25_ranking, start=1)}
        vector_rank = {doc_id: i for i, doc_id in enumerate(vector_ranking, start=1)}
        fused_rank = {doc_id: i for i, doc_id in enumerate(fused_ranking, start=1)}

        hits = []
        for position, doc_id in enumerate(chosen, start=1):
            chunk = self.by_id[doc_id]
            hits.append(
                {
                    **chunk,
                    "rank": position,
                    "scores": {
                        "bm25": round(bm25_scores.get(doc_id, 0.0), 4),
                        "bm25_rank": bm25_rank.get(doc_id),
                        "vector": round(vector_scores.get(doc_id, 0.0), 4),
                        "vector_rank": vector_rank.get(doc_id),
                        "fused": round(fused_scores.get(doc_id, 0.0), 5),
                        "fused_rank": fused_rank.get(doc_id),
                    },
                }
            )
        return hits
