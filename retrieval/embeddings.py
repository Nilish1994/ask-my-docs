"""Dense retrieval: BAAI/bge-small-en-v1.5 via fastembed (ONNX on CPU).

Vectors are L2-normalised and stored as one float32 NumPy matrix, so cosine
similarity is a single matrix-vector product. At ~500 chunks x 384 dims the
whole index is under 1 MB and exact search takes well under a millisecond;
an ANN library would add a dependency without a measurable benefit here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE v1.5 recommends this instruction on short queries (not on passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class FastEmbedEmbedder:
    """Lazy-loads the ONNX model on first use (about 130 MB, cached on disk)."""

    dimensions = 384

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str | None = None, threads: int | None = None):
        self.name = model_name
        self._cache_dir = cache_dir
        self._threads = threads
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.name, cache_dir=self._cache_dir, threads=self._threads)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return _normalise(np.stack(list(self.model.embed(texts, batch_size=32))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(np.stack(list(self.model.embed([QUERY_INSTRUCTION + text])))[0])


class VectorIndex:
    """Exact cosine search over a normalised (n_chunks, dims) matrix."""

    def __init__(self, matrix: np.ndarray, ids: list[int]):
        if len(matrix) != len(ids):
            raise ValueError("matrix rows and ids must line up")
        self.matrix = _normalise(matrix) if len(matrix) else matrix.astype(np.float32)
        self.ids = list(ids)

    @classmethod
    def build(cls, embedder: Embedder, texts: list[str], ids: list[int]) -> VectorIndex:
        return cls(embedder.embed_documents(texts), ids)

    def scores(self, query_vector: np.ndarray) -> dict[int, float]:
        if not len(self.ids):
            return {}
        similarities = self.matrix @ query_vector
        return {doc_id: float(score) for doc_id, score in zip(self.ids, similarities, strict=True)}

    def save(self, path: Path) -> None:
        np.save(path, self.matrix)

    @classmethod
    def load(cls, path: Path, ids: list[int]) -> VectorIndex:
        return cls(np.load(path), ids)
