import numpy as np
import pytest

from retrieval.bm25 import BM25Index
from retrieval.embeddings import VectorIndex
from retrieval.hybrid import rank, reciprocal_rank_fusion
from retrieval.index import Retriever


def test_bm25_ranks_keyword_match_first():
    docs = [
        {"id": 1, "title": "CORS", "text": "configure CORSMiddleware origins"},
        {"id": 2, "title": "Query", "text": "query parameters are optional"},
    ]
    index = BM25Index(docs)
    assert [d["id"] for d in index.search("optional query", top_k=2)] == [2]
    assert set(index.scores("origins query")) == {1, 2}


def test_bm25_stemming_is_optional():
    docs = [{"id": 1, "title": "", "text": "running servers"}]
    assert BM25Index(docs).scores("run server") == {}
    assert 1 in BM25Index(docs, stemming=True).scores("run server")


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]], k=60)
    assert rank(fused)[0] == 2
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)


def test_rank_breaks_ties_by_id():
    assert rank({5: 1.0, 3: 1.0, 9: 2.0}) == [9, 3, 5]


def test_vector_index_normalises_and_scores():
    index = VectorIndex(np.array([[3.0, 0.0], [0.0, 2.0]]), [10, 20])
    scores = index.scores(np.array([1.0, 0.0], dtype=np.float32))
    assert scores == {10: pytest.approx(1.0), 20: pytest.approx(0.0)}


def test_vector_index_rejects_misaligned_ids():
    with pytest.raises(ValueError):
        VectorIndex(np.zeros((2, 3)), [1])


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_retriever_modes_return_all_scores(retriever, mode):
    hits = retriever.search("How do I allow my browser frontend to call the API with CORS?", mode=mode, top_k=3)
    assert hits, mode
    assert hits[0]["page"] == "tutorial/cors"
    assert set(hits[0]["scores"]) == {"bm25", "bm25_rank", "vector", "vector_rank", "fused", "fused_rank"}
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))


def test_retriever_rejects_unknown_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")


def test_retriever_save_and_load_round_trip(tmp_path, retriever, embedder):
    retriever.save(tmp_path, {"corpus": "test"})
    assert Retriever.exists(tmp_path)
    loaded = Retriever.load(tmp_path, embedder)
    query = "optional query parameter default"
    assert [h["chunk_id"] for h in loaded.search(query)] == [h["chunk_id"] for h in retriever.search(query)]
