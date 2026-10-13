import numpy as np

from rag.eval.search import Reranker, rrf, BM25Index, dense_rank, kiwi_tokens


def test_dense_rank_orders_by_dot_product_with_stable_ties():
    docs = np.array([[1, 0], [0, 1], [0.6, 0.8], [1, 0]], dtype=np.float32)
    queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert dense_rank(queries, docs, k=3) == [[0, 3, 2], [1, 2, 0]]


def test_dense_rank_k_larger_than_corpus():
    docs = np.eye(2, dtype=np.float32)
    assert dense_rank(np.array([[1, 0]], dtype=np.float32), docs, k=10) == [[0, 1]]


def test_bm25_with_simple_tokenizer():
    texts = ["소금 섭취 제한", "운동 권고 유산소", "소금 소금 나트륨"]
    index = BM25Index(texts, tokenize=str.split)
    assert index.rank("소금", k=5) == [2, 0]
    assert index.rank("없는단어", k=5) == []


def test_kiwi_tokens_keeps_content_morphemes():
    tokens = kiwi_tokens("하루 소금 섭취량을 5 g 이하로 제한한다.")
    assert "소금" in tokens
    assert "제한" in tokens
    assert "을" not in tokens


def test_rrf_combines_and_breaks_ties_by_index():
    assert rrf([[2, 0, 1], [0, 2]]) == [0, 2, 1]
    assert rrf([[1], [0]]) == [0, 1]


class _FakeCross:
    def predict(self, pairs):
        return [len(text) for _, text in pairs]


def test_reranker_orders_by_score_and_times_calls():
    r = Reranker("bge-reranker-v2-m3", model=_FakeCross())
    assert r.rerank("q", ["a", "ccc", "bb", "dddd"], [0, 1, 2]) == [1, 2, 0]
    assert r.ms_per_call() >= 0.0
