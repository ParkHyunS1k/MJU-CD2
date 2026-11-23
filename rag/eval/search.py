"""numpy 전수 벡터 검색과 kiwi 토큰 BM25."""

import time
from collections.abc import Callable
from functools import cache, lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

_KEEP_TAGS = ("NN", "NR", "NP", "VV", "VA", "SL", "SN", "XR", "SH")


def dense_rank(query_vecs: np.ndarray, doc_vecs: np.ndarray, k: int) -> list[list[int]]:
    scores = query_vecs @ doc_vecs.T
    order = np.lexsort((np.broadcast_to(np.arange(doc_vecs.shape[0]), scores.shape), -scores), axis=1)
    return [row[:k].tolist() for row in order]


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def kiwi_tokens(text: str) -> list[str]:
    return [t.form for t in _kiwi().tokenize(text) if t.tag.startswith(_KEEP_TAGS)]


class BM25Index:
    def __init__(self, texts: list[str], tokenize: Callable[[str], list[str]] = kiwi_tokens):
        self.tokenize = tokenize
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def rank(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(self.tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [i for i in order[:k] if scores[i] > 0]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for r, i in enumerate(ranking):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores, key=lambda i: (-scores[i], i))


RERANKERS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-v2-m3-ko": "dragonkue/bge-reranker-v2-m3-ko",
}


class Reranker:
    def __init__(self, name: str, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(RERANKERS[name], max_length=512)
        self.name = name
        self._model = model
        self._seconds = 0.0
        self._calls = 0

    def rerank(self, query: str, texts: list[str], candidates: list[int]) -> list[int]:
        start = time.perf_counter()
        scores = self._model.predict([(query, texts[i]) for i in candidates])
        self._seconds += time.perf_counter() - start
        self._calls += 1
        order = sorted(range(len(candidates)), key=lambda j: (-scores[j], j))
        return [candidates[j] for j in order]

    def ms_per_call(self) -> float:
        return self._seconds * 1000 / self._calls if self._calls else 0.0


@cache
def reranker(name: str) -> Reranker:
    return Reranker(name)
