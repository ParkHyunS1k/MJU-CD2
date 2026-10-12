"""numpy 전수 벡터 검색과 kiwi 토큰 BM25."""

from collections.abc import Callable
from functools import lru_cache

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
