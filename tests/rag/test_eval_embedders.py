import numpy as np

from rag.eval.embedders import REGISTRY, CachedEmbedder, l2_normalize


class CountingEmbedder:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def encode(self, texts, kind):
        self.calls += 1
        return l2_normalize(np.array([[len(t), 1.0 if kind == "query" else 2.0] for t in texts], dtype=np.float32))


def test_l2_normalize_rows():
    m = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert m.dtype == np.float32


def test_cache_hit_skips_inner(tmp_path):
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    first = cached.encode(["가", "나다"], "doc")
    second = CachedEmbedder(inner, tmp_path).encode(["가", "나다"], "doc")
    assert inner.calls == 1
    assert np.array_equal(first, second)
    assert cached.name == "fake"


def test_eval_cache_key_changes_with_texts(tmp_path):
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    cached.encode(["가"], "doc")
    cached.encode(["가 바뀜"], "doc")
    cached.encode(["가"], "query")
    assert inner.calls == 3


def test_registry_names():
    assert set(REGISTRY) == {
        "gemini-768", "gemini-1536", "gemini-3072", "bge-m3", "kure-v1", "qwen3-0.6b", "e5-large-instruct",
    }
