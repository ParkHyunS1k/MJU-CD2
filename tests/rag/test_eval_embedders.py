from pathlib import Path

import pytest
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


class _QuotaError(Exception):
    code = 429


class FakeGeminiClient:
    def __init__(self, fail_first=1):
        self.batch_sizes = []
        self.fail_first = fail_first
        self.models = self

    def embed_content(self, model, contents, config):
        if self.fail_first:
            self.fail_first -= 1
            raise _QuotaError("RESOURCE_EXHAUSTED")
        self.batch_sizes.append(len(contents))
        values = [type("E", (), {"values": [1.0, float(i)]})() for i in range(len(contents))]
        return type("R", (), {"embeddings": values})()


def test_gemini_embedder_small_batches_and_retries_on_quota(monkeypatch):
    from rag.eval import llm
    from rag.eval.embedders import GeminiEmbedder

    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    client = FakeGeminiClient(fail_first=1)
    vectors = GeminiEmbedder(2, client=client).encode([f"t{i}" for i in range(45)], "doc")
    assert vectors.shape == (45, 2)
    assert client.batch_sizes == [20, 20, 5]
    assert len(sleeps) == 1 and sleeps[0] >= 15


class RecordingEmbedder:
    name = "rec"

    def __init__(self, fail_after: int | None = None):
        self.seen: list[list[str]] = []
        self.fail_after = fail_after

    def encode(self, texts, kind):
        if self.fail_after is not None and len(self.seen) >= self.fail_after:
            raise RuntimeError("quota")
        self.seen.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)


def test_cache_is_per_text_so_subsets_hit(tmp_path):
    inner = RecordingEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    full = cached.encode(["가", "나다", "라마바"], "query")
    sub = cached.encode(["라마바", "가", "사"], "query")
    assert inner.seen == [["가", "나다", "라마바"], ["사"]]
    assert np.array_equal(sub[:2], full[[2, 0]])


def test_cache_keeps_finished_batches_when_inner_fails(tmp_path):
    texts = [f"문장{i}" for i in range(5)]
    with pytest.raises(RuntimeError):
        CachedEmbedder(RecordingEmbedder(fail_after=1), tmp_path, batch=2).encode(texts, "doc")
    inner = RecordingEmbedder()
    CachedEmbedder(inner, tmp_path, batch=2).encode(texts, "doc")
    assert inner.seen == [["문장2", "문장3"], ["문장4"]]


def test_cache_write_failure_keeps_previous_cache(tmp_path, monkeypatch):
    CachedEmbedder(RecordingEmbedder(), tmp_path).encode(["가"], "doc")

    def broken_savez(file, **arrays):
        Path(file).write_bytes(b"partial zip")
        raise KeyboardInterrupt

    monkeypatch.setattr(np, "savez", broken_savez)
    with pytest.raises(KeyboardInterrupt):
        CachedEmbedder(RecordingEmbedder(), tmp_path).encode(["나"], "doc")
    monkeypatch.undo()
    inner = RecordingEmbedder()
    CachedEmbedder(inner, tmp_path).encode(["가"], "doc")
    assert inner.seen == []
