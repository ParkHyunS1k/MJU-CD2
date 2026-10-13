"""비교할 임베딩 모델들. 모두 encode(texts, kind) -> 행 정규화 행렬 형태로 맞춘다."""

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str], kind: str) -> np.ndarray: ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class CachedEmbedder:
    """텍스트마다 벡터를 캐시한다. 부분 집합 요청도 캐시를 쓰고, batch마다 저장해 중간에 끊겨도 이어서 한다."""

    def __init__(self, inner: Embedder, cache_dir: Path, batch: int = 100):
        self.inner = inner
        self.name = inner.name
        self.cache_dir = cache_dir
        self.batch = batch

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        path = self.cache_dir / f"{self.name}-{kind}.npz"
        store: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as f:
                store = dict(zip(f["keys"].tolist(), f["vectors"]))
        keys = [_text_key(t) for t in texts]
        todo = list(dict.fromkeys(t for k, t in zip(keys, texts) if k not in store))
        for i in range(0, len(todo), self.batch):
            part = todo[i : i + self.batch]
            store.update(zip(map(_text_key, part), self.inner.encode(part, kind)))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")  # 쓰다 끊겨도 기존 캐시는 남도록 바꿔치기한다
            np.savez(tmp, keys=np.array(list(store)), vectors=np.stack(list(store.values())))
            os.replace(tmp, path)
        return np.stack([store[k] for k in keys])


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class GeminiEmbedder:
    _BATCH = 20  # 청크 100개는 무료 등급 분당 토큰 한도를 넘는다

    def __init__(self, dims: int, client=None):
        self.dims = dims
        self.name = f"gemini-{dims}"
        self._client = client

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        from google.genai import types

        from rag.eval.llm import call_with_retry, get_client

        client = self._client or get_client()
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        rows = []
        for i in range(0, len(texts), self._BATCH):
            batch = texts[i : i + self._BATCH]
            res = call_with_retry(lambda: client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch,
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dims),
            ))
            rows.extend(e.values for e in res.embeddings)
        return l2_normalize(np.array(rows))


class STEmbedder:
    def __init__(self, name: str, model_id: str, query_prefix: str = "", doc_prefix: str = "",
                 query_prompt_name: str | None = None):
        self.name = name
        self.model_id = model_id
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.query_prompt_name = query_prompt_name
        self._model = None

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.model_id)
        prefix = self.query_prefix if kind == "query" else self.doc_prefix
        kwargs = {"prompt_name": self.query_prompt_name} if kind == "query" and self.query_prompt_name else {}
        vectors = self._model.encode([prefix + t for t in texts], batch_size=8, show_progress_bar=True, **kwargs)
        return l2_normalize(vectors)


_E5_QUERY = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "

REGISTRY: dict[str, Callable[[], Embedder]] = {
    "gemini-768": lambda: GeminiEmbedder(768),
    "gemini-1536": lambda: GeminiEmbedder(1536),
    "gemini-3072": lambda: GeminiEmbedder(3072),
    "bge-m3": lambda: STEmbedder("bge-m3", "BAAI/bge-m3"),
    "kure-v1": lambda: STEmbedder("kure-v1", "nlpai-lab/KURE-v1"),
    "qwen3-0.6b": lambda: STEmbedder("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B", query_prompt_name="query"),
    "e5-large-instruct": lambda: STEmbedder("e5-large-instruct", "intfloat/multilingual-e5-large-instruct",
                                            query_prefix=_E5_QUERY),
}


def load(name: str, cache_dir: Path = Path("rag/eval/cache/embeddings")) -> Embedder:
    return CachedEmbedder(REGISTRY[name](), cache_dir)
