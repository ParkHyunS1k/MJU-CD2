"""비교할 임베딩 모델들. 모두 encode(texts, kind) -> 행 정규화 행렬 형태로 맞춘다."""

import hashlib
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
    def __init__(self, inner: Embedder, cache_dir: Path):
        self.inner = inner
        self.name = inner.name
        self.cache_dir = cache_dir

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        key = hashlib.sha256("\x1f".join([self.name, kind, *texts]).encode()).hexdigest()
        path = self.cache_dir / f"{self.name}-{kind}-{key[:24]}.npy"
        if path.exists():
            return np.load(path)
        vectors = self.inner.encode(texts, kind)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors


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
