"""Gemini 임베딩. 앱의 쿼리 임베딩과 rag.load의 문서 임베딩이 같은 설정을 쓰도록 한 곳에 둔다."""

import math
from typing import Literal

from google.genai import types

MODEL = "gemini-embedding-001"
DIMS = 768
_BATCH = 20  # 무료 등급 분당 토큰 한도 안에 들어가는 크기
_TASK = {"query": "RETRIEVAL_QUERY", "document": "RETRIEVAL_DOCUMENT"}


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def embed(texts: list[str], kind: Literal["query", "document"], client=None) -> list[list[float]]:
    """768차원 이하는 정규화되어 오지 않으므로 L2 정규화해 돌려준다."""
    task = _TASK[kind]
    if not texts:
        return []
    if client is None:
        from app.services.ai_service import get_gemini

        client = get_gemini()
    config = types.EmbedContentConfig(task_type=task, output_dimensionality=DIMS)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        res = client.models.embed_content(model=MODEL, contents=texts[i : i + _BATCH], config=config)
        vectors.extend(_normalize(list(e.values)) for e in res.embeddings)
    return vectors
