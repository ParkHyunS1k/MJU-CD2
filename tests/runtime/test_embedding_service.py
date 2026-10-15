import math
from types import SimpleNamespace

import pytest

from app.services import embedding_service as es


class FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, model, contents, config):
        self.calls.append((model, list(contents), config.task_type, config.output_dimensionality))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0] + [0.0] * 766) for _ in contents])


def _client():
    return SimpleNamespace(models=FakeModels())


def test_embed_batches_by_20_and_normalizes():
    client = _client()
    vecs = es.embed([f"t{i}" for i in range(45)], "document", client=client)
    assert [len(c[1]) for c in client.models.calls] == [20, 20, 5]
    assert all(c[0] == "gemini-embedding-001" and c[2] == "RETRIEVAL_DOCUMENT" and c[3] == 768
               for c in client.models.calls)
    assert len(vecs) == 45 and len(vecs[0]) == 768
    assert math.isclose(vecs[0][0], 0.6) and math.isclose(vecs[0][1], 0.8)


def test_query_kind_and_empty_input():
    client = _client()
    es.embed(["고혈압"], "query", client=client)
    assert client.models.calls[0][2] == "RETRIEVAL_QUERY"
    assert es.embed([], "query", client=client) == []


def test_unknown_kind_rejected():
    with pytest.raises(KeyError):
        es.embed(["x"], "passage", client=_client())
