import json
import time
from pathlib import Path

import pytest

from app.services import retrieval_service as rs
from app.services.embedding_service import DIMS, MODEL
from app.services.rag_taxonomy import Query

Q1 = Query("hypertension", "diet", "고혈압 예방 식이요법 식사 관리")
Q2 = Query("diabetes", "diet", "당뇨병 예방 식이요법 식사 관리")


def _row(cid, sim, doc="kha-hypertension-2026"):
    return {"chunk_id": cid, "document_id": doc, "title": "고혈압 진료지침", "publisher": "대한고혈압학회",
            "published_year": 2026, "url": "https://x", "heading_path": "생활요법", "page_start": 74,
            "page_end": 75, "content": f"본문 {cid}", "similarity": sim}


def test_config_matches_selected_eval_config():
    sel = json.loads(Path("rag/eval/selected.json").read_text(encoding="utf-8"))
    assert sel["embedding_model"] == f"gemini-{rs.EMBEDDING_DIMS}" and rs.EMBEDDING_DIMS == sel["dimensions"] == DIMS
    assert rs.EMBEDDING_MODEL == MODEL
    assert rs.FINAL_K == sel["final_k"] and rs.MIN_SIMILARITY == sel["threshold"]
    assert sel["hybrid"] is False and sel["reranker"] is None


def test_select_round_robin_by_rank_dedupes_and_numbers():
    results = [[_row(1, .9), _row(2, .8), _row(3, .75)], [_row(2, .85), _row(4, .8)]]
    ev = rs.select_evidence([Q1, Q2], results)
    assert [e.chunk_id for e in ev] == [1, 2, 4, 3]
    assert [e.sid for e in ev] == ["S1", "S2", "S3", "S4"]
    assert ev[1].condition == "diabetes"  # 순위 순서로 처음 고른 쿼리(Q2의 1위)의 태그
    assert ev[0].year == 2026 and ev[0].content == "본문 1"


def test_select_respects_limit():
    results = [[_row(i * 10 + r, .9 - r / 10) for r in range(3)] for i in range(8)]
    assert len(rs.select_evidence([Q1] * 8, results)) == rs.MAX_EVIDENCE


def test_retrieve_embeds_once_and_calls_rpc_per_query():
    seen = []

    def fake_embed(texts, kind):
        assert kind == "query"
        seen.append(texts)
        return [[float(i)] * 3 for i, _ in enumerate(texts)]

    ev = rs.retrieve([Q1, Q2], embed_fn=fake_embed, match_fn=lambda v: [_row(int(v[0]) + 1, .9)])
    assert seen == [[Q1.text, Q2.text]] and [e.chunk_id for e in ev] == [1, 2]


def test_retrieve_empty_queries_skips_calls():
    assert rs.retrieve([], embed_fn=lambda *_: pytest.fail("no call"), match_fn=lambda v: []) == []


def test_rpc_failure_raises_retrieval_error():
    def boom(v):
        raise RuntimeError("rpc down")

    with pytest.raises(rs.RetrievalError):
        rs.retrieve([Q1], embed_fn=lambda t, k: [[0.0]], match_fn=boom)


def test_hanging_rpc_times_out_quickly():
    start = time.monotonic()
    with pytest.raises(rs.RetrievalError):
        rs.retrieve([Q1], embed_fn=lambda t, k: [[0.0]], match_fn=lambda v: time.sleep(2) or [], timeout=0.2)
    assert time.monotonic() - start < 1.0


def test_vector_literal_format():
    assert rs._vector_literal([0.5, -0.25]) == "[0.500000,-0.250000]"
