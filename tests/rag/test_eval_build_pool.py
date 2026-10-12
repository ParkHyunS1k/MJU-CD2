import numpy as np

from rag.eval.build_pool import build_review_items, rank_all
from rag.eval.judge import CandidateJudgment, JudgeCache, QueryJudgment
from rag.eval.search import BM25Index

CHUNKS = [
    {"content_hash": "h-salt", "doc_id": "kha", "title": "고혈압", "heading_path": "소금", "page_start": 74,
     "page_end": 74, "content": "고혈압 > 소금\n하루 소금 섭취량을 5 g 이하로 제한한다."},
    {"content_hash": "h-run", "doc_id": "kha", "title": "고혈압", "heading_path": "운동", "page_start": 77,
     "page_end": 77, "content": "고혈압 > 운동\n주 5회 유산소 운동을 한다."},
]
QUERIES = [{"query_id": "q-salt", "text": "소금 제한", "form": "keyword", "condition": "hypertension",
            "category": "diet", "split": "tune"}]


class KeywordEmbedder:
    name = "kw"

    def encode(self, texts, kind):
        return np.array([[1.0 if "소금" in t else 0.0, 1.0 if "운동" in t else 0.0] for t in texts], dtype=np.float32)


def test_rank_all_includes_embedders_and_bm25():
    docs = [c["content"] for c in CHUNKS] + ["기타 문서 내용"]  # 문서 2개면 BM25 IDF가 0이 됨
    rankings = rank_all({"kw": KeywordEmbedder()}, docs, ["소금 제한"], k=1, bm25=BM25Index(docs, tokenize=str.split))
    assert rankings == {"kw": [[0]], "bm25": [[0]]}


def test_build_review_items_judges_once_and_uses_cache(tmp_path):
    calls = []

    def ask(prompt):
        calls.append(prompt)
        return QueryJudgment(judgments=[
            CandidateJudgment(candidate_id="c1", relevant=True, evidence=["하루 소금 섭취량을 5 g 이하로 제한한다."]),
            CandidateJudgment(candidate_id="c2", relevant=False, evidence=[]),
        ])

    rankings = {"kw": [[0, 1]], "bm25": [[1]]}
    cache = JudgeCache(tmp_path / "j.jsonl")
    items = build_review_items(CHUNKS, QUERIES, rankings, per_system=2, ask=ask, cache=cache)
    again = build_review_items(CHUNKS, QUERIES, rankings, per_system=2, ask=ask, cache=JudgeCache(tmp_path / "j.jsonl"))

    assert len(calls) == 1
    assert items == again
    [query] = items["queries"]
    assert query["query_id"] == "q-salt"
    assert [c["content_hash"] for c in query["candidates"]] == ["h-salt", "h-run"]
    salt = query["candidates"][0]
    assert salt["systems"] == ["kw"]
    assert salt["judged_relevant"] is True
    assert salt["judged_evidence"] == ["하루 소금 섭취량을 5 g 이하로 제한한다."]
    assert salt["page_start"] == 74
    assert query["candidates"][1]["systems"] == ["bm25", "kw"]
