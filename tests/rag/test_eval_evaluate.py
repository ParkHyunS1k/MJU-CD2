import json

from rag.eval.evaluate import (evaluate, find_evidence, groups, load_eval, pick, preservation, summarize,
                               table)

EVIDENCE = "하루 소금 섭취량을 5 g 이하로 제한한다."
CHUNKS = [
    {"doc_id": "a", "content": f"앞 내용. {EVIDENCE} 뒤 내용."},
    {"doc_id": "b", "content": EVIDENCE},
    {"doc_id": "a", "content": "운동은 주 5회 이상 한다. 관계없는 문장."},
]
LABELS = {"q1": [{"doc_id": "a", "evidence": EVIDENCE, "source": "human"}],
          "q2": [{"doc_id": "a", "evidence": "Resistance exercise twice weekly", "source": "llm"}]}


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_load_eval_drops_unanswerable_and_excluded(tmp_path):
    q = lambda qid, form="keyword", excluded=False: {"query_id": qid, "text": qid, "form": form,
                                                       "split": "tune", "excluded": excluded}
    queries = _jsonl(tmp_path / "q.jsonl", [q("q1"), q("q2", excluded=True), q("none-1", "unanswerable"), q("q3")])
    labels = _jsonl(tmp_path / "l.jsonl", [{"query_id": "q1", "doc_id": "a", "evidence": EVIDENCE, "source": "human"},
                                          {"query_id": "q2", "doc_id": "a", "evidence": EVIDENCE, "source": "llm"}])
    all_q, scored, by_q = load_eval(queries, labels)
    assert len(all_q) == 4
    assert [x["query_id"] for x in scored] == ["q1"]
    assert set(by_q) == {"q1", "q2"}


def test_find_evidence_same_doc_only():
    assert find_evidence(CHUNKS, LABELS["q1"]) == {0: {0}}


def test_find_evidence_split_across_chunks():
    # 앞 조각은 10-gram 6/9(0.67), 뒤 조각은 0/9를 담는다
    chunks = [{"doc_id": "a", "content": EVIDENCE[:-3]}, {"doc_id": "a", "content": EVIDENCE[-8:]}]
    assert find_evidence(chunks, LABELS["q1"]) == {0: {0}}


def test_preservation_and_evaluate():
    queries = [{"query_id": "q1"}, {"query_id": "q2"}]
    assert preservation(CHUNKS, queries, LABELS) == 0.5
    per_query = evaluate(CHUNKS, queries, LABELS, [[2, 0], [0, 1]], ks=(1,))
    assert [p["recall@1"] for p in per_query] == [0.0, 0.0]
    assert per_query[0]["mrr"] == 0.5 and per_query[0]["query_id"] == "q1"


def test_groups_human_and_english():
    queries = [{"query_id": "q1"}, {"query_id": "q2"}]
    assert groups(queries, LABELS) == {"사람 검수": {"q1"}, "영문 근거": {"q2"}}


def test_summarize_empty_group_renders_dash():
    per_query = [{"query_id": "q1", "ndcg@10": 1.0, "mrr": 1.0}]
    row = summarize("base", per_query, ["ndcg@10", "mrr"], {"사람 검수": set()}, {"chunks": 3})
    assert row["ndcg@10"] == (1.0, 1.0, 1.0) and row["chunks"] == 3
    md = table([row])
    assert "| 구성 | ndcg@10 | mrr | ndcg@10 (사람 검수) | chunks |" in md
    assert "| base | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | – | 3 |" in md


def test_pick_prefers_simplest_overlapping():
    rows = [{"name": "A", "m": (0.70, 0.60, 0.80)}, {"name": "B", "m": (0.75, 0.65, 0.85)},
            {"name": "C", "m": (0.90, 0.85, 0.95)}]
    assert pick(rows, "m")["name"] == "B"
    rows[2]["m"] = (0.90, 0.86, 0.95)
    assert pick(rows, "m")["name"] == "C"


def test_pick_tie_prefers_first():
    rows = [{"name": n, "m": [0.5, 0.4, 0.6]} for n in "XYZ"]
    assert pick(rows, "m")["name"] == "X"
