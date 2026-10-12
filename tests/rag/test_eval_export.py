from rag.eval.export_labels import export


def _cand(h, page):
    return {"content_hash": h, "doc_id": "kha", "title": "t", "heading_path": None, "page_start": page,
            "page_end": page, "content": "본문", "systems": ["a"], "judged_relevant": None, "judged_evidence": []}


ITEMS = {"queries": [
    {"query_id": "q1", "text": "소금", "form": "keyword", "condition": "hypertension", "category": "diet",
     "split": "tune", "candidates": [_cand("h1", 74), _cand("h2", 75), _cand("h3", 76)]},
    {"query_id": "q2", "text": "질문 초안", "form": "question", "condition": "hypertension", "category": "diet",
     "split": "tune", "candidates": [_cand("h1", 74)]},
    {"query_id": "none-1", "text": "통풍", "form": "unanswerable", "condition": "none", "category": "none",
     "split": "tune", "candidates": [_cand("h2", 75)]},
]}


def test_export_labels_and_query_edits():
    decisions = [
        {"query_id": "q1", "content_hash": "h1", "relevant": True, "evidence": ["소금을 5 g 이하로", "소금을 5g 이하로"]},
        {"query_id": "q1", "content_hash": "h2", "relevant": False, "evidence": []},
        {"query_id": "none-1", "content_hash": "h2", "relevant": False, "evidence": []},
    ]
    edits = [{"query_id": "q2", "text": "고친 질문?", "excluded": False}]
    labels, queries, stats = export(ITEMS, decisions, edits)

    assert labels == [{"query_id": "q1", "doc_id": "kha", "page_start": 74, "page_end": 74, "content_hash": "h1",
                       "evidence": "소금을 5 g 이하로", "source": "human"}]
    q2 = next(q for q in queries if q["query_id"] == "q2")
    assert q2["text"] == "고친 질문?" and q2["excluded"] is False
    assert "candidates" not in q2
    assert stats["answerable_queries_without_labels"] == ["q2"]


def test_eval_export_skips_unreviewed_and_evidence_less():
    decisions = [
        {"query_id": "q1", "content_hash": "h1", "relevant": True, "evidence": []},
        {"query_id": "q9", "content_hash": "zz", "relevant": True, "evidence": ["x"]},
    ]
    labels, _, stats = export(ITEMS, decisions, [])
    assert labels == []
    assert stats["candidates"] == 5
    assert stats["reviewed"] == 1
    assert stats["unreviewed"] == 4
    assert stats["relevant_without_evidence"] == 1
    assert stats["unknown_decisions"] == 1


def test_excluded_query_kept_with_flag_and_not_counted():
    labels, queries, stats = export(ITEMS, [], [{"query_id": "q2", "text": None, "excluded": True}])
    q2 = next(q for q in queries if q["query_id"] == "q2")
    assert q2["excluded"] is True and q2["text"] == "질문 초안"
    assert stats["excluded_queries"] == ["q2"]
    assert "q2" not in stats["answerable_queries_without_labels"]
