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
        {"query_id": "q1", "content_hash": "h1", "relevant": True, "evidence": ["소금은 하루 5 g 이하로 줄인다", "소금은 하루 5g 이하로 줄인다"]},
        {"query_id": "q1", "content_hash": "h2", "relevant": False, "evidence": []},
        {"query_id": "none-1", "content_hash": "h2", "relevant": False, "evidence": []},
    ]
    edits = [{"query_id": "q2", "text": "고친 질문?", "excluded": False}]
    labels, queries, stats = export(ITEMS, decisions, edits)

    assert labels == [{"query_id": "q1", "doc_id": "kha", "page_start": 74, "page_end": 74, "content_hash": "h1",
                       "evidence": "소금은 하루 5 g 이하로 줄인다", "source": "human"}]
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


def _items_with(content, judged_relevant=None, judged_evidence=()):
    c = _cand("h1", 74) | {"content": content, "judged_relevant": judged_relevant,
                           "judged_evidence": list(judged_evidence)}
    return {"queries": [{"query_id": "q1", "text": "운동", "form": "keyword", "condition": "hypertension",
                         "category": "exercise", "split": "tune", "candidates": [c]}]}


CONTENT = "운동\n속보, 조깅은 혈압을 낮추는\n효과가 있다. 주 5일 운동한다.\n연구 결과 다른 내용이다. 운동"


def test_contiguous_fragments_merge_and_short_runs_drop():
    frags = ["운동", "속보, 조깅은 혈압을 낮추는", "효과가 있다.", "운동"]
    labels, _, _ = export(_items_with(CONTENT), [{"query_id": "q1", "content_hash": "h1", "relevant": True,
                                                  "evidence": frags}], [])
    assert [l["evidence"] for l in labels] == ["운동 속보, 조깅은 혈압을 낮추는 효과가 있다."]


def test_unreviewed_candidates_fall_back_to_llm_judgment():
    items = _items_with(CONTENT, True, ["속보, 조깅은 혈압을 낮추는\n효과가 있다.", "주 5일 운동한다."])
    labels, _, stats = export(items, [], [])
    assert [(l["evidence"], l["source"]) for l in labels] == [
        ("속보, 조깅은 혈압을 낮추는\n효과가 있다.", "llm")]
    assert stats["reviewed"] == 0 and stats["llm_filled"] == 1 and stats["unreviewed"] == 1


def test_llm_sentences_stay_separate_spans():
    items = _items_with(CONTENT, True, ["속보, 조깅은 혈압을 낮추는\n효과가 있다.", "주 5일 운동한다는 문장이다."])
    items["queries"][0]["candidates"][0]["content"] = CONTENT.replace("주 5일 운동한다.", "주 5일 운동한다는 문장이다.")
    labels, _, _ = export(items, [], [])
    assert [l["evidence"] for l in labels] == ["속보, 조깅은 혈압을 낮추는\n효과가 있다.", "주 5일 운동한다는 문장이다."]


def test_evidence_matching_many_chunks_is_dropped():
    tag = "[무작위대조군연구, 일반적권고]"
    items = _items_with(CONTENT + tag)
    decision = {"query_id": "q1", "content_hash": "h1", "relevant": True,
                "evidence": ["속보, 조깅은 혈압을 낮추는 효과가 있다.", tag]}
    corpus = [CONTENT] + [f"권고 {i} {tag}" for i in range(4)]
    labels, _, stats = export(items, [decision], [], corpus)
    assert [l["evidence"] for l in labels] == ["속보, 조깅은 혈압을 낮추는 효과가 있다."]
    assert stats["nonspecific_dropped"] == 1
