from collections import Counter

from rag.eval.queries import SCENARIOS, build_queries, keyword_query


def fake_draft(condition, category):
    return f"{condition}-{category} 질문?"


def test_scenarios_cover_spec_table():
    assert len(SCENARIOS) == 24
    assert ("sarcopenia", "exercise") in SCENARIOS
    assert ("general", "diet") in SCENARIOS


def test_keyword_query_templates():
    assert keyword_query("hypertension", "diet") == "고혈압 예방 식이요법 식사 관리"
    assert keyword_query("general", "exercise") == "성인 신체활동 권장량 유산소 근력운동"


def test_build_queries_shapes_and_counts():
    queries = build_queries(fake_draft)
    forms = Counter(q["form"] for q in queries)
    assert forms == {"keyword": 24, "question": 24, "inbody": 3, "unanswerable": 3}
    assert len({q["query_id"] for q in queries}) == len(queries)
    assert set(queries[0]) == {"query_id", "text", "form", "condition", "category", "split"}


def test_question_text_comes_from_drafter():
    queries = build_queries(fake_draft)
    q = next(q for q in queries if q["query_id"] == "dyslipidemia-diet-question")
    assert q["text"] == "dyslipidemia-diet 질문?"


def test_scenario_pairs_share_split_and_splits_are_balanced():
    queries = build_queries(fake_draft)
    by_id = {q["query_id"]: q for q in queries}
    for condition, category in SCENARIOS:
        a = by_id[f"{condition}-{category}-keyword"]
        b = by_id[f"{condition}-{category}-question"]
        assert a["split"] == b["split"]
    splits = Counter(q["split"] for q in queries)
    assert abs(splits["tune"] - splits["test"]) <= 2


def test_drafter_whitespace_is_stripped():
    queries = build_queries(lambda c, k: "  질문입니다?\n")
    assert all(q["text"] == "질문입니다?" for q in queries if q["form"] == "question")
