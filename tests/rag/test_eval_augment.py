from rag.eval.augment import apply_judgments, collect

EV = "하루 소금 섭취량을 5 g 이하로 제한한다."
CHUNKS = [
    {"content_hash": "h0", "doc_id": "a", "page_start": 1, "page_end": 1, "content": f"식사. {EV}"},
    {"content_hash": "h1", "doc_id": "a", "page_start": 2, "page_end": 2, "content": "나트륨이 많은 가공식품과 국물 섭취를 줄이도록 권고한다."},
    {"content_hash": "h2", "doc_id": "a", "page_start": 3, "page_end": 3, "content": "고혈압의 역학."},
    {"content_hash": "h3", "doc_id": "b", "page_start": 4, "page_end": 4, "content": "통풍 환자는 퓨린을 줄인다."},
]
QUERIES = [{"query_id": "q1", "text": "소금 섭취 제한"}]
LABELS = {"q1": [{"doc_id": "a", "evidence": EV, "source": "human"}]}


def test_collect_takes_unlabeled_chunks_in_top_k():
    items = collect(CHUNKS, QUERIES, LABELS, [[0, 1, 2, 3]], k=3)
    assert items == [{"query_id": "q1", "text": "소금 섭취 제한", "candidates": [
        {"candidate_id": "c1", **CHUNKS[1]}, {"candidate_id": "c2", **CHUNKS[2]}]}]


def test_collect_skips_queries_without_new_candidates():
    assert collect(CHUNKS, QUERIES, LABELS, [[0]], k=1) == []


def test_apply_keeps_only_verbatim_specific_new_evidence():
    items = collect(CHUNKS, QUERIES, LABELS, [[0, 1, 2]], k=3)
    judgments = {"q1": [
        {"candidate_id": "c1", "relevant": True, "evidence": [
            "나트륨이 많은 가공식품과 국물 섭취를 줄이도록 권고한다.",  # 그대로 → 채택
            "가공식품 섭취를 줄인다.",  # 원문에 없음 → 버림
            "국물"]},  # 너무 짧음 → 버림
        {"candidate_id": "c2", "relevant": False, "evidence": []},
        {"candidate_id": "c9", "relevant": True, "evidence": ["없는 후보"]},
    ]}
    corpus = [c["content"] for c in CHUNKS]
    rows, stats = apply_judgments(items, judgments, LABELS, corpus)
    assert rows == [{"query_id": "q1", "doc_id": "a", "page_start": 2, "page_end": 2, "content_hash": "h1",
                     "evidence": "나트륨이 많은 가공식품과 국물 섭취를 줄이도록 권고한다.", "source": "llm-gpt"}]
    assert stats == {"candidates": 2, "judged": 2, "relevant": 1, "added": 1, "dropped_evidence": 2}


def test_apply_drops_evidence_matching_many_chunks():
    tag = "[전문가의견, 일반적권고] 근거 등급 표시"
    chunks = [{"content_hash": f"t{i}", "doc_id": "a", "page_start": i, "page_end": i, "content": tag} for i in range(5)]
    items = collect(chunks, QUERIES, LABELS, [[0]], k=1)
    rows, stats = apply_judgments(items, {"q1": [{"candidate_id": "c1", "relevant": True, "evidence": [tag]}]},
                                  LABELS, [c["content"] for c in chunks])
    assert rows == [] and stats["dropped_evidence"] == 1


def test_judge_items_maps_judge_results_and_resumes(tmp_path):
    from rag.eval.augment import judge_items
    from rag.eval.judge import CandidateJudgment, QueryJudgment

    calls = []

    def ask(prompt):
        calls.append(prompt)
        return QueryJudgment(judgments=[
            CandidateJudgment(candidate_id="c1", relevant=True, evidence=["나트륨이 많은 가공식품과 국물 섭취를 줄이도록 권고한다."]),
            CandidateJudgment(candidate_id="c2", relevant=False, evidence=[])])

    items = collect(CHUNKS, QUERIES, LABELS, [[0, 1, 2]], k=3)
    cache = tmp_path / "j.json"
    judgments = judge_items(items, ask, cache)
    assert judgments == {"q1": [
        {"candidate_id": "c1", "relevant": True, "evidence": ["나트륨이 많은 가공식품과 국물 섭취를 줄이도록 권고한다."]},
        {"candidate_id": "c2", "relevant": False, "evidence": []}]}
    assert judge_items(items, ask, cache) == judgments and len(calls) == 1  # 이미 판정한 쿼리는 다시 묻지 않는다
