from app.services import supplement_evidence as se


def _ing(name, tags, notified=True, no=None):
    return {"name": name, "notified": notified, "recognition_no": no, "function": "기능", "intake": "1 g",
            "cautions": "주의", "tags": tags, "updated": "20260101"}


def RISK(key, level):
    return {"disease_key": key, "risk_level": level}


def test_needed_tags_from_risks_and_body_type():
    risks = [RISK("diabetes", "high"), RISK("hypertension", "low"), RISK("triglyceride", "medium")]
    assert se.needed_tags(risks, {"body_type": {"type": "지방→근육형"}}) == \
        ["blood_glucose", "triglyceride", "body_fat", "muscle"]


def test_unknown_muscle_with_fat_excess_gets_body_fat_only():
    assert se.needed_tags([], {"body_type": {"type": "근육 판정 불가", "fat_excess": True}}) == ["body_fat"]
    assert se.needed_tags([], {"body_type": {"type": "근육 판정 불가", "fat_excess": False}}) == []


def test_all_low_and_no_type_gives_no_tags():
    assert se.needed_tags([RISK("diabetes", "low")], {"body_type": {"type": "해당 없음"}}) == []
    assert se.needed_tags([RISK("diabetes", "low")], {}) == []


def test_notified_first_cap_per_tag_round_robin_dedupe_limit():
    ings = [_ing(f"개별{i}", ["blood_glucose"], False, f"2020-{i}") for i in range(3)] + \
           [_ing(f"고시{i}", ["blood_glucose"]) for i in range(6)] + \
           [_ing("공통", ["blood_glucose", "cholesterol"]), _ing("콜1", ["cholesterol"])]
    picked = se.select_ingredients(ings, ["blood_glucose", "cholesterol"], per_tag=5, limit=10)
    names = [i["name"] for _, i in picked]
    assert names[:2] == ["고시0", "공통"]  # 태그별 1위부터 번갈아
    assert len(names) == len(set(names))  # 공통은 한 번만
    assert [n for n in names if n.startswith("고시")] == [f"고시{i}" for i in range(5)]
    assert not any(n.startswith("개별") for n in names)  # 고시형 5개가 먼저 차서 개별인정형은 밀림
    assert len(se.select_ingredients(ings * 3, ["blood_glucose", "cholesterol"], limit=4)) <= 4


def test_same_name_with_two_recognitions_is_picked_once():
    ings = [_ing("저분자유청단백가수분해물", ["muscle"], False, "2023-21"),
            _ing("저분자유청단백가수분해물", ["muscle"], False, "2023-22"), _ing("오미자추출물", ["muscle"], False, "2018-9")]
    assert [i["name"] for _, i in se.select_ingredients(ings, ["muscle"])] == ["저분자유청단백가수분해물", "오미자추출물"]


def test_evidence_numbering_and_fields():
    ings = [_ing("바나바잎 추출물", ["blood_glucose"]), _ing("Nopal 추출물", ["blood_glucose"], False, "2009-39")]
    ev = se.supplement_evidence([RISK("diabetes", "high")], {}, start=16, ingredients=ings)
    assert [e.sid for e in ev] == ["S16", "S17"]
    assert {e.chunk_category for e in ev} == {"supplement"} and {e.document_id for e in ev} == {"mfds"}
    assert ev[0].heading_path == "고시형" and ev[1].heading_path == "개별인정형 제2009-39호"
    assert ev[0].publisher == "식품의약품안전처" and ev[0].year == 2026 and ev[0].page_start is None
    assert ev[0].content.split("\n", 1)[1].startswith("주된 기능성: 기능")
    assert ev[0].trait == "혈당 기능성 원료"
    assert all(e.chunk_id < 0 for e in ev)


def test_missing_file_gives_no_evidence(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(se, "DATA", tmp_path / "none.json")
    se.load_ingredients.cache_clear()
    try:
        assert se.supplement_evidence([RISK("diabetes", "high")], {}, start=1) == []
        assert "mfds" in caplog.text
    finally:
        se.load_ingredients.cache_clear()


def test_real_file_loads_and_has_every_tag():
    se.load_ingredients.cache_clear()
    tags = {t for i in se.load_ingredients() for t in i["tags"]}
    assert tags == {"blood_glucose", "blood_pressure", "cholesterol", "triglyceride", "body_fat", "muscle"}
