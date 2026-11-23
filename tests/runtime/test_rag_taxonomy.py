from app.services.rag_taxonomy import SCENARIOS, Query, build_queries, keyword_query


def _risk(key, level):
    return {"disease_key": key, "risk_level": level}


def _pairs(queries):
    return [(q.condition, q.category) for q in queries]


def test_medium_or_high_risks_map_to_conditions_in_scenario_order():
    qs = build_queries([_risk("hypertension", "high"), _risk("diabetes", "medium"), _risk("cholesterol", "low")],
                       {"bmi": 22}, "male")
    assert _pairs(qs) == [("hypertension", "criteria"), ("hypertension", "diet"), ("hypertension", "lifestyle"),
                          ("hypertension", "exercise"), ("diabetes", "diet"), ("diabetes", "exercise"),
                          ("diabetes", "lifestyle")]
    assert qs[1] == Query("hypertension", "diet", "고혈압 예방 식이요법 식사 관리")


def test_cholesterol_and_triglyceride_become_one_dyslipidemia():
    qs = build_queries([_risk("cholesterol", "medium"), _risk("triglyceride", "high")], {}, "female")
    assert [c for c, _ in _pairs(qs)] == ["dyslipidemia"] * 4


def test_body_composition_triggers():
    assert ("obesity", "criteria") in _pairs(build_queries([], {"bmi": 25.0}, "male"))
    assert ("obesity", "criteria") not in _pairs(build_queries([], {"bmi": 24.9}, "male"))
    assert ("metabolic_syndrome", "criteria") in _pairs(build_queries([], {"abs_fat": 0.90}, "male"))
    assert ("metabolic_syndrome", "criteria") not in _pairs(build_queries([], {"abs_fat": 0.89}, "male"))
    assert ("metabolic_syndrome", "criteria") in _pairs(build_queries([], {"abs_fat": 0.85}, "female"))
    assert ("metabolic_syndrome", "criteria") in _pairs(build_queries([], {"fat_level": 10}, "female"))


def test_missing_values_and_gender_skip_rules_without_error():
    qs = build_queries([], {"bmi": None, "abs_fat": 0.95, "fat_level": None}, None)
    assert _pairs(qs) == [("general", "diet"), ("general", "exercise")]


def test_all_low_gives_general_queries():
    qs = build_queries([_risk("hypertension", "low")], {"bmi": 21}, "male")
    assert [q.text for q in qs] == ["한국인 영양소 섭취기준 권장섭취량 상한섭취량", "성인 신체활동 권장량 유산소 근력운동"]


def test_unknown_disease_key_is_ignored():
    assert _pairs(build_queries([_risk("gout", "high")], {}, "male")) == [("general", "diet"), ("general", "exercise")]


def test_every_query_pair_is_an_evaluated_scenario():
    qs = build_queries([_risk(k, "high") for k in ("hypertension", "diabetes", "cholesterol")],
                       {"bmi": 30, "fat_level": 12}, "male")
    assert len(qs) == 16 and all(p in SCENARIOS for p in _pairs(qs))
    assert all(q.text == keyword_query(q.condition, q.category) for q in qs)


from app.services.personal_metrics import compute  # noqa: E402
from app.services.rag_taxonomy import trait_queries  # noqa: E402

SUPP = "영양제 건강기능식품 선택 주의사항"


def _traits(inbody, profile, risks=(), trend=None):
    return trait_queries(list(risks), profile, compute(inbody, profile), trend)


def test_traits_follow_profile():
    prof = {"gender": "male", "fh_hypertension": True, "fh_diabetes": None}
    texts = [q.text for q in _traits({"bmi": 24.0, "body_fat_percentage": 23.0}, prof, [_risk("cholesterol", "medium")])]
    assert "이상지질혈증 경계 수치 생활습관 교정" in texts
    assert "고혈압 가족력 고위험군 조기 예방" in texts
    assert "비만 전단계 과체중 체중 증가 예방" in texts
    assert not any("당뇨병 가족력" in t for t in texts)
    assert texts[-1] == SUPP


def test_one_per_trait_type_first():
    prof = {"gender": "male", "fh_hypertension": True, "fh_diabetes": True, "fh_dyslipidemia": True}
    risks = [_risk("hypertension", "medium"), _risk("diabetes", "medium"), _risk("triglyceride", "medium")]
    qs = _traits({"bmi": 22.0, "abs_fat": 0.95}, prof, risks)
    assert [q.trait for q in qs[:4]] == ["복부비만", "전단계 위험", "가족력", "여러 위험 동반"]


def test_trait_cap_and_supplement_always():
    prof = {"gender": "male", "fh_hypertension": True, "fh_diabetes": True, "fh_dyslipidemia": True}
    risks = [_risk("hypertension", "medium"), _risk("diabetes", "high")]
    trend = {"feature_trends": [{"feature_kor": "체중", "direction": "up"}]}
    qs = _traits({"bmi": 22.0, "body_fat_percentage": 27.0, "abs_fat": 0.95, "muscle_control": 2.0}, prof, risks, trend)
    assert len(qs) == 5 and qs[-1].trait == "영양제" and qs[-1].category == "supplement"
    assert [q.trait for q in qs[:4]] == ["지방→근육형", "복부비만", "전단계 위험", "가족력"]
    assert [q.text for q in _traits({}, {"gender": "female"})] == [SUPP]


def test_weight_trend_trait():
    up = {"feature_trends": [{"feature_kor": "체중", "direction": "up"}]}
    stable = {"feature_trends": [{"feature_kor": "체중", "direction": "stable"}]}
    assert "체중 증가 예방 체중 유지 관리" in [q.text for q in _traits({}, {"gender": "male"}, trend=up)]
    assert "체중 증가 예방 체중 유지 관리" not in [q.text for q in _traits({}, {"gender": "male"}, trend=stable)]


def test_lean_body_fat_trait_needs_bmi_under_23():
    lean = [q.text for q in _traits({"bmi": 22.0, "body_fat_percentage": 26.0}, {"gender": "male"})]
    heavy = [q.text for q in _traits({"bmi": 24.0, "body_fat_percentage": 26.0}, {"gender": "male"})]
    assert "체지방률 감소 근력운동 체성분 개선" in lean and "체지방률 감소 근력운동 체성분 개선" not in heavy


def _parts(**over):
    return {"left_arm": "표준", "right_arm": "표준", "trunk": "표준", "left_leg": "표준", "right_leg": "표준", **over}


def test_lower_body_muscle_trait():
    qs = _traits({"segmental_lean": _parts(left_leg="표준이하")}, {"gender": "male"})
    assert qs[0].trait == "부위 근육 부족" and qs[0].text == "근감소증 운동요법 하체 근력운동 권고"


def test_upper_body_only_uses_large_muscle_query():
    qs = _traits({"segmental_lean": _parts(left_arm="표준이하", right_arm="표준이하")}, {"gender": "male"})
    assert qs[0].text == "근감소증 운동요법 상체 근력운동 권고"


def test_trunk_fat_triggers_abdominal_and_order_after_whole_body_muscle():
    qs = _traits({"segmental_lean": _parts(right_leg="표준이하"), "segmental_fat": _parts(trunk="표준이상"),
                  "muscle_control": 1.0}, {"gender": "male"})
    assert [q.trait for q in qs[:3]] == ["근육 부족", "부위 근육 부족", "복부비만"]


def test_body_type_queries():
    def texts(**inbody):
        return [(q.trait, q.text) for q in _traits(inbody, {"gender": "male"})]

    assert texts(body_fat_percentage=27.0, muscle_control=0.0)[:3] == [
        ("지방 감량형", "비만 유산소 운동 체지방 감소 권고"),
        ("지방 감량형", "체중 감량 식사요법 에너지 섭취 감소 권고"),
        ("지방 감량형", "유산소 운동 혈압 중성지방 감소 효과")]
    assert texts(body_fat_percentage=27.0, muscle_control=2.0)[:3] == [
        ("지방→근육형", "저항운동 강도 점진적 증가 권고"),
        ("지방→근육형", "단백질 섭취 근육량 증가 권고"),
        ("지방→근육형", "저항운동 인슐린 저항성 혈당 개선")]
    assert texts(body_fat_percentage=15.0, muscle_control=2.0)[:2] == [
        ("근육 보강형", "근육량 증가 저항운동 단백질 섭취"),
        ("근육 보강형", "근육량 감소 당뇨병 대사증후군 위험")]
    assert texts(body_fat_percentage=15.0, muscle_control=0.0) == [("영양제", SUPP)]


def test_unknown_muscle_keeps_lean_body_fat_query_only():
    qs = _traits({"bmi": 22.0, "body_fat_percentage": 27.0}, {"gender": "male"})
    assert [q.trait for q in qs] == ["체지방 과다", "영양제"]
