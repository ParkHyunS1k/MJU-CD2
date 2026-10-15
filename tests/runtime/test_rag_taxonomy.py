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
