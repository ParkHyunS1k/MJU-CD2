from app.services.risk_change import compute

ZERO = {"age": 0.0, "female": 0.0, "bmi": 0.0, "body_fat_percentage": 0.0,
        "fh_hypertension": 0.4, "fh_diabetes": 0.0, "fh_dyslipidemia": 0.0}


def _stored(p, bmi, pbf, c, key="hypertension"):
    return {"inputs": {"bmi": bmi, "body_fat_percentage": pbf}, "age": 35,
            "diseases": [{"disease": "고혈압", "disease_key": key, "probability": p, "risk_level": "medium",
                          "contributions": {**ZERO, **c}}]}


PREV = _stored(0.30, 27.0, 30.0, {"age": 0.10, "bmi": 0.30, "body_fat_percentage": 0.50})
NOW = _stored(0.25, 26.0, 27.5, {"age": 0.15, "bmi": 0.20, "body_fat_percentage": 0.30})


def test_change_and_shares_sum_to_change():
    r = compute(PREV, NOW, "2025-06-02T00:00:00+00:00")[0]
    assert r["disease"] == "고혈압" and r["change_pp"] == -5.0 and r["previous_measured_at"] == "2025-06-02"
    assert (r["previous_pct"], r["current_pct"]) == (30.0, 25.0)
    assert r["share_pp"] == {"BMI": -2.0, "체지방률": -4.0, "나이": 1.0}
    assert r["previous_inputs"] == {"bmi": 27.0, "body_fat_percentage": 30.0}
    assert r["current_inputs"] == {"bmi": 26.0, "body_fat_percentage": 27.5}


def test_zero_logodds_change_has_no_shares():
    now = _stored(0.30, 28.0, 29.0, {"age": 0.10, "bmi": 0.40, "body_fat_percentage": 0.40})
    r = compute(PREV, now, "2025-06-02")[0]
    assert r["share_pp"] is None and r["change_pp"] == 0.0


def test_only_common_diseases_are_paired():
    other = _stored(0.2, 26.0, 27.5, {}, key="diabetes")
    assert compute(PREV, other, "2025-06-02") == []
