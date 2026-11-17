import pytest

from rag.eval.report_preview import jaccard_mean, mentions_user_number


def test_jaccard_mean():
    a, b, c = {1, 2}, {2, 3}, {4}
    # (a,b)=1/3, (a,c)=0, (b,c)=0
    assert jaccard_mean([a, b, c]) == pytest.approx(1 / 9)
    assert jaccard_mean([a]) == 0.0 and jaccard_mean([set(), set()]) == 0.0


def test_action_mentions_user_number():
    values = [{"weight": 74.9, "bmi": 24.9}, {"weight_loss_5pct_kg": 3.7, "bmi_band": "비만 전단계"}]
    assert mentions_user_number("현재 체중 74.9 kg의 5%인 3.7 kg 감량", values)
    assert not mentions_user_number("중강도 유산소 운동을 주 150분 하세요", values)
    assert mentions_user_number("체중 80 kg 기준", [{"weight": 80.0}])


def test_segmental_labels_by_sex_percentile():
    import pandas as pd

    from rag.eval.report_preview import segmental_labels
    n = 10
    df = pd.DataFrame({"sex": [1] * n, "HE_ht": [170.0] * n, "BIA_FFM": [60.0] * n,
                       **{c: [3.0] * n for c in ("BIA_LRA", "BIA_LLA", "BIA_LRL", "BIA_FRA", "BIA_FLA", "BIA_FT",
                                                 "BIA_FRL", "BIA_FLL")},
                       "BIA_LLL": [float(v) for v in range(1, n + 1)]})  # 왼다리 근육만 사람마다 다르다
    labels = segmental_labels(df)
    left_leg = [lean["left_leg"] for lean, _ in labels]
    assert left_leg[:2] == ["표준이하"] * 2 and left_leg[-2:] == ["표준이상"] * 2 and left_leg[2:-2] == ["표준"] * 6
    assert labels[0][0]["right_leg"] == "표준" and set(labels[0][1]) == {"left_arm", "right_arm", "trunk",
                                                                         "left_leg", "right_leg"}


def test_with_previous_cycles_improved_worsened_similar():
    from rag.eval.report_preview import PREVIOUS_MEASURED_AT, with_previous
    cases = [{"case": str(i), "profile": {}, "inbody": {"bmi": 25.0, "body_fat_percentage": 30.0}} for i in range(3)]
    prev = [c["previous"] for c in with_previous(cases)]
    assert [(p["bmi"], p["body_fat_percentage"]) for p in prev] == [(26.5, 33.0), (23.5, 27.0), (25.1, 29.8)]
    assert all(p["measured_at"] == PREVIOUS_MEASURED_AT for p in prev)
    assert "previous" not in cases[0]  # 원본은 그대로


def test_case_risk_change_uses_previous_record():
    from app.services.prediction_service import predict_all
    from rag.eval.report_preview import case_risk_change, with_previous
    case = with_previous([{"case": "a", "profile": {"gender": "male", "birth_year": 1990},
                           "inbody": {"bmi": 27.0, "body_fat_percentage": 28.0}}])[0]
    change = case_risk_change(case, predict_all(case["inbody"], case["profile"]))
    assert len(change) == 4 and all(c["change_pp"] < 0 for c in change)  # 지난번보다 BMI·체지방률이 낮아짐
    assert case_risk_change({**case, "previous": None}, []) is None
