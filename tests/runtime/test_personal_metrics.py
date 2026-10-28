from app.services.personal_metrics import METRIC_NAMES, compute

MALE, FEMALE = {"gender": "male"}, {"gender": "female"}


def test_height_derived_from_weight_and_bmi():
    m = compute({"weight": 80.0, "bmi": 23.9}, MALE)
    assert m["height_cm"] == 183.0 and m["height_source"] == "derived"
    m = compute({"weight": 80.0, "bmi": 23.9, "height": 183}, MALE)
    assert m["height_cm"] == 183 and m["height_source"] == "measured"


def test_bmi_band_boundaries():
    cases = {18.4: "저체중", 18.5: "정상", 22.9: "정상", 23.0: "비만 전단계", 24.9: "비만 전단계",
             25.0: "1단계 비만", 30.0: "2단계 비만", 35.0: "3단계 비만"}
    for bmi, band in cases.items():
        assert compute({"bmi": bmi}, MALE)["bmi_band"] == band, bmi


def test_weight_targets_only_from_bmi_23():
    m = compute({"weight": 74.9, "bmi": 24.9}, MALE)
    assert m["weight_loss_5pct_kg"] == 3.7 and m["weight_after_5pct"] == 71.2
    assert m["weight_loss_10pct_kg"] == 7.5 and m["weight_after_10pct"] == 67.4
    assert m["kg_to_normal_bmi"] > 0
    m = compute({"weight": 60.0, "bmi": 22.0}, MALE)
    assert not {"weight_loss_5pct_kg", "weight_loss_10pct_kg", "kg_to_normal_bmi"} & m.keys()


def test_body_fat_status_by_sex():
    def st(pbf, prof):
        return compute({"body_fat_percentage": pbf}, prof)["body_fat_status"]
    assert st(20.0, MALE) == {"status": "표준", "over_standard_pp": 0.0}
    assert st(23.0, MALE) == {"status": "경도비만", "over_standard_pp": 3.0}
    assert st(9.0, MALE)["status"] == "낮음"
    assert st(30.0, FEMALE) == {"status": "경도비만", "over_standard_pp": 2.0}
    assert st(34.0, FEMALE)["status"] == "비만"


def test_muscle_and_abdominal():
    assert compute({"muscle_control": 1.5}, MALE)["muscle_status"] == {"status": "표준 이하", "increase_kg": 1.5}
    assert compute({"muscle_control": 0.0}, MALE)["muscle_status"]["status"] == "표준"
    assert "muscle_status" not in compute({}, MALE)
    ab = compute({"abs_fat": 0.91, "fat_level": 10}, MALE)["abdominal"]
    assert [a["metric"] for a in ab] == ["복부지방률", "내장지방레벨"]
    assert ab[0] == {"metric": "복부지방률", "value": 0.91, "threshold": 0.90}
    assert "abdominal" not in compute({"abs_fat": 0.89, "fat_level": 9}, MALE)
    assert "abdominal" in compute({"abs_fat": 0.86}, FEMALE)


def test_missing_inputs_are_omitted_not_errors():
    assert compute({}, MALE) == {"missing_metrics": list(METRIC_NAMES.values())}
    m = compute({"bmi": 24.0}, MALE)
    assert "height_cm" not in m and "kg_to_normal_bmi" not in m and m["bmi_band"] == "비만 전단계"
    assert len(METRIC_NAMES) == 13 and METRIC_NAMES["height"] == "키" and METRIC_NAMES["muscle_control"] == "근육조절"
