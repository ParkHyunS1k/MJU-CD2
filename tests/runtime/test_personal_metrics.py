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


STD = "표준"


def _seg(**over):
    base = {"left_arm": STD, "right_arm": STD, "trunk": STD, "left_leg": STD, "right_leg": STD}
    return {**base, **over}


def test_segmental_summary_lists_non_standard_parts():
    m = compute({"segmental_lean": _seg(left_leg="표준이하", right_leg="표준이하"),
                 "segmental_fat": _seg(trunk="표준이상")}, MALE)
    assert m["segmental"] == {"근육 표준이하": ["왼다리", "오른다리"], "체지방 표준이상": ["몸통"]}


def test_segmental_left_right_difference():
    m = compute({"segmental_lean": _seg(left_leg="표준이하"), "segmental_fat": _seg(left_arm="표준이상")}, MALE)
    assert m["segmental"]["근육 좌우 차이"] == ["다리"] and m["segmental"]["체지방 좌우 차이"] == ["팔"]


def test_segmental_absent_or_all_standard_is_omitted():
    assert "segmental" not in compute({}, MALE)
    assert "segmental" not in compute({"segmental_lean": _seg(), "segmental_fat": _seg()}, MALE)


def test_body_type_from_fat_and_muscle():
    def bt(**inbody):
        return compute(inbody, MALE).get("body_type")

    assert bt(body_fat_percentage=27.0, muscle_control=0.0)["type"] == "지방 감량형"
    assert bt(body_fat_percentage=23.0, muscle_control=1.5)["type"] == "지방→근육형"  # 경도비만
    assert bt(body_fat_percentage=15.0, muscle_control=1.5)["type"] == "근육 보강형"
    assert bt(body_fat_percentage=15.0, muscle_control=0.0)["type"] == "해당 없음"
    unknown = bt(body_fat_percentage=27.0)
    assert unknown == {"type": "근육 판정 불가", "fat_excess": True}
    assert bt(muscle_control=1.5) is None and "body_type" not in compute({"body_fat_percentage": 27.0}, {})


def test_body_type_fat_loss_kg_from_fat_control():
    m = compute({"body_fat_percentage": 27.0, "muscle_control": 0.0, "fat_control": -6.2}, MALE)
    assert m["body_type"]["fat_loss_kg"] == 6.2
    m = compute({"body_fat_percentage": 27.0, "muscle_control": 0.0, "fat_control": 0.0}, MALE)
    assert "fat_loss_kg" not in m["body_type"]
