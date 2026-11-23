from app.schemas.inbody import InBodyCreate, InBodyUpdate
from app.services.inbody_service import INBODY_VALUE_FIELDS


def test_height_and_muscle_control_round_trip():
    rec = InBodyCreate(weight=80, height=183, muscle_control=1.5)
    assert rec.model_dump()["height"] == 183 and rec.model_dump()["muscle_control"] == 1.5
    assert InBodyUpdate(height=None).height is None


def test_upload_mapping_carries_new_fields():
    assert {"height", "muscle_control"} <= set(INBODY_VALUE_FIELDS)


SHEET = {  # 사용자 결과지(InBody270, 2023-06-23)에 적힌 값
    "inbody_score": 80, "target_weight": 78.6, "weight_control": -4.4, "fat_control": -4.4,
    "fat_free_mass": 66.8, "obesity_degree": 113, "recommended_calories": None,
    "bmi_evaluation": "과체중", "body_fat_evaluation": "표준",
    "segmental_lean": {"left_arm": "표준", "right_arm": "표준", "trunk": "표준", "left_leg": "표준", "right_leg": "표준"},
    "segmental_fat": {"left_arm": "표준", "right_arm": "표준", "trunk": "표준이상", "left_leg": "표준", "right_leg": "표준"},
    "impedance": {"20kHz": {"RA": 309.4, "LA": 307.1, "TR": 25.7, "RL": 271.6, "LL": 269.4},
                  "100kHz": {"RA": 272.9, "LA": 271.6, "TR": 21.7, "RL": 235.5, "LL": 233.5}},
}


def test_all_sheet_values_round_trip():
    dumped = InBodyCreate(weight=83.0, **SHEET).model_dump()
    assert {k: dumped[k] for k in SHEET} == SHEET
    assert set(SHEET) <= set(INBODY_VALUE_FIELDS)


def test_extraction_schema_reads_new_sheet_fields():
    from app.schemas.inbody import InBodyExtraction
    from app.services.ai_service import _parse_json_response
    from app.services.inbody_service import _record_data_from_extraction
    import json

    sheet = {k: v for k, v in SHEET.items() if k != "impedance"}  # 임피던스는 리포트에 쓰지 않아 추출하지 않는다
    raw = json.dumps({"is_inbody": True, "confidence": 0.9, "weight": 83.0, "height": 180.0, "muscle_control": 0.0,
                      **sheet})
    extraction = _parse_json_response(raw, InBodyExtraction)
    data = _record_data_from_extraction("u1", "u1/a.png", extraction)
    assert data["muscle_control"] == 0.0 and data["height"] == 180.0 and data["fat_control"] == -4.4
    assert data["segmental_fat"]["trunk"] == "표준이상" and data["body_fat_evaluation"] == "표준"
