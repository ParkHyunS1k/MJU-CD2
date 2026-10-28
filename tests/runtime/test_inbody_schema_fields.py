from app.schemas.inbody import InBodyCreate, InBodyUpdate
from app.services.inbody_service import INBODY_VALUE_FIELDS


def test_height_and_muscle_control_round_trip():
    rec = InBodyCreate(weight=80, height=183, muscle_control=1.5)
    assert rec.model_dump()["height"] == 183 and rec.model_dump()["muscle_control"] == 1.5
    assert InBodyUpdate(height=None).height is None


def test_upload_mapping_carries_new_fields():
    assert {"height", "muscle_control"} <= set(INBODY_VALUE_FIELDS)
