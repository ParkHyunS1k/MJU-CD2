from fastapi.testclient import TestClient

from inbody_ocr.api import app

client = TestClient(app)


def test_server_revalidates_inconsistent_values():
    # 사용자가 BMI를 잘못 고쳐서 보냈다고 가정
    r = client.post("/v1/inbody/measurements", json={
        "input_type": "photo",
        "values": {"weight_kg": 70.0, "height_cm": 172.0, "bmi": 27.7},
        "edited_fields": ["bmi"],
    })
    assert r.status_code == 200
    assert set(r.json()["needs_review"]) == {"weight_kg", "height_cm", "bmi"}


def test_rejects_unknown_fields():
    r = client.post("/v1/inbody/measurements", json={"input_type": "photo", "values": {"name": 1}})
    assert r.status_code == 422
