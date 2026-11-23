from types import SimpleNamespace

import pytest

import app.services.inbody_service as svc
from app.schemas.inbody import InBodyCreate, InBodyUpdate

ROW = {"id": "r1", "user_id": "u1", "weight": 70.0, "bmi": 24.0, "body_fat_percentage": 22.0, "ai_report": None}


class FakeTable:
    def insert(self, data):
        return self

    def update(self, data):
        return self

    def eq(self, *_):
        return self

    def execute(self):
        return SimpleNamespace(data=[ROW])


@pytest.fixture
def refreshed(monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "get_supabase_admin", lambda: SimpleNamespace(table=lambda _: FakeTable()))
    monkeypatch.setattr(svc, "refresh_record", lambda sb, record, profile=None: calls.append(record))
    return calls


def test_create_record_refreshes_risks(refreshed):
    assert svc.create_inbody_record("u1", InBodyCreate(weight=70.0))["id"] == "r1"
    assert refreshed == [ROW]


def test_update_record_refreshes_risks(refreshed):
    svc.update_inbody_record("u1", "r1", InBodyUpdate(bmi=24.0))
    assert refreshed == [ROW]


def test_upload_record_refreshes_risks(refreshed, monkeypatch):
    monkeypatch.setattr(svc, "extract_inbody_from_image",
                        lambda b, t: {"is_inbody": True, "weight": 70.0, "confidence": 0.9})
    monkeypatch.setattr(svc, "upload_inbody_file", lambda u, b, t: "u1/a.png")
    assert svc.create_inbody_from_upload("u1", b"x", "image/png").record.id == "r1"
    assert refreshed == [ROW]


def test_create_record_succeeds_when_refresh_raises(monkeypatch, caplog):
    def boom(sb, record, profile=None):
        raise RuntimeError("risk save broke")

    monkeypatch.setattr(svc, "get_supabase_admin", lambda: SimpleNamespace(table=lambda _: FakeTable()))
    monkeypatch.setattr(svc, "refresh_record", boom)
    assert svc.create_inbody_record("u1", InBodyCreate(weight=70.0))["id"] == "r1"
    assert "risk save broke" in caplog.text
