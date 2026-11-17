from types import SimpleNamespace

import pytest

import app.services.risk_store as rs

PROFILE = {"gender": "male", "birth_year": 1990, "fh_hypertension": True, "fh_diabetes": None, "fh_dyslipidemia": False}
RECORD = {"id": "r1", "user_id": "u1", "measured_at": "2025-06-02T00:00:00+00:00", "bmi": 26.0,
          "body_fat_percentage": 27.5}
CONTRIB = {"age": 0.1, "female": 0.0, "bmi": 0.2, "body_fat_percentage": 0.3,
           "fh_hypertension": 0.4, "fh_diabetes": 0.0, "fh_dyslipidemia": 0.0}
PRED = [{"disease": "고혈압", "disease_key": "hypertension", "probability": 0.3, "risk_level": "medium",
         "top_factors": [], "contributions": CONTRIB}]


class FakeQuery:
    def __init__(self, sb, name):
        self.sb, self.name, self.filters, self.data = sb, name, {}, None

    def select(self, *_):
        return self

    def eq(self, key, value):
        self.filters[key] = value
        return self

    def order(self, *_, **__):
        return self

    def limit(self, *_):
        return self

    def update(self, data):
        self.data = data
        return self

    def execute(self):
        if self.data is not None:
            if self.sb.fail_update:
                raise RuntimeError("db down")
            self.sb.updates.append((self.name, self.data, dict(self.filters)))
            return SimpleNamespace(data=[{**self.filters, **self.data}])
        rows = [r for r in self.sb.rows.get(self.name, [])
                if all(r.get(k) == v for k, v in self.filters.items())]
        return SimpleNamespace(data=rows)


class FakeSB:
    def __init__(self, rows=None, fail_update=False):
        self.rows, self.fail_update, self.updates = rows or {}, fail_update, []

    def table(self, name):
        return FakeQuery(self, name)


@pytest.fixture
def predict(monkeypatch):
    calls = []

    def fake(inbody, user_profile):
        calls.append(inbody)
        return PRED

    monkeypatch.setattr(rs, "predict_all", fake)
    return calls


def test_inputs_of_snapshot():
    assert rs.inputs_of(RECORD, PROFILE) == {
        "gender": "male", "birth_year": 1990, "fh_hypertension": True, "fh_diabetes": None,
        "fh_dyslipidemia": False, "measured_year": 2025, "bmi": 26.0, "body_fat_percentage": 27.5}


def test_compute_shape(predict):
    value = rs.compute(RECORD, PROFILE)
    assert value["inputs"] == rs.inputs_of(RECORD, PROFILE) and value["age"] == 35
    assert value["diseases"] == [{"disease": "고혈압", "disease_key": "hypertension", "probability": 0.3,
                                  "risk_level": "medium", "contributions": CONTRIB}]


@pytest.mark.parametrize("result", [[], ValueError("범위")])
def test_compute_none_when_uncomputable(monkeypatch, result):
    def fake(inbody, user_profile):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(rs, "predict_all", fake)
    assert rs.compute(RECORD, PROFILE) is None


def test_ensure_returns_stored_when_inputs_match(predict):
    stored = {"inputs": rs.inputs_of(RECORD, PROFILE), "age": 35, "diseases": []}
    sb = FakeSB()
    assert rs.ensure(sb, {**RECORD, rs.COLUMN: stored}, PROFILE) is stored
    assert sb.updates == [] and predict == []


def test_ensure_recomputes_when_profile_changed(predict):
    stored = {"inputs": rs.inputs_of(RECORD, PROFILE), "age": 35, "diseases": []}
    sb = FakeSB()
    value = rs.ensure(sb, {**RECORD, rs.COLUMN: stored}, {**PROFILE, "fh_diabetes": True})
    assert value["inputs"]["fh_diabetes"] is True and value["diseases"]
    assert sb.updates == [("inbody_logs", {rs.COLUMN: value}, {"id": "r1"})]


def test_ensure_recomputes_when_record_values_changed(predict):
    stored = {"inputs": rs.inputs_of(RECORD, PROFILE), "age": 35, "diseases": []}
    sb = FakeSB()
    value = rs.ensure(sb, {**RECORD, "bmi": 24.0, rs.COLUMN: stored}, PROFILE)
    assert value["inputs"]["bmi"] == 24.0 and len(sb.updates) == 1


def test_ensure_returns_value_even_if_save_fails(predict, caplog):
    value = rs.ensure(FakeSB(fail_update=True), RECORD, PROFILE)
    assert value["diseases"] and "db down" in caplog.text


def test_refresh_record_swallows_errors(monkeypatch, caplog):
    def boom(inbody, user_profile):
        raise RuntimeError("model broke")

    monkeypatch.setattr(rs, "predict_all", boom)
    rs.refresh_record(FakeSB(), RECORD, PROFILE)
    assert "model broke" in caplog.text


def test_refresh_record_reads_profile_when_missing(predict):
    sb = FakeSB({"users": [{"id": "u1", **PROFILE}]})
    rs.refresh_record(sb, RECORD)
    assert sb.updates[0][1][rs.COLUMN]["inputs"]["gender"] == "male"


def test_refresh_user_updates_every_record(predict):
    sb = FakeSB({"inbody_logs": [RECORD, {**RECORD, "id": "r2"}]})
    rs.refresh_user(sb, "u1", PROFILE)
    assert [u[2]["id"] for u in sb.updates] == ["r1", "r2"]


def test_trend_orders_by_date_and_drops_uncomputable(monkeypatch):
    def fake(inbody, user_profile):
        return [] if inbody["id"] == "bad" else PRED

    monkeypatch.setattr(rs, "predict_all", fake)
    later = {**RECORD, "id": "r2", "measured_at": "2025-09-01T00:00:00+00:00"}
    bad = {**RECORD, "id": "bad", "measured_at": "2025-07-01T00:00:00+00:00"}
    sb = FakeSB({"users": [{"id": "u1", **PROFILE}], "inbody_logs": [later, bad, RECORD]})
    out = rs.trend(sb, "u1")
    assert [p["inbody_id"] for p in out] == ["r1", "r2"]
    assert out[0]["measured_at"] == RECORD["measured_at"]


def test_trend_has_no_contributions(predict):
    sb = FakeSB({"users": [{"id": "u1", **PROFILE}], "inbody_logs": [RECORD]})
    assert rs.trend(sb, "u1")[0]["risks"] == [
        {"disease": "고혈압", "disease_key": "hypertension", "probability": 0.3, "risk_level": "medium"}]


def test_trend_without_user_is_empty():
    assert rs.trend(FakeSB(), "u1") == []


def test_backfill_counts_records(predict):
    sb = FakeSB({"users": [{"id": "u1", **PROFILE}], "inbody_logs": [RECORD, {**RECORD, "id": "r2"}]})
    assert rs.backfill(sb) == 2 and len(sb.updates) == 2


def test_risk_trend_route_registered_before_record_route():
    from main import app
    paths = list(app.openapi()["paths"])
    assert paths.index("/inbody/risk-trend") < paths.index("/inbody/{inbody_id}")
