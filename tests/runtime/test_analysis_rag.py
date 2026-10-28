import pytest

import app.services.inbody_analysis_service as svc
from app.services.retrieval_service import Evidence, RetrievalError

USER = {"id": "u1", "gender": "male", "birth_year": 1990}
TARGET = {"id": "r2", "measured_at": "2026-10-10T00:00:00+00:00", "bmi": 27.0, "image_url": "img.png"}
OLDER = {"id": "r1", "measured_at": "2026-09-01T00:00:00+00:00",
         "ai_report": {"summary": {"key_focus": ["체중"]}, "lifestyle_plan": [{"title": "식단"}]}}
RISKS = [{"disease": "고혈압", "disease_key": "hypertension", "probability": 0.4, "risk_level": "high",
          "top_factors": []}]
EV = Evidence(sid="S1", chunk_id=1, document_id="d", title="고혈압 진료지침", publisher="학회", year=2026, url=None,
              heading_path=None, page_start=74, page_end=74, content="본문", similarity=0.8,
              condition="hypertension", category="diet")
CONTENT = {"summary": {"title": "t", "overall_message": "m", "key_focus": []},
           "body_composition_report": {"strengths": [], "watch_points": []},
           "risk_report": [{"disease": "고혈압", "risk_level": "high", "probability": 0.4, "main_factors": [],
                            "message": "m", "source_ids": ["S1"]}],
           "lifestyle_plan": [], "supplement_candidates": [],
           "cautions": {"medical_disclaimer": "d", "consult_doctor_if": []}}


@pytest.fixture
def wired(monkeypatch):
    calls = {"saved": None, "generate": None}
    monkeypatch.setattr(svc, "predict_all", lambda inbody, user_profile: RISKS)
    monkeypatch.setattr(svc, "_build_trend_analysis", lambda records, risks: None)
    monkeypatch.setattr(svc, "_save_report_and_cleanup_image", lambda sb, t, r: calls.__setitem__("saved", r))

    def fake_generate(**kw):
        calls["generate"] = kw
        return CONTENT

    monkeypatch.setattr(svc, "generate_report", fake_generate)
    return calls


def test_report_gets_sources_and_prompt_gets_evidence(wired, monkeypatch):
    monkeypatch.setattr(svc, "retrieve", lambda queries: [EV])
    report = svc._analyze_inbody_record(None, USER, [TARGET, OLDER], TARGET)
    assert [s.sid for s in report.sources] == ["S1"]
    assert wired["generate"]["evidence"] == [EV]
    assert wired["generate"]["previous_report"] == {"key_focus": ["체중"], "lifestyle_titles": ["식단"],
                                                    "supplements": []}
    assert wired["saved"] is report


def test_analysis_puts_disease_queries_first_and_passes_metrics(wired, monkeypatch):
    # 상한이 쿼리 수 이상이라 모든 쿼리의 1위는 순서와 관계없이 들어간다. 순서는 2위 이하에만 영향을 주므로
    # 질환 쿼리를 앞에 둬야 특성의 2위 근거가 질환의 2·3위 근거를 밀어내지 않는다
    seen = []
    monkeypatch.setattr(svc, "retrieve", lambda queries: seen.extend(queries) or [EV])
    svc._analyze_inbody_record(None, {**USER, "fh_hypertension": True}, [TARGET], TARGET)
    diseases = [q for q in seen if q.trait is None]
    assert diseases and seen[:len(diseases)] == diseases  # 질환 쿼리가 앞에, 특성 쿼리가 뒤에
    assert any(q.trait for q in seen)
    assert "missing_metrics" in wired["generate"]["personal_metrics"]


def test_retrieval_failure_is_503_and_nothing_saved(wired, monkeypatch):
    def boom(queries):
        raise RetrievalError("down")

    monkeypatch.setattr(svc, "retrieve", boom)
    with pytest.raises(svc.AnalysisServiceError) as e:
        svc._analyze_inbody_record(None, USER, [TARGET], TARGET)
    assert e.value.status_code == 503 and "참고 자료" in e.value.detail
    assert wired["saved"] is None and wired["generate"] is None


def test_zero_evidence_still_generates_and_warns(wired, monkeypatch, caplog):
    monkeypatch.setattr(svc, "retrieve", lambda queries: [])
    report = svc._analyze_inbody_record(None, USER, [TARGET], TARGET)
    assert report.sources == [] and report.risk_report[0].source_ids == []
    assert "근거 0개" in caplog.text


def test_previous_summary_tolerates_old_schema_and_picks_latest_earlier():
    older2 = {"id": "r0", "measured_at": "2026-08-01T00:00:00+00:00", "ai_report": {"summary": {}}}
    newer = {"id": "r3", "measured_at": "2026-11-01T00:00:00+00:00", "ai_report": {"summary": {"key_focus": ["x"]}}}
    no_report = {"id": "r1b", "measured_at": "2026-09-15T00:00:00+00:00", "ai_report": None}
    records = [newer, TARGET, no_report, OLDER, older2]
    assert svc._previous_report_summary(records, TARGET)["key_focus"] == ["체중"]
    assert svc._previous_report_summary([older2], TARGET) == {"key_focus": [], "lifestyle_titles": [],
                                                              "supplements": []}
    assert svc._previous_report_summary([TARGET], TARGET) is None


def test_previous_summary_ignores_records_without_measured_at():
    undated = {"id": "rx", "measured_at": None, "ai_report": {"summary": {"key_focus": ["날짜 없음"]}}}
    assert svc._previous_report_summary([TARGET, undated, OLDER], TARGET)["key_focus"] == ["체중"]
    assert svc._previous_report_summary([TARGET, undated], TARGET) is None
