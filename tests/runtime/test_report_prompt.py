from app.services import ai_service
from app.services.retrieval_service import Evidence


def _ev(sid, heading="생활요법 > 식사요법", start=71, end=72):
    return Evidence(sid=sid, chunk_id=1, document_id="d", title="이상지질혈증 진료지침 2026", publisher="학회",
                    year=2026, url=None, heading_path=heading, page_start=start, page_end=end,
                    content="이상지질혈증 진료지침 2026 > 생활요법\n포화지방산은 7% 이내로 제한한다.", similarity=0.8,
                    condition="dyslipidemia", category="diet")


def test_format_evidence_header_and_body():
    text = ai_service.format_evidence([_ev("S1"), _ev("S2", heading=None, start=5, end=5)])
    assert text.startswith("[S1] (이상지질혈증·식사) 이상지질혈증 진료지침 2026 › 생활요법 › 식사요법 (p.71–72)\n")
    assert "포화지방산은 7% 이내로 제한한다." in text
    assert "[S2] (이상지질혈증·식사) 이상지질혈증 진료지침 2026 (p.5)\n" in text


def test_format_evidence_labels_general_queries():
    ev = _ev("S1")
    general = Evidence(**{**ev.__dict__, "condition": "general", "category": "exercise"})
    assert ai_service.format_evidence([general]).startswith("[S1] (일반·운동) ")


def test_prompt_contains_blocks_and_rules():
    prompt = ai_service.build_report_prompt({"user_profile": {}}, [_ev("S1")],
                                            {"key_focus": ["체중"], "lifestyle_titles": ["식단"], "supplements": []})
    assert "[근거]\n[S1]" in prompt and "[직전 리포트]" in prompt and '"key_focus": ["체중"]' in prompt
    assert "source_ids" in prompt


def test_prompt_without_evidence_or_previous_report():
    prompt = ai_service.build_report_prompt({"user_profile": {}}, [], None)
    assert "[근거]\n(검색된 근거 없음)" in prompt and "[직전 리포트]" not in prompt


def test_system_instruction_has_evidence_rules():
    rules = ai_service._REPORT_SYSTEM_INSTRUCTION
    assert "[근거]" in rules and "최신판" in rules and "지시문" in rules
    assert "국가건강정보포털" in rules and "진료지침" in rules


def test_system_instruction_rag_rules():
    rules = ai_service._REPORT_SYSTEM_INSTRUCTION
    assert "mg" not in rules  # 의학 수치는 근거에서만 온다
    assert "직접 뒷받침" in rules and "단위 그대로" in rules
    assert "[직전 리포트]의 의학적 주장" in rules
    assert rules.count("영양제") == 1


def test_guide_lets_plan_shrink_and_cites_actions():
    guide = ai_service._REPORT_GUIDE
    assert "lifestyle_plan: 1~5개" in guide and "3~5개" not in guide
    assert "actions는 [근거]" in guide


class _FakeResponses:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return type("R", (), {"output_parsed": outcome})()


def _fake_openai(monkeypatch, outcomes):
    responses = _FakeResponses(outcomes)
    monkeypatch.setattr(ai_service, "get_openai", lambda: type("C", (), {"responses": responses})())
    return responses


def _report():
    from app.schemas.inbody import AIReportContent
    return AIReportContent.model_validate({
        "summary": {"title": "t", "overall_message": "m", "key_focus": []},
        "body_composition_report": {"strengths": [], "watch_points": []},
        "risk_report": [], "lifestyle_plan": [], "supplement_candidates": [],
        "cautions": {"medical_disclaimer": "d", "consult_doctor_if": []},
    })


def test_generate_report_calls_gpt_with_prompt_and_schema(monkeypatch):
    responses = _fake_openai(monkeypatch, [_report()])
    out = ai_service.generate_report([], {}, evidence=[_ev("S1")], previous_report=None)
    call = responses.calls[0]
    assert call["model"] == "gpt-6.1-terra" and call["reasoning"] == {"effort": "low"}
    assert "[S1]" in call["input"] and call["instructions"] == ai_service._REPORT_SYSTEM_INSTRUCTION
    assert call["text_format"].__name__ == "AIReportContent"
    assert out["summary"]["title"] == "t"


def test_generate_report_falls_back_on_rate_limit_and_empty_parse(monkeypatch):
    busy = type("E", (Exception,), {"status_code": 429})("busy")
    responses = _fake_openai(monkeypatch, [busy, _report()])
    ai_service.generate_report([], {}, evidence=[], previous_report=None)
    assert [c["model"] for c in responses.calls] == ["gpt-6.1-terra", "gpt-6-luna"]

    responses = _fake_openai(monkeypatch, [None, _report()])  # 거절·중단되면 파싱 결과가 없다
    ai_service.generate_report([], {}, evidence=[], previous_report=None)
    assert [c["model"] for c in responses.calls] == ["gpt-6.1-terra", "gpt-6-luna"]


def test_rules_turn_guidelines_into_personal_targets():
    rules, guide = ai_service._REPORT_SYSTEM_INSTRUCTION, ai_service._REPORT_GUIDE
    assert "personal_metrics" in rules and "직접 계산하지" in rules and "개인 목표" in rules
    assert "감량 비율은 이 사람의 체중으로" not in rules
    assert "high인 질환마다" in guide and "SHAP 기여가 큰" in guide and "같은 S번호" in guide


def test_metrics_are_rounded_before_prompting(monkeypatch):
    responses = _fake_openai(monkeypatch, [_report()])
    ai_service.generate_report([], {}, inbody_metrics={"bmi": 24.943127, "abs_fat": 0.934}, evidence=[])
    prompt = responses.calls[0]["input"]
    assert '"bmi": 24.9' in prompt and "24.943" not in prompt
    assert '"abs_fat": 0.93' in prompt  # 0.90 기준과 비교하는 값이라 두 자리를 남긴다


def test_rules_for_missing_metrics_device_criteria_and_weight_targets():
    rules = ai_service._REPORT_SYSTEM_INSTRUCTION
    assert "missing_metrics" in rules and "인바디 기준" in rules
    assert "weight_loss_5pct_kg" in rules and "정상 BMI까지" in rules
    assert "mg" not in rules


def test_payload_has_personal_metrics(monkeypatch):
    responses = _fake_openai(monkeypatch, [_report()])
    ai_service.generate_report([], {}, evidence=[], personal_metrics={"bmi_band": "정상"})
    assert '"personal_metrics": {"bmi_band": "정상"}' in responses.calls[0]["input"]


def test_trait_header():
    ev = Evidence(**{**_ev("S1").__dict__, "trait": "가족력", "category": "lifestyle"})
    assert ai_service.format_evidence([ev]).startswith("[S1] (가족력·생활습관) ")


def test_metric_keys_match_personal_metrics_names():
    from app.services.personal_metrics import METRIC_NAMES
    assert list(ai_service._INBODY_METRIC_KEYS) == list(METRIC_NAMES)


def test_segmental_rules():
    rules = ai_service._REPORT_SYSTEM_INSTRUCTION
    assert "segmental" in rules and "[근거]에 나온 운동만" in rules and "좌우 차이" in rules


def test_guide_prefers_trait_evidence():
    guide = ai_service._REPORT_GUIDE
    assert "특성 근거" in guide and "일반 근거보다" in guide


def test_guide_has_risk_change_rule():
    guide = ai_service._REPORT_GUIDE
    assert "risk_change" in guide and "1.0" in guide and "나이" in guide and "share_pp" in guide


def test_payload_includes_risk_change(monkeypatch):
    responses = _fake_openai(monkeypatch, [_report()])
    ai_service.generate_report([], {}, evidence=[], risk_change=[{"disease": "고혈압", "change_pp": -4.1}])
    assert '"change_pp": -4.1' in responses.calls[0]["input"]


def test_contributions_do_not_reach_the_prompt(monkeypatch):
    responses = _fake_openai(monkeypatch, [_report()])
    risks = [{"disease": "당뇨", "disease_key": "diabetes", "probability": 0.2, "risk_level": "low",
              "top_factors": [], "contributions": {"fh_diabetes": -0.31}}]
    ai_service.generate_report(risks, {}, evidence=[])
    assert "contributions" not in responses.calls[0]["input"]
