from app.services import ai_service
from rag.eval import grounding as g

_REPORT = {
    "summary": {"title": "t", "overall_message": "요약", "key_focus": ["초점1"]},
    "body_composition_report": {"strengths": [{"metric": "BMI", "message": "정상"}],
                                "watch_points": [{"metric": "체지방률", "current_value": "39%",
                                                  "interpretation": "높음", "priority": "high", "source_ids": []}]},
    "risk_report": [{"disease": "고혈압", "risk_level": "high", "probability": 0.5, "main_factors": [],
                     "message": "위험", "source_ids": ["S1"]}],
    "lifestyle_plan": [{"category": "exercise", "title": "운동", "reason": "이유",
                        "actions": ["주 150분", "근력운동"], "source_ids": ["S2", "S3"]}],
    "supplement_candidates": [],
    "cautions": {"medical_disclaimer": "면책", "consult_doctor_if": ["상담"]},
}


def test_plain_text_drops_rag_sentences_from_real_prompts():
    plain = g.plain_text(ai_service._REPORT_SYSTEM_INSTRUCTION) + g.plain_text(ai_service._REPORT_GUIDE)
    assert not any(word in plain for word in g.BANNED)
    assert "missing_metrics" in plain and "personal_metrics" in plain and "lifestyle_plan: 1~5개" in plain


def test_report_fields_ids_and_item_of():
    ids = [fid for fid, _ in g.report_fields(_REPORT)]
    assert ids == ["summary.overall_message", "summary.key_focus[0]", "strengths[0]",
                   "watch_points[0].interpretation", "risk[0].message", "plan[0].reason",
                   "plan[0].actions[0]", "plan[0].actions[1]"]
    assert g.item_of("plan[0].actions[1]") == "plan[0]" and g.item_of("summary.key_focus[0]") == "summary"
    assert g.item_of("strengths[0]") == "strengths[0]"


def test_item_sources_only_citable():
    assert g.item_sources(_REPORT) == {"watch_points[0]": [], "risk[0]": ["S1"], "plan[0]": ["S2", "S3"]}


def test_apply_verdicts_ignores_unknown_ids_and_marks_missing():
    claims = [{"id": "c1"}, {"id": "c2"}]
    verdicts = g.Verdicts(verdicts=[
        g.Verdict(claim_id="c1", verdict="supported", passage_ids=["p1", "p7"]),
        g.Verdict(claim_id="c9", verdict="supported", passage_ids=[]),
    ])
    out = g.apply_verdicts(claims, verdicts, {"c1": ["p1", "p2"], "c2": ["p1"]})
    assert out == {"c1": {"verdict": "supported", "passage_ids": ["p1"]},
                   "c2": {"verdict": "missing", "passage_ids": []}}


def test_clean_claims_drops_unknown_fields_and_numbers_them():
    fields = g.report_fields(_REPORT)
    claims = g.Claims(claims=[g.Claim(field_id="plan[0].actions[0]", text="주 150분", kind="medical"),
                              g.Claim(field_id="nope", text="x", kind="medical")])
    assert g.clean_claims(claims, fields) == [
        {"id": "c1", "field_id": "plan[0].actions[0]", "text": "주 150분", "kind": "medical"}]


def test_prompts_do_not_name_the_version():
    fields = g.report_fields(_REPORT)
    judge = g.judge_prompt([{"id": "c1", "text": "주 150분", "passages": [{"pid": "p1", "content": "본문"}]}])
    for prompt in (g.extract_prompt(fields), judge):
        assert "RAG" not in prompt and "단순" not in prompt


def test_codex_schemas_are_strict():
    for model in (g.Claims, g.Verdicts):
        schema = g.schema_of(model)
        assert "additionalProperties" in str(schema) and "$ref" not in str(schema)
