from app.schemas.inbody import HealthReport
from app.services.citation_service import validate_citations
from app.services.retrieval_service import Evidence


def _ev(sid, start=74, end=75):
    return Evidence(sid=sid, chunk_id=int(sid[1:]), document_id="d", title=f"문서{sid}", publisher="학회",
                    year=2026, url="https://x", heading_path="생활요법", page_start=start, page_end=end,
                    content="본문", similarity=0.8, condition="hypertension", category="diet")


def _content(**over):
    base = {
        "summary": {"title": "t", "overall_message": "m", "key_focus": ["a"]},
        "body_composition_report": {"strengths": [], "watch_points": [
            {"metric": "BMI", "current_value": "27", "interpretation": "i", "priority": "high", "source_ids": ["S2"]}]},
        "risk_report": [{"disease": "고혈압", "risk_level": "high", "probability": 0.4, "main_factors": [],
                         "message": "m", "source_ids": [" s1", "S9", "S1"]}],
        "lifestyle_plan": [{"category": "nutrition", "title": "t", "reason": "r", "actions": [], "source_ids": []}],
        "supplement_candidates": [
            {"name": "오메가3", "purpose": "p", "related_metrics": [], "evidence_level": "limited", "why_consider": "w",
             "cautions": [], "not_recommended_if": [], "source_ids": ["S7"]},
            {"name": "홍국", "purpose": "p", "related_metrics": [], "evidence_level": "limited", "why_consider": "w",
             "cautions": [], "not_recommended_if": [], "source_ids": ["S2"]}],
        "cautions": {"medical_disclaimer": "d", "consult_doctor_if": []},
    }
    return {**base, **over}


def test_unknown_duplicate_and_messy_ids_are_cleaned():
    content, _ = validate_citations(_content(), [_ev("S1"), _ev("S2")])
    assert content["risk_report"][0]["source_ids"] == ["S1"]


def test_supplement_without_valid_citation_is_dropped():
    content, _ = validate_citations(_content(), [_ev("S1"), _ev("S2")])
    assert [s["name"] for s in content["supplement_candidates"]] == ["홍국"]


def test_items_without_citations_are_kept():
    content, _ = validate_citations(_content(), [_ev("S1")])
    assert len(content["lifestyle_plan"]) == 1 and content["body_composition_report"]["watch_points"][0]["source_ids"] == []


def test_sources_hold_only_cited_evidence_in_sid_order():
    _, sources = validate_citations(_content(), [_ev("S1"), _ev("S2", 71, 71), _ev("S3")])
    assert [s["sid"] for s in sources] == ["S1", "S2"]
    assert sources[0]["page"] == "74–75" and sources[1]["page"] == "71" and sources[0]["title"] == "문서S1"


def test_no_evidence_drops_all_supplements():
    content, sources = validate_citations(_content(), [])
    assert content["supplement_candidates"] == [] and sources == []


def test_old_report_without_new_fields_still_parses():
    old = {k: v for k, v in _content().items()}
    for item in old["risk_report"] + old["lifestyle_plan"] + old["supplement_candidates"]:
        item.pop("source_ids")
    old["body_composition_report"]["watch_points"][0].pop("source_ids")
    report = HealthReport(disease_risks=[], **old)
    assert report.sources == [] and report.risk_report[0].source_ids == []
