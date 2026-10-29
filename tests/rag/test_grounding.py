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


def _entry(version, claims, corpus, cited=None, report=_REPORT):
    return {"case": "A", "version": version, "report": report, "evidence": [],
            "claims": [{"id": cid, "field_id": fid, "text": cid, "kind": kind} for cid, fid, kind in claims],
            "corpus": {cid: {"verdict": v, "passage_ids": [], "passages": []} for cid, v in corpus.items()},
            "cited": {cid: {"verdict": v, "passage_ids": []} for cid, v in (cited or {}).items()}}


_PLAIN_REPORT = {**_REPORT, "risk_report": [{**_REPORT["risk_report"][0], "source_ids": []}],
                 "lifestyle_plan": [{**_REPORT["lifestyle_plan"][0], "source_ids": []}]}


def test_metrics_counts_and_rates():
    plain = _entry("plain", [("c1", "plan[0].actions[0]", "medical"), ("c2", "plan[0].actions[1]", "medical"),
                             ("c3", "risk[0].message", "medical"), ("c4", "plan[0].reason", "user_data")],
                   {"c1": "not_found", "c2": "not_found", "c3": "supported"}, report=_PLAIN_REPORT)
    m = g.metrics([plain])
    assert m["medical"] == 3 and m["citation_rate"] == 0.0 and m["unsupported_rate"] == 2 / 3
    assert m["cite_accuracy"] is None

    rag = _entry("rag", [("c1", "plan[0].actions[0]", "medical"), ("c2", "plan[0].actions[1]", "medical"),
                         ("c3", "plan[0].reason", "medical"), ("c4", "summary.overall_message", "medical")],
                 {"c1": "supported", "c2": "supported", "c3": "supported", "c4": "contradicted"},
                 {"c1": "supported", "c2": "supported", "c3": "not_found"})
    m = g.metrics([rag])
    assert (m["citable"], m["cited"], m["citation_rate"]) == (3, 3, 1.0)
    assert (m["unsupported"], m["contradicted"], m["unsupported_rate"]) == (1, 1, 0.25)
    assert (m["cite_judged"], m["cite_supported"], m["cite_accuracy"]) == (3, 2, 2 / 3)


def test_metrics_no_medical_claims_gives_none():
    m = g.metrics([_entry("plain", [("c1", "plan[0].reason", "user_data")], {})])
    assert m["medical"] == 0 and m["citation_rate"] is None and m["unsupported_rate"] is None


def test_metrics_excludes_missing_from_rate():
    e = _entry("plain", [("c1", "risk[0].message", "medical"), ("c2", "plan[0].reason", "medical")],
               {"c1": "not_found", "c2": "missing"}, report=_PLAIN_REPORT)
    m = g.metrics([e])
    assert (m["judged"], m["missing"], m["unsupported_rate"]) == (1, 1, 1.0)


def test_render_summary_has_both_versions_and_examples():
    plain = _entry("plain", [("c1", "plan[0].actions[0]", "medical")], {"c1": "not_found"}, report=_PLAIN_REPORT)
    rag = _entry("rag", [("c1", "plan[0].actions[0]", "medical")], {"c1": "supported"}, {"c1": "supported"})
    assert g.examples([plain, rag]) == [(plain["claims"][0], rag["claims"][0])]
    md = g.render_summary([plain, rag])
    assert "단순 프롬프트" in md and "RAG" in md and "근거 없는 주장 비율" in md


class _Fake:
    def __init__(self, result):
        self.result, self.prompts = result, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.result


def _fakes():
    claims = _Fake(g.Claims(claims=[g.Claim(field_id="plan[0].actions[0]", text="주 150분 운동", kind="medical"),
                                    g.Claim(field_id="plan[0].reason", text="체지방률 39%", kind="user_data")]))
    verdicts = _Fake(g.Verdicts(verdicts=[g.Verdict(claim_id="c1", verdict="supported", passage_ids=["p1"])]))
    searched = []

    def search(texts):
        searched.append(texts)
        return [[{"chunk_id": 9, "title": "운동", "heading_path": "권고", "content": "주 150분"}] for _ in texts]

    return claims, verdicts, search, searched


def test_evaluate_fills_stages_and_skips_done_ones():
    claims, verdicts, search, searched = _fakes()
    entry = {"case": "A", "version": "rag", "report": _REPORT,
             "evidence": [{"sid": "S2", "chunk_id": 1, "title": "t", "heading_path": "h", "content": "주 150분"}]}
    g.evaluate(entry, claims, verdicts, search)
    assert [c["kind"] for c in entry["claims"]] == ["medical", "user_data"]
    assert searched == [["주 150분 운동"]]
    assert entry["corpus"]["c1"]["verdict"] == "supported" and entry["corpus"]["c1"]["passages"][0]["pid"] == "p1"
    assert "(S2) 주 150분" in verdicts.prompts[-1]  # 인용 판정은 그 항목이 인용한 근거만 본다
    assert set(entry["cited"]) == {"c1"}
    calls = (len(claims.prompts), len(verdicts.prompts), len(searched))
    g.evaluate(entry, claims, verdicts, search)
    assert (len(claims.prompts), len(verdicts.prompts), len(searched)) == calls


def test_evaluate_plain_has_no_cited_stage():
    claims, verdicts, search, _ = _fakes()
    entry = {"case": "A", "version": "plain", "report": _REPORT, "evidence": []}
    g.evaluate(entry, claims, verdicts, search)
    assert entry["cited"] == {} and len(verdicts.prompts) == 1


def test_generate_plain_prompt_has_no_evidence_block(monkeypatch):
    from types import SimpleNamespace

    from app.schemas.inbody import AIReportContent

    seen = {}

    def parse(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(output_parsed=AIReportContent.model_validate(_REPORT))

    monkeypatch.setattr(ai_service, "get_openai", lambda: SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    monkeypatch.setattr("app.services.prediction_service.predict_all", lambda inbody, profile: [])
    case = {"case": "A", "profile": {"gender": "male", "birth_year": 1995}, "inbody": {"weight": 70.0, "bmi": 22.0}}
    report, evidence = g.generate(case, "plain")
    text = seen["instructions"] + seen["input"]
    assert not any(word in text for word in ("[근거]", "S번호", "source_ids")) and "[사용자 데이터]" in text
    assert evidence == [] and report["lifestyle_plan"][0]["source_ids"] == [] and report["risk_report"][0]["source_ids"] == []
    assert ai_service._REPORT_GUIDE in ai_service.build_report_prompt({}, [], None)  # 패치가 끝나면 원래대로


def test_plain_rewrites_keep_non_rag_rules():
    for old, _ in g.PLAIN_REWRITES:  # 앱 프롬프트가 바뀌면 여기서 알 수 있다
        assert old in ai_service._REPORT_SYSTEM_INSTRUCTION + ai_service._REPORT_GUIDE
    system, guide = g.plain_prompts()
    plain = system + guide
    assert not any(word in plain for word in g.BANNED)
    for rule in ("일반 권고는 이 사람의 현재 상태", "kg_to_normal_bmi", "personal_metrics.segmental",
                 "일반론으로 채우지 마세요", "SHAP 기여가 큰 요인부터", "보수적인 보충제 후보",
                 "supplement_candidates: 최대 3개", "해당 섹션을 일반 조언으로 채우지"):
        assert rule in plain, rule


def test_render_summary_reports_supplement_claims_separately():
    plain = _entry("plain", [("c1", "risk[0].message", "medical"), ("c2", "supp[0].purpose", "medical")],
                   {"c1": "supported", "c2": "not_found"}, report=_PLAIN_REPORT)
    md = g.render_summary([plain])
    assert "영양제 의학 주장" in md and "영양제 제외" in md
    assert g.metrics(g.without_supplements([plain]))["unsupported_rate"] == 0.0


def test_generate_pins_the_report_model(monkeypatch):
    from types import SimpleNamespace

    from app.schemas.inbody import AIReportContent

    seen = []

    def parse(**kwargs):
        seen.append(ai_service.REPORT_MODELS)
        return SimpleNamespace(output_parsed=AIReportContent.model_validate(_REPORT))

    monkeypatch.setattr(ai_service, "get_openai", lambda: SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    monkeypatch.setattr("app.services.prediction_service.predict_all", lambda inbody, profile: [])
    g.generate({"case": "A", "profile": {"gender": "male", "birth_year": 1995}, "inbody": {"bmi": 22.0}}, "plain")
    assert seen == [(g.REPORT_MODEL,)]


def test_spotcheck_hides_verdicts(tmp_path):
    import json

    e = _entry("rag", [("c1", "plan[0].actions[0]", "medical")], {"c1": "contradicted"})
    (tmp_path / "state.json").write_text(json.dumps({"A|rag": e}, ensure_ascii=False))
    g.spotcheck(tmp_path)
    assert "contradicted" not in (tmp_path / "spotcheck.md").read_text()
    assert json.loads((tmp_path / "spotcheck_key.json").read_text()) == [
        {"n": 1, "key": "A|rag", "claim_id": "c1", "verdict": "contradicted"}]


def test_load_skips_unfinished_entries(tmp_path):
    import json

    done = _entry("rag", [("c1", "plan[0].actions[0]", "medical")], {"c1": "supported"})
    (tmp_path / "state.json").write_text(json.dumps({"A|rag": done, "B|plain": {"case": "B", "version": "plain"}},
                                                    ensure_ascii=False))
    assert [e["case"] for e in g._load(tmp_path)] == ["A"]
