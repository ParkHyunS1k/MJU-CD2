from rag.eval.judge import CandidateJudgment, JudgeCache, QueryJudgment, judge_prompt, judge_query

CANDIDATES = [
    {"content_hash": "h1", "content": "고혈압 진료지침 > 소금\n하루 소금 섭취량을 5 g 이하로 제한한다. 칼륨 섭취를 늘린다."},
    {"content_hash": "h2", "content": "운동 장\n주 5회 이상 유산소 운동을 한다."},
]


def ask_returning(*judgments):
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return QueryJudgment(judgments=list(judgments))

    ask.prompts = prompts
    return ask


def test_prompt_numbers_candidates():
    prompt = judge_prompt("소금 얼마나?", CANDIDATES)
    assert "[c1]" in prompt and "[c2]" in prompt
    assert "하루 소금 섭취량" in prompt


def test_verbatim_evidence_kept_and_irrelevant_cleared():
    ask = ask_returning(
        CandidateJudgment(candidate_id="c1", relevant=True, evidence=["하루 소금 섭취량을 5 g 이하로 제한한다."]),
        CandidateJudgment(candidate_id="c2", relevant=False, evidence=["주 5회 이상 유산소 운동을 한다."]),
    )
    result = judge_query(ask, "소금 얼마나?", CANDIDATES)
    assert result == {
        "h1": {"relevant": True, "evidence": ["하루 소금 섭취량을 5 g 이하로 제한한다."]},
        "h2": {"relevant": False, "evidence": []},
    }


def test_eval_judge_drops_non_verbatim_evidence():
    ask = ask_returning(
        CandidateJudgment(candidate_id="c1", relevant=True,
                          evidence=["소금은 하루 5g 미만으로", "하루소금섭취량을5 g 이하로 제한한다."]),
        CandidateJudgment(candidate_id="c2", relevant=False, evidence=[]),
    )
    result = judge_query(ask, "q", CANDIDATES)
    assert result["h1"] == {"relevant": True, "evidence": ["하루소금섭취량을5 g 이하로 제한한다."]}


def test_eval_judge_missing_and_unknown_candidates():
    ask = ask_returning(CandidateJudgment(candidate_id="c9", relevant=True, evidence=[]))
    result = judge_query(ask, "q", CANDIDATES)
    assert result == {"h1": {"relevant": None, "evidence": []}, "h2": {"relevant": None, "evidence": []}}


def test_eval_judge_cache_resumes(tmp_path):
    cache = JudgeCache(tmp_path / "judgments.jsonl")
    assert cache.get("q1") is None
    cache.put("q1", {"h1": {"relevant": True, "evidence": ["a"]}})
    cache.put("q1", {"h1": {"relevant": False, "evidence": []}})
    reopened = JudgeCache(tmp_path / "judgments.jsonl")
    assert reopened.get("q1") == {"h1": {"relevant": False, "evidence": []}}


def test_codex_ask_runs_read_only_exec_and_parses_output():
    import json
    from pathlib import Path

    from rag.eval.judge import codex_ask

    seen = {}

    def fake_run(cmd, input, text, capture_output, timeout, check):
        seen["cmd"], seen["input"] = cmd, input
        out = Path(cmd[cmd.index("-o") + 1])
        schema = json.loads(Path(cmd[cmd.index("--output-schema") + 1]).read_text())
        seen["schema_required"] = schema["required"]
        out.write_text(json.dumps({"judgments": [{"candidate_id": "c1", "relevant": True, "evidence": ["x"]}]}))

    result = codex_ask(run=fake_run)("판정 프롬프트")
    assert result == QueryJudgment(judgments=[CandidateJudgment(candidate_id="c1", relevant=True, evidence=["x"])])
    assert seen["cmd"][:2] == ["codex", "exec"]
    assert seen["cmd"][seen["cmd"].index("--sandbox") + 1] == "read-only"
    assert seen["cmd"][-1] == "-"
    assert seen["input"] == "판정 프롬프트"
    assert seen["schema_required"] == ["judgments"]
