import numpy as np
import pandas as pd
import pytest

from training.compare import checklist_score, consistency_stats, sens_at_spec


def _X(rows):
    cols = ["age", "female", "bmi", "body_fat_percentage", "fh_hypertension", "fh_diabetes", "fh_dyslipidemia"]
    return pd.DataFrame(rows, columns=cols, dtype=float)


def test_checklist_score_boundaries():
    X = _X([
        [40, 0, 25.0, 25.0, 1, 0, 0],   # 남: 모든 항목 경계값에서 해당 → 4
        [39, 1, 24.9, 29.9, 0, 1, 1],   # 여: 경계 바로 아래, 고혈압 가족력 없음 → 0
        [30, 1, 20.0, 30.0, 0, 0, 0],   # 여 체지방률 30 → 1
    ])
    assert checklist_score(X, "hypertension").tolist() == [4, 0, 1]
    assert checklist_score(X, "cholesterol").tolist() == [3, 1, 1]   # 고지혈증 가족력을 본다
    assert checklist_score(X, "diabetes").tolist() == [3, 1, 1]


def test_sens_at_spec_with_ties():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([1, 2, 3, 4, 3, 5])
    r = sens_at_spec(y, s, 0.75)
    assert r["threshold"] == 4 and r["specificity"] == 0.75 and r["sensitivity"] == 0.5
    tied = sens_at_spec(np.array([0, 0, 1, 1]), np.array([1, 1, 1, 1]), 0.5)
    assert tied["threshold"] == np.inf and tied["specificity"] == 1.0 and tied["sensitivity"] == 0.0


def test_consistency_stats():
    probs = np.array([[0.1] * 5, [0.1, 0.3, 0.1, 0.1, 0.1]])
    r = consistency_stats(probs, 0.2)
    assert r["n"] == 2 and r["flip_rate"] == 0.5
    assert r["range_max"] == pytest.approx(20.0) and r["range_median"] == pytest.approx(10.0)
    assert consistency_stats(np.empty((0, 5)), 0.2) == {"n": 0}


from training.compare import build_results, compare_disease, disease_predictions, write_report


def test_disease_predictions_columns(knhanes_like):
    pred = disease_predictions(knhanes_like, "hypertension", False)
    assert list(pred.columns) == ["y", "age", "base", "rule", "lr", "level", "xgb"]
    assert pred.age.between(25, 44).all() and len(pred) > 0
    assert set(pred.rule.unique()) <= {0, 1, 2, 3, 4}


def test_compare_disease_marks_small_subgroups(knhanes_like):
    pred = disease_predictions(knhanes_like, "hypertension", False)
    young = pred.age <= 34
    pred.loc[young, "y"] = 0.0
    pred.loc[pred[young].index[:5], "y"] = 1.0   # 25–34세 양성 5명
    r = compare_disease(pred, ["base", "rule", "lr", "xgb"])
    assert r["subgroups"]["25-34"]["auroc"]["lr"] is None
    assert r["subgroups"]["35-44"]["auroc"]["lr"] is not None
    assert "auroc_diff" not in r["models"]["lr"] and "auroc_diff" in r["models"]["xgb"]
    assert "brier" not in r["models"]["rule"]
    assert r["models"]["xgb"]["at_spec"]["specificity"] >= r["models"]["lr"]["at_spec"]["specificity"]


def test_report_without_llm(knhanes_like, tmp_path):
    preds = {"hypertension": disease_predictions(knhanes_like, "hypertension", False)}
    res = build_results(preds, None, [])
    out = tmp_path / "compare.md"
    write_report(res, out)
    text = out.read_text(encoding="utf-8")
    assert "## 요약" in text and "② 배포 로지스틱" in text and "④ GPT" not in text


from training.compare import KEYS, REPEATS, parse_response, plot_consistency, repeat_matrix, with_llm


def test_parse_response_accepts_valid():
    assert parse_response('{"hypertension": 15, "diabetes": 3, "cholesterol": 8, "triglyceride": 100}') == \
        {"hypertension": 0.15, "diabetes": 0.03, "cholesterol": 0.08, "triglyceride": 1.0}


def test_parse_response_rejects_bad_output():
    bad = ['{"hypertension": -1, "diabetes": -1, "cholesterol": -1, "triglyceride": -1}',
           '{"hypertension": 101, "diabetes": 3, "cholesterol": 8, "triglyceride": 1}',
           '{"hypertension": 15, "diabetes": 3, "cholesterol": 8}',
           '{"hypertension": 15.5, "diabetes": 3, "cholesterol": 8, "triglyceride": 1}',
           '{"hypertension": true, "diabetes": 3, "cholesterol": 8, "triglyceride": 1}',
           "확률을 추정할 수 없습니다.", ""]
    assert all(parse_response(t) is None for t in bad)


def _rec(idx, rep, probs):
    return {"idx": idx, "rep": rep, "probs": probs, "error": None if probs else "x", "tokens": 1, "seconds": 1.0}


def _fake_cache(preds, rep_ids, noise=0.05):
    """②의 예측에 잡음을 더한 가짜 GPT 응답."""
    rng = np.random.default_rng(1)
    ids = sorted(set().union(*(p.index for p in preds.values())))
    cache = {}
    for i in ids:
        for rep in range(REPEATS if i in rep_ids else 1):
            probs = {k: float(np.clip((preds[k].lr.get(i, 0.1) if k in preds else 0.1) + rng.normal(0, noise), 0, 1))
                     for k in KEYS}
            cache[(int(i), rep)] = _rec(int(i), rep, probs)
    return cache


def test_with_llm_drops_missing_rows(knhanes_like):
    pred = disease_predictions(knhanes_like, "hypertension", False)
    cache = _fake_cache({"hypertension": pred}, [])
    first, second = pred.index[0], pred.index[1]
    cache[(int(first), 0)] = _rec(int(first), 0, None)
    del cache[(int(second), 0)]
    out, missing = with_llm(pred, cache, "hypertension")
    # first: 표본이지만 실패 → 누락 1명. second: 호출 대상(표본)이 아님 → 누락으로 세지 않는다
    assert missing == 1 and first not in out.index and second not in out.index
    assert len(out) == len(pred) - 2


def test_repeat_matrix_requires_all_reps():
    full = {k: 0.1 for k in KEYS}
    cache = {(1, r): _rec(1, r, full) for r in range(REPEATS)}
    cache.update({(2, r): _rec(2, r, full) for r in range(REPEATS - 1)})        # 반복 하나 누락
    cache.update({(3, r): _rec(3, r, full if r else None) for r in range(REPEATS)})  # 1회 실패
    m = repeat_matrix(cache, [1, 2, 3], "diabetes")
    assert m.index.tolist() == [1] and m.shape == (1, REPEATS)


def test_report_and_plot_with_llm(knhanes_like, tmp_path):
    preds = {k: disease_predictions(knhanes_like, k, False) for k in ("hypertension", "diabetes")}
    rep_ids = [int(i) for i in preds["hypertension"].index[:60]]
    cache = _fake_cache(preds, rep_ids)
    res = build_results(preds, cache, rep_ids)
    h = res["diseases"]["hypertension"]
    assert "llm" in h["models"] and h["consistency"]["n"] == 60 and h["llm_missing"] == 0
    write_report(res, tmp_path / "c.md")
    text = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert "④ GPT" in text and "반복 응답 편차" in text and "프롬프트 원문" in text
    plot_consistency(preds, cache, rep_ids, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").exists()


import json as _json

import training.compare as cmp
from training.compare import load_cache, pilot_summary, run_llm


def _people():
    return pd.DataFrame([[34, 0, 26.1, 27.4, 1, 0, 0], [41, 1, 22.0, 31.0, 0, 1, 0]],
                        columns=["age", "female", "bmi", "body_fat_percentage",
                                 "fh_hypertension", "fh_diabetes", "fh_dyslipidemia"], index=[10, 11])


def test_build_prompt_has_inputs_and_definitions():
    p = cmp.build_prompt(_people().loc[10])
    assert "34세" in p and "남성" in p and "BMI 26.1" in p and "체지방률 27.4%" in p
    assert "고혈압 가족력 있음" in p and "당뇨 가족력 없음" in p and "140/90" in p


def test_run_llm_resumes_and_retries_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(cmp.time, "sleep", lambda s: None)
    good = '{"hypertension": 20, "diabetes": 5, "cholesterol": 10, "triglyceride": 12}'
    calls = []

    def flaky(prompt):
        calls.append(prompt)
        if "41세" in prompt and len(calls) <= 4:
            return '{"hypertension": -1, "diabetes": -1, "cholesterol": -1, "triglyceride": -1}', 100
        return good, 100

    cache = tmp_path / "c.jsonl"
    tasks = [(10, 0), (11, 0), (10, 1)]
    run_llm(tasks, _people(), cache, caller=flaky, workers=1)
    c = load_cache(cache)
    assert c[(10, 0)]["probs"]["hypertension"] == 0.2 and c[(11, 0)]["probs"] is None
    n_first = len(calls)
    run_llm(tasks, _people(), cache, caller=flaky, workers=1)   # 성공분은 건너뛰고 실패분만 다시 한다
    assert len(calls) == n_first + 1 and load_cache(cache)[(11, 0)]["probs"] is not None


def test_ask_retries_codex_errors(monkeypatch):
    monkeypatch.setattr(cmp.time, "sleep", lambda s: None)

    def broken(prompt):
        raise cmp.CodexError("Selected model is at capacity")

    r = cmp.ask("p", broken)
    assert r["probs"] is None and "capacity" in r["error"]


def test_load_cache_skips_truncated_line(tmp_path):
    path = tmp_path / "c.jsonl"
    ok = {"idx": 1, "rep": 0, "probs": {k: 0.1 for k in KEYS}, "error": None, "tokens": 5, "seconds": 1.0}
    path.write_text(_json.dumps(ok) + "\n" + '{"idx": 2, "rep": 0, "pro', encoding="utf-8")
    assert list(load_cache(path)) == [(1, 0)]
    assert load_cache(tmp_path / "none.jsonl") == {}


def test_pilot_summary_counts():
    cache = {(1, 0): {"idx": 1, "rep": 0, "probs": {k: 0.1 for k in KEYS}, "error": None, "tokens": 20000, "seconds": 8.0},
             (2, 0): {"idx": 2, "rep": 0, "probs": None, "error": "parse", "tokens": None, "seconds": 9.0}}
    s = pilot_summary(cache, [(1, 0), (2, 0)])
    assert "성공 1/2" in s and "20,000" in s


def test_run_llm_stops_on_usage_limit(tmp_path):
    calls = []

    def limited(prompt):
        calls.append(prompt)
        raise cmp.CodexError("ERROR: You've hit your usage limit. Try again later.")

    cache = tmp_path / "c.jsonl"
    with pytest.raises(cmp.UsageLimitError):
        run_llm([(10, 0), (11, 0), (10, 1)], _people(), cache, caller=limited, workers=1)
    assert len(calls) == 1 and load_cache(cache) == {}


def test_codex_files_written_even_if_schema_exists(tmp_path, monkeypatch):
    io = tmp_path / "io"
    io.mkdir()
    monkeypatch.setattr(cmp, "_WORKDIR", tmp_path / "empty")
    monkeypatch.setattr(cmp, "_SCHEMA_PATH", io / "schema.json")
    monkeypatch.setattr(cmp, "_INSTR_PATH", io / "instructions.md")
    (io / "schema.json").write_text("{}", encoding="utf-8")   # 이전 실행이 남긴 스키마만 있는 상태
    cmp._ensure_codex_files()
    assert (io / "instructions.md").read_text(encoding="utf-8") == cmp.INSTRUCTIONS
    assert _json.loads((io / "schema.json").read_text(encoding="utf-8")) == cmp.SCHEMA
    assert (tmp_path / "empty").is_dir()


def test_report_shows_actual_specificity(knhanes_like, tmp_path):
    preds = {"hypertension": disease_predictions(knhanes_like, "hypertension", False)}
    res = build_results(preds, None, [])
    write_report(res, tmp_path / "c.md")
    text = (tmp_path / "c.md").read_text(encoding="utf-8")
    spec = res["diseases"]["hypertension"]["models"]["rule"]["at_spec"]["specificity"]
    assert "실제 특이도" in text and f"특이도 {100 * spec:.1f}%" in text


def test_run_llm_stops_when_interrupted(tmp_path):
    """Ctrl-C는 메인 스레드에 온다. 그 뒤로 남은 호출을 하지 않아야 한다."""
    import _thread
    import time as _time
    calls = []

    def interrupted(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            _thread.interrupt_main()
            _time.sleep(0.3)
        return '{"hypertension": 20, "diabetes": 5, "cholesterol": 10, "triglyceride": 12}', 1

    with pytest.raises(KeyboardInterrupt):
        run_llm([(10, 0), (11, 0), (10, 1)], _people(), tmp_path / "c.jsonl", caller=interrupted, workers=1)
    assert len(calls) == 1


def test_llm_paths_are_per_model():
    a, b = cmp.llm_paths("gpt-5.6-luna"), cmp.llm_paths("gpt-5.6-terra")
    assert a["cache"].name == "gpt-5.6-luna.jsonl" and b["cache"].name == "gpt-5.6-terra.jsonl"
    assert b["tag"] == "compare-llm-gpt-5.6-terra" and b["figure"] == "llm_consistency_gpt-5.6-terra.png"


def test_report_links_model_figure(knhanes_like, tmp_path):
    preds = {"hypertension": disease_predictions(knhanes_like, "hypertension", False)}
    rep_ids = [int(i) for i in preds["hypertension"].index[:10]]
    res = build_results(preds, _fake_cache(preds, rep_ids), rep_ids, model="m-x")
    write_report(res, tmp_path / "c.md")
    assert res["llm"]["model"] == "m-x"
    assert "figures/llm_consistency_m-x.png" in (tmp_path / "c.md").read_text(encoding="utf-8")
