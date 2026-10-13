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
    assert missing == 2 and first not in out.index and second not in out.index
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
