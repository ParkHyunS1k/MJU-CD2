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
