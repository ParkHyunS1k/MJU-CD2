from training.evaluate import (
    check_criteria, evaluate_disease, nested_predictions, temporal, write_report,
)
from training.features import training_rows


def test_nested_predictions_cover_all_rows(knhanes_like):
    X, y, info = training_rows(knhanes_like, "hypertension")
    p, level = nested_predictions(X, y, info.age.to_numpy(), "hypertension", False)
    assert p.shape == y.shape and ((p > 0) & (p < 1)).all()
    assert set(level) <= {"low", "medium", "high"}


def _main(mean_pred, prev_ci, slope, prevs):
    return {"mean_pred": mean_pred, "prevalence_ci": prev_ci, "cal_slope": slope,
            "grades": [{"level": lv, "prevalence": pv} for lv, pv in zip(("low", "medium", "high"), prevs)]}


def test_check_criteria():
    ok = check_criteria(_main(0.10, (0.08, 0.12), 1.0, (0.02, 0.1, 0.3)), (0.01, -0.01, 0.03))
    assert ok == {"calibration": True, "contribution": True, "grades": True}
    bad = check_criteria(_main(0.20, (0.08, 0.12), 1.3, (0.02, 0.3, 0.1)), (-0.01, -0.03, 0.01))
    assert bad == {"calibration": False, "contribution": False, "grades": False}
    missing = check_criteria(_main(0.10, (0.08, 0.12), 1.0, (0.02, 0.1, None)), (0.01, 0.0, 0.02))
    assert missing["grades"] is False


def test_evaluate_disease_and_report(knhanes_like, tmp_path):
    r = evaluate_disease(knhanes_like, "hypertension", old_dir=None, fig_dir=tmp_path / "figures")
    assert set(r["main"]) == {"linear", "spline"} and isinstance(r["spline_adopt"], bool)
    assert set(r["compare"]) == {"spline", "rf", "xgb"}
    assert "old_model" not in r
    assert (tmp_path / "figures" / "calibration_hypertension.png").exists()
    r["temporal"] = temporal(knhanes_like, "hypertension", r["spline_adopt"])
    results = {"date": "2026-10-12", "diseases": {"hypertension": r}, "external": {"skipped": "test"}}
    out = tmp_path / "eval.md"
    write_report(results, out)
    text = out.read_text(encoding="utf-8")
    assert "## 요약" in text and "고혈압" in text and "건너뜀: test" in text
