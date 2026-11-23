import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from training.metrics import (
    bootstrap_ci, calibration_slope_intercept, grade_table, paired_diff_ci, risk_levels, sens_spec, wilson_ci,
)


def test_wilson_known_value():
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)


def test_calibrated_predictions_have_slope_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.5, 20000)
    y = (rng.random(20000) < p).astype(float)
    slope, intercept = calibration_slope_intercept(y, p)
    assert 0.9 < slope < 1.1 and abs(intercept) < 0.15


def test_bootstrap_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.3).astype(float)
    p = np.clip(0.3 + 0.2 * (y - 0.3) + rng.normal(0, 0.1, 400), 0.01, 0.99)
    est, lo, hi = bootstrap_ci(y, p, roc_auc_score, n_boot=200)
    assert lo <= est <= hi
    assert bootstrap_ci(y, p, roc_auc_score, n_boot=200) == (est, lo, hi)


def test_paired_diff_of_identical_predictions_is_zero():
    rng = np.random.default_rng(2)
    y = (rng.random(300) < 0.4).astype(float)
    p = rng.random(300)
    assert paired_diff_ci(y, p, p, roc_auc_score, n_boot=100) == (0.0, 0.0, 0.0)


def test_risk_levels_boundaries():
    got = risk_levels(np.array([0.05, 0.1, 0.2, 0.3, 0.9]), 0.1, 0.3).tolist()
    assert got == ["low", "medium", "medium", "high", "high"]


def test_grade_table_and_sens_spec():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=float)
    level = np.array(["low", "low", "low", "low", "medium", "medium", "high", "high"], dtype=object)
    rows = {r["level"]: r for r in grade_table(y, level)}
    assert rows["low"]["n"] == 4 and rows["low"]["prevalence"] == 0.25
    assert rows["high"]["prevalence"] == 1.0 and rows["medium"]["share"] == 0.25
    s = sens_spec(y, level)
    assert s["sensitivity"] == 0.75 and s["specificity"] == 0.75 and s["high_ppv"] == 1.0
