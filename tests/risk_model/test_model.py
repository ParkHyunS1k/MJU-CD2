import pytest

from training.features import FEATURES, training_rows
from training.model import build_pipeline, feature_groups, fit_bundle


def test_pipeline_steps():
    assert [n for n, _ in build_pipeline(False).steps] == ["scaler", "clf"]
    assert [n for n, _ in build_pipeline(True).steps] == ["spline", "scaler", "clf"]


@pytest.mark.parametrize("spline", [False, True])
def test_fit_bundle_contract(knhanes_like, spline):
    X, y, info = training_rows(knhanes_like, "hypertension")
    b = fit_bundle(X, y, info.age.to_numpy(), "hypertension", spline)
    assert b["version"] == 2 and b["disease_key"] == "hypertension" and b["target"] == "고혈압"
    assert b["features"] == FEATURES
    assert b["feat_kor"]["bmi"] == "BMI" and b["feat_kor"]["body_fat_percentage"] == "체지방률"
    n_cols = b["model"][:-1].transform(X.to_numpy()).shape[1]
    assert len(b["background"]) == n_cols
    cols = sorted(i for idx in b["feature_groups"].values() for i in idx)
    assert cols == list(range(n_cols))
    assert set(b["feature_groups"]) == set(FEATURES)
    assert 0 < b["thresh_medium"] < b["thresh_high"] < 1
    assert b["meta"]["n_train"] == len(y) and b["meta"]["n_pos"] == int(y.sum())
    assert b["meta"]["age_spline"] is spline and b["meta"]["train_age"] == [19, 59]


def test_spline_groups_age_columns(knhanes_like):
    X, y, info = training_rows(knhanes_like, "diabetes")
    pipe = build_pipeline(True).fit(X.to_numpy(), y)
    groups = feature_groups(pipe, FEATURES)
    assert groups["age"] == [0, 1, 2, 3, 4, 5]
    assert groups["female"] == [6]
