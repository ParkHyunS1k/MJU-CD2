import numpy as np
import pandas as pd
import pytest

from training.features import (
    BASE_FEATURES, DISEASES, FEATURES, build_features, build_labels, choose_thresholds, training_rows,
)


def _labels_df(**cols):
    n = len(next(iter(cols.values())))
    base = {c: [np.nan] * n for c in ("HE_HP", "HE_DM_HbA1c", "HE_HCHOL", "HE_HTG")}
    base.update(cols)
    return pd.DataFrame(base)


def test_constants():
    assert list(DISEASES) == ["hypertension", "cholesterol", "diabetes", "triglyceride"]
    assert FEATURES[0] == "age" and len(FEATURES) == 7
    assert BASE_FEATURES == ["age", "female", "bmi", "body_fat_percentage"]


def test_hypertension_label_excludes_prehypertension():
    df = _labels_df(HE_HP=[1, 2, 3, 4, np.nan])
    got = build_labels(df)["hypertension"].tolist()
    assert got[:4] == [0.0, 0.0, 0.0, 1.0] and np.isnan(got[4])


def test_diabetes_label_uses_hba1c_prevalence_code():
    df = _labels_df(HE_DM_HbA1c=[1, 2, 3, np.nan])
    got = build_labels(df)["diabetes"].tolist()
    assert got[:3] == [0.0, 0.0, 1.0] and np.isnan(got[3])


def test_lipid_labels_drop_non_binary_codes():
    df = _labels_df(HE_HCHOL=[0, 1, 9, np.nan], HE_HTG=[1, 0, 8, np.nan])
    lab = build_labels(df)
    assert lab["cholesterol"].tolist()[:2] == [0.0, 1.0] and lab["cholesterol"].isna().tolist()[2:] == [True, True]
    assert lab["triglyceride"].tolist()[:2] == [1.0, 0.0] and lab["triglyceride"].isna().tolist()[2:] == [True, True]


def test_build_features_family_history_and_sex():
    df = pd.DataFrame({
        "age": [30, 40], "sex": [1, 2], "HE_BMI": [22.0, 27.0], "BIA_PBF": [18.0, 33.0],
        "HE_HPfh1": [9, 0], "HE_HPfh2": [1, 0], "HE_HPfh3": [0, 8],
        "HE_DMfh1": [0, 0], "HE_DMfh2": [0, 1], "HE_DMfh3": [9, 0],
        "HE_HLfh1": [9, 9], "HE_HLfh2": [9, 9], "HE_HLfh3": [9, 9],
    })
    X = build_features(df)
    assert list(X.columns) == FEATURES
    assert X.female.tolist() == [0.0, 1.0]
    assert X.fh_hypertension.tolist() == [1.0, 0.0]
    assert X.fh_diabetes.tolist() == [0.0, 1.0]
    assert X.fh_dyslipidemia.tolist() == [0.0, 0.0]


def test_training_rows_filters_age_and_missing(knhanes_like):
    df = knhanes_like.copy()
    df.loc[0, "age"] = 60
    df.loc[1, "HE_BMI"] = np.nan
    df.loc[2, "HE_HP"] = np.nan
    X, y, info = training_rows(df, "hypertension")
    assert {0, 1, 2}.isdisjoint(X.index)
    assert info.age.between(19, 59).all()
    assert len(X) == len(y) == len(info) and list(info.columns) == ["age", "sex", "yr"]
    Xb, _, _ = training_rows(df, "diabetes", features=BASE_FEATURES)
    assert list(Xb.columns) == BASE_FEATURES


def test_choose_thresholds_youden_and_negative_quantile():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.2).astype(float)
    p = np.clip(rng.normal(0.15 + 0.25 * y, 0.08), 0.001, 0.999)
    medium, high = choose_thresholds(y, p)
    cands = np.sort(np.unique(p))[::-1]
    j = [(p[y == 1] >= t).mean() - (p[y == 0] >= t).mean() for t in cands]
    assert medium == pytest.approx(cands[int(np.argmax(j))])
    assert high == pytest.approx(np.quantile(p[y == 0], 0.95))


def test_choose_thresholds_rejects_inverted():
    with pytest.raises(ValueError):
        choose_thresholds(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.1, 0.9, 0.2, 0.95]))


def test_family_history_all_missing_is_nan_and_excluded(knhanes_like):
    """가족력 문항 3개가 모두 비어 있으면 '없음'(0)이 아니라 결측이다. 학습에서 빠진다."""
    df = knhanes_like.copy()
    df.loc[0, ["HE_HPfh1", "HE_HPfh2", "HE_HPfh3"]] = np.nan            # 전부 결측 → NaN
    df.loc[1, ["HE_DMfh1", "HE_DMfh2"]] = np.nan                         # 일부만 결측 → 남은 응답으로 판단
    df.loc[1, "HE_DMfh3"] = 0
    df.loc[2, ["HE_HLfh1", "HE_HLfh2"]] = np.nan
    df.loc[2, "HE_HLfh3"] = 1
    X = build_features(df)
    assert np.isnan(X.loc[0, "fh_hypertension"])
    assert X.loc[1, "fh_diabetes"] == 0.0 and X.loc[2, "fh_dyslipidemia"] == 1.0
    Xt, _, _ = training_rows(df, "diabetes")
    assert 0 not in Xt.index
