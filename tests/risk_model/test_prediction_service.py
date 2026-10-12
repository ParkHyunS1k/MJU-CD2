from datetime import date

import joblib
import numpy as np
import pytest

import app.services.prediction_service as ps
from training.features import DISEASES, training_rows
from training.model import fit_bundle

AGE35 = date.today().year - 35
PROFILE = {"gender": "male", "birth_year": AGE35}
INBODY = {"bmi": 27.0, "body_fat_percentage": 28.0}


def _write_bundles(df, directory, spline=False):
    for key in DISEASES:
        X, y, info = training_rows(df, key)
        joblib.dump(fit_bundle(X, y, info.age.to_numpy(), key, spline), directory / f"{key}.pkl")


@pytest.fixture
def models_dir(knhanes_like, tmp_path, monkeypatch):
    _write_bundles(knhanes_like, tmp_path)
    monkeypatch.setattr(ps, "MODELS_DIR", tmp_path)
    ps._load_bundle.cache_clear()
    yield tmp_path
    ps._load_bundle.cache_clear()


def test_output_shape(models_dir):
    res = ps.predict_all(INBODY, PROFILE)
    assert [r["disease"] for r in res] == ["고혈압", "고콜레스테롤", "당뇨", "고중성지방"]
    for r in res:
        assert set(r) == {"disease", "probability", "risk_level", "top_factors"}
        assert 0 < r["probability"] < 1 and r["risk_level"] in {"low", "medium", "high"}
        assert 1 <= len(r["top_factors"]) <= 3
        for f in r["top_factors"]:
            assert set(f) == {"feature_kor", "value", "shap_value", "direction"}
            assert f["feature_kor"] != "성별"
            assert f["direction"] == ("up" if f["shap_value"] > 0 else "down")


def test_risk_level_matches_thresholds(models_dir):
    for r in ps.predict_all(INBODY, PROFILE):
        key = next(k for k, v in DISEASES.items() if v == r["disease"])
        b = joblib.load(models_dir / f"{key}.pkl")
        p = r["probability"]
        expected = "high" if p >= b["thresh_high"] else "medium" if p >= b["thresh_medium"] else "low"
        assert r["risk_level"] == expected


@pytest.mark.parametrize("spline", [False, True])
def test_contributions_sum_to_logit_difference(knhanes_like, tmp_path, spline):
    X, y, info = training_rows(knhanes_like, "diabetes")
    b = fit_bundle(X, y, info.age.to_numpy(), "diabetes", spline)
    x = np.array([[35.0, 1.0, 27.0, 31.0, 1.0, 0.0, 0.0]])
    contrib = ps._contributions(b, x)
    assert set(contrib) == set(b["features"])
    clf = b["model"][-1]
    p = b["model"].predict_proba(x)[0, 1]
    base_logit = clf.intercept_[0] + clf.coef_[0] @ np.asarray(b["background"])
    assert sum(contrib.values()) == pytest.approx(np.log(p / (1 - p)) - base_logit, abs=1e-9)


def test_family_history_none_or_missing_is_zero(models_dir):
    a = ps.predict_all(INBODY, PROFILE)
    b = ps.predict_all(INBODY, {**PROFILE, "fh_hypertension": None, "fh_diabetes": None, "fh_dyslipidemia": None})
    c = ps.predict_all(INBODY, {**PROFILE, "fh_diabetes": True})
    assert [r["probability"] for r in a] == [r["probability"] for r in b]
    assert [r["probability"] for r in a] != [r["probability"] for r in c]


def test_missing_bmi_returns_empty(models_dir):
    assert ps.predict_all({"body_fat_percentage": 28.0}, PROFILE) == []
    assert ps.predict_all({"bmi": 27.0, "body_fat_percentage": None}, PROFILE) == []


@pytest.mark.parametrize("profile, msg", [
    ({"birth_year": AGE35}, "성별"),
    ({"gender": "male"}, "출생연도"),
    ({"gender": "female", "birth_year": date.today().year - 18}, "성인"),
])
def test_invalid_profile_raises(models_dir, profile, msg):
    with pytest.raises(ValueError, match=msg):
        ps.predict_all(INBODY, profile)


@pytest.mark.parametrize("inbody", [
    {"bmi": 240.0, "body_fat_percentage": 28.0},   # 24.0 오독
    {"bmi": 5.0, "body_fat_percentage": 28.0},
    {"bmi": 27.0, "body_fat_percentage": 85.0},
    {"bmi": 27.0, "body_fat_percentage": 1.0},
])
def test_out_of_range_inputs_raise(models_dir, inbody):
    with pytest.raises(ValueError, match="범위"):
        ps.predict_all(inbody, PROFILE)


def test_no_models_raises(models_dir):
    for p in models_dir.glob("*.pkl"):
        p.unlink()
    with pytest.raises(ps.ModelBundleError):
        ps.load_models()
    with pytest.raises(ps.ModelBundleError):
        ps.predict_all(INBODY, PROFILE)


def test_load_models_returns_available(models_dir):
    assert [b["disease_key"] for b in ps.load_models()] == list(DISEASES)


def test_extreme_inputs_predict_finite(models_dir):
    res = ps.predict_all({"bmi": 45.0, "body_fat_percentage": 50.0},
                         {"gender": "female", "birth_year": date.today().year - 80})
    assert len(res) == 4 and all(0 < r["probability"] < 1 for r in res)


def test_integer_inputs(models_dir):
    a = ps.predict_all({"bmi": 24, "body_fat_percentage": 20}, PROFILE)
    b = ps.predict_all({"bmi": 24.0, "body_fat_percentage": 20.0}, PROFILE)
    assert [r["probability"] for r in a] == [r["probability"] for r in b]


def test_missing_model_file_skips_disease(models_dir):
    (models_dir / "cholesterol.pkl").unlink()
    res = ps.predict_all(INBODY, PROFILE)
    assert [r["disease"] for r in res] == ["고혈압", "당뇨", "고중성지방"]


def test_old_gender_files_are_ignored(models_dir):
    joblib.dump({"not": "a bundle"}, models_dir / "hypertension_male.pkl")
    assert len(ps.predict_all(INBODY, PROFILE)) == 4


def test_bad_version_raises(models_dir):
    b = joblib.load(models_dir / "hypertension.pkl")
    joblib.dump({**b, "version": 1}, models_dir / "hypertension.pkl")
    with pytest.raises(ps.ModelBundleError):
        ps.predict_all(INBODY, PROFILE)


def test_feature_mismatch_raises(models_dir):
    b = joblib.load(models_dir / "diabetes.pkl")
    joblib.dump({**b, "features": b["features"][:4]}, models_dir / "diabetes.pkl")
    with pytest.raises(ps.ModelBundleError):
        ps.predict_all(INBODY, PROFILE)


def test_real_models_if_present():
    paths = [ps.MODELS_DIR / f"{k}.pkl" for k in DISEASES]
    present = [p for p in paths if p.exists() and joblib.load(p).get("version") == 2]
    if not present:
        pytest.skip("v2 모델 파일이 아직 없음")
    ps._load_bundle.cache_clear()
    res = ps.predict_all(INBODY, {**PROFILE, "fh_hypertension": True})
    assert len(res) == len(present)


FH_LABELS = {"고혈압 가족력", "당뇨 가족력", "고지혈증 가족력"}
OWN_FH = {"고혈압": "고혈압 가족력", "당뇨": "당뇨 가족력",
          "고콜레스테롤": "고지혈증 가족력", "고중성지방": "고지혈증 가족력"}


def test_unanswered_family_history_not_in_factors(models_dir):
    """가족력을 받지 않은 사용자에게 가족력을 위험 요인으로 보여주지 않는다."""
    for r in ps.predict_all(INBODY, PROFILE):
        assert not {f["feature_kor"] for f in r["top_factors"]} & FH_LABELS


def test_only_own_disease_family_history_in_factors(knhanes_like, tmp_path, monkeypatch):
    """질환별로 자기 질환의 가족력만 요인으로 보여준다(다른 질환 가족력은 계수 방향이 상식과 어긋날 수 있다)."""
    _write_bundles(knhanes_like, tmp_path)
    for p in tmp_path.glob("*.pkl"):  # 가족력 계수를 키워 상위 3개에 들게 만든다
        b = joblib.load(p)
        b["model"][-1].coef_[0][4:] = [3.0, -3.0, 3.0]
        joblib.dump(b, p)
    monkeypatch.setattr(ps, "MODELS_DIR", tmp_path)
    ps._load_bundle.cache_clear()
    profile = {**PROFILE, "fh_hypertension": True, "fh_diabetes": True, "fh_dyslipidemia": True}
    for r in ps.predict_all(INBODY, profile):
        shown = {f["feature_kor"] for f in r["top_factors"]} & FH_LABELS
        assert shown == {OWN_FH[r["disease"]]}
    ps._load_bundle.cache_clear()
