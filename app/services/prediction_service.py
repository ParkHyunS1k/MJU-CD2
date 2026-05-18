from __future__ import annotations

import warnings
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import shap

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

DISEASES = ["hypertension", "cholesterol", "diabetes", "triglyceride"]

# triglyceride는 성별 통합 모델 하나
_GENDER_SPLIT = {"hypertension", "cholesterol", "diabetes"}


@lru_cache(maxsize=None)
def _load_bundle(path: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def _bundle(disease: str, sex: str) -> dict:
    if disease in _GENDER_SPLIT:
        fname = f"{disease}_{sex}.pkl"
    else:
        fname = f"{disease}.pkl"
    return _load_bundle(str(MODELS_DIR / fname))


def _risk_level(prob: float, bundle: dict) -> str:
    if prob >= bundle["thresh_high"]:
        return "high"
    if prob >= bundle["thresh_binary"]:
        return "medium"
    return "low"


def _shap_values(bundle: dict, X: np.ndarray) -> np.ndarray:
    model = bundle["model"]

    if hasattr(model, "steps"):
        # Pipeline(smote, sc, clf) — smote은 학습 전용이라 predict 시 건너뜀
        # StandardScaler 후 0벡터(=훈련 데이터 평균)를 배경으로 LinearExplainer 적용
        sc = model.named_steps["sc"]
        clf = model.named_steps["clf"]
        X_scaled = sc.transform(X)
        background = np.zeros_like(X_scaled)
        explainer = shap.LinearExplainer(clf, background)
        vals = explainer.shap_values(X_scaled)
    else:
        # GradientBoostingClassifier
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
        if isinstance(vals, list):
            vals = vals[1]  # class=1 (양성)

    return vals[0]  # 단일 샘플


def predict_all(inbody: dict, user_profile: dict) -> list[dict]:
    """인바디 수치 + 유저 프로필로 4개 성인병 위험도를 예측한다."""
    sex = user_profile.get("gender")
    if not sex:
        raise ValueError("성별 정보가 없습니다.")

    birth_year = user_profile.get("birth_year")
    age = date.today().year - birth_year if birth_year else None

    weight = inbody.get("weight")
    fat = inbody.get("body_fat_mass")
    ffm = (weight - fat) if (weight is not None and fat is not None) else None

    raw = {
        "age": age,
        "sex": 1 if sex == "female" else 0,
        "HE_BMI": inbody.get("bmi"),
        "BIA_PBF": inbody.get("body_fat_percentage"),
        "BIA_FFM": ffm,
        "BIA_TBW": inbody.get("body_water"),
        "SMM": inbody.get("skeletal_muscle_mass"),
    }

    results = []
    for disease in DISEASES:
        bundle = _bundle(disease, sex)
        features = bundle["features"]
        feat_kor = bundle["feat_kor"]

        vals = [raw.get(f) for f in features]
        if any(v is None for v in vals):
            continue  # 필수 수치 누락 시 해당 질병 스킵

        X = np.array([vals], dtype=float)
        prob = float(bundle["model"].predict_proba(X)[0][1])
        level = _risk_level(prob, bundle)
        shap_vals = _shap_values(bundle, X)

        top_factors = []
        for i, feat in enumerate(features):
            if feat == "sex":
                continue
            top_factors.append({
                "feature_kor": feat_kor[feat],
                "value": float(vals[i]),
                "shap_value": float(shap_vals[i]),
                "direction": "up" if shap_vals[i] > 0 else "down",
            })
        top_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        results.append({
            "disease": bundle["target"],
            "probability": prob,
            "risk_level": level,
            "top_factors": top_factors[:3],
        })

    return results
