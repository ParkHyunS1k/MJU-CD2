from __future__ import annotations

import warnings
from datetime import date
from functools import lru_cache
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np

_CACHE_DIR = Path(tempfile.gettempdir()) / "healthcare-api-cache"
_MPLCONFIG_DIR = _CACHE_DIR / "matplotlib"
_XDG_CACHE_HOME = _CACHE_DIR / "xdg"
_MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import shap

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

DISEASES = ["hypertension", "cholesterol", "diabetes", "triglyceride"]

# cholesterol은 통합 모델(sex 포함), 나머지는 성별 분리
_GENDER_SPLIT = {"hypertension", "diabetes", "triglyceride"}


@lru_cache(maxsize=None)
def _load_bundle(path: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def _bundle_path(disease: str, sex: str) -> Path:
    if disease in _GENDER_SPLIT:
        fname = f"{disease}_{sex}.pkl"
    else:
        fname = f"{disease}.pkl"
    return MODELS_DIR / fname


def _risk_level(prob: float, bundle: dict) -> str:
    if prob >= bundle["thresh_high"]:
        return "high"
    if prob >= bundle["thresh_binary"]:
        return "medium"
    return "low"


@lru_cache(maxsize=None)
def _shap_explainer(bundle_path: str):
    bundle = _load_bundle(bundle_path)
    model = bundle["model"]

    if hasattr(model, "steps"):
        steps = model.named_steps
        clf = steps["clf"]
        if "sc" in steps:
            # SMOTE + StandardScaler + LR → LinearExplainer
            background = np.zeros((1, len(bundle["features"])))
            return "linear", steps["sc"], shap.LinearExplainer(clf, background)
        else:
            # SMOTE + ET (StandardScaler 없음) → TreeExplainer on clf
            return "tree", None, shap.TreeExplainer(clf)

    return "tree", None, shap.TreeExplainer(model)


def _shap_values(bundle_path: str, X: np.ndarray) -> np.ndarray:
    kind, transformer, explainer = _shap_explainer(bundle_path)
    if kind == "linear":
        X = transformer.transform(X)

    vals = explainer.shap_values(X)
    # list → 양성 클래스(index 1) 선택
    if isinstance(vals, list):
        vals = vals[1]
    # (n_samples, n_features, n_classes) 형태일 경우 양성 클래스 슬라이스
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals[0]  # 단일 샘플 → (n_features,)


def predict_all(inbody: dict, user_profile: dict) -> list[dict]:
    """인바디 수치 + 유저 프로필로 4개 성인병 위험도를 예측한다."""
    sex = user_profile.get("gender")
    if not sex:
        raise ValueError("성별 정보가 없습니다.")

    birth_year = user_profile.get("birth_year")
    age = date.today().year - birth_year if birth_year else None

    raw = {
        "age":     age,
        "sex":     1 if sex == "female" else 0,
        "BIA_PBF": inbody.get("body_fat_percentage"),
        "SMM":     inbody.get("skeletal_muscle_mass"),
    }

    results = []
    for disease in DISEASES:
        path = _bundle_path(disease, sex)
        bundle = _load_bundle(str(path))
        features = bundle["features"]
        feat_kor = bundle["feat_kor"]

        vals = [raw.get(f) for f in features]
        if any(v is None for v in vals):
            continue  # 필수 수치 누락 시 해당 질병 스킵

        X = np.array([vals], dtype=float)
        prob = float(bundle["model"].predict_proba(X)[0][1])
        level = _risk_level(prob, bundle)
        shap_vals = _shap_values(str(path), X)

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
