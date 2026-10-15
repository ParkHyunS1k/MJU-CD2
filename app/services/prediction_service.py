"""인바디 수치 + 프로필로 성인병 4종의 현재 유병 확률·등급·기여 요인을 계산한다.
모델: training/train.py가 만든 models/{disease}.pkl(version 2).
설계: docs/superpowers/specs/2026-10-12-risk-model-redesign-design.md 6절."""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DISEASES = ["hypertension", "cholesterol", "diabetes", "triglyceride"]
FEATURES = ["age", "female", "bmi", "body_fat_percentage",
            "fh_hypertension", "fh_diabetes", "fh_dyslipidemia"]
FAMILY_HISTORY = FEATURES[4:]
# 요인 목록에는 자기 질환의 가족력만 보여준다. 다른 질환 가족력은 계수 방향이 상식과 어긋날 수 있다.
OWN_FAMILY_HISTORY = {"hypertension": "fh_hypertension", "diabetes": "fh_diabetes",
                      "cholesterol": "fh_dyslipidemia", "triglyceride": "fh_dyslipidemia"}
MIN_AGE = 19
TOP_K = 3
# KNHANES 19–59세 실제 범위(BMI 14.5–48.0, 체지방률 6.7–54.6)에 여유를 둔 값. 벗어나면 OCR 오독으로 본다.
BMI_RANGE = (12.0, 60.0)
PBF_RANGE = (3.0, 70.0)


class ModelBundleError(RuntimeError):
    """학습 코드와 서비스 코드의 모델 형식이 맞지 않는다."""


@lru_cache(maxsize=None)
def _load_bundle(path: str) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("version") != 2:
        raise ModelBundleError(f"{path}: version 2 번들이 아닙니다")
    if bundle.get("features") != FEATURES:
        raise ModelBundleError(f"{path}: 입력 변수가 다릅니다 {bundle.get('features')}")
    return bundle


def load_models() -> list[dict]:
    """있는 모델을 모두 불러온다. 앱 시작 시 호출해 형식 오류와 '모델 없음'을 배포 단계에서 드러낸다."""
    out = []
    for key in DISEASES:
        path = MODELS_DIR / f"{key}.pkl"
        if not path.exists():
            log.warning("모델 파일이 없어 건너뜁니다: %s", path)
            continue
        out.append(_load_bundle(str(path)))
    if not out:
        raise ModelBundleError(f"{MODELS_DIR}에 사용할 모델이 없습니다")
    return out


def _check_range(name: str, value: float, bounds: tuple[float, float]) -> float:
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} {value}이(가) 허용 범위({bounds[0]:g}–{bounds[1]:g})를 벗어났습니다. "
                         "인바디 수치를 확인해 주세요.")
    return value


def _inputs(inbody: dict, user_profile: dict) -> np.ndarray | None:
    gender = user_profile.get("gender")
    if not gender:
        raise ValueError("성별 정보가 없습니다.")
    birth_year = user_profile.get("birth_year")
    if not birth_year:
        raise ValueError("출생연도 정보가 없습니다.")
    age = date.today().year - int(birth_year)
    if age < MIN_AGE:
        raise ValueError("성인(19세 이상)만 위험도를 예측할 수 있습니다.")

    bmi, pbf = inbody.get("bmi"), inbody.get("body_fat_percentage")
    if bmi is None or pbf is None:
        return None
    bmi = _check_range("BMI", float(bmi), BMI_RANGE)
    pbf = _check_range("체지방률", float(pbf), PBF_RANGE)
    family = [1.0 if user_profile.get(k) else 0.0 for k in FAMILY_HISTORY]
    return np.array([[age, 1.0 if gender == "female" else 0.0, bmi, pbf, *family]])


def _contributions(bundle: dict, x: np.ndarray) -> dict[str, float]:
    """로지스틱 회귀의 로그 오즈 기여 = 계수 × (변환값 − 학습 평균). SHAP 값과 같다."""
    pipe = bundle["model"]
    z = pipe[:-1].transform(x)[0]
    c = pipe[-1].coef_[0] * (z - np.asarray(bundle["background"]))
    return {f: float(c[idx].sum()) for f, idx in bundle["feature_groups"].items()}


def _risk_level(prob: float, bundle: dict) -> str:
    if prob >= bundle["thresh_high"]:
        return "high"
    if prob >= bundle["thresh_medium"]:
        return "medium"
    return "low"


def predict_all(inbody: dict, user_profile: dict) -> list[dict]:
    """질환별 {disease, probability, risk_level, top_factors}. 필수 수치가 없으면 빈 리스트."""
    x = _inputs(inbody, user_profile)
    if x is None:
        return []

    answered = {k for k in FAMILY_HISTORY if user_profile.get(k) is not None}
    results = []
    for bundle in load_models():
        prob = float(bundle["model"].predict_proba(x)[0, 1])
        # 성별, 받지 않은 가족력, 다른 질환의 가족력은 설명에서 뺀다(예측에는 쓴다)
        shown = {f for f in FEATURES if f != "female" and f not in FAMILY_HISTORY}
        own = OWN_FAMILY_HISTORY[bundle["disease_key"]]
        if own in answered:
            shown.add(own)
        factors = [
            {"feature_kor": bundle["feat_kor"][f], "value": float(x[0, FEATURES.index(f)]),
             "shap_value": c, "direction": "up" if c > 0 else "down"}
            for f, c in _contributions(bundle, x).items() if f in shown
        ]
        factors.sort(key=lambda d: abs(d["shap_value"]), reverse=True)
        results.append({"disease": bundle["target"], "disease_key": bundle["disease_key"], "probability": prob,
                        "risk_level": _risk_level(prob, bundle), "top_factors": factors[:TOP_K]})
    return results
