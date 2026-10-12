"""KNHANES 원자료 → 학습용 라벨·입력, 등급 기준값. train.py와 evaluate.py가 함께 쓴다.
설계: docs/superpowers/specs/2026-10-12-risk-model-redesign-design.md 3절, 4.3절."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

DISEASES = {
    "hypertension": "고혈압",
    "cholesterol": "고콜레스테롤",
    "diabetes": "당뇨",
    "triglyceride": "고중성지방",
}
FEATURES = ["age", "female", "bmi", "body_fat_percentage",
            "fh_hypertension", "fh_diabetes", "fh_dyslipidemia"]
BASE_FEATURES = FEATURES[:4]  # 가족력 없는 외부 검증용
FEAT_KOR = {
    "age": "나이",
    "female": "성별",
    "bmi": "BMI",                     # trend_service._FEATURE_MAP과 같은 글자
    "body_fat_percentage": "체지방률",  # 〃
    "fh_hypertension": "고혈압 가족력",
    "fh_diabetes": "당뇨 가족력",
    "fh_dyslipidemia": "고지혈증 가족력",
}
TRAIN_AGE = (19, 59)
THRESHOLD_AGE = (25, 44)


def _binary(s: pd.Series) -> pd.Series:
    return s.astype(float).where(s.isin([0, 1]))


def build_labels(df: pd.DataFrame) -> dict[str, pd.Series]:
    """HE_HP 4=고혈압(3은 전단계), HE_DM_HbA1c 3=당뇨(진단·치료 포함)."""
    return {
        "hypertension": (df.HE_HP == 4).astype(float).where(df.HE_HP.notna()),
        "cholesterol": _binary(df.HE_HCHOL),
        "diabetes": (df.HE_DM_HbA1c == 3).astype(float).where(df.HE_DM_HbA1c.notna()),
        "triglyceride": _binary(df.HE_HTG),
    }


def _any_family(df: pd.DataFrame, prefix: str) -> pd.Series:
    """부·모·형제 중 한 명이라도 1이면 1. 모름(9)·비해당(8)은 0."""
    return df[[f"{prefix}{i}" for i in (1, 2, 3)]].eq(1).any(axis=1).astype(float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "age": df.age.astype(float),
        "female": (df.sex == 2).astype(float),
        "bmi": df.HE_BMI.astype(float),
        "body_fat_percentage": df.BIA_PBF.astype(float),
        "fh_hypertension": _any_family(df, "HE_HPfh"),
        "fh_diabetes": _any_family(df, "HE_DMfh"),
        "fh_dyslipidemia": _any_family(df, "HE_HLfh"),
    }, index=df.index)


def training_rows(df, disease_key, features=FEATURES, age=TRAIN_AGE):
    """라벨·입력이 모두 있고 나이가 범위 안인 행. (X, y, info[age, sex, yr])."""
    X = build_features(df)[list(features)]
    y = build_labels(df)[disease_key]
    m = y.notna() & X.notna().all(axis=1) & df.age.between(*age)
    return X[m], y[m].to_numpy(dtype=float), df.loc[m, ["age", "sex", "yr"]]


def choose_thresholds(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """보통 = Youden J 최대 지점, 높음 = 음성 예측의 95번째 백분위."""
    fpr, tpr, th = roc_curve(y, p)
    medium = float(th[np.argmax(tpr - fpr)])
    high = float(np.quantile(p[y == 0], 0.95))
    if high <= medium:
        raise ValueError(f"thresh_high({high:.4f}) <= thresh_medium({medium:.4f})")
    return medium, high
