#!/usr/bin/env python3
"""
최종 모델 재훈련 (SMM 피처셋, 다중공선성 해소)

확정 구성:
  고혈압       female=LR,  male=LR   (성별 분리)
  고콜레스테롤  all=ET               (통합, sex 포함)
  당뇨         female=LR,  male=ET   (성별 분리)
  고중성지방    female=LR,  male=ET   (성별 분리)

피처: age, BIA_PBF, SMM  (+sex 는 통합 모델만)
sex 인코딩: 0=남성, 1=여성  (앱 기준)
"""
from __future__ import annotations

import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

DATASET_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR  = Path(__file__).resolve().parent.parent / "models"

FEAT_KOR = {
    "age":     "나이(세)",
    "BIA_PBF": "체지방률",
    "SMM":     "골격근량",
    "sex":     "성별(0=남,1=여)",
}

# ── 데이터 로드 ──────────────────────────────────────
def load_data() -> pd.DataFrame:
    need = ["age","sex","BIA_PBF","BIA_LRA","BIA_LLA","BIA_LRL","BIA_LLL",
            "HE_HP","HE_HCHOL","HE_glu","HE_HbA1c","HE_HTG"]
    dfs = []
    for yr in [22, 23, 24]:
        d = pd.read_sas(DATASET_DIR / f"hn{yr}_all.sas7bdat", encoding="latin1")
        dfs.append(d[[c for c in need if c in d.columns]].copy())
    df = pd.concat(dfs, ignore_index=True)

    limb = ["BIA_LRA","BIA_LLA","BIA_LRL","BIA_LLL"]
    df["SMM"] = df[limb].sum(axis=1)
    df.loc[df[limb].isna().any(axis=1), "SMM"] = np.nan

    # sex 재인코딩: NHANES(1=남,2=여) → 앱(0=남,1=여)
    df["sex"] = (df["sex"] == 2).astype(float)
    return df

# ── 모델 팩토리 ──────────────────────────────────────
def lr_pipeline(k_neighbors: int) -> ImbPipeline:
    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
        ("sc",    StandardScaler()),
        ("clf",   LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ])

def et_pipeline(k_neighbors: int) -> ImbPipeline:
    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
        ("clf",   ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        )),
    ])

# ── 임계값 (Youden's J / 양성 75th pct) ────────────
def calc_thresholds(probs: np.ndarray, y: np.ndarray):
    fpr, tpr, ths = roc_curve(y, probs)
    j_idx = np.argmax(tpr - fpr)
    return float(ths[j_idx]), float(np.percentile(probs[y == 1], 75))

# ── 학습 + 저장 ──────────────────────────────────────
def train_one(df, feats, y_series, gender, model_fn, disease_key, kor):
    # 마스크: 타겟 + 피처 모두 유효한 행
    valid = y_series.notna() & df[feats].notna().all(axis=1)
    if gender == "female":
        valid &= df["sex"] == 1   # 재인코딩 후 1=여성
        fname  = f"{disease_key}_female.pkl"
    elif gender == "male":
        valid &= df["sex"] == 0   # 0=남성
        fname  = f"{disease_key}_male.pkl"
    else:
        fname = f"{disease_key}.pkl"

    sub = df[valid]
    y   = y_series[valid].values.astype(float)
    X   = sub[feats].values.astype(float)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"  [{kor} | {gender}]  n={len(y)}, 양성={n_pos}({100*n_pos/len(y):.1f}%)")

    k_neighbors = min(5, n_pos - 1)
    model = model_fn(k_neighbors)
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    tb, th = calc_thresholds(probs, y)

    bundle = {
        "model":         model,
        "features":      feats,
        "feat_kor":      {f: FEAT_KOR[f] for f in feats},
        "target":        kor,
        "thresh_binary": tb,
        "thresh_high":   th,
    }
    joblib.dump(bundle, MODELS_DIR / fname)
    print(f"    → {fname}  (thresh_binary={tb:.3f}, thresh_high={th:.3f})")

# ── 메인 ────────────────────────────────────────────
def main():
    print("데이터 로딩...")
    df = load_data()

    hp_y = (df["HE_HP"] >= 3).astype(float).where(df["HE_HP"].notna())
    ch_y = pd.to_numeric(df["HE_HCHOL"], errors="coerce")
    dm_y = ((df["HE_glu"] >= 126) | (df["HE_HbA1c"] >= 6.5)).astype(float)\
             .where(df[["HE_glu","HE_HbA1c"]].notna().all(axis=1))
    tg_y = pd.to_numeric(df["HE_HTG"], errors="coerce")

    BASE = ["age", "BIA_PBF", "SMM"]
    BASE_SEX = BASE + ["sex"]

    configs = [
        (hp_y, "female", BASE,     lr_pipeline, "hypertension", "고혈압"),
        (hp_y, "male",   BASE,     lr_pipeline, "hypertension", "고혈압"),
        (ch_y, "all",    BASE_SEX, et_pipeline, "cholesterol",  "고콜레스테롤"),
        (dm_y, "female", BASE,     lr_pipeline, "diabetes",     "당뇨"),
        (dm_y, "male",   BASE,     et_pipeline, "diabetes",     "당뇨"),
        (tg_y, "female", BASE,     lr_pipeline, "triglyceride", "고중성지방"),
        (tg_y, "male",   BASE,     et_pipeline, "triglyceride", "고중성지방"),
    ]

    print("\n학습 시작")
    for y_s, gender, feats, pipe_fn, key, kor in configs:
        train_one(df, feats, y_s, gender, pipe_fn, key, kor)

    # 기존 불필요 파일 제거
    old = ["cholesterol_female.pkl", "cholesterol_male.pkl", "triglyceride.pkl"]
    print("\n구 파일 삭제:")
    for name in old:
        p = MODELS_DIR / name
        if p.exists():
            p.unlink()
            print(f"  삭제: {name}")

    print("\n최종 models/ 목록:")
    for f in sorted(MODELS_DIR.glob("*.pkl")):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
