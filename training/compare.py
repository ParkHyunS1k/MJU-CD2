"""위험 예측 방식 비교: ⓪ 나이+성별, ① 체크리스트, ② 배포 로지스틱, ③ XGBoost, ④ GPT(Codex CLI).
실행: python -m training.compare [--skip-llm] [--limit N]
설계: docs/superpowers/specs/2026-10-13-model-comparison-design.md"""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.metrics import wilson_ci

FH_FOR = {"hypertension": "fh_hypertension", "diabetes": "fh_diabetes",
          "cholesterol": "fh_dyslipidemia", "triglyceride": "fh_dyslipidemia"}


def checklist_score(X: pd.DataFrame, disease_key: str) -> np.ndarray:
    """BMI≥25, 체지방률(남 25·여 30 이상), 해당 질환 가족력, 40세 이상을 각 1점으로 센다."""
    pbf_cut = np.where(X.female == 1, 30.0, 25.0)
    return ((X.bmi >= 25).astype(int) + (X.body_fat_percentage >= pbf_cut).astype(int)
            + (X[FH_FOR[disease_key]] == 1).astype(int) + (X.age >= 40).astype(int)).to_numpy()


def sens_at_spec(y, score, target_spec: float) -> dict:
    """특이도가 target_spec 이상이 되는 가장 낮은 기준값(점수 ≥ 기준 = 양성)과 그 지점의 민감도."""
    neg, pos = score[y == 0], score[y == 1]
    t, spec = np.inf, 1.0
    for cand in np.unique(score):  # 오름차순: 기준값이 오를수록 특이도가 오른다
        s = float((neg < cand).mean())
        if s >= target_spec:
            t, spec = float(cand), s
            break
    k = int((pos >= t).sum())
    return {"threshold": t, "specificity": spec, "sensitivity": k / len(pos), "sensitivity_ci": wilson_ci(k, len(pos))}


def consistency_stats(probs: np.ndarray, threshold: float) -> dict:
    """같은 입력 반복 응답의 사람별 편차. probs는 (사람, 반복) 0–1 배열."""
    if len(probs) == 0:
        return {"n": 0}
    sd = probs.std(axis=1, ddof=1) * 100
    rng = (probs.max(axis=1) - probs.min(axis=1)) * 100
    pos = probs >= threshold
    flip = pos.any(axis=1) & ~pos.all(axis=1)
    return {"n": len(probs), "sd_median": float(np.median(sd)), "sd_p90": float(np.percentile(sd, 90)),
            "range_median": float(np.median(rng)), "range_max": float(rng.max()), "flip_rate": float(flip.mean())}
