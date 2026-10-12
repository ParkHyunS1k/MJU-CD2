"""외부 검증 데이터 → BASE_FEATURES + 라벨. 라벨 정의는 KNHANES(features.build_labels)에 맞춘다.
출처: NHANES 2003–04(github.com/vandomed/nhanesdata), Kaggle Body Performance(국민체력100)."""
from functools import reduce

import numpy as np
import pandas as pd

from training.analysis.common import EXT_DIR


def _rda(name: str) -> pd.DataFrame:
    import pyreadr
    path = EXT_DIR / "nhanes" / f"{name}.rda"
    if not path.exists():
        raise FileNotFoundError(path)
    return next(iter(pyreadr.read_r(path).values()))


def load_nhanes() -> pd.DataFrame:
    sy = [f"BPXSY{i}" for i in range(1, 5)]
    di = [f"BPXDI{i}" for i in range(1, 5)]
    parts = [
        _rda("bix_c")[["SEQN", "BIDPFAT"]],
        _rda("demo_c")[["SEQN", "RIAGENDR", "RIDAGEYR"]],
        _rda("bmx_c")[["SEQN", "BMXBMI"]],
        _rda("bpx_c")[["SEQN"] + sy + di],
        _rda("bpq_c")[["SEQN", "BPQ050A", "BPQ100D"]],
        _rda("diq_c")[["SEQN", "DIQ010", "DIQ050", "DIQ070"]],
        _rda("l10_c")[["SEQN", "LBXGH"]],
        _rda("l10am_c")[["SEQN", "LBXGLU"]],
        _rda("l13_c")[["SEQN", "LBXTC"]],
        _rda("l13am_c")[["SEQN", "LBXTR"]],
    ]
    d = reduce(lambda a, b: a.merge(b, on="SEQN", how="left"), parts)
    d = d[(d.RIDAGEYR >= 20) & d.BIDPFAT.notna() & d.BMXBMI.notna()]
    sbp = d[sy].mean(axis=1)
    dbp = d[di].replace(0, np.nan).mean(axis=1)  # 이완기 0은 측정 실패
    yes = lambda s: s == 1
    return pd.DataFrame({
        "age": d.RIDAGEYR.astype(float),
        "female": (d.RIAGENDR == 2).astype(float),
        "bmi": d.BMXBMI.astype(float),
        "body_fat_percentage": d.BIDPFAT.astype(float),
        "hypertension": ((sbp >= 140) | (dbp >= 90) | yes(d.BPQ050A)).astype(float).where(sbp.notna()),
        "diabetes": (yes(d.DIQ010) | yes(d.DIQ050) | yes(d.DIQ070) | (d.LBXGH >= 6.5) | (d.LBXGLU >= 126))
                    .astype(float).where(d.LBXGH.notna() | d.DIQ010.notna()),
        "cholesterol": ((d.LBXTC >= 240) | yes(d.BPQ100D)).astype(float).where(d.LBXTC.notna()),
        "triglyceride": (d.LBXTR >= 200).astype(float).where(d.LBXTR.notna()),  # 공복 하위표본
    })


def load_kaggle() -> pd.DataFrame:
    path = EXT_DIR / "bodyPerformance.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    k = pd.read_csv(path)
    bad = ((k.systolic <= 0) | (k.diastolic <= 0) | (k.diastolic >= k.systolic)
           | (k.systolic > 250) | (k.diastolic < 30))
    k = k[~bad]
    return pd.DataFrame({
        "age": k.age.astype(float),
        "female": (k.gender == "F").astype(float),
        "bmi": k.weight_kg / (k.height_cm / 100) ** 2,
        "body_fat_percentage": k["body fat_%"].astype(float),
        "hypertension": ((k.systolic >= 140) | (k.diastolic >= 90)).astype(float),
    })
