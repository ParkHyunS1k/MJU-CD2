"""Kaggle Body Performance(국민체력100): 품질, 고혈압 라벨 잡음, KNHANES 모델 외부검증."""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import EXT_DIR, auc_any_direction, cv_auc, load_knhanes, logreg

warnings.simplefilter("ignore")
AGE_BINS, AGE_LABELS = [19, 29, 39, 49, 59, 70], ["20대", "30대", "40대", "50대", "60대"]

k = pd.read_csv(EXT_DIR / "bodyPerformance.csv")
print("shape", k.shape, "결측", int(k.isna().sum().sum()), "중복", int(k.duplicated().sum()))
print(k.describe().T[["mean", "min", "50%", "max"]].round(1).to_string())
print("gender", k.gender.value_counts().to_dict(), "class", k["class"].value_counts().to_dict())
bad = (k.systolic <= 0) | (k.diastolic <= 0) | (k.diastolic >= k.systolic) | (k.systolic > 250) | (k.diastolic < 30)
print("혈압 이상치", int(bad.sum()))
print(k[bad][["age", "systolic", "diastolic"]].to_string())
k = k[~bad].copy()
k["female"] = (k.gender == "F").astype(float)
k["BMI"] = k.weight_kg / (k.height_cm / 100) ** 2
k["htn"] = ((k.systolic >= 140) | (k.diastolic >= 90)).astype(float)
k["ag"] = pd.cut(k.age, AGE_BINS, labels=AGE_LABELS)

# KNHANES도 같은 연령대·같은 정의(측정혈압만)로 맞춘다
d = load_knhanes()
d = d[(d.age >= 21) & (d.age <= 64)].copy()
d["gender"] = np.where(d.sex == 2, "F", "M")
d["ag"] = pd.cut(d.age, AGE_BINS, labels=AGE_LABELS)
d["htn_m"] = ((d.HE_sbp >= 140) | (d.HE_dbp >= 90)).astype(float).where(d.HE_sbp.notna())

print("\n고혈압(측정 ≥140/90) %  Kaggle")
print((k.pivot_table(index="ag", columns="gender", values="htn", aggfunc="mean") * 100).round(1).to_string())
print("같은 정의 KNHANES 21–64세")
print((d.pivot_table(index="ag", columns="gender", values="htn_m", aggfunc="mean") * 100).round(1).to_string())

print("\n체지방률 중앙값")
t = pd.concat({"Kaggle": k.groupby(["ag", "gender"])["body fat_%"].median(),
               "KNHANES": d.groupby(["ag", "gender"])["BIA_PBF"].median()}, axis=1).round(1)
t["차이"] = (t.Kaggle - t.KNHANES).round(1)
print(t.to_string())
print("BMI 중앙값 Kaggle %.1f / KNHANES %.1f" % (k.BMI.median(), d.HE_BMI.median()))

print("\n단일 변수 AUROC (남/여)")
for v in ["age", "BMI", "body fat_%", "weight_kg"]:
    print(f"  {v:12s}", "/".join(f"{auc_any_direction(k.htn[k.gender == g], k.loc[k.gender == g, v]):.2f}" for g in "MF"))

SETS = {  # (Kaggle 열, KNHANES 열)
    "나이+성별": (["age", "female"], ["age", "female"]),
    "+BMI": (["age", "female", "BMI"], ["age", "female", "HE_BMI"]),
    "+체지방률": (["age", "female", "body fat_%"], ["age", "female", "BIA_PBF"]),
    "+BMI+체지방률": (["age", "female", "BMI", "body fat_%"], ["age", "female", "HE_BMI", "BIA_PBF"]),
}
print("\nKaggle 내부 5-fold CV AUROC")
for n, (fk, _) in SETS.items():
    print(f"  {n:14s} {cv_auc(k[fk].values, k.htn.values):.3f}")
print("\n외부검증: KNHANES(21–64세, 측정혈압 라벨) 학습 → Kaggle 평가")
for n, (fk, fd) in SETS.items():
    m = d.htn_m.notna() & d[fd].notna().all(axis=1)
    p = logreg().fit(d.loc[m, fd].values, d.htn_m[m].values).predict_proba(k[fk].values)[:, 1]
    print(f"  {n:14s} AUROC {roc_auc_score(k.htn, p):.3f}  예측평균 {p.mean():.3f} vs 실제 {k.htn.mean():.3f}")
