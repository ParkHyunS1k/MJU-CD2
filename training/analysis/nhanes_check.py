"""NHANES 2003–04 BIA(BIX_C) + 질환 파일: 라벨 생성, 예측력, KNHANES 모델 외부검증.
파일 출처: github.com/vandomed/nhanesdata (data/*.rda)
"""
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import pyreadr
from sklearn.metrics import roc_auc_score

from common import EXT_DIR, auc_any_direction, cv_auc, knhanes_labels, load_knhanes, logreg

warnings.simplefilter("ignore")


def rda(name: str) -> pd.DataFrame:
    return next(iter(pyreadr.read_r(EXT_DIR / "nhanes" / f"{name}.rda").values()))


parts = [
    rda("bix_c")[["SEQN", "BIAEXSTS", "BIDFIT", "BIDPFAT", "BIDFAT", "BIDFFM", "BIDTBW"]],
    rda("demo_c")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "WTMEC2YR"]],
    rda("bmx_c")[["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"]],
    rda("bpx_c")[["SEQN"] + [f"BPXSY{i}" for i in range(1, 5)] + [f"BPXDI{i}" for i in range(1, 5)]],
    rda("bpq_c")[["SEQN", "BPQ020", "BPQ050A", "BPQ100D"]],
    rda("diq_c")[["SEQN", "DIQ010", "DIQ050", "DIQ070"]],
    rda("l10_c"),
    rda("l10am_c")[["SEQN", "LBXGLU"]],
    rda("l13_c")[["SEQN", "LBXTC", "LBXHDD"]],
    rda("l13am_c")[["SEQN", "LBXTR", "LBDLDL"]],
]
df = reduce(lambda x, y: x.merge(y, on="SEQN", how="left"), parts)
print("BIX 전체", len(df), "| 체지방률 있음", int(df.BIDPFAT.notna().sum()), "| 나이", df.RIDAGEYR.min(), "-", df.RIDAGEYR.max())

a = df[(df.RIDAGEYR >= 20) & df.BIDPFAT.notna()].copy()
print("성인(20+)", len(a), "| BIDFIT(0=정상)", a.BIDFIT.value_counts().to_dict())
print("인종 RIDRETH1", a.RIDRETH1.value_counts().to_dict(), "(1=멕시코계,2=기타히스패닉,3=백인,4=흑인,5=기타)")
a["female"] = (a.RIAGENDR == 2).astype(float)
a["age"] = a.RIDAGEYR
a["sbp"] = a[[f"BPXSY{i}" for i in range(1, 5)]].mean(axis=1)
a["dbp"] = a[[f"BPXDI{i}" for i in range(1, 5)]].replace(0, np.nan).mean(axis=1)  # 이완기 0은 측정 실패

yes = lambda s: s == 1
# KNHANES 라벨과 같은 정의로 맞춘다
L = {
    "고혈압": ((a.sbp >= 140) | (a.dbp >= 90) | yes(a.BPQ050A)).astype(float).where(a.sbp.notna()),
    "당뇨": (yes(a.DIQ010) | yes(a.DIQ050) | yes(a.DIQ070) | (a.LBXGH >= 6.5) | (a.LBXGLU >= 126))
            .astype(float).where(a.LBXGH.notna() | a.DIQ010.notna()),
    "고콜레스테롤": ((a.LBXTC >= 240) | yes(a.BPQ100D)).astype(float).where(a.LBXTC.notna()),
    "고중성지방": (a.LBXTR >= 200).astype(float).where(a.LBXTR.notna()),  # 공복 하위표본만
}
dysl = ((a.LBXTC >= 240) | (a.LBDLDL >= 160) | (a.LBXTR >= 200) | (a.LBXHDD < 40) | yes(a.BPQ100D)
        ).astype(float).where(a.LBXTC.notna())

print("\n라벨: n / 양성 / 유병률 / 가중 유병률(WTMEC2YR)")
for name, y in {**L, "이상지질혈증(복합)": dysl}.items():
    m = y.notna()
    print(f"  {name:12s} {int(m.sum()):5d} {int(y.sum()):4d} {100 * y.mean():5.1f}%"
          f"  {100 * np.average(y[m], weights=a.WTMEC2YR[m]):5.1f}%")

k = load_knhanes()
k = k[k.age <= 49]
print("\n체지방률 중앙값 남/여: NHANES %.1f/%.1f, KNHANES 20–49세 %.1f/%.1f" % (
    *a.groupby("female").BIDPFAT.median(), *k[k.age >= 20].groupby("sex").BIA_PBF.median()))
print("BMI 중앙값 NHANES %.1f / KNHANES 20–49세 %.1f" % (a.BMXBMI.median(), k[k.age >= 20].HE_BMI.median()))

print("\n단일 변수 AUROC (남/여)")
for name, y in L.items():
    cells = []
    for v in ["age", "BMXBMI", "BMXWAIST", "BIDPFAT", "BIDFAT", "BIDFFM"]:
        s = "/".join(f"{auc_any_direction(y[m], a.loc[m, v]):.2f}"
                     for f in (0, 1) for m in [y.notna() & a[v].notna() & (a.female == f)])
        cells.append(f"{v}={s}")
    print(f"  {name:8s}", "  ".join(cells))

BIA = ["BIDPFAT", "BIDFAT", "BIDFFM", "BIDTBW", "BMXWT", "BMXHT"]
SETS = {
    "나이+성별": ["age", "female"],
    "+BMI": ["age", "female", "BMXBMI"],
    "+허리": ["age", "female", "BMXWAIST"],
    "+BMI+허리": ["age", "female", "BMXBMI", "BMXWAIST"],
    "+체지방률": ["age", "female", "BIDPFAT"],
    "+BIA전체": ["age", "female"] + BIA,
    "+BIA전체+허리": ["age", "female"] + BIA + ["BMXWAIST"],
}
allf = sorted({f for s in SETS.values() for f in s})
rows = []
for name, y in L.items():
    m = y.notna() & a[allf].notna().all(axis=1)
    row = {"라벨": name, "n": int(m.sum()), "양성": int(y[m].sum())}
    for sn, fs in SETS.items():
        row[sn] = round(cv_auc(a.loc[m, fs].values, y[m].values), 3)
    rows.append(row)
pd.set_option("display.width", 250)
print("\nNHANES 내부 5-fold CV AUROC")
print(pd.DataFrame(rows).set_index("라벨").to_string())

print("\n외부검증: KNHANES 20–49세 학습 → NHANES 평가 (AUROC, 예측평균 vs 실제)")
k = k[k.age >= 20]
KL = knhanes_labels(k)
M = {  # (KNHANES 열, NHANES 열)
    "+BMI+허리": (["age", "female", "HE_BMI", "HE_wc"], ["age", "female", "BMXBMI", "BMXWAIST"]),
    "+체지방률": (["age", "female", "BIA_PBF"], ["age", "female", "BIDPFAT"]),
    "+체지방률+BMI+허리": (["age", "female", "BIA_PBF", "HE_BMI", "HE_wc"], ["age", "female", "BIDPFAT", "BMXBMI", "BMXWAIST"]),
}
for name, y in L.items():
    cells = []
    for n, (fk, fn) in M.items():
        mk = KL[name].notna() & k[fk].notna().all(axis=1)
        mn = y.notna() & a[fn].notna().all(axis=1)
        p = logreg().fit(k.loc[mk, fk].values, KL[name][mk].values).predict_proba(a.loc[mn, fn].values)[:, 1]
        cells.append(f"{n} {roc_auc_score(y[mn], p):.3f} ({p.mean():.2f} vs {y[mn].mean():.2f})")
    print(f"  {name:8s}", " | ".join(cells))
