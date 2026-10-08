"""KNHANES: 체성분(BIA) 변수가 BMI·허리둘레보다 질환을 더 잘 가려내는지 비교."""
import warnings

import pandas as pd

from common import auc_any_direction, cv_auc, knhanes_labels, load_knhanes

warnings.simplefilter("ignore")

a = load_knhanes()
Y = knhanes_labels(a)
# 현행 retrain_final.py 라벨과 비교용
Y["고혈압_현행(HE_HP>=3)"] = (a.HE_HP >= 3).astype(float).where(a.HE_HP.notna())
Y["당뇨_현행(검사치만)"] = ((a.HE_glu >= 126) | (a.HE_HbA1c >= 6.5)).astype(float).where(
    a[["HE_glu", "HE_HbA1c"]].notna().all(axis=1))

BIA = ["BIA_PBF", "BIA_BFM", "BIA_FFM", "ASM", "BIA_TBW", "BIA_WBPA50", "HE_wt", "HE_ht"]
SETS = {
    "나이+성별": ["age", "female"],
    "+BMI": ["age", "female", "HE_BMI"],
    "+허리둘레": ["age", "female", "HE_wc"],
    "+BMI+허리": ["age", "female", "HE_BMI", "HE_wc"],
    "+체지방률+ASM(현행)": ["age", "female", "BIA_PBF", "ASM"],
    "+BIA전체": ["age", "female"] + BIA,
    "+BIA전체+허리": ["age", "female"] + BIA + ["HE_wc"],
}
allf = sorted({f for s in SETS.values() for f in s})

rows = []
for name, y in Y.items():
    m = y.notna() & a[allf].notna().all(axis=1)  # 모든 세트에 같은 표본
    row = {"라벨": name, "n": int(m.sum()), "유병률%": round(100 * y[m].mean(), 1)}
    for sn, fs in SETS.items():
        row[sn] = round(cv_auc(a.loc[m, fs].values, y[m].values), 3)
    rows.append(row)
pd.set_option("display.width", 250)
print("5-fold CV AUROC (로지스틱 회귀)")
print(pd.DataFrame(rows).set_index("라벨").to_string())

print("\n단일 변수 AUROC (남/여)")
out = []
for name, y in knhanes_labels(a).items():
    row = {"라벨": name}
    for v in ["age", "HE_BMI", "HE_wc", "BIA_PBF", "BIA_BFM", "ASM", "BIA_WBPA50"]:
        row[v] = "/".join(
            f"{auc_any_direction(y[m], a.loc[m, v]):.2f}"
            for sx in (1, 2)
            for m in [y.notna() & a[v].notna() & (a.sex == sx)]
        )
    out.append(row)
print(pd.DataFrame(out).set_index("라벨").to_string())
