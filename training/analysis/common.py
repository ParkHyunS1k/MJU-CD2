"""분석 스크립트 공통: 데이터 경로, KNHANES 로드, 교차검증 AUROC."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXT_DIR = DATA_DIR / "ext"

KNHANES_COLS = [
    "ID", "age", "sex", "HE_wt", "HE_ht", "HE_BMI", "HE_wc", "BIA_ex",
    "BIA_PBF", "BIA_BFM", "BIA_FFM", "BIA_TBW", "BIA_LRA", "BIA_LLA", "BIA_LRL", "BIA_LLL", "BIA_WBPA50",
    "HE_sbp", "HE_dbp", "HE_HP", "HE_glu", "HE_HbA1c", "HE_DM_HbA1c", "HE_chol", "HE_TG", "HE_HCHOL", "HE_HTG",
]


def load_knhanes() -> pd.DataFrame:
    """2022–2024 원자료에서 체성분검사를 받은 19세 이상만. 2024년은 40세 이상만 검사했다."""
    dfs = []
    for yr in (22, 23, 24):
        d = pd.read_sas(DATA_DIR / f"hn{yr}_all.sas7bdat", encoding="latin1")
        d = d[KNHANES_COLS].copy()
        d["yr"] = 2000 + yr
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df.age >= 19) & (df.BIA_ex == 1)].copy()
    df["female"] = (df.sex == 2).astype(float)
    df["ASM"] = df[["BIA_LRA", "BIA_LLA", "BIA_LRL", "BIA_LLL"]].sum(axis=1, min_count=4)
    return df


def knhanes_labels(df: pd.DataFrame) -> dict[str, pd.Series]:
    """재설계용 라벨. HE_HP 4=고혈압(3은 전단계), HE_DM_HbA1c 3=당뇨(진단·치료 포함)."""
    return {
        "고혈압": (df.HE_HP == 4).astype(float).where(df.HE_HP.notna()),
        "당뇨": (df.HE_DM_HbA1c == 3).astype(float).where(df.HE_DM_HbA1c.notna()),
        "고콜레스테롤": df.HE_HCHOL,
        "고중성지방": df.HE_HTG,
    }


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def cv_auc(X, y) -> float:
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    p = cross_val_predict(logreg(), X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def auc_any_direction(y, x) -> float:
    a = roc_auc_score(y, x)
    return max(a, 1 - a)
