#!/usr/bin/env python3
"""
성인병 위험도 ML 재훈련 스크립트

피처셋 : SMM 버전 / TBW 버전
모델   : LR, RF, GBM, ExtraTrees, XGBoost, LightGBM
LUPI   : 교사(LightGBM + 특권피처) → 학생(XGBoost 지식증류) OOF 기반
평가   : 5-fold 층화 CV → AUROC / PR-AUC / Brier Score
"""

from __future__ import annotations

import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────
DATASET_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR  = Path(__file__).resolve().parent / "output"
MODEL_DIR   = OUTPUT_DIR / "models_new"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 피처 설정
# ──────────────────────────────────────────────
# 성별 분리 모델 (sex 제외 — 이미 분리하므로)
BASE_FEATS_SMM = ["age", "BIA_PBF", "SMM"]
BASE_FEATS_TBW = ["age", "BIA_PBF", "BIA_TBW"]

FEATURE_SETS = {
    "smm": BASE_FEATS_SMM,
    "tbw": BASE_FEATS_TBW,
}
# 통합(중성지방) 모델은 sex 추가
SEX_FEAT = "sex"

# 특권 피처 후보 (타겟 관련 컬럼은 질병별로 제외)
ALL_PRIVILEGED = [
    "HE_chol", "HE_TG", "HE_glu", "HE_HbA1c",
    "HE_HDL_st2", "HE_LDL_drct", "HE_sbp", "HE_dbp",
]

# ──────────────────────────────────────────────
# 타겟 정의
# ──────────────────────────────────────────────
TARGETS: dict[str, dict] = {
    "hypertension": {
        "kor": "고혈압",
        # HE_HP: 1=정상, 2=주의, 3=고혈압1단계, 4=고혈압2단계
        "build": lambda df: (df["HE_HP"] >= 3).astype(int),
        "gender_split": True,
        # 특권 피처에서 타겟 직접 관련 컬럼(혈압) 제외
        "excl_priv": ["HE_sbp", "HE_dbp"],
        "need_cols": ["HE_HP"],
    },
    "cholesterol": {
        "kor": "고콜레스테롤",
        "build": lambda df: pd.to_numeric(df["HE_HCHOL"], errors="coerce").astype("Int64").astype(float),
        "gender_split": True,
        "excl_priv": ["HE_chol"],
        "need_cols": ["HE_HCHOL"],
    },
    "diabetes": {
        "kor": "당뇨",
        "build": lambda df: (
            (df["HE_glu"] >= 126) | (df["HE_HbA1c"] >= 6.5)
        ).astype(int),
        "gender_split": True,
        "excl_priv": ["HE_glu", "HE_HbA1c"],
        "need_cols": ["HE_glu", "HE_HbA1c"],
    },
    "triglyceride": {
        "kor": "고중성지방",
        "build": lambda df: pd.to_numeric(df["HE_HTG"], errors="coerce").astype("Int64").astype(float),
        "gender_split": False,
        "excl_priv": ["HE_TG"],
        "need_cols": ["HE_HTG"],
    },
}

# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    load_cols = [
        "age", "sex", "BIA_PBF", "BIA_TBW",
        "BIA_LRA", "BIA_LLA", "BIA_LRL", "BIA_LLL",
        "HE_HP", "HE_chol", "HE_TG", "HE_glu", "HE_HbA1c",
        "HE_HDL_st2", "HE_LDL_drct", "HE_sbp", "HE_dbp",
        "HE_HCHOL", "HE_HTG",
    ]
    dfs = []
    for year in [22, 23, 24]:
        path = DATASET_DIR / f"hn{year}_all.sas7bdat"
        df = pd.read_sas(path, encoding="latin1")
        cols = [c for c in load_cols if c in df.columns]
        dfs.append(df[cols].copy())

    df = pd.concat(dfs, ignore_index=True)

    limb = ["BIA_LRA", "BIA_LLA", "BIA_LRL", "BIA_LLL"]
    df["SMM"] = df[limb].sum(axis=1)
    df.loc[df[limb].isna().any(axis=1), "SMM"] = np.nan

    return df

# ──────────────────────────────────────────────
# sklearn 파이프라인 (SMOTE + 모델)
# ──────────────────────────────────────────────
def make_sklearn_pipelines() -> dict:
    smote = SMOTE(random_state=42, k_neighbors=5)
    return {
        "lr": ImbPipeline([
            ("smote", smote),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ]),
        "rf": ImbPipeline([
            ("smote", smote),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=20,
                random_state=42, n_jobs=-1,
            )),
        ]),
        "gbm": ImbPipeline([
            ("smote", smote),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=4, random_state=42,
            )),
        ]),
        "et": ImbPipeline([
            ("smote", smote),
            ("clf", ExtraTreesClassifier(
                n_estimators=300, min_samples_leaf=20,
                random_state=42, n_jobs=-1,
            )),
        ]),
    }

# ──────────────────────────────────────────────
# XGBoost / LightGBM 헬퍼
# ──────────────────────────────────────────────
_XGB_BASE_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=42,
    verbosity=0,
)

_LGB_BASE_PARAMS = dict(
    objective="binary",
    metric="average_precision",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=-1,
    seed=42,
)


def _train_xgb_native(X_tr, y_tr, X_val, y_val, extra_params=None):
    params = {**_XGB_BASE_PARAMS, **(extra_params or {})}
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)
    model  = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return model, model.predict(xgb.DMatrix(X_val))


def _train_lgb_native(X_tr, y_tr, X_val, y_val, extra_params=None):
    params  = {**_LGB_BASE_PARAMS, **(extra_params or {})}
    ds_tr   = lgb.Dataset(X_tr, label=y_tr)
    ds_val  = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    cbs     = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
    model   = lgb.train(
        params, ds_tr, num_boost_round=500,
        valid_sets=[ds_val], callbacks=cbs,
    )
    return model, model.predict(X_val)

# ──────────────────────────────────────────────
# LUPI: 교사-학생 지식 증류
# ──────────────────────────────────────────────
def _oof_teacher_probs(X_full, y, n_splits=3, spw=1.0) -> np.ndarray:
    """훈련셋 내부에서 3-fold OOF 교사 예측 (소프트 레이블)."""
    oof = np.zeros(len(y))
    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr2, val2 in inner.split(X_full, y):
        m, _ = _train_lgb_native(
            X_full[tr2], y[tr2],
            X_full[val2], y[val2],
            extra_params={"is_unbalance": True},
        )
        oof[val2] = m.predict(X_full[val2])
    return oof


def train_lupi_xgb(
    X_reg_tr, y_tr, teacher_probs_tr,
    X_reg_val, y_val,
    alpha: float = 0.5,
    spw: float = 1.0,
):
    """
    alpha * CE(y_hard) + (1-alpha) * CE(y_teacher_soft) 를 최소화.
    gradient/hessian 을 직접 정의해 XGBoost custom objective 로 전달.
    """
    t = teacher_probs_tr  # closure

    def distillation_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        p   = 1.0 / (1.0 + np.exp(-y_pred))
        w   = np.where(y_true == 1, spw, 1.0)
        g   = w * (alpha * (p - y_true) + (1 - alpha) * (p - t))
        h   = w * p * (1 - p)
        return g, h

    dtrain = xgb.DMatrix(X_reg_tr, label=y_tr)
    dval   = xgb.DMatrix(X_reg_val, label=y_val)

    params = dict(
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=42,
        verbosity=0,
        eval_metric="logloss",
    )
    model = xgb.train(
        params, dtrain, num_boost_round=300,
        obj=distillation_obj,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    # custom objective → raw margin 반환, sigmoid 변환 필요
    raw = model.predict(xgb.DMatrix(X_reg_val), output_margin=True)
    prob = 1.0 / (1.0 + np.exp(-raw))
    return model, prob

# ──────────────────────────────────────────────
# 평가 지표
# ──────────────────────────────────────────────
def metrics(y_true, y_prob) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"auroc": np.nan, "pr_auc": np.nan, "brier": np.nan}
    return {
        "auroc":  roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier":  brier_score_loss(y_true, y_prob),
    }

# ──────────────────────────────────────────────
# 5-fold 실험
# ──────────────────────────────────────────────
def run_cv(X_reg, y, X_full, spw: float) -> dict[str, list]:
    """
    X_reg  : 정규 피처만
    X_full : 정규 + 특권 피처
    """
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows: dict[str, list] = {}

    def _add(name, m_dict):
        rows.setdefault(name, []).append(m_dict)

    for fold, (tr, val) in enumerate(skf.split(X_reg, y), 1):
        Xr_tr, Xr_val = X_reg[tr],  X_reg[val]
        Xf_tr, Xf_val = X_full[tr], X_full[val]
        y_tr,  y_val  = y[tr],      y[val]

        # ── sklearn 파이프라인 ──
        for name, pipe in make_sklearn_pipelines().items():
            pipe.fit(Xr_tr, y_tr)
            _add(name, metrics(y_val, pipe.predict_proba(Xr_val)[:, 1]))

        # ── XGBoost ──
        try:
            _, p = _train_xgb_native(Xr_tr, y_tr, Xr_val, y_val,
                                     {"scale_pos_weight": spw})
            _add("xgb", metrics(y_val, p))
        except Exception as e:
            print(f"    [fold {fold}] XGB 오류: {e}")

        # ── LightGBM ──
        try:
            _, p = _train_lgb_native(Xr_tr, y_tr, Xr_val, y_val,
                                     {"is_unbalance": True})
            _add("lgbm", metrics(y_val, p))
        except Exception as e:
            print(f"    [fold {fold}] LGBM 오류: {e}")

        # ── LUPI (교사 OOF → 학생 지식증류) ──
        try:
            t_probs = _oof_teacher_probs(Xf_tr, y_tr, n_splits=3, spw=spw)
            _, p = train_lupi_xgb(Xr_tr, y_tr, t_probs,
                                  Xr_val, y_val, alpha=0.5, spw=spw)
            _add("lupi_xgb", metrics(y_val, p))
        except Exception as e:
            print(f"    [fold {fold}] LUPI 오류: {e}")

    # fold 평균
    summary: dict[str, dict] = {}
    for name, fold_list in rows.items():
        keys = fold_list[0].keys()
        summary[name] = {
            k: float(np.mean([f[k] for f in fold_list])) for k in keys
        }
        summary[name]["auroc_std"] = float(
            np.std([f["auroc"] for f in fold_list])
        )
    return summary

# ──────────────────────────────────────────────
# 최적 모델 재훈련 후 번들 저장
# ──────────────────────────────────────────────
FEAT_KOR = {
    "age":     "나이(세)",
    "sex":     "성별(0=남,1=여)",
    "BIA_PBF": "체지방률",
    "SMM":     "골격근량",
    "BIA_TBW": "체수분",
}


def _retrain_and_save(best_info: dict, X_reg, y, X_full, spw: float,
                      feat_ver: str, disease: str, gender: str, kor: str,
                      reg_feats: list[str]):
    """결과 테이블의 최고 모델을 전체 데이터로 재훈련하고 pkl 번들 저장."""
    best_model_name = best_info["model"]
    save_name = f"{disease}_{feat_ver}_{gender}"

    model_obj = None

    if best_model_name in ("lr", "rf", "gbm", "et"):
        pipe = make_sklearn_pipelines()[best_model_name]
        pipe.fit(X_reg, y)
        model_obj = pipe

    elif best_model_name == "xgb":
        # 전체 데이터로 훈련 (val 없이 고정 300 라운드)
        dtrain = xgb.DMatrix(X_reg, label=y)
        params = {**_XGB_BASE_PARAMS, "scale_pos_weight": spw}
        params.pop("eval_metric", None)
        model_obj = xgb.train(params, dtrain, num_boost_round=300)

    elif best_model_name == "lgbm":
        ds = lgb.Dataset(X_reg, label=y)
        params = {**_LGB_BASE_PARAMS, "is_unbalance": True}
        params.pop("metric", None)
        model_obj = lgb.train(params, ds, num_boost_round=300)

    elif best_model_name == "lupi_xgb":
        t_probs = _oof_teacher_probs(X_full, y, n_splits=3, spw=spw)
        dtrain  = xgb.DMatrix(X_reg, label=y)
        t       = t_probs

        def obj(y_pred, dtrain_inner):
            y_true = dtrain_inner.get_label()
            p = 1.0 / (1.0 + np.exp(-y_pred))
            w = np.where(y_true == 1, spw, 1.0)
            return (w * (0.5*(p-y_true) + 0.5*(p-t)),
                    w * p * (1-p))

        params = dict(learning_rate=0.05, max_depth=5,
                      subsample=0.8, colsample_bytree=0.8,
                      seed=42, verbosity=0)
        model_obj = xgb.train(params, dtrain, num_boost_round=300, obj=obj)

    if model_obj is None:
        return

    # 임계값: Youden's J 기반 (val이 없으므로 전체 데이터로 근사)
    if hasattr(model_obj, "predict_proba"):
        probs = model_obj.predict_proba(X_reg)[:, 1]
    elif isinstance(model_obj, xgb.Booster):
        raw = model_obj.predict(xgb.DMatrix(X_reg), output_margin=True)
        probs = 1.0 / (1.0 + np.exp(-raw))
    else:
        probs = model_obj.predict(X_reg)

    from sklearn.metrics import roc_curve
    fpr, tpr, ths = roc_curve(y, probs)
    j_idx        = np.argmax(tpr - fpr)
    thresh_binary = float(ths[j_idx])
    thresh_high   = float(np.percentile(probs[y == 1], 75))

    bundle = {
        "model":        model_obj,
        "features":     reg_feats,
        "feat_kor":     {f: FEAT_KOR.get(f, f) for f in reg_feats},
        "target":       kor,
        "feat_ver":     feat_ver,
        "best_model":   best_model_name,
        "thresh_binary": thresh_binary,
        "thresh_high":   thresh_high,
    }
    out_path = MODEL_DIR / f"{save_name}.pkl"
    joblib.dump(bundle, out_path)
    print(f"    → 저장: {out_path}")

# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    print("데이터 로딩 중...")
    df = load_data()
    print(f"전체 샘플: {len(df)}\n")

    all_rows: list[dict] = []
    best_candidates: list[dict] = []  # 최적 모델 재훈련용

    for disease, tcfg in TARGETS.items():
        kor          = tcfg["kor"]
        gender_split = tcfg["gender_split"]
        excl_priv    = set(tcfg["excl_priv"])
        priv_feats   = [f for f in ALL_PRIVILEGED if f not in excl_priv and f in df.columns]

        y_all = tcfg["build"](df)
        # 타겟 결측치가 있는 행 제거
        valid_target = y_all.notna() & df[tcfg["need_cols"]].notna().all(axis=1)

        print(f"{'='*60}")
        print(f"[{kor}] 특권 피처: {priv_feats}")
        print(f"{'='*60}")

        for feat_ver, base_feats in FEATURE_SETS.items():
            # 항상 여성 / 남성 / 통합 세 가지 모두 실험
            subsets = [
                ("female", df["sex"] == 2,                           base_feats),
                ("male",   df["sex"] == 1,                           base_feats),
                ("all",    pd.Series(True, index=df.index), base_feats + [SEX_FEAT]),
            ]

            for gender_label, gender_mask, reg_feats in subsets:
                full_feats = reg_feats + priv_feats

                mask   = valid_target & gender_mask
                sub    = df[mask]
                y_sub  = y_all[mask].values.astype(float)

                # 피처 결측치 제거
                all_needed = list(set(reg_feats + full_feats))
                valid_feat = sub[all_needed].notna().all(axis=1)
                sub   = sub[valid_feat]
                y_sub = y_sub[valid_feat.values]

                n_pos  = int(y_sub.sum())
                n_neg  = int((y_sub == 0).sum())
                n_tot  = len(y_sub)
                spw    = n_neg / n_pos if n_pos > 0 else 1.0

                tag = f"[{kor}|{feat_ver}|{gender_label}]"
                print(f"\n  {tag}  n={n_tot}, 양성={n_pos}({100*n_pos/n_tot:.1f}%), spw={spw:.1f}")

                if n_pos < 50 or n_neg < 50:
                    print("  → 샘플 부족, 스킵")
                    continue

                X_reg  = sub[reg_feats].values.astype(float)
                X_full = sub[full_feats].values.astype(float)

                summary = run_cv(X_reg, y_sub, X_full, spw)

                for model_name, m in summary.items():
                    row = {
                        "disease": disease, "kor": kor,
                        "feat_ver": feat_ver, "gender": gender_label,
                        "model": model_name,
                        **m,
                    }
                    all_rows.append(row)

                # 이 조합의 최고 모델 (AUROC 기준)
                best_name = max(summary, key=lambda k: summary[k]["auroc"])
                best_candidates.append({
                    "disease": disease, "feat_ver": feat_ver,
                    "gender": gender_label, "kor": kor,
                    "model": best_name,
                    "X_reg": X_reg, "y": y_sub,
                    "X_full": X_full, "spw": spw,
                    "reg_feats": reg_feats,
                })

    # ── 결과 출력 ──────────────────────────────
    rdf = pd.DataFrame(all_rows)

    print("\n\n" + "="*80)
    print("▶ 5-fold CV 결과")
    print("="*80)

    model_order = ["lr", "rf", "gbm", "et", "xgb", "lgbm", "lupi_xgb"]

    for (disease, feat_ver, gender), grp in rdf.groupby(
        ["disease", "feat_ver", "gender"], sort=False
    ):
        kor = grp["kor"].iloc[0]
        print(f"\n  [{kor} | {feat_ver.upper()} | {gender}]")
        print(f"  {'모델':<12} {'AUROC':>7} {'±σ':>6} {'PR-AUC':>8} {'Brier':>7}")
        print(f"  {'-'*48}")
        grp_sorted = grp.set_index("model").reindex(model_order).dropna(how="all")
        for model_name, row in grp_sorted.iterrows():
            if pd.isna(row.get("auroc")):
                continue
            best_mark = " ★" if model_name == grp.loc[
                grp["auroc"].idxmax(), "model"] else ""
            print(
                f"  {model_name:<12} "
                f"{row['auroc']:>7.4f} "
                f"{row.get('auroc_std', 0):>6.4f} "
                f"{row['pr_auc']:>8.4f} "
                f"{row['brier']:>7.4f}"
                f"{best_mark}"
            )

    # ── CSV 저장 ──────────────────────────────
    csv_path = OUTPUT_DIR / "experiment_results.csv"
    rdf.to_csv(csv_path, index=False)
    print(f"\n결과 CSV 저장: {csv_path}")

    # ── 최적 모델 번들 저장 ───────────────────
    print("\n최적 모델 재훈련 & 저장 중...")
    for info in best_candidates:
        print(f"  [{info['kor']} | {info['feat_ver']} | {info['gender']}] best={info['model']}")
        _retrain_and_save(
            best_info=info,
            X_reg=info["X_reg"], y=info["y"],
            X_full=info["X_full"], spw=info["spw"],
            feat_ver=info["feat_ver"],
            disease=info["disease"], gender=info["gender"],
            kor=info["kor"], reg_feats=info["reg_feats"],
        )

    print("\n완료.")
    return rdf


if __name__ == "__main__":
    main()
