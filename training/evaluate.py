"""위험 예측 모델 평가(spec 5절). 실행: python -m training.evaluate
결과: training/reports/eval-YYYY-MM-DD.md, metrics.json, figures/calibration_{key}.png"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from training.analysis.common import load_knhanes
from training.external import load_kaggle, load_nhanes
from training.features import BASE_FEATURES, DISEASES, THRESHOLD_AGE, training_rows
from training.metrics import (
    bootstrap_ci, calibration_slope_intercept, grade_table, paired_diff_ci, risk_levels, sens_spec, wilson_ci,
)
from training.model import SEED, build_pipeline, fit_bundle

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OLD_MODELS = ROOT / "output" / "old_models"
VARIANTS = ("linear", "spline")
COMPARE_MODELS = {
    "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=30, n_jobs=-1, random_state=SEED),
    "xgb": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, random_state=SEED, verbosity=0),
}


def _outer() -> StratifiedKFold:
    return StratifiedKFold(5, shuffle=True, random_state=SEED)


def nested_predictions(X, y, age, disease_key, age_spline):
    """바깥 5-fold의 평가 예측. 각 폴드는 fit_bundle(안쪽 OOF로 기준값 결정 → 학습)을 쓴다."""
    Xa = X.to_numpy(dtype=float)
    p = np.zeros(len(y))
    level = np.empty(len(y), dtype=object)
    for tr, te in _outer().split(Xa, y):
        b = fit_bundle(X.iloc[tr], y[tr], age[tr], disease_key, age_spline)
        p[te] = b["model"].predict_proba(Xa[te])[:, 1]
        level[te] = risk_levels(p[te], b["thresh_medium"], b["thresh_high"])
    return p, level


def outer_predictions(X, y, make_model):
    Xa = X.to_numpy(dtype=float)
    p = np.zeros(len(y))
    for tr, te in _outer().split(Xa, y):
        p[te] = make_model().fit(Xa[tr], y[tr]).predict_proba(Xa[te])[:, 1]
    return p


def summarize(y, p, level=None) -> dict:
    k, n = int(y.sum()), len(y)
    slope, intercept = calibration_slope_intercept(y, p)
    out = {"n": n, "n_pos": k,
           "auroc": bootstrap_ci(y, p, roc_auc_score), "brier": bootstrap_ci(y, p, brier_score_loss),
           "mean_pred": float(p.mean()), "prevalence": k / n, "prevalence_ci": wilson_ci(k, n),
           "cal_slope": slope, "cal_intercept": intercept}
    if level is not None:
        out["grades"] = grade_table(y, level)
        out["sens_spec"] = sens_spec(y, level)
    return out


def check_criteria(main: dict, auroc_diff: tuple) -> dict[str, bool]:
    """spec 5.4. thresh_high > thresh_medium은 choose_thresholds가 이미 강제한다."""
    lo, hi = main["prevalence_ci"]
    prevs = [g["prevalence"] for g in main["grades"]]
    return {
        "calibration": bool(lo <= main["mean_pred"] <= hi and 0.8 <= main["cal_slope"] <= 1.2),
        "contribution": bool(auroc_diff[0] >= 0),
        "grades": bool(None not in prevs and prevs[0] < prevs[1] < prevs[2]),
    }


def old_model_predictions(raw: pd.DataFrame, key: str, old_dir: Path | None):
    """기존 배포 모델(성별 분리·SMOTE, 입력 age/BIA_PBF/SMM=사지근육합/sex)의 예측."""
    if old_dir is None or not old_dir.exists():
        return None
    feats = pd.DataFrame({
        "age": raw.age, "BIA_PBF": raw.BIA_PBF,
        "SMM": raw[["BIA_LRA", "BIA_LLA", "BIA_LRL", "BIA_LLL"]].sum(axis=1),
        "sex": (raw.sex == 2).astype(float),
    })
    if key == "cholesterol":
        b = joblib.load(old_dir / "cholesterol.pkl")
        return b["model"].predict_proba(feats[b["features"]].to_numpy())[:, 1]
    p = np.full(len(raw), np.nan)
    for code, name in ((1, "male"), (2, "female")):
        m = (raw.sex == code).to_numpy()
        b = joblib.load(old_dir / f"{key}_{name}.pkl")
        p[m] = b["model"].predict_proba(feats.loc[m, b["features"]].to_numpy())[:, 1]
    return p


def plot_calibration(y, p, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.DataFrame({"p": p, "y": y})
    d["bin"] = pd.qcut(d.p, 10, labels=False, duplicates="drop")
    g = d.groupby("bin").mean()
    top = max(g.p.max(), g.y.max()) * 1.1
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, top], [0, top], "--", color="gray")
    ax.plot(g.p, g.y, "o-")
    ax.set(xlabel="predicted", ylabel="observed", title=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_disease(df, key, old_dir=OLD_MODELS, fig_dir=REPORTS / "figures") -> dict:
    X, y, info = training_rows(df, key)
    age, sex = info.age.to_numpy(), info.sex.to_numpy()
    tgt = (age >= THRESHOLD_AGE[0]) & (age <= THRESHOLD_AGE[1])
    groups = {"25-34": (age >= 25) & (age <= 34), "35-44": (age >= 35) & (age <= 44),
              "남(25-44)": tgt & (sex == 1), "여(25-44)": tgt & (sex == 2)}

    preds = {v: nested_predictions(X, y, age, key, v == "spline") for v in VARIANTS}
    p_base = outer_predictions(X[["age", "female"]], y, lambda: build_pipeline(False))  # AUROC 비교만 하므로 기준값 불필요
    res = {"main": {}, "subgroups": {}, "criteria": {},
           "baseline_age_sex": {"auroc": float(roc_auc_score(y[tgt], p_base[tgt]))}}
    for v, (p, level) in preds.items():
        res["main"][v] = summarize(y[tgt], p[tgt], level[tgt])
        res["subgroups"][v] = {g: summarize(y[m], p[m], level[m])
                               for g, m in groups.items() if 0 < y[m].sum() < m.sum()}
        diff = paired_diff_ci(y[tgt], p[tgt], p_base[tgt], roc_auc_score)
        res["baseline_age_sex"][f"auroc_diff_{v}"] = diff
        res["criteria"][v] = check_criteria(res["main"][v], diff)

    lin, spl = res["main"]["linear"], res["main"]["spline"]
    res["spline_adopt"] = bool(spl["brier"][0] < lin["brier"][0] and spl["auroc"][0] >= lin["auroc"][0])

    p_lin = preds["linear"][0]
    others = {"spline": preds["spline"][0], **{n: outer_predictions(X, y, mk) for n, mk in COMPARE_MODELS.items()}}
    res["compare"] = {n: {"auroc_diff": paired_diff_ci(y[tgt], pc[tgt], p_lin[tgt], roc_auc_score),
                          "brier_diff": paired_diff_ci(y[tgt], pc[tgt], p_lin[tgt], brier_score_loss),
                          "cal_slope": calibration_slope_intercept(y[tgt], pc[tgt])[0]}
                      for n, pc in others.items()}

    p_old = old_model_predictions(df.loc[info.index], key, old_dir)
    if p_old is not None:
        res["old_model"] = summarize(y[tgt], p_old[tgt])

    chosen = "spline" if res["spline_adopt"] else "linear"
    plot_calibration(y[tgt], preds[chosen][0][tgt], f"{key} ({chosen}, age 25-44)",
                     fig_dir / f"calibration_{key}.png")
    return res


def temporal(df, key, age_spline) -> dict:
    """2022–23년(19–59세) 학습 → 2024년(40–59세) 평가."""
    X, y, info = training_rows(df, key)
    early = (info.yr < 2024).to_numpy()
    b = fit_bundle(X[early], y[early], info.age.to_numpy()[early], key, age_spline)
    p = b["model"].predict_proba(X[~early].to_numpy(dtype=float))[:, 1]
    return summarize(y[~early], p)


def external(df, adopt: dict[str, bool]) -> dict:
    """가족력 제외 모델을 KNHANES 19–59세로 학습해 NHANES(4개)·Kaggle(고혈압)에서 평가."""
    try:
        nh, kg = load_nhanes(), load_kaggle()
    except (FileNotFoundError, ImportError) as e:
        return {"skipped": str(e)}
    out = {}
    for key in DISEASES:
        X, y, info = training_rows(df, key, features=BASE_FEATURES)
        b = fit_bundle(X, y, info.age.to_numpy(), key, adopt[key])
        m = nh[key].notna().to_numpy()
        yn = nh.loc[m, key].to_numpy()
        pn = b["model"].predict_proba(nh.loc[m, BASE_FEATURES].to_numpy())[:, 1]
        out[key] = {"nhanes": {"n": int(m.sum()), "n_pos": int(yn.sum()),
                               "auroc": bootstrap_ci(yn, pn, roc_auc_score),
                               "mean_pred": float(pn.mean()), "prevalence": float(yn.mean())}}
        if key == "hypertension":
            pk = b["model"].predict_proba(kg[BASE_FEATURES].to_numpy())[:, 1]
            out[key]["kaggle"] = {"n": len(kg), "auroc": bootstrap_ci(kg.hypertension.to_numpy(), pk, roc_auc_score)}
    return out


def _ci(t, d=3) -> str:
    return f"{t[0]:.{d}f} ({t[1]:.{d}f}–{t[2]:.{d}f})"


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _criteria(c: dict) -> str:
    return "✅" if all(c.values()) else "❌ " + ", ".join(k for k, ok in c.items() if not ok)


def _chosen(r: dict) -> str:
    return "spline" if r["spline_adopt"] else "linear"


def write_report(results: dict, path: Path) -> None:
    L = [f"# 위험 예측 모델 평가 ({results['date']})", "",
         "설계: `docs/superpowers/specs/2026-10-12-risk-model-redesign-design.md` 5절. "
         "주 평가는 19–59세 학습, 25–44세 중첩 교차검증이다.", "",
         "## 요약", "", "| 질환 | 채택 변형 | 통과 | AUROC (25–44) | 예측 평균 vs 실제 |", "|---|---|---|---|---|"]
    for key, r in results["diseases"].items():
        v = _chosen(r)
        m = r["main"][v]
        L.append(f"| {DISEASES[key]} | {v} | {_criteria(r['criteria'][v])} | {_ci(m['auroc'])} | "
                 f"{_pct(m['mean_pred'])} vs {_pct(m['prevalence'])} |")
    for key, r in results["diseases"].items():
        v = _chosen(r)
        L += ["", f"## {DISEASES[key]} (`{key}`)", "", "### 주 평가 (25–44세)", "",
              "| 변형 | n / 양성 | AUROC | Brier | 예측 평균 vs 실제 (95% CI) | 보정 기울기 / 절편 | 통과 기준 |",
              "|---|---|---|---|---|---|---|"]
        for var, m in r["main"].items():
            lo, hi = m["prevalence_ci"]
            L.append(f"| {var} | {m['n']} / {m['n_pos']} | {_ci(m['auroc'])} | {_ci(m['brier'], 4)} | "
                     f"{_pct(m['mean_pred'])} vs {_pct(m['prevalence'])} ({_pct(lo)}–{_pct(hi)}) | "
                     f"{m['cal_slope']:.2f} / {m['cal_intercept']:.2f} | {_criteria(r['criteria'][var])} |")
        L += ["", f"나이 스플라인 채택: **{'예' if r['spline_adopt'] else '아니오'}** (규칙: Brier 감소 그리고 AUROC 비감소)",
              "", f"### 등급 ({v})", "", "| 등급 | 인원 | 비율 | 실제 유병률 (95% CI) |", "|---|---|---|---|"]
        for g in r["main"][v]["grades"]:
            ci = g["prevalence_ci"]
            ci_txt = f" ({_pct(ci[0])}–{_pct(ci[1])})" if ci else ""
            L.append(f"| {g['level']} | {g['n']} | {_pct(g['share'])} | {_pct(g['prevalence'])}{ci_txt} |")
        s = r["main"][v]["sens_spec"]
        L += ["", f"보통 이상 민감도 {_pct(s['sensitivity'])} ({_pct(s['sensitivity_ci'][0])}–{_pct(s['sensitivity_ci'][1])}), "
                  f"특이도 {_pct(s['specificity'])} ({_pct(s['specificity_ci'][0])}–{_pct(s['specificity_ci'][1])}), "
                  f"높음 적중률 {_pct(s['high_ppv'])}",
              "", f"### 하위군 ({v})", "", "| 구간 | n / 양성 | AUROC | 예측 평균 vs 실제 | 민감도 / 특이도 |", "|---|---|---|---|---|"]
        for g, m in r["subgroups"][v].items():
            L.append(f"| {g} | {m['n']} / {m['n_pos']} | {_ci(m['auroc'])} | {_pct(m['mean_pred'])} vs "
                     f"{_pct(m['prevalence'])} | {_pct(m['sens_spec']['sensitivity'])} / {_pct(m['sens_spec']['specificity'])} |")
        b = r["baseline_age_sex"]
        L += ["", "### 비교", "", f"- 나이+성별 모델 AUROC {b['auroc']:.3f}, 차이({v}) {_ci(b['auroc_diff_' + v])}"]
        if "old_model" in r:
            o = r["old_model"]
            L.append(f"- 기존 배포 모델(새 라벨 기준, 전체 데이터로 학습돼 낙관적): AUROC {_ci(o['auroc'])}, "
                     f"예측 평균 {_pct(o['mean_pred'])} vs 실제 {_pct(o['prevalence'])}, 보정 기울기 {o['cal_slope']:.2f}")
        L += ["", "| 모델 (vs 로지스틱 회귀) | AUROC 차이 | Brier 차이 | 보정 기울기 |", "|---|---|---|---|"]
        for n, c in r["compare"].items():
            L.append(f"| {n} | {_ci(c['auroc_diff'])} | {_ci(c['brier_diff'], 4)} | {c['cal_slope']:.2f} |")
        t = r["temporal"]
        L += ["", "### 시간 검증 (2022–23 학습 → 2024, 40–59세)", "",
              f"n {t['n']} / 양성 {t['n_pos']}, AUROC {_ci(t['auroc'])}, 예측 평균 {_pct(t['mean_pred'])} vs "
              f"실제 {_pct(t['prevalence'])}, 보정 기울기 {t['cal_slope']:.2f}",
              "", f"![calibration](figures/calibration_{key}.png)"]
    ext = results["external"]
    L += ["", "## 외부 검증 (가족력 제외 모델)", ""]
    if "skipped" in ext:
        L.append(f"건너뜀: {ext['skipped']}")
    else:
        L += ["| 질환 | NHANES n / 양성 | NHANES AUROC | 예측 평균 vs 실제 | Kaggle AUROC (고혈압) |", "|---|---|---|---|---|"]
        for key, e in ext.items():
            nh, kg = e["nhanes"], e.get("kaggle")
            L.append(f"| {DISEASES[key]} | {nh['n']} / {nh['n_pos']} | {_ci(nh['auroc'])} | "
                     f"{_pct(nh['mean_pred'])} vs {_pct(nh['prevalence'])} | {_ci(kg['auroc']) if kg else '-'} |")
        L += ["", "Kaggle은 단회 측정 혈압이라 고혈압 비율이 부풀어 있어 보정을 평가하지 않는다."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    df = load_knhanes()
    results = {"date": date.today().isoformat(), "diseases": {}}
    for key in DISEASES:
        print(f"[{key}] 평가 중…", flush=True)
        r = evaluate_disease(df, key)
        r["temporal"] = temporal(df, key, r["spline_adopt"])
        results["diseases"][key] = r
    results["external"] = external(df, {k: r["spline_adopt"] for k, r in results["diseases"].items()})
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    path = REPORTS / f"eval-{results['date']}.md"
    write_report(results, path)
    print(f"보고서: {path}")


if __name__ == "__main__":
    main()
