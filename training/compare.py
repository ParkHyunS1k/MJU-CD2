"""위험 예측 방식 비교: ⓪ 나이+성별, ① 체크리스트, ② 배포 로지스틱, ③ XGBoost, ④ GPT(Codex CLI).
실행: python -m training.compare [--skip-llm] [--limit N]
설계: docs/superpowers/specs/2026-10-13-model-comparison-design.md"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from training.analysis.common import load_knhanes
from training.evaluate import COMPARE_MODELS, _ci, _json_default, _pct, nested_predictions, outer_predictions
from training.features import DISEASES, THRESHOLD_AGE, training_rows
from training.metrics import bootstrap_ci, paired_diff_ci, sens_spec, wilson_ci
from training.model import SEED, build_pipeline

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
NAMES = {"base": "⓪ 나이+성별", "rule": "① 체크리스트", "lr": "② 배포 로지스틱", "xgb": "③ XGBoost", "llm": "④ GPT"}
ML_MODELS = ["base", "rule", "lr", "xgb"]
SUBGROUPS = {"25-34": (25, 34), "35-44": (35, 44)}
MIN_POS = 10

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


def disease_predictions(df: pd.DataFrame, key: str, spline: bool) -> pd.DataFrame:
    """19–59세로 학습한 바깥 5-fold 예측 중 25–44세만. 기존 평가(evaluate.py)와 같은 분할이다."""
    X, y, info = training_rows(df, key)
    age = info.age.to_numpy()
    p_lr, level = nested_predictions(X, y, age, key, spline)
    pred = pd.DataFrame({
        "y": y, "age": age,
        "base": outer_predictions(X[["age", "female"]], y, lambda: build_pipeline(False)),
        "rule": checklist_score(X, key),
        "lr": p_lr, "level": level,
        "xgb": outer_predictions(X, y, COMPARE_MODELS["xgb"]),
    }, index=X.index)
    return pred[pred.age.between(*THRESHOLD_AGE)]


def compare_disease(pred: pd.DataFrame, models: list[str]) -> dict:
    y = pred.y.to_numpy()
    p_lr = pred.lr.to_numpy(dtype=float)
    lr_at = sens_spec(y, pred.level.to_numpy())
    res = {"n": len(y), "n_pos": int(y.sum()), "prevalence": float(y.mean()), "models": {}, "subgroups": {}}
    for m in models:
        p = pred[m].to_numpy(dtype=float)
        r = {"auroc": bootstrap_ci(y, p, roc_auc_score), "auprc": bootstrap_ci(y, p, average_precision_score)}
        if m != "lr":
            r["auroc_diff"] = paired_diff_ci(y, p, p_lr, roc_auc_score)
        if m != "rule":  # 점수는 확률이 아니다
            r["brier"] = bootstrap_ci(y, p, brier_score_loss)
            r["mean_pred"] = float(p.mean())
        r["at_spec"] = lr_at if m == "lr" else sens_at_spec(y, p, lr_at["specificity"])
        res["models"][m] = r
    for g, (lo, hi) in SUBGROUPS.items():
        s = pred[pred.age.between(lo, hi)]
        ys = s.y.to_numpy()
        k = int(ys.sum())
        ok = MIN_POS <= k < len(ys)
        res["subgroups"][g] = {"n": len(ys), "n_pos": k, "auroc": {
            m: bootstrap_ci(ys, s[m].to_numpy(dtype=float), roc_auc_score) if ok else None for m in models}}
    return res


def build_results(preds: dict[str, pd.DataFrame], cache: dict | None, rep_ids: list[int]) -> dict:
    res = {"date": date.today().isoformat(), "llm": None, "diseases": {}}
    for key, pred in preds.items():
        res["diseases"][key] = compare_disease(pred, ML_MODELS)
    return res


def _model_row(m: str, x: dict) -> str:
    diff = _ci(x["auroc_diff"]) if "auroc_diff" in x else "기준"
    brier = _ci(x["brier"], 4) if "brier" in x else "해당 없음"
    mean = _pct(x["mean_pred"]) if "mean_pred" in x else "해당 없음"
    at = x["at_spec"]
    sens = f"{_pct(at['sensitivity'])} ({_pct(at['sensitivity_ci'][0])}–{_pct(at['sensitivity_ci'][1])})"
    return f"| {NAMES[m]} | {_ci(x['auroc'])} | {diff} | {_ci(x['auprc'])} | {brier} | {mean} | {sens} |"


def write_report(results: dict, path: Path) -> None:
    ms = [m for m in NAMES if m in next(iter(results["diseases"].values()))["models"]]
    L = [f"# 위험 예측 방식 비교 ({results['date']})", "",
         "설계: `docs/superpowers/specs/2026-10-13-model-comparison-design.md`. "
         "19–59세 학습, 25–44세 바깥 5-fold 예측. 모든 모델이 같은 사람을 평가한다.", "",
         "## 요약 (AUROC, 95% CI)", "",
         "| 질환 | n / 양성 | " + " | ".join(NAMES[m] for m in ms) + " |",
         "|---|---|" + "---|" * len(ms)]
    for key, r in results["diseases"].items():
        L.append(f"| {DISEASES[key]} | {r['n']} / {r['n_pos']} | "
                 + " | ".join(_ci(r["models"][m]["auroc"]) for m in ms) + " |")
    for key, r in results["diseases"].items():
        L += ["", f"## {DISEASES[key]} (`{key}`)", "",
              f"실제 유병률 {_pct(r['prevalence'])}. 민감도는 ②의 '보통 이상' 판정 특이도"
              f"({_pct(r['models']['lr']['at_spec']['specificity'])})에 맞춘 지점이다.", "",
              "| 모델 | AUROC | ② 대비 AUROC 차이 | AUPRC | Brier | 평균 예측 | 민감도 |",
              "|---|---|---|---|---|---|---|"]
        L += [_model_row(m, r["models"][m]) for m in ms]
        L += ["", "| 하위군 | n / 양성 | " + " | ".join(NAMES[m] for m in ms) + " |",
              "|---|---|" + "---|" * len(ms)]
        for g, sub in r["subgroups"].items():
            L.append(f"| {g} | {sub['n']} / {sub['n_pos']} | "
                     + " | ".join("표본 부족" if sub["auroc"][m] is None else _ci(sub["auroc"][m]) for m in ms) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _spline_flags() -> dict[str, bool]:
    d = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    return {k: bool(v["spline_adopt"]) for k, v in d["diseases"].items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-llm", action="store_true", help="⓪①②③만 계산한다(1단계)")
    args = ap.parse_args(argv)
    if not args.skip_llm:
        raise SystemExit("LLM 단계는 Task 4에서 추가한다. --skip-llm으로 실행하라.")
    df = load_knhanes()
    spline = _spline_flags()
    preds = {}
    for key in DISEASES:
        print(f"[{key}] 예측 중…", flush=True)
        preds[key] = disease_predictions(df, key, spline[key])
    results = build_results(preds, None, [])
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "compare-metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    path = REPORTS / f"compare-{results['date']}.md"
    write_report(results, path)
    print(f"보고서: {path}")


if __name__ == "__main__":
    main()
