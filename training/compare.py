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
KEYS = ("hypertension", "diabetes", "cholesterol", "triglyceride")
MODEL = "gpt-5.6-luna"
REPEATS = 5
REPEAT_N = 300
PROMPT = """당신은 의료 위험도를 추정하는 도우미입니다.
아래 사람의 정보만으로, 이 사람이 현재 각 질환에 해당할 확률을 0-100 정수로 답하세요.
정보가 제한적이어도 반드시 최선의 추정치를 숫자로 답하세요.

[정보]
나이 {age}세, {sex}, BMI {bmi:.1f}, 체지방률 {pbf:.1f}%
고혈압 가족력 {fh_hp}, 당뇨 가족력 {fh_dm}, 고지혈증 가족력 {fh_hl}

[질환 정의]
- hypertension(고혈압): 측정 혈압 140/90mmHg 이상 또는 혈압약 복용
- diabetes(당뇨): 공복혈당 126mg/dL 이상, 당화혈색소 6.5% 이상, 의사 진단, 또는 약·인슐린 치료
- cholesterol(고콜레스테롤): 총콜레스테롤 240mg/dL 이상 또는 약 복용
- triglyceride(고중성지방): 12시간 이상 공복 시 중성지방 200mg/dL 이상"""
EXAMPLE = {"age": 34, "female": 0, "bmi": 26.1, "body_fat_percentage": 27.4,
           "fh_hypertension": 1, "fh_diabetes": 0, "fh_dyslipidemia": 0}

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


def build_prompt(row) -> str:
    yn = lambda v: "있음" if v == 1 else "없음"
    return PROMPT.format(age=int(row["age"]), sex="여성" if row["female"] == 1 else "남성",
                         bmi=row["bmi"], pbf=row["body_fat_percentage"], fh_hp=yn(row["fh_hypertension"]),
                         fh_dm=yn(row["fh_diabetes"]), fh_hl=yn(row["fh_dyslipidemia"]))


def parse_response(text: str) -> dict[str, float] | None:
    """출력 스키마를 따른 0–100 정수 4개면 0–1 확률로, 아니면 None(실패)."""
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict) or set(d) != set(KEYS):
        return None
    if not all(type(d[k]) is int and 0 <= d[k] <= 100 for k in KEYS):
        return None
    return {k: d[k] / 100 for k in KEYS}


def with_llm(pred: pd.DataFrame, cache: dict, key: str) -> tuple[pd.DataFrame, int]:
    """1회차 GPT 확률을 붙인다. 실패·누락인 사람은 모든 모델 비교에서 뺀다."""
    p = pd.Series({i: r["probs"][key] for (i, rep), r in cache.items() if rep == 0 and r["probs"]}, dtype=float)
    out = pred.assign(llm=p.reindex(pred.index))
    missing = int(out.llm.isna().sum())
    return out.dropna(subset=["llm"]), missing


def repeat_matrix(cache: dict, rep_ids: list[int], key: str) -> pd.DataFrame:
    """반복 REPEATS회가 모두 성공한 사람만."""
    rows = {}
    for i in rep_ids:
        recs = [cache.get((i, r)) for r in range(REPEATS)]
        if all(rc and rc["probs"] for rc in recs):
            rows[i] = [rc["probs"][key] for rc in recs]
    return pd.DataFrame.from_dict(rows, orient="index", columns=range(REPEATS))


def plot_consistency(preds: dict, cache: dict, rep_ids: list[int], path: Path, n_show: int = 40) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(preds)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 3.8), squeeze=False)
    for ax, key in zip(axes[0], keys):
        m = repeat_matrix(cache, rep_ids, key)
        common = m.index.intersection(preds[key].index)
        show = np.random.default_rng(SEED).choice(np.asarray(common), min(n_show, len(common)), replace=False)
        lr = preds[key].lr.loc[show].sort_values()
        x = np.arange(len(lr))
        vals = m.loc[lr.index].to_numpy() * 100
        ax.vlines(x, vals.min(axis=1), vals.max(axis=1), color="lightgray", lw=3)
        for j in range(vals.shape[1]):
            ax.scatter(x, vals[:, j], s=10, color="tab:orange", label="GPT (5 runs)" if j == 0 else None)
        ax.scatter(x, lr.to_numpy() * 100, marker="D", s=18, color="tab:blue", label="ML (logistic)")
        ax.set(title=key, xlabel="people (sorted by ML risk)", ylabel="predicted risk (%)", xticks=[])
    axes[0][0].legend(loc="upper left", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_results(preds: dict[str, pd.DataFrame], cache: dict | None, rep_ids: list[int]) -> dict:
    res = {"date": date.today().isoformat(), "llm": None, "diseases": {}}
    if cache is not None:
        res["llm"] = {"model": MODEL, "calls": len(cache), "failed": sum(r["probs"] is None for r in cache.values()),
                      "repeat_people": len(rep_ids), "prompt_example": build_prompt(EXAMPLE)}
    for key, pred in preds.items():
        if cache is None:
            res["diseases"][key] = compare_disease(pred, ML_MODELS)
            continue
        pred, missing = with_llm(pred, cache, key)
        r = compare_disease(pred, ML_MODELS + ["llm"])
        r["llm_missing"] = missing
        thr = r["models"]["llm"]["at_spec"]["threshold"]
        r["consistency"] = consistency_stats(repeat_matrix(cache, rep_ids, key).to_numpy(), thr)
        res["diseases"][key] = r
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
    if results["llm"]:
        info = results["llm"]
        L += ["", "## GPT 반복 응답 편차 (같은 입력 5회)", "",
              f"모델 `{info['model']}`(Codex CLI). 무작위 {info['repeat_people']}명에게 같은 입력을 5회 넣었다. "
              "판정 변경은 위 '같은 특이도' 기준값으로 5회 응답을 판정했을 때 양성·음성이 섞인 사람의 비율이다. "
              "ML(②)은 결정적이라 모두 0이다.", "",
              "| 질환 | 인원 | 표준편차 중앙값 / 90% (%p) | 범위 중앙값 / 최대 (%p) | 판정 변경 |",
              "|---|---|---|---|---|"]
        for key, r in results["diseases"].items():
            c = r["consistency"]
            if c["n"] == 0:
                L.append(f"| {DISEASES[key]} | 0 | - | - | - |")
                continue
            L.append(f"| {DISEASES[key]} | {c['n']} | {c['sd_median']:.1f} / {c['sd_p90']:.1f} | "
                     f"{c['range_median']:.1f} / {c['range_max']:.1f} | {_pct(c['flip_rate'])} |")
        L += ["", "![consistency](figures/llm_consistency.png)", "",
              "## 한계", "",
              f"- GPT 호출 {info['calls']}회 중 실패 {info['failed']}회. 1회차가 실패한 사람"
              f"({', '.join(f'{DISEASES[k]} {r["llm_missing"]}명' for k, r in results['diseases'].items())})은 "
              "모든 모델의 비교에서 뺐다.",
              "- Codex CLI는 에이전트 시스템 프롬프트를 붙인다. 일반 ChatGPT 대화와 조건이 같지 않다.",
              "- 프롬프트는 아래 1개로 고정했다. 이 실험은 LLM의 상한이 아니라, 같은 정보로 일반적인 질문을 했을 때를 측정한다.",
              "- KNHANES 참가자의 입력 7개(ID·연도·라벨 제외)를 외부 LLM으로 보냈다. 이용약관 판단은 사용자가 했다(2026-10-13).",
              "- 단면 자료라서 '현재 유병일 확률'을 비교한 것이다.",
              "", "## 프롬프트 원문 (예시 입력)", "", "```", info["prompt_example"], "```"]
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
