# 위험 예측 방식 비교실험 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 배포 로지스틱 모델(②)을 같은 사람, 같은 입력에서 나이·성별 로지스틱(⓪), 체크리스트 점수(①), XGBoost(③), GPT 제로샷(④)과 비교한다. 결과는 발표용 AUROC 표 1장과 GPT 출력 편차 그림 1장이다.

**Architecture:**
- 실행 모듈은 `training/compare.py` 하나다.
- ML 예측은 기존 `training.evaluate`의 `nested_predictions`·`outer_predictions`를 그대로 불러 쓴다. 지표는 `training.metrics`를 쓴다.
- GPT는 `codex exec`를 subprocess로 호출한다. 응답은 JSONL로 캐시하므로 중단해도 이어서 실행할 수 있다.
- 순수 함수(점수·지표·파싱·집계·보고서)와 CLI 호출(`call_codex`)을 분리한다. 테스트에서는 가짜 호출 함수를 주입한다.

**Tech Stack:** Python 3.13(`.venv`), scikit-learn 1.8.0, xgboost 3.2.0, pandas, numpy, matplotlib, pytest, Codex CLI 0.162.1(ChatGPT 로그인).

**Spec:** `docs/superpowers/specs/2026-10-13-model-comparison-design.md`

## Global Constraints

- 명령은 모두 저장소 루트에서 `.venv/bin/python`으로 실행한다. base 파이썬은 sklearn 1.7.2라서 쓰지 않는다.
- 코호트·라벨·입력은 `training.features.training_rows`를 그대로 쓴다(19–59세 학습).
- 평가 연령은 25–44세다. 하위군은 25–34세와 35–44세이고, 양성이 10명 미만이면 "표본 부족"으로 적는다.
- ②의 나이 스플라인 여부는 `training/reports/metrics.json`의 `diseases[key].spline_adopt`에서 읽는다(현재 고콜레스테롤만 `true`).
- ①의 규칙:
  - BMI ≥ 25
  - 체지방률이 남 ≥ 25, 여 ≥ 30
  - 해당 질환의 가족력 == 1(고혈압→`fh_hypertension`, 당뇨→`fh_diabetes`, 고콜레스테롤·고중성지방→`fh_dyslipidemia`)
  - 나이 ≥ 40
  - 각 1점, 0–4점
- 민감도 비교 지점은 ②의 '보통 이상' 판정 특이도다. 다른 모델은 그 특이도 이상이 되는 가장 낮은 기준값을 쓴다.
- LLM 설정:
  - `MODEL = "gpt-5.6-luna"`, `model_reasoning_effort=low`
  - 동시 호출 4개
  - 1인당 반복 5회(`REPEATS = 5`), 반복 대상 300명(`REPEAT_N = 300`)
  - `SEED = 42`
- Codex 명령: `codex exec --ignore-user-config -m gpt-5.6-luna -c model_reasoning_effort=low -s read-only --skip-git-repo-check --ephemeral --output-schema <schema> -o <out> "<prompt>"`
  - stdin은 `DEVNULL`로 닫는다.
  - 작업 폴더는 빈 임시 폴더로 둔다.
- GPT에 보내는 것은 입력 7개뿐이다. ID, 조사연도, 라벨은 보내지 않는다.
- 캐시는 `training/output/llm/gpt-5.6-luna.jsonl`이다. `training/output/`은 gitignore 대상이며 커밋하지 않는다.
- 기존 `training/*.py`는 수정하지 않는다. 불러와서 쓰기만 한다(`_ci`, `_pct`, `_json_default` 포함).
- 진행 순서: ① 표(LLM 없음) → ② 시범 20회 → ③ 전체 실행 → ④ 보고서 확정. 각 단계가 끝나면 **멈추고 사용자에게 보고한다.** 다음 단계를 앞질러 실행하지 않는다.

## Review Focus

1. **실행이 중간에 끊긴 뒤 다시 실행할 때.** 성공한 호출은 다시 하지 않는다. 실패한 호출만 다시 시도한다. 마지막 줄이 잘린 캐시 파일도 읽을 수 있어야 한다. → Task 4 `test_run_llm_resumes_and_retries_failures`, `test_load_cache_skips_truncated_line`
2. **GPT가 `-1`, 범위 밖 값, JSON이 아닌 문장을 반환할 때.** 실패로 기록하고, 그 사람은 모든 모델의 평가에서 뺀다. → Task 3 `test_parse_response_rejects_bad_output`, `test_with_llm_drops_missing_rows`
3. **반복 5회 중 일부만 성공한 사람.** 편차 통계에서 그 사람을 뺀다. 0으로 채우지 않는다. → Task 3 `test_repeat_matrix_requires_all_reps`
4. **하위군 양성이 10명 미만일 때.** 오류 없이 "표본 부족"을 표시한다. → Task 2 `test_compare_disease_marks_small_subgroups`
5. **체크리스트 경계값과 동점.** 경계값(25.0, 29.9, 40)이 규칙대로 판정되고, 정수 점수의 동점에서도 같은 특이도 기준값이 정해진다. → Task 1 `test_checklist_score_boundaries`, `test_sens_at_spec_with_ties`

---

### Task 1: 규칙 점수와 비교 지표 함수

**Files:**
- Create: `training/compare.py`
- Test: `tests/risk_model/test_compare.py`

**Interfaces:**
- Consumes: `training.metrics.wilson_ci(k: int, n: int) -> tuple[float, float]`
- Produces:
  - `FH_FOR: dict[str, str]`
  - `checklist_score(X: pd.DataFrame, disease_key: str) -> np.ndarray`(int, 0–4)
  - `sens_at_spec(y: np.ndarray, score: np.ndarray, target_spec: float) -> dict`. 키는 `threshold`, `specificity`, `sensitivity`, `sensitivity_ci`다.
  - `consistency_stats(probs: np.ndarray, threshold: float) -> dict`. `probs`는 (n, 5) 0–1 배열이고, 키는 `n`, `sd_median`, `sd_p90`, `range_median`, `range_max`(%p), `flip_rate`다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/risk_model/test_compare.py`:

```python
import numpy as np
import pandas as pd
import pytest

from training.compare import checklist_score, consistency_stats, sens_at_spec


def _X(rows):
    cols = ["age", "female", "bmi", "body_fat_percentage", "fh_hypertension", "fh_diabetes", "fh_dyslipidemia"]
    return pd.DataFrame(rows, columns=cols, dtype=float)


def test_checklist_score_boundaries():
    X = _X([
        [40, 0, 25.0, 25.0, 1, 0, 0],   # 남: 모든 항목 경계값에서 해당 → 4
        [39, 1, 24.9, 29.9, 0, 1, 1],   # 여: 경계 바로 아래, 고혈압 가족력 없음 → 0
        [30, 1, 20.0, 30.0, 0, 0, 0],   # 여 체지방률 30 → 1
    ])
    assert checklist_score(X, "hypertension").tolist() == [4, 0, 1]
    assert checklist_score(X, "cholesterol").tolist() == [3, 1, 1]   # 고지혈증 가족력을 본다
    assert checklist_score(X, "diabetes").tolist() == [3, 1, 1]


def test_sens_at_spec_with_ties():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([1, 2, 3, 4, 3, 5])
    r = sens_at_spec(y, s, 0.75)
    assert r["threshold"] == 4 and r["specificity"] == 0.75 and r["sensitivity"] == 0.5
    tied = sens_at_spec(np.array([0, 0, 1, 1]), np.array([1, 1, 1, 1]), 0.5)
    assert tied["threshold"] == np.inf and tied["specificity"] == 1.0 and tied["sensitivity"] == 0.0


def test_consistency_stats():
    probs = np.array([[0.1] * 5, [0.1, 0.3, 0.1, 0.1, 0.1]])
    r = consistency_stats(probs, 0.2)
    assert r["n"] == 2 and r["flip_rate"] == 0.5
    assert r["range_max"] == pytest.approx(20.0) and r["range_median"] == pytest.approx(10.0)
    assert consistency_stats(np.empty((0, 5)), 0.2) == {"n": 0}
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_compare.py -v`
Expected: FAIL. `ModuleNotFoundError: No module named 'training.compare'`

- [ ] **Step 3: 최소 구현**

`training/compare.py`:

```python
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
```

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_compare.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add training/compare.py tests/risk_model/test_compare.py
git commit -m "feat(compare): add checklist score, matched-specificity sensitivity, consistency stats

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: ML 비교 표(⓪①②③)와 보고서, `--skip-llm` 실행 → 1단계에서 멈춤

**Files:**
- Modify: `training/compare.py`(아래 코드를 추가하고, import 블록을 교체한다)
- Test: `tests/risk_model/test_compare.py`(테스트 추가)
- Create(실행 결과): `training/reports/compare-YYYY-MM-DD.md`, `training/reports/compare-metrics.json`

**Interfaces:**
- Consumes:
  - `training_rows(df, key) -> (X, y, info)`
  - `nested_predictions(X, y, age, key, age_spline) -> (p, level)`
  - `outer_predictions(X, y, make_model) -> p`
  - `COMPARE_MODELS["xgb"]`, `build_pipeline(False)`
  - `bootstrap_ci`, `paired_diff_ci`, `sens_spec`
  - `_ci`, `_pct`, `_json_default`(`training.evaluate`)
  - `load_knhanes`, `DISEASES`
- Produces:
  - `disease_predictions(df, key, spline: bool) -> pd.DataFrame`. 25–44세 행만 담고, 열은 `y, age, base, rule, lr, level, xgb`, 인덱스는 `df` 원래 인덱스다.
  - `compare_disease(pred: pd.DataFrame, models: list[str]) -> dict`
  - `build_results(preds: dict[str, pd.DataFrame], cache: dict | None, rep_ids: list[int]) -> dict`. 이 Task에서는 `cache=None`만 처리한다. Task 3에서 확장한다.
  - `write_report(results: dict, path: Path) -> None`
  - `NAMES: dict[str, str]`, `ML_MODELS = ["base", "rule", "lr", "xgb"]`
  - `main(argv)`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/risk_model/test_compare.py` 끝에 추가한다:

```python
from training.compare import build_results, compare_disease, disease_predictions, write_report


def test_disease_predictions_columns(knhanes_like):
    pred = disease_predictions(knhanes_like, "hypertension", False)
    assert list(pred.columns) == ["y", "age", "base", "rule", "lr", "level", "xgb"]
    assert pred.age.between(25, 44).all() and len(pred) > 0
    assert set(pred.rule.unique()) <= {0, 1, 2, 3, 4}


def test_compare_disease_marks_small_subgroups(knhanes_like):
    pred = disease_predictions(knhanes_like, "hypertension", False)
    young = pred.age <= 34
    pred.loc[young, "y"] = 0.0
    pred.loc[pred[young].index[:5], "y"] = 1.0   # 25–34세 양성 5명
    r = compare_disease(pred, ["base", "rule", "lr", "xgb"])
    assert r["subgroups"]["25-34"]["auroc"]["lr"] is None
    assert r["subgroups"]["35-44"]["auroc"]["lr"] is not None
    assert "auroc_diff" not in r["models"]["lr"] and "auroc_diff" in r["models"]["xgb"]
    assert "brier" not in r["models"]["rule"]
    assert r["models"]["xgb"]["at_spec"]["specificity"] >= r["models"]["lr"]["at_spec"]["specificity"]


def test_report_without_llm(knhanes_like, tmp_path):
    preds = {"hypertension": disease_predictions(knhanes_like, "hypertension", False)}
    res = build_results(preds, None, [])
    out = tmp_path / "compare.md"
    write_report(res, out)
    text = out.read_text(encoding="utf-8")
    assert "## 요약" in text and "② 배포 로지스틱" in text and "④ GPT" not in text
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_compare.py -v`
Expected: FAIL. `ImportError: cannot import name 'build_results'`

- [ ] **Step 3: 구현**

`training/compare.py`의 import 블록을 아래로 교체한다:

```python
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
```

파일 끝에 추가한다:

```python
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
```

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run: `.venv/bin/python -m pytest tests/risk_model -q`
Expected: 기존 49개 + 새 6개 = 55 passed

- [ ] **Step 5: 커밋**

```bash
git add training/compare.py tests/risk_model/test_compare.py
git commit -m "feat(compare): add ML comparison table and report (no LLM)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: 실제 데이터로 1단계 실행**

Run: `.venv/bin/python -m training.compare --skip-llm`
Expected: `보고서: .../training/reports/compare-2026-10-13.md`(날짜는 실행일)

- [ ] **Step 7: 기존 평가와 수치가 같은지 검증**

②와 XGB의 수치가 같은 분할을 쓴 기존 평가 결과와 일치해야 한다.

Run:
```bash
.venv/bin/python -c "
import json
a=json.load(open('training/reports/metrics.json'))['diseases']; b=json.load(open('training/reports/compare-metrics.json'))['diseases']
for k in a:
    v='spline' if a[k]['spline_adopt'] else 'linear'
    old_lr, new_lr = a[k]['main'][v]['auroc'][0], b[k]['models']['lr']['auroc'][0]
    old_x = a[k]['compare']['xgb']['auroc_diff'][0] if v=='linear' else None
    new_x = b[k]['models']['xgb']['auroc_diff'][0]
    print(k, f'lr {old_lr:.4f} vs {new_lr:.4f}', f'xgb diff {old_x} vs {new_x:.4f}', 'OK' if abs(old_lr-new_lr)<=0.002 and (old_x is None or abs(old_x-new_x)<=0.002) else 'MISMATCH')
"
```
Expected: 네 줄 모두 `OK`. 고콜레스테롤은 기존 XGB 차이가 linear 기준이라 `None`으로 건너뛴다. `MISMATCH`가 나오면 원인을 찾을 때까지 다음으로 넘어가지 않는다(superpowers:systematic-debugging).

- [ ] **Step 8: 보고서 커밋**

```bash
git add training/reports/compare-*.md training/reports/compare-metrics.json
git commit -m "docs(compare): add stage-1 comparison report without LLM

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 9: 멈추고 보고**

사용자에게 다음을 보고하고 **승인을 기다린다.**
- 요약 표(질환 × ⓪①②③ AUROC)
- ①과 ⓪의 AUROC가 ②보다 얼마나 낮은지(차이 CI)
- Step 7 검증 결과

---

### Task 3: LLM 결과 집계, 편차 그림, 보고서 확장(캐시 주입, CLI 호출 없음)

**Files:**
- Modify: `training/compare.py`
- Test: `tests/risk_model/test_compare.py`

**Interfaces:**
- Consumes: Task 1·2의 함수
- Produces:
  - `KEYS = ("hypertension", "diabetes", "cholesterol", "triglyceride")`
  - `REPEATS = 5`, `MODEL`, `REPEAT_N`, `PROMPT`, `EXAMPLE`
  - `build_prompt(row: pd.Series | dict) -> str`
  - `parse_response(text: str) -> dict[str, float] | None`(0–1 확률)
  - `with_llm(pred, cache, key) -> tuple[pd.DataFrame, int]`. `llm` 열을 붙이고 누락 행을 뺀다. 두 번째 값은 누락 수다.
  - `repeat_matrix(cache, rep_ids, key) -> pd.DataFrame`. 인덱스는 사람, 열은 0..4다.
  - `plot_consistency(preds, cache, rep_ids, path, n_show=40) -> None`
  - `build_results(preds, cache, rep_ids)`가 `cache`가 있으면 ④와 편차를 포함하도록 확장된다.
  - 캐시 형식은 `dict[tuple[int, int], dict]`다. 키는 `(idx, rep)`, 값은 `{"idx", "rep", "probs": dict | None, "error", "tokens", "seconds"}`다.

- [ ] **Step 1: 실패하는 테스트 추가**

```python
from training.compare import KEYS, REPEATS, parse_response, plot_consistency, repeat_matrix, with_llm


def test_parse_response_accepts_valid():
    assert parse_response('{"hypertension": 15, "diabetes": 3, "cholesterol": 8, "triglyceride": 100}') == \
        {"hypertension": 0.15, "diabetes": 0.03, "cholesterol": 0.08, "triglyceride": 1.0}


def test_parse_response_rejects_bad_output():
    bad = ['{"hypertension": -1, "diabetes": -1, "cholesterol": -1, "triglyceride": -1}',
           '{"hypertension": 101, "diabetes": 3, "cholesterol": 8, "triglyceride": 1}',
           '{"hypertension": 15, "diabetes": 3, "cholesterol": 8}',
           '{"hypertension": 15.5, "diabetes": 3, "cholesterol": 8, "triglyceride": 1}',
           '{"hypertension": true, "diabetes": 3, "cholesterol": 8, "triglyceride": 1}',
           "확률을 추정할 수 없습니다.", ""]
    assert all(parse_response(t) is None for t in bad)


def _rec(idx, rep, probs):
    return {"idx": idx, "rep": rep, "probs": probs, "error": None if probs else "x", "tokens": 1, "seconds": 1.0}


def _fake_cache(preds, rep_ids, noise=0.05):
    """②의 예측에 잡음을 더한 가짜 GPT 응답."""
    rng = np.random.default_rng(1)
    ids = sorted(set().union(*(p.index for p in preds.values())))
    cache = {}
    for i in ids:
        for rep in range(REPEATS if i in rep_ids else 1):
            probs = {k: float(np.clip((preds[k].lr.get(i, 0.1) if k in preds else 0.1) + rng.normal(0, noise), 0, 1))
                     for k in KEYS}
            cache[(int(i), rep)] = _rec(int(i), rep, probs)
    return cache


def test_with_llm_drops_missing_rows(knhanes_like):
    pred = disease_predictions(knhanes_like, "hypertension", False)
    cache = _fake_cache({"hypertension": pred}, [])
    first, second = pred.index[0], pred.index[1]
    cache[(int(first), 0)] = _rec(int(first), 0, None)
    del cache[(int(second), 0)]
    out, missing = with_llm(pred, cache, "hypertension")
    assert missing == 2 and first not in out.index and second not in out.index
    assert len(out) == len(pred) - 2


def test_repeat_matrix_requires_all_reps():
    full = {k: 0.1 for k in KEYS}
    cache = {(1, r): _rec(1, r, full) for r in range(REPEATS)}
    cache.update({(2, r): _rec(2, r, full) for r in range(REPEATS - 1)})        # 반복 하나 누락
    cache.update({(3, r): _rec(3, r, full if r else None) for r in range(REPEATS)})  # 1회 실패
    m = repeat_matrix(cache, [1, 2, 3], "diabetes")
    assert m.index.tolist() == [1] and m.shape == (1, REPEATS)


def test_report_and_plot_with_llm(knhanes_like, tmp_path):
    preds = {k: disease_predictions(knhanes_like, k, False) for k in ("hypertension", "diabetes")}
    rep_ids = [int(i) for i in preds["hypertension"].index[:60]]
    cache = _fake_cache(preds, rep_ids)
    res = build_results(preds, cache, rep_ids)
    h = res["diseases"]["hypertension"]
    assert "llm" in h["models"] and h["consistency"]["n"] == 60 and h["llm_missing"] == 0
    write_report(res, tmp_path / "c.md")
    text = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert "④ GPT" in text and "반복 응답 편차" in text and "프롬프트 원문" in text
    plot_consistency(preds, cache, rep_ids, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").exists()
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_compare.py -v`
Expected: FAIL. `ImportError: cannot import name 'KEYS'`

- [ ] **Step 3: 구현**

`ML_MODELS` 정의 아래에 상수와 프롬프트를 추가한다:

```python
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
```

`build_results` 위에 아래 함수들을 추가한다:

```python
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
```

`build_results`를 교체한다:

```python
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
```

`write_report`에서 `path.parent.mkdir(...)` 바로 위에 아래 블록을 넣는다:

```python
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
              f"({', '.join(f'{DISEASES[k]} {r['llm_missing']}명' for k, r in results['diseases'].items())})은 "
              "모든 모델의 비교에서 뺐다.",
              "- Codex CLI는 에이전트 시스템 프롬프트를 붙인다. 일반 ChatGPT 대화와 조건이 같지 않다.",
              "- 프롬프트는 아래 1개로 고정했다. 이 실험은 LLM의 상한이 아니라, 같은 정보로 일반적인 질문을 했을 때를 측정한다.",
              "- KNHANES 참가자의 입력 7개(ID·연도·라벨 제외)를 외부 LLM으로 보냈다. 이용약관 판단은 사용자가 했다(2026-10-13).",
              "- 단면 자료라서 '현재 유병일 확률'을 비교한 것이다.",
              "", "## 프롬프트 원문 (예시 입력)", "", "```", info["prompt_example"], "```"]
```

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run: `.venv/bin/python -m pytest tests/risk_model -q`
Expected: 60 passed

- [ ] **Step 5: 커밋**

```bash
git add training/compare.py tests/risk_model/test_compare.py
git commit -m "feat(compare): aggregate GPT responses, consistency figure and report sections

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Codex CLI 호출, 캐시, `main` 완성 → 2단계 시범 호출에서 멈춤

**Files:**
- Modify: `training/compare.py`
- Test: `tests/risk_model/test_compare.py`

**Interfaces:**
- Consumes: `build_prompt`, `parse_response`, `KEYS`, `MODEL`, `REPEATS`, `REPEAT_N`, `build_results`, `plot_consistency`
- Produces:
  - `class CodexError(RuntimeError)`
  - `call_codex(prompt: str) -> tuple[str, int | None]`. 반환값은 (응답 텍스트, 사용 토큰)이다.
  - `ask(prompt, caller, attempts=3) -> dict`. 캐시 레코드에서 `idx`·`rep`을 뺀 나머지다.
  - `run_llm(tasks: list[tuple[int, int]], X: pd.DataFrame, cache_path: Path, caller=call_codex, workers=4) -> None`
  - `load_cache(path: Path) -> dict[tuple[int, int], dict]`
  - `pilot_summary(cache: dict, tasks) -> str`
  - `CACHE = ROOT / "output" / "llm" / f"{MODEL}.jsonl"`

- [ ] **Step 1: 실패하는 테스트 추가**

```python
import json as _json

import training.compare as cmp
from training.compare import load_cache, pilot_summary, run_llm


def _people():
    return pd.DataFrame([[34, 0, 26.1, 27.4, 1, 0, 0], [41, 1, 22.0, 31.0, 0, 1, 0]],
                        columns=["age", "female", "bmi", "body_fat_percentage",
                                 "fh_hypertension", "fh_diabetes", "fh_dyslipidemia"], index=[10, 11])


def test_build_prompt_has_inputs_and_definitions():
    p = cmp.build_prompt(_people().loc[10])
    assert "34세" in p and "남성" in p and "BMI 26.1" in p and "체지방률 27.4%" in p
    assert "고혈압 가족력 있음" in p and "당뇨 가족력 없음" in p and "140/90" in p


def test_run_llm_resumes_and_retries_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(cmp.time, "sleep", lambda s: None)
    good = '{"hypertension": 20, "diabetes": 5, "cholesterol": 10, "triglyceride": 12}'
    calls = []

    def flaky(prompt):
        calls.append(prompt)
        if "41세" in prompt and len(calls) <= 4:
            return '{"hypertension": -1, "diabetes": -1, "cholesterol": -1, "triglyceride": -1}', 100
        return good, 100

    cache = tmp_path / "c.jsonl"
    tasks = [(10, 0), (11, 0), (10, 1)]
    run_llm(tasks, _people(), cache, caller=flaky, workers=1)
    c = load_cache(cache)
    assert c[(10, 0)]["probs"]["hypertension"] == 0.2 and c[(11, 0)]["probs"] is None
    n_first = len(calls)
    run_llm(tasks, _people(), cache, caller=flaky, workers=1)   # 성공분은 건너뛰고 실패분만 다시 한다
    assert len(calls) == n_first + 1 and load_cache(cache)[(11, 0)]["probs"] is not None


def test_ask_retries_codex_errors(monkeypatch):
    monkeypatch.setattr(cmp.time, "sleep", lambda s: None)

    def broken(prompt):
        raise cmp.CodexError("Selected model is at capacity")

    r = cmp.ask("p", broken)
    assert r["probs"] is None and "capacity" in r["error"]


def test_load_cache_skips_truncated_line(tmp_path):
    path = tmp_path / "c.jsonl"
    ok = {"idx": 1, "rep": 0, "probs": {k: 0.1 for k in KEYS}, "error": None, "tokens": 5, "seconds": 1.0}
    path.write_text(_json.dumps(ok) + "\n" + '{"idx": 2, "rep": 0, "pro', encoding="utf-8")
    assert list(load_cache(path)) == [(1, 0)]
    assert load_cache(tmp_path / "none.jsonl") == {}


def test_pilot_summary_counts():
    cache = {(1, 0): {"idx": 1, "rep": 0, "probs": {k: 0.1 for k in KEYS}, "error": None, "tokens": 20000, "seconds": 8.0},
             (2, 0): {"idx": 2, "rep": 0, "probs": None, "error": "parse", "tokens": None, "seconds": 9.0}}
    s = pilot_summary(cache, [(1, 0), (2, 0)])
    assert "성공 1/2" in s and "20,000" in s
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_compare.py -v`
Expected: FAIL. `ImportError: cannot import name 'load_cache'`

- [ ] **Step 3: 구현**

import 블록에 추가한다:

```python
import re
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from training.features import build_features
```

`REPEAT_N` 아래에 추가한다:

```python
CACHE = ROOT / "output" / "llm" / f"{MODEL}.jsonl"
SCHEMA = {"type": "object", "additionalProperties": False, "required": list(KEYS),
          "properties": {k: {"type": "integer", "minimum": 0, "maximum": 100} for k in KEYS}}
_TMP = Path(tempfile.gettempdir()) / "compare-codex"
_WORKDIR = _TMP / "empty"     # Codex 작업 폴더: 비워 둔다
_IO = _TMP / "io"             # 스키마와 응답 파일
_SCHEMA_PATH = _IO / "schema.json"
_setup_lock = threading.Lock()
```

`build_results` 위에 추가한다:

```python
class CodexError(RuntimeError):
    pass


def call_codex(prompt: str) -> tuple[str, int | None]:
    """codex exec 1회. 사용자 설정을 끄고 빈 폴더에서 stdin을 닫고 실행한다(닫지 않으면 멈춘다)."""
    with _setup_lock:  # 여러 스레드가 동시에 스키마를 쓰지 않게 한다
        if not _SCHEMA_PATH.exists():
            _WORKDIR.mkdir(parents=True, exist_ok=True)
            _IO.mkdir(parents=True, exist_ok=True)
            _SCHEMA_PATH.write_text(json.dumps(SCHEMA), encoding="utf-8")
    schema = _SCHEMA_PATH
    out = _IO / f"{uuid.uuid4().hex}.txt"
    r = subprocess.run(
        ["codex", "exec", "--ignore-user-config", "-m", MODEL, "-c", "model_reasoning_effort=low",
         "-s", "read-only", "--skip-git-repo-check", "--ephemeral",
         "--output-schema", str(schema), "-o", str(out), prompt],
        cwd=_WORKDIR, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    log = r.stdout + r.stderr
    if r.returncode != 0 or not out.exists():
        raise CodexError(log[-500:])
    text = out.read_text(encoding="utf-8").strip()
    out.unlink()
    m = re.search(r"tokens used\s*\n\s*([\d,]+)", log)
    return text, int(m.group(1).replace(",", "")) if m else None


def ask(prompt: str, caller, attempts: int = 3) -> dict:
    t0, err = time.monotonic(), None
    for _ in range(attempts):
        try:
            text, tokens = caller(prompt)
        except (CodexError, subprocess.TimeoutExpired) as e:
            err = str(e)
            time.sleep(30 if "capacity" in err else 5)
            continue
        probs = parse_response(text)
        if probs is not None:
            return {"probs": probs, "error": None, "tokens": tokens, "seconds": round(time.monotonic() - t0, 1)}
        err = f"parse: {text[:200]}"
    return {"probs": None, "error": err, "tokens": None, "seconds": round(time.monotonic() - t0, 1)}


def load_cache(path: Path) -> dict[tuple[int, int], dict]:
    """같은 키가 여러 번 있으면 마지막 기록을 쓴다. 끊겨서 잘린 줄은 건너뛴다."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[(r["idx"], r["rep"])] = r
    return out


def run_llm(tasks: list[tuple[int, int]], X: pd.DataFrame, cache_path: Path, caller=call_codex, workers: int = 4) -> None:
    """캐시에 성공 기록이 없는 (사람, 반복)만 호출해 JSONL에 덧붙인다."""
    done = {k for k, r in load_cache(cache_path).items() if r["probs"] is not None}
    todo = [t for t in tasks if t not in done]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def work(task):
        idx, rep = task
        rec = {"idx": int(idx), "rep": int(rep), **ask(build_prompt(X.loc[idx]), caller)}
        with lock, cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"LLM 호출 {len(todo)}건 (캐시 {len(tasks) - len(todo)}건 건너뜀)", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(work, todo), 1):
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}", flush=True)


def pilot_summary(cache: dict, tasks) -> str:
    recs = [cache[t] for t in tasks if t in cache]
    ok = [r for r in recs if r["probs"]]
    toks = [r["tokens"] for r in ok if r["tokens"]]
    L = [f"성공 {len(ok)}/{len(tasks)}",
         f"호출당 시간 중앙값 {np.median([r['seconds'] for r in recs]):.1f}초" if recs else "기록 없음",
         f"호출당 토큰 중앙값 {int(np.median(toks)):,}" if toks else "토큰 기록 없음"]
    for k in KEYS:
        v = [r["probs"][k] * 100 for r in ok]
        if v:
            L.append(f"{k}: 최소 {min(v):.0f} / 중앙 {np.median(v):.0f} / 최대 {max(v):.0f}")
    L += [f"실패 예: {r['error'][:120]}" for r in recs if not r["probs"]][:3]
    return "\n".join(L)
```

`main`을 교체한다:

```python
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-llm", action="store_true", help="⓪①②③만 계산한다(1단계)")
    ap.add_argument("--limit", type=int, help="앞의 N회만 호출하고 시범 요약만 출력한다(2단계)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    df = load_knhanes()
    spline = _spline_flags()
    preds = {}
    for key in DISEASES:
        print(f"[{key}] 예측 중…", flush=True)
        preds[key] = disease_predictions(df, key, spline[key])

    cache, rep_ids = None, []
    if not args.skip_llm:
        ids = sorted(int(i) for i in set().union(*(p.index for p in preds.values())))
        rng = np.random.default_rng(SEED)
        rep_ids = sorted(int(i) for i in rng.choice(ids, min(REPEAT_N, len(ids)), replace=False))
        tasks = [(i, 0) for i in ids] + [(i, r) for i in rep_ids for r in range(1, REPEATS)]
        X = build_features(df.loc[ids])
        if args.limit:
            tasks = tasks[:args.limit]
        run_llm(tasks, X, CACHE, workers=args.workers)
        cache = load_cache(CACHE)
        if args.limit:
            print(pilot_summary(cache, tasks))
            return

    results = build_results(preds, cache, rep_ids)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "compare-metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    if cache is not None:
        plot_consistency(preds, cache, rep_ids, REPORTS / "figures" / "llm_consistency.png")
    path = REPORTS / f"compare-{results['date']}.md"
    write_report(results, path)
    print(f"보고서: {path}")
```

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run: `.venv/bin/python -m pytest tests/risk_model -q`
Expected: 65 passed

- [ ] **Step 5: 커밋**

```bash
git add training/compare.py tests/risk_model/test_compare.py
git commit -m "feat(compare): call GPT through Codex CLI with resumable JSONL cache

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: 시범 호출 전 한도 기록 요청**

사용자에게 Codex의 5시간·주간 한도 사용률을 확인해 달라고 요청한다. Codex 앱이나 터미널 `codex`의 `/status`에서 볼 수 있다. 값을 받은 뒤 다음 단계로 간다.

- [ ] **Step 7: 시범 호출 20회**

Run: `.venv/bin/python -m training.compare --limit 20`
Expected:
- 마지막에 `성공 N/20`, 호출당 시간·토큰 중앙값, 질환별 응답 분포가 출력된다.
- 보고서 파일은 바뀌지 않는다.

- [ ] **Step 8: 멈추고 보고**

사용자에게 다음을 보고하고, 한도 사용률을 다시 확인해 달라고 요청한다.
- 시범 요약
- 실패 예
- 4,100회 기준 예상 소요 시간: 시간 중앙값 × 4,100 / 4

20회에 쓴 한도 %를 205배 하면 전체 한도 소모를 추정할 수 있다. 이 추정이 한도를 넘으면 사용자와 다시 정한다(AUROC 대상 축소 또는 묶음 호출). 사용자 승인 없이 전체 실행하지 않는다.

---

### Task 5: 전체 실행과 보고서 확정 → 3·4단계

**Files:**
- Create(실행 결과): `training/reports/compare-YYYY-MM-DD.md`(1단계 보고서를 덮어쓴다), `training/reports/compare-metrics.json`, `training/reports/figures/llm_consistency.png`

**Interfaces:**
- Consumes: `python -m training.compare`(Task 4의 `main`)

- [ ] **Step 1: 전체 실행(백그라운드)**

Run(긴 작업이므로 `run_in_background`로 실행한다): `.venv/bin/python -m training.compare`
Expected:
- `LLM 호출 ~4,080건 (캐시 20건 건너뜀)` 다음에 50건마다 진행 상황이 출력된다.
- 마지막에 `보고서: ...`가 출력된다.
- 중간에 끊기면 같은 명령을 다시 실행한다. 캐시에 있는 호출은 건너뛴다.

- [ ] **Step 2: 실패 수 확인**

Run: `.venv/bin/python -c "import json;print(json.load(open('training/reports/compare-metrics.json'))['llm'])"`
Expected: `failed`가 전체의 1% 미만이다. 그 이상이면 같은 명령을 한 번 더 실행해 실패분을 다시 시도한 뒤 확인한다. 그래도 1% 이상이면 보고서에 그대로 남기고 사용자에게 알린다.

- [ ] **Step 3: 테스트 재실행**

Run: `.venv/bin/python -m pytest tests/risk_model -q`
Expected: 65 passed

- [ ] **Step 4: 결과 커밋**

캐시(`training/output/`)는 커밋하지 않는다.

```bash
git add training/reports/compare-*.md training/reports/compare-metrics.json training/reports/figures/llm_consistency.png
git status --short training/output   # 아무것도 나오지 않아야 한다(gitignore)
git commit -m "docs(compare): add full rule/ML/GPT comparison report and consistency figure

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: 멈추고 보고**

사용자에게 다음을 보고하고 결과를 해석한다.
- 요약 표
- ④의 AUROC 차이와 Brier·평균 예측(과대 또는 과소 추정 여부)
- 편차 표
- 그림 파일(SendUserFile)

결과는 나온 그대로 전한다. GPT가 ②와 비슷하더라도 숨기지 않는다.
