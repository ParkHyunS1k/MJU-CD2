# 성인병 위험 예측 모델 재설계 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** KNHANES 2022–2024로 질환별 로지스틱 회귀 4개를 다시 학습하고, 사전에 정한 기준으로 평가한다. 통과한 모델은 `prediction_service`에 반영한다.

**Architecture:**
- `training/` 아래 순수 함수 모듈 네 개를 둔다. 데이터 변환(`features.py`), 모델·번들(`model.py`), 지표(`metrics.py`), 외부 데이터(`external.py`)다.
- 실행 스크립트는 두 개다. `train.py`가 `models/*.pkl`을 쓰고, `evaluate.py`가 `training/reports/`를 쓴다.
- 백엔드 `app/services/prediction_service.py`는 v2 번들을 읽고 기여도를 직접 계산한다. `shap`을 쓰지 않는다. 출력 형식은 지금과 같다.

**Tech Stack:** Python 3.13, scikit-learn 1.8.0, pandas, numpy, joblib, xgboost(비교용), imbalanced-learn(기존 모델 평가용, 학습 전용), pyreadr, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-risk-model-redesign-design.md`

## Global Constraints

- 학습 연령은 19–59세다. 기준값 결정과 주 평가 연령은 25–44세다.
- 입력은 7개이고 순서가 고정이다: `["age", "female", "bmi", "body_fat_percentage", "fh_hypertension", "fh_diabetes", "fh_dyslipidemia"]`. `age`는 반드시 0번 열이다(스플라인이 0번 열에 적용된다).
- 라벨: 고혈압 `HE_HP == 4`, 당뇨 `HE_DM_HbA1c == 3`, 고콜레스테롤 `HE_HCHOL == 1`, 고중성지방 `HE_HTG == 1`.
- 모델: `StandardScaler` + `LogisticRegression(C=1.0, max_iter=2000)`, `random_state=42`. SMOTE와 클래스 가중치는 쓰지 않는다.
- 등급 기준값: `thresh_medium` = 25–44세 OOF의 Youden J 최대 지점, `thresh_high` = 25–44세 음성 예측의 95번째 백분위. `thresh_high <= thresh_medium`이면 실패다.
- 번들 형식은 spec 4.4의 `version: 2` 딕셔너리다. 파일명은 `models/{disease_key}.pkl`이다.
- `feat_kor`에서 `bmi`는 `"BMI"`, `body_fat_percentage`는 `"체지방률"`이다(`trend_service._FEATURE_MAP`과 같은 글자).
- 서버와 학습 모두 `scikit-learn==1.8.0`을 쓴다. 서버 `requirements.txt`에서 `shap`, `imbalanced-learn`을 뺀다.
- `predict_all(inbody, user_profile)`의 시그니처와 출력 키(`disease`, `probability`, `risk_level`, `top_factors[{feature_kor, value, shap_value, direction}]`)는 바꾸지 않는다.
- 통과 기준(spec 5.4)은 결과를 본 뒤 바꾸지 않는다.
- 명령은 모두 저장소 루트에서 `.venv/bin/python`으로 실행한다.

## Review Focus

- **통과하지 못해 pkl이 없는 질환**: 서비스는 그 질환만 건너뛰고 나머지를 반환해야 한다. Task 7 `test_missing_model_file_skips_disease`에서 검사한다.
- **`models/`에 남은 옛 파일**(`hypertension_male.pkl`, version 없는 `cholesterol.pkl`): 성별 파일은 무시하고, 같은 이름의 옛 형식은 로드할 때 바로 실패해야 한다. Task 7 `test_old_gender_files_are_ignored`와 `test_bad_version_raises`에서 검사한다.
- **DB에서 `null`로 오는 가족력**(③ 이후 nullable boolean): 값이 없을 때와 같이 0으로 처리해야 한다. Task 7 `test_family_history_none_or_missing_is_zero`에서 검사한다.
- **60세 이상 또는 극단값**(나이 80, BMI 45, 체지방률 50): 예외 없이 0~1 사이의 확률을 내야 한다. Task 7 `test_extreme_inputs_predict_finite`에서 검사한다.
- **정수로 오는 수치**(Supabase가 `bmi: 24`로 반환): 실수와 같게 처리해야 한다. Task 7 `test_integer_inputs`에서 검사한다.
- **추가로 보완한 것**(사용자 요청): 모델이 하나도 없을 때(`test_no_models_raises`), 앱 시작 시 사전 로드(Task 7 Step 4, Task 9 Step 4), OCR 오독 같은 범위 밖 입력(`test_out_of_range_inputs_raise`)을 다룬다.

---

## File Structure

| 파일 | 상태 | 책임 |
|---|---|---|
| `training/__init__.py` | 생성 | `training`을 패키지로 만든다(빈 파일) |
| `training/analysis/common.py` | 수정 | `KNHANES_COLS`에 가족력 9개 열을 추가한다 |
| `training/features.py` | 생성 | 상수, `build_labels`, `build_features`, `training_rows`, `choose_thresholds` |
| `training/model.py` | 생성 | `build_pipeline`, `feature_groups`, `fit_bundle` |
| `training/metrics.py` | 생성 | `wilson_ci`, `bootstrap_ci`, `paired_diff_ci`, `calibration_slope_intercept`, `risk_levels`, `grade_table`, `sens_spec` |
| `training/external.py` | 생성 | `load_nhanes`, `load_kaggle` |
| `training/train.py` | 생성 | CLI. `models/{key}.pkl` 저장 |
| `training/evaluate.py` | 생성 | CLI. 중첩 CV·비교·외부 검증·보고서 |
| `training/requirements.txt` | 수정 | `matplotlib` 추가 |
| `training/README.md` | 수정 | 새 사용법 |
| `training/retrain_final.py`, `models/*_male.pkl`, `models/*_female.pkl` | 삭제 | 기존 모델 |
| `app/services/prediction_service.py` | 재작성 | v2 번들 로드, 입력 구성, 기여도 계산 |
| `requirements.txt` | 수정 | `scikit-learn==1.8.0`, `shap`·`imbalanced-learn` 제거 |
| `tests/risk_model/conftest.py` | 생성 | 합성 KNHANES 데이터 fixture |
| `tests/risk_model/test_features.py` 등 | 생성 | 단위 테스트 |
| `training/reports/` | 생성(실행 산출물) | 평가 보고서·`metrics.json`·그림 |

---

### Task 0: 실행 환경과 공통 준비

**Files:**
- Create: `training/__init__.py`
- Modify: `training/analysis/common.py` (`KNHANES_COLS`)
- Modify: `training/requirements.txt`

**Interfaces:**
- Produces: `training.analysis.common.load_knhanes()`가 `HE_HPfh1..3`, `HE_DMfh1..3`, `HE_HLfh1..3` 열을 포함한 DataFrame을 반환한다. 열은 `age`, `sex`, `yr`, `HE_BMI`, `BIA_PBF`, `BIA_LRA..LLL`, 라벨 열 등이다.

- [ ] **Step 1: 가상환경 만들기**

```bash
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt -r training/requirements.txt matplotlib==3.10.6
.venv/bin/python -c "import sklearn, imblearn, xgboost, pyreadr, matplotlib; print(sklearn.__version__)"
```
Expected: `1.8.0`

- [ ] **Step 2: 패키지 파일과 의존성 추가**

`training/__init__.py`를 빈 파일로 만든다. `training/requirements.txt` 맨 끝에 한 줄을 추가한다.

```
matplotlib==3.10.6  # evaluate.py 보정 곡선
```

- [ ] **Step 3: `common.py`에 가족력 열 추가**

`training/analysis/common.py`의 `KNHANES_COLS` 리스트에서 마지막 줄 `"HE_HCHOL", "HE_HTG",` 다음에 한 줄을 추가한다.

```python
    "HE_HPfh1", "HE_HPfh2", "HE_HPfh3", "HE_DMfh1", "HE_DMfh2", "HE_DMfh3", "HE_HLfh1", "HE_HLfh2", "HE_HLfh3",
```

- [ ] **Step 4: 로드 확인**

```bash
.venv/bin/python -c "from training.analysis.common import load_knhanes; d=load_knhanes(); print(len(d), d[['HE_HPfh1','HE_DMfh3','HE_HLfh2']].notna().mean().round(2).to_dict())"
```
Expected: `13785` 근처의 행 수와 1.0에 가까운 비율

- [ ] **Step 5: 기존 모델 백업 (기존 모델 평가용, Git 제외 폴더)**

```bash
mkdir -p training/output/old_models && cp models/*.pkl training/output/old_models/ && ls training/output/old_models
```
Expected: 7개 파일

- [ ] **Step 6: Commit**

```bash
git add training/__init__.py training/analysis/common.py training/requirements.txt
git commit -m "chore(training): package training dir, add family history columns"
```

---

### Task 1: `features.py` — 라벨·입력·기준값

**Files:**
- Create: `training/features.py`
- Create: `tests/risk_model/conftest.py`
- Test: `tests/risk_model/test_features.py`

**Interfaces:**
- Consumes: KNHANES 열(`age`, `sex`, `HE_BMI`, `BIA_PBF`, `HE_*fh1..3`, `HE_HP`, `HE_DM_HbA1c`, `HE_HCHOL`, `HE_HTG`, `yr`)
- Produces:
  - `DISEASES: dict[str, str]`: `{"hypertension": "고혈압", "cholesterol": "고콜레스테롤", "diabetes": "당뇨", "triglyceride": "고중성지방"}`(이 순서)
  - `FEATURES: list[str]`, `BASE_FEATURES: list[str]`(앞 4개), `FEAT_KOR: dict[str, str]`
  - `TRAIN_AGE = (19, 59)`, `THRESHOLD_AGE = (25, 44)`
  - `build_labels(df) -> dict[str, pd.Series]`(값은 0.0/1.0/NaN)
  - `build_features(df) -> pd.DataFrame`(열 = `FEATURES`)
  - `training_rows(df, disease_key, features=FEATURES, age=TRAIN_AGE) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]`. 반환값은 `X`(열 = features), `y`(float), `info`(열 `age`, `sex`, `yr`, 인덱스는 원래 df와 같음)다.
  - `choose_thresholds(y, p) -> tuple[float, float]`. `(medium, high)`를 반환하고, `high <= medium`이면 `ValueError`를 던진다.

- [ ] **Step 1: 합성 데이터 fixture 작성**

`tests/risk_model/conftest.py`:

```python
import numpy as np
import pandas as pd
import pytest

FH_COLS = [f"HE_{d}fh{i}" for d in ("HP", "DM", "HL") for i in (1, 2, 3)]


@pytest.fixture
def knhanes_like():
    """KNHANES 열 이름을 가진 합성 데이터. 나이·BMI가 높을수록 질환 확률이 오른다."""
    rng = np.random.default_rng(0)
    n = 3000
    age = rng.integers(19, 60, n).astype(float)
    sex = rng.integers(1, 3, n).astype(float)
    bmi = rng.normal(24, 3.5, n)
    pbf = rng.normal(27, 7, n)

    def label(shift):
        lin = -9 + 0.07 * age + 0.18 * bmi + 0.02 * pbf + shift
        return (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)

    fh = {c: rng.choice([0.0, 1.0, 9.0], n, p=[0.7, 0.2, 0.1]) for c in FH_COLS}
    return pd.DataFrame({
        "age": age, "sex": sex, "HE_BMI": bmi, "BIA_PBF": pbf, **fh,
        "HE_HP": np.where(label(0.0) == 1, 4.0, rng.choice([1.0, 2.0, 3.0], n)),
        "HE_DM_HbA1c": np.where(label(-1.0) == 1, 3.0, rng.choice([1.0, 2.0], n)),
        "HE_HCHOL": label(0.5),
        "HE_HTG": label(-0.5),
        "yr": rng.choice([2022.0, 2023.0, 2024.0], n),
    })
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/risk_model/test_features.py`:

```python
import numpy as np
import pandas as pd
import pytest

from training.features import (
    BASE_FEATURES, DISEASES, FEATURES, build_features, build_labels, choose_thresholds, training_rows,
)


def _labels_df(**cols):
    base = {"HE_HP": np.nan, "HE_DM_HbA1c": np.nan, "HE_HCHOL": np.nan, "HE_HTG": np.nan}
    base.update(cols)
    return pd.DataFrame(base)


def test_constants():
    assert list(DISEASES) == ["hypertension", "cholesterol", "diabetes", "triglyceride"]
    assert FEATURES[0] == "age" and len(FEATURES) == 7
    assert BASE_FEATURES == ["age", "female", "bmi", "body_fat_percentage"]


def test_hypertension_label_excludes_prehypertension():
    df = _labels_df(HE_HP=[1, 2, 3, 4, np.nan])
    got = build_labels(df)["hypertension"].tolist()
    assert got[:4] == [0.0, 0.0, 0.0, 1.0] and np.isnan(got[4])


def test_diabetes_label_uses_hba1c_prevalence_code():
    df = _labels_df(HE_DM_HbA1c=[1, 2, 3, np.nan])
    got = build_labels(df)["diabetes"].tolist()
    assert got[:3] == [0.0, 0.0, 1.0] and np.isnan(got[3])


def test_lipid_labels_drop_non_binary_codes():
    df = _labels_df(HE_HCHOL=[0, 1, 9, np.nan], HE_HTG=[1, 0, 8, np.nan])
    lab = build_labels(df)
    assert lab["cholesterol"].tolist()[:2] == [0.0, 1.0] and lab["cholesterol"].isna().tolist()[2:] == [True, True]
    assert lab["triglyceride"].tolist()[:2] == [1.0, 0.0] and lab["triglyceride"].isna().tolist()[2:] == [True, True]


def test_build_features_family_history_and_sex():
    df = pd.DataFrame({
        "age": [30, 40], "sex": [1, 2], "HE_BMI": [22.0, 27.0], "BIA_PBF": [18.0, 33.0],
        "HE_HPfh1": [9, 0], "HE_HPfh2": [1, 0], "HE_HPfh3": [0, 8],
        "HE_DMfh1": [0, 0], "HE_DMfh2": [0, 1], "HE_DMfh3": [9, 0],
        "HE_HLfh1": [9, 9], "HE_HLfh2": [9, 9], "HE_HLfh3": [9, 9],
    })
    X = build_features(df)
    assert list(X.columns) == FEATURES
    assert X.female.tolist() == [0.0, 1.0]
    assert X.fh_hypertension.tolist() == [1.0, 0.0]
    assert X.fh_diabetes.tolist() == [0.0, 1.0]
    assert X.fh_dyslipidemia.tolist() == [0.0, 0.0]


def test_training_rows_filters_age_and_missing(knhanes_like):
    df = knhanes_like.copy()
    df.loc[0, "age"] = 60
    df.loc[1, "HE_BMI"] = np.nan
    df.loc[2, "HE_HP"] = np.nan
    X, y, info = training_rows(df, "hypertension")
    assert {0, 1, 2}.isdisjoint(X.index)
    assert info.age.between(19, 59).all()
    assert len(X) == len(y) == len(info) and list(info.columns) == ["age", "sex", "yr"]
    Xb, _, _ = training_rows(df, "diabetes", features=BASE_FEATURES)
    assert list(Xb.columns) == BASE_FEATURES


def test_choose_thresholds_youden_and_negative_quantile():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.2).astype(float)
    p = np.clip(rng.normal(0.15 + 0.25 * y, 0.08), 0.001, 0.999)
    medium, high = choose_thresholds(y, p)
    cands = np.sort(np.unique(p))[::-1]
    j = [(p[y == 1] >= t).mean() - (p[y == 0] >= t).mean() for t in cands]
    assert medium == pytest.approx(cands[int(np.argmax(j))])
    assert high == pytest.approx(np.quantile(p[y == 0], 0.95))


def test_choose_thresholds_rejects_inverted():
    with pytest.raises(ValueError):
        choose_thresholds(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.1, 0.9, 0.2, 0.95]))
```

- [ ] **Step 3: 실패 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_features.py -q`
Expected: `ModuleNotFoundError: No module named 'training.features'`

- [ ] **Step 4: 구현**

`training/features.py`:

```python
"""KNHANES 원자료 → 학습용 라벨·입력, 등급 기준값. train.py와 evaluate.py가 함께 쓴다.
설계: docs/superpowers/specs/2026-10-12-risk-model-redesign-design.md 3절, 4.3절."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

DISEASES = {
    "hypertension": "고혈압",
    "cholesterol": "고콜레스테롤",
    "diabetes": "당뇨",
    "triglyceride": "고중성지방",
}
FEATURES = ["age", "female", "bmi", "body_fat_percentage",
            "fh_hypertension", "fh_diabetes", "fh_dyslipidemia"]
BASE_FEATURES = FEATURES[:4]  # 가족력 없는 외부 검증용
FEAT_KOR = {
    "age": "나이",
    "female": "성별",
    "bmi": "BMI",                     # trend_service._FEATURE_MAP과 같은 글자
    "body_fat_percentage": "체지방률",  # 〃
    "fh_hypertension": "고혈압 가족력",
    "fh_diabetes": "당뇨 가족력",
    "fh_dyslipidemia": "고지혈증 가족력",
}
TRAIN_AGE = (19, 59)
THRESHOLD_AGE = (25, 44)


def _binary(s: pd.Series) -> pd.Series:
    return s.astype(float).where(s.isin([0, 1]))


def build_labels(df: pd.DataFrame) -> dict[str, pd.Series]:
    """HE_HP 4=고혈압(3은 전단계), HE_DM_HbA1c 3=당뇨(진단·치료 포함)."""
    return {
        "hypertension": (df.HE_HP == 4).astype(float).where(df.HE_HP.notna()),
        "cholesterol": _binary(df.HE_HCHOL),
        "diabetes": (df.HE_DM_HbA1c == 3).astype(float).where(df.HE_DM_HbA1c.notna()),
        "triglyceride": _binary(df.HE_HTG),
    }


def _any_family(df: pd.DataFrame, prefix: str) -> pd.Series:
    """부·모·형제 중 한 명이라도 1이면 1. 모름(9)·비해당(8)은 0."""
    return df[[f"{prefix}{i}" for i in (1, 2, 3)]].eq(1).any(axis=1).astype(float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "age": df.age.astype(float),
        "female": (df.sex == 2).astype(float),
        "bmi": df.HE_BMI.astype(float),
        "body_fat_percentage": df.BIA_PBF.astype(float),
        "fh_hypertension": _any_family(df, "HE_HPfh"),
        "fh_diabetes": _any_family(df, "HE_DMfh"),
        "fh_dyslipidemia": _any_family(df, "HE_HLfh"),
    }, index=df.index)


def training_rows(df, disease_key, features=FEATURES, age=TRAIN_AGE):
    """라벨·입력이 모두 있고 나이가 범위 안인 행. (X, y, info[age, sex, yr])."""
    X = build_features(df)[list(features)]
    y = build_labels(df)[disease_key]
    m = y.notna() & X.notna().all(axis=1) & df.age.between(*age)
    return X[m], y[m].to_numpy(dtype=float), df.loc[m, ["age", "sex", "yr"]]


def choose_thresholds(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """보통 = Youden J 최대 지점, 높음 = 음성 예측의 95번째 백분위."""
    fpr, tpr, th = roc_curve(y, p)
    medium = float(th[np.argmax(tpr - fpr)])
    high = float(np.quantile(p[y == 0], 0.95))
    if high <= medium:
        raise ValueError(f"thresh_high({high:.4f}) <= thresh_medium({medium:.4f})")
    return medium, high
```

- [ ] **Step 5: 통과 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_features.py -q`
Expected: `8 passed`

- [ ] **Step 6: Commit**

```bash
git add training/features.py tests/risk_model/conftest.py tests/risk_model/test_features.py
git commit -m "feat(training): add label, feature and threshold builders"
```

---

### Task 2: `model.py` — 파이프라인과 v2 번들

**Files:**
- Create: `training/model.py`
- Test: `tests/risk_model/test_model.py`

**Interfaces:**
- Consumes: `training.features.{DISEASES, FEAT_KOR, THRESHOLD_AGE, TRAIN_AGE, choose_thresholds}`
- Produces:
  - `SEED = 42`
  - `build_pipeline(age_spline: bool) -> sklearn.pipeline.Pipeline`. 단계 이름은 `["spline"]? + ["scaler", "clf"]`다.
  - `feature_groups(pipe, features) -> dict[str, list[int]]`. 변환 후 열 번호를 원래 변수로 묶는다.
  - `fit_bundle(X: pd.DataFrame, y: np.ndarray, age: np.ndarray, disease_key: str, age_spline: bool) -> dict`. spec 4.4의 v2 번들이다. `X`의 첫 열은 `age`여야 한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/risk_model/test_model.py`:

```python
import numpy as np
import pytest

from training.features import FEATURES, training_rows
from training.model import build_pipeline, feature_groups, fit_bundle


def test_pipeline_steps():
    assert [n for n, _ in build_pipeline(False).steps] == ["scaler", "clf"]
    assert [n for n, _ in build_pipeline(True).steps] == ["spline", "scaler", "clf"]


@pytest.mark.parametrize("spline", [False, True])
def test_fit_bundle_contract(knhanes_like, spline):
    X, y, info = training_rows(knhanes_like, "hypertension")
    b = fit_bundle(X, y, info.age.to_numpy(), "hypertension", spline)
    assert b["version"] == 2 and b["disease_key"] == "hypertension" and b["target"] == "고혈압"
    assert b["features"] == FEATURES
    assert b["feat_kor"]["bmi"] == "BMI" and b["feat_kor"]["body_fat_percentage"] == "체지방률"
    n_cols = b["model"][:-1].transform(X.to_numpy()).shape[1]
    assert len(b["background"]) == n_cols
    cols = sorted(i for idx in b["feature_groups"].values() for i in idx)
    assert cols == list(range(n_cols))
    assert set(b["feature_groups"]) == set(FEATURES)
    assert 0 < b["thresh_medium"] < b["thresh_high"] < 1
    assert b["meta"]["n_train"] == len(y) and b["meta"]["n_pos"] == int(y.sum())
    assert b["meta"]["age_spline"] is spline and b["meta"]["train_age"] == [19, 59]


def test_spline_groups_age_columns(knhanes_like):
    X, y, info = training_rows(knhanes_like, "diabetes")
    pipe = build_pipeline(True).fit(X.to_numpy(), y)
    groups = feature_groups(pipe, FEATURES)
    assert groups["age"] == [0, 1, 2, 3, 4, 5]
    assert groups["female"] == [6]
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_model.py -q`
Expected: `ModuleNotFoundError: No module named 'training.model'`

- [ ] **Step 3: 구현**

`training/model.py`:

```python
"""로지스틱 회귀 파이프라인과 v2 모델 번들. 설계: spec 4.1–4.4."""
from datetime import datetime, timezone

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from training.features import DISEASES, FEAT_KOR, THRESHOLD_AGE, TRAIN_AGE, choose_thresholds

SEED = 42


def build_pipeline(age_spline: bool) -> Pipeline:
    """age_spline이면 0번 열(age)에 3차 스플라인(매듭 4개)을 적용한다."""
    steps = []
    if age_spline:
        steps.append(("spline", ColumnTransformer(
            [("age", SplineTransformer(n_knots=4, degree=3), [0])], remainder="passthrough")))
    steps += [("scaler", StandardScaler()),
              ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))]
    return Pipeline(steps)


def feature_groups(pipe: Pipeline, features: list[str]) -> dict[str, list[int]]:
    """변환 후 열 번호 → 원래 변수. ColumnTransformer는 스플라인 열을 먼저, 나머지를 뒤에 둔다."""
    if "spline" not in pipe.named_steps:
        return {f: [i] for i, f in enumerate(features)}
    n_age = pipe.named_steps["spline"].named_transformers_["age"].n_features_out_
    groups = {features[0]: list(range(n_age))}
    for i, f in enumerate(features[1:]):
        groups[f] = [n_age + i]
    return groups


def fit_bundle(X, y, age, disease_key, age_spline) -> dict:
    """OOF 예측으로 25–44세 기준값을 정한 뒤 전체로 다시 학습해 번들을 만든다."""
    features = list(X.columns)
    Xa = X.to_numpy(dtype=float)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(build_pipeline(age_spline), Xa, y, cv=cv, method="predict_proba")[:, 1]
    tgt = (age >= THRESHOLD_AGE[0]) & (age <= THRESHOLD_AGE[1])
    medium, high = choose_thresholds(y[tgt], oof[tgt])
    pipe = build_pipeline(age_spline).fit(Xa, y)
    return {
        "version": 2,
        "disease_key": disease_key,
        "target": DISEASES[disease_key],
        "model": pipe,
        "features": features,
        "feat_kor": {f: FEAT_KOR[f] for f in features},
        "feature_groups": feature_groups(pipe, features),
        "background": pipe[:-1].transform(Xa).mean(axis=0).tolist(),
        "thresh_medium": medium,
        "thresh_high": high,
        "meta": {
            "data": "KNHANES 2022-2024",
            "train_age": list(TRAIN_AGE),
            "threshold_age": list(THRESHOLD_AGE),
            "n_train": int(len(y)),
            "n_pos": int(np.sum(y)),
            "age_spline": bool(age_spline),
            "sklearn": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_model.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add training/model.py tests/risk_model/test_model.py
git commit -m "feat(training): add logistic pipeline and v2 bundle builder"
```

---

### Task 3: `metrics.py` — 평가 지표

**Files:**
- Create: `training/metrics.py`
- Test: `tests/risk_model/test_metrics.py`

**Interfaces:**
- Produces:
  - `LEVELS = ("low", "medium", "high")`
  - `wilson_ci(k: int, n: int) -> tuple[float, float]`
  - `bootstrap_ci(y, p, metric, n_boot=1000, seed=42) -> tuple[float, float, float]`. `(점추정, 하한, 상한)`이다.
  - `paired_diff_ci(y, p_new, p_ref, metric, n_boot=1000, seed=42) -> tuple[float, float, float]`. `metric(new) − metric(ref)`이다.
  - `calibration_slope_intercept(y, p) -> tuple[float, float]`
  - `risk_levels(p, medium, high) -> np.ndarray[object]`. 값은 `"low"/"medium"/"high"`다.
  - `grade_table(y, level) -> list[dict]`. 키는 `level, n, share, prevalence, prevalence_ci`다.
  - `sens_spec(y, level) -> dict`. 키는 `sensitivity, sensitivity_ci, specificity, specificity_ci, high_ppv`다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/risk_model/test_metrics.py`:

```python
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from training.metrics import (
    bootstrap_ci, calibration_slope_intercept, grade_table, paired_diff_ci, risk_levels, sens_spec, wilson_ci,
)


def test_wilson_known_value():
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)


def test_calibrated_predictions_have_slope_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.5, 20000)
    y = (rng.random(20000) < p).astype(float)
    slope, intercept = calibration_slope_intercept(y, p)
    assert 0.9 < slope < 1.1 and abs(intercept) < 0.15


def test_bootstrap_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.3).astype(float)
    p = np.clip(0.3 + 0.2 * (y - 0.3) + rng.normal(0, 0.1, 400), 0.01, 0.99)
    est, lo, hi = bootstrap_ci(y, p, roc_auc_score, n_boot=200)
    assert lo <= est <= hi
    assert bootstrap_ci(y, p, roc_auc_score, n_boot=200) == (est, lo, hi)


def test_paired_diff_of_identical_predictions_is_zero():
    rng = np.random.default_rng(2)
    y = (rng.random(300) < 0.4).astype(float)
    p = rng.random(300)
    assert paired_diff_ci(y, p, p, roc_auc_score, n_boot=100) == (0.0, 0.0, 0.0)


def test_risk_levels_boundaries():
    got = risk_levels(np.array([0.05, 0.1, 0.2, 0.3, 0.9]), 0.1, 0.3).tolist()
    assert got == ["low", "medium", "medium", "high", "high"]


def test_grade_table_and_sens_spec():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=float)
    level = np.array(["low", "low", "low", "low", "medium", "medium", "high", "high"], dtype=object)
    rows = {r["level"]: r for r in grade_table(y, level)}
    assert rows["low"]["n"] == 4 and rows["low"]["prevalence"] == 0.25
    assert rows["high"]["prevalence"] == 1.0 and rows["medium"]["share"] == 0.25
    s = sens_spec(y, level)
    assert s["sensitivity"] == 0.75 and s["specificity"] == 0.75 and s["high_ppv"] == 1.0
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_metrics.py -q`
Expected: `ModuleNotFoundError: No module named 'training.metrics'`

- [ ] **Step 3: 구현**

`training/metrics.py`:

```python
"""평가 지표와 신뢰구간. 설계: spec 5.1."""
import numpy as np
from sklearn.linear_model import LogisticRegression

LEVELS = ("low", "medium", "high")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def _resample(y, n_boot, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():  # 한 클래스만 뽑히면 AUROC를 못 구한다
            yield i


def bootstrap_ci(y, p, metric, n_boot=1000, seed=42) -> tuple[float, float, float]:
    vals = [metric(y[i], p[i]) for i in _resample(y, n_boot, seed)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(metric(y, p)), float(lo), float(hi))


def paired_diff_ci(y, p_new, p_ref, metric, n_boot=1000, seed=42) -> tuple[float, float, float]:
    vals = [metric(y[i], p_new[i]) - metric(y[i], p_ref[i]) for i in _resample(y, n_boot, seed)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(metric(y, p_new) - metric(y, p_ref)), float(lo), float(hi))


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """logit(p)에 대한 무규제 로지스틱 회귀. 기울기 1, 절편 0이 이상적이다."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(penalty=None, max_iter=1000).fit(lp, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def risk_levels(p, medium, high) -> np.ndarray:
    return np.where(p >= high, "high", np.where(p >= medium, "medium", "low")).astype(object)


def grade_table(y, level) -> list[dict]:
    rows = []
    for lv in LEVELS:
        m = level == lv
        k, n = int(y[m].sum()), int(m.sum())
        rows.append({"level": lv, "n": n, "share": n / len(y),
                     "prevalence": k / n if n else None,
                     "prevalence_ci": wilson_ci(k, n) if n else None})
    return rows


def sens_spec(y, level) -> dict:
    """민감도 = 양성 중 보통 이상, 특이도 = 음성 중 낮음, high_ppv = 높음 중 양성 비율."""
    pos, neg = y == 1, y == 0
    tp, tn = int((level[pos] != "low").sum()), int((level[neg] == "low").sum())
    hi = level == "high"
    return {"sensitivity": tp / int(pos.sum()), "sensitivity_ci": wilson_ci(tp, int(pos.sum())),
            "specificity": tn / int(neg.sum()), "specificity_ci": wilson_ci(tn, int(neg.sum())),
            "high_ppv": float(y[hi].mean()) if hi.any() else None}
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_metrics.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add training/metrics.py tests/risk_model/test_metrics.py
git commit -m "feat(training): add evaluation metrics with confidence intervals"
```

---

### Task 4: `external.py` — NHANES·Kaggle 로더

**Files:**
- Create: `training/external.py`
- Test: `tests/risk_model/test_external.py`

**Interfaces:**
- Consumes: `training.analysis.common.EXT_DIR`, 파일 `training/data/ext/nhanes/*.rda`, `training/data/ext/bodyPerformance.csv`
- Produces:
  - `load_nhanes() -> pd.DataFrame`. 열은 `BASE_FEATURES` + `hypertension, diabetes, cholesterol, triglyceride`(0/1/NaN)이고 20–49세다.
  - `load_kaggle() -> pd.DataFrame`. 열은 `BASE_FEATURES` + `hypertension`(측정 140/90 이상, 0/1)이다.
  - 파일이 없으면 `FileNotFoundError`를, `pyreadr`가 없으면 `ImportError`를 던진다.

- [ ] **Step 1: 실패하는 테스트 작성 (데이터 파일이 없으면 skip)**

`tests/risk_model/test_external.py`:

```python
import pytest

from training.analysis.common import EXT_DIR
from training.features import BASE_FEATURES

LABELS = ["hypertension", "diabetes", "cholesterol", "triglyceride"]


@pytest.mark.skipif(not (EXT_DIR / "nhanes" / "bix_c.rda").exists(), reason="NHANES 파일 없음")
def test_load_nhanes():
    from training.external import load_nhanes
    d = load_nhanes()
    assert list(d.columns) == BASE_FEATURES + LABELS
    assert 1500 < len(d) <= 1668 and d.age.between(20, 49).all()
    for c in LABELS:
        assert set(d[c].dropna().unique()) <= {0.0, 1.0}
    assert d.diabetes.sum() >= 70


@pytest.mark.skipif(not (EXT_DIR / "bodyPerformance.csv").exists(), reason="Kaggle 파일 없음")
def test_load_kaggle():
    from training.external import load_kaggle
    d = load_kaggle()
    assert list(d.columns) == BASE_FEATURES + ["hypertension"]
    assert 13300 < len(d) < 13393
    assert d.bmi.between(10, 60).all()
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_external.py -q`
Expected: `ModuleNotFoundError: No module named 'training.external'`

- [ ] **Step 3: 구현**

`training/external.py`:

```python
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
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_external.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add training/external.py tests/risk_model/test_external.py
git commit -m "feat(training): add NHANES and Kaggle external validation loaders"
```

---

### Task 5: `train.py` — 모델 저장 CLI

**Files:**
- Create: `training/train.py`
- Test: `tests/risk_model/test_train_cli.py`

**Interfaces:**
- Consumes: `load_knhanes`, `DISEASES`, `training_rows`, `fit_bundle`
- Produces:
  - CLI `python -m training.train [--age-spline KEY ...] [--only KEY ...]`
  - `MODELS_DIR`(모듈 변수, 테스트에서 교체 가능)
  - `main(argv: list[str] | None = None) -> None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/risk_model/test_train_cli.py`:

```python
import joblib

import training.train as train


def test_main_writes_selected_bundles(knhanes_like, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "load_knhanes", lambda: knhanes_like)
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    train.main(["--only", "hypertension", "diabetes", "--age-spline", "diabetes"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["diabetes.pkl", "hypertension.pkl"]
    assert joblib.load(tmp_path / "hypertension.pkl")["meta"]["age_spline"] is False
    assert joblib.load(tmp_path / "diabetes.pkl")["meta"]["age_spline"] is True
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_train_cli.py -q`
Expected: `ModuleNotFoundError: No module named 'training.train'`

- [ ] **Step 3: 구현**

`training/train.py`:

```python
"""질병 위험 모델을 학습해 models/{disease}.pkl로 저장한다.
실행: python -m training.train [--age-spline KEY ...] [--only KEY ...]
설계: docs/superpowers/specs/2026-10-12-risk-model-redesign-design.md 4절."""
import argparse
from pathlib import Path

import joblib

from training.analysis.common import load_knhanes
from training.features import DISEASES, training_rows
from training.model import fit_bundle

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--age-spline", nargs="*", default=[], choices=list(DISEASES),
                    help="나이 스플라인을 쓸 질환(evaluate.py 결과로 정한다)")
    ap.add_argument("--only", nargs="*", choices=list(DISEASES),
                    help="저장할 질환. 생략하면 전부. 통과 기준(spec 5.4)을 넘은 질환만 넣는다")
    args = ap.parse_args(argv)

    df = load_knhanes()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for key in args.only or list(DISEASES):
        X, y, info = training_rows(df, key)
        b = fit_bundle(X, y, info.age.to_numpy(), key, key in args.age_spline)
        joblib.dump(b, MODELS_DIR / f"{key}.pkl")
        print(f"{key:13s} n={b['meta']['n_train']:5d} 양성={b['meta']['n_pos']:5d} "
              f"spline={b['meta']['age_spline']} 보통≥{b['thresh_medium']:.3f} 높음≥{b['thresh_high']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_train_cli.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add training/train.py tests/risk_model/test_train_cli.py
git commit -m "feat(training): add train CLI writing v2 model bundles"
```

---

### Task 6: `evaluate.py` — 중첩 CV·비교·외부 검증·보고서

**Files:**
- Create: `training/evaluate.py`
- Test: `tests/risk_model/test_evaluate.py`

**Interfaces:**
- Consumes: Task 1–4의 함수 전부, `load_knhanes`
- Produces:
  - `nested_predictions(X, y, age, disease_key, age_spline) -> tuple[np.ndarray, np.ndarray]`. `(p, level)`이고 모든 행이 바깥 폴드 평가 예측이다.
  - `outer_predictions(X, y, make_model) -> np.ndarray`
  - `summarize(y, p, level=None) -> dict`
  - `check_criteria(main: dict, auroc_diff: tuple) -> dict[str, bool]`. 키는 `calibration, contribution, grades`다.
  - `evaluate_disease(df, key, old_dir, fig_dir) -> dict`. 키는 `main{linear,spline}, subgroups{linear,spline}, criteria{linear,spline}, spline_adopt, baseline_age_sex, compare, old_model?`이다.
  - `temporal(df, key, age_spline) -> dict`, `external(df, adopt: dict[str, bool]) -> dict`
  - `write_report(results: dict, path: Path) -> None`
  - CLI `python -m training.evaluate` → `training/reports/eval-YYYY-MM-DD.md`, `metrics.json`, `figures/calibration_{key}.png`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/risk_model/test_evaluate.py`:

```python
import numpy as np

from training.evaluate import (
    check_criteria, evaluate_disease, nested_predictions, temporal, write_report,
)
from training.features import training_rows


def test_nested_predictions_cover_all_rows(knhanes_like):
    X, y, info = training_rows(knhanes_like, "hypertension")
    p, level = nested_predictions(X, y, info.age.to_numpy(), "hypertension", False)
    assert p.shape == y.shape and ((p > 0) & (p < 1)).all()
    assert set(level) <= {"low", "medium", "high"}


def _main(mean_pred, prev_ci, slope, prevs):
    return {"mean_pred": mean_pred, "prevalence_ci": prev_ci, "cal_slope": slope,
            "grades": [{"level": lv, "prevalence": pv} for lv, pv in zip(("low", "medium", "high"), prevs)]}


def test_check_criteria():
    ok = check_criteria(_main(0.10, (0.08, 0.12), 1.0, (0.02, 0.1, 0.3)), (0.01, -0.01, 0.03))
    assert ok == {"calibration": True, "contribution": True, "grades": True}
    bad = check_criteria(_main(0.20, (0.08, 0.12), 1.3, (0.02, 0.3, 0.1)), (-0.01, -0.03, 0.01))
    assert bad == {"calibration": False, "contribution": False, "grades": False}
    missing = check_criteria(_main(0.10, (0.08, 0.12), 1.0, (0.02, 0.1, None)), (0.01, 0.0, 0.02))
    assert missing["grades"] is False


def test_evaluate_disease_and_report(knhanes_like, tmp_path):
    r = evaluate_disease(knhanes_like, "hypertension", old_dir=None, fig_dir=tmp_path / "figures")
    assert set(r["main"]) == {"linear", "spline"} and isinstance(r["spline_adopt"], bool)
    assert set(r["compare"]) == {"spline", "rf", "xgb"}
    assert "old_model" not in r
    assert (tmp_path / "figures" / "calibration_hypertension.png").exists()
    r["temporal"] = temporal(knhanes_like, "hypertension", r["spline_adopt"])
    results = {"date": "2026-10-12", "diseases": {"hypertension": r}, "external": {"skipped": "test"}}
    out = tmp_path / "eval.md"
    write_report(results, out)
    text = out.read_text(encoding="utf-8")
    assert "## 요약" in text and "고혈압" in text and "건너뜀: test" in text
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_evaluate.py -q`
Expected: `ModuleNotFoundError: No module named 'training.evaluate'`

- [ ] **Step 3: 구현**

`training/evaluate.py`:

```python
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
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_evaluate.py -q`
Expected: `3 passed` (RandomForest 때문에 수십 초가 걸릴 수 있다)

- [ ] **Step 5: Commit**

```bash
git add training/evaluate.py tests/risk_model/test_evaluate.py
git commit -m "feat(training): add nested-CV evaluation, comparisons and report"
```

---

### Task 7: `prediction_service` 재작성과 서버 의존성

**Files:**
- Modify (전체 교체): `app/services/prediction_service.py`
- Modify: `main.py` (시작 시 모델 로드)
- Modify: `requirements.txt`
- Test: `tests/risk_model/test_prediction_service.py`

**Interfaces:**
- Consumes: `models/{key}.pkl` v2 번들(Task 2 형식). 테스트는 `training.model.fit_bundle`로 번들을 만든다.
- Produces:
  - `predict_all(inbody: dict, user_profile: dict) -> list[dict]`(형식은 기존과 같다)
  - `load_models() -> list[dict]`. 있는 모델을 모두 불러오고, 하나도 없으면 `ModelBundleError`를 던진다. `main.py`가 앱 시작 시 호출한다.
  - `ModelBundleError(RuntimeError)`
  - 모듈 변수 `MODELS_DIR`, `DISEASES`, `FEATURES`, `BMI_RANGE = (12.0, 60.0)`, `PBF_RANGE = (3.0, 70.0)`, 내부 함수 `_load_bundle`(lru_cache), `_contributions(bundle, x) -> dict[str, float]`

**빈틈 보완(사용자 요청)**
- 모델이 하나도 없으면 빈 리스트(호출부의 잘못된 422 안내) 대신 `ModelBundleError`를 던진다.
- 앱 시작 시 `load_models()`를 호출한다. 그러면 형식이 틀린 모델은 배포가 실패하는 단계에서 드러난다.
- BMI·체지방률이 범위를 벗어나면(OCR 오독 등) 예측하지 않고 `ValueError`를 던진다. 호출부가 400으로 응답한다. 범위의 근거는 KNHANES 19–59세 실제 범위(BMI 14.5–48.0, 체지방률 6.7–54.6)에 여유를 둔 것이다.
- 주의: 이 작업을 커밋한 뒤부터 Task 9까지는 `models/`에 옛 `cholesterol.pkl`이 있어서 로컬에서 앱이 시작되지 않는다. Task 9에서 해소된다. push는 Task 9 이후에만 한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/risk_model/test_prediction_service.py`:

```python
from datetime import date

import joblib
import numpy as np
import pytest

import app.services.prediction_service as ps
from training.features import DISEASES, training_rows
from training.model import fit_bundle

AGE35 = date.today().year - 35
PROFILE = {"gender": "male", "birth_year": AGE35}
INBODY = {"bmi": 27.0, "body_fat_percentage": 28.0}


def _write_bundles(df, directory, spline=False):
    for key in DISEASES:
        X, y, info = training_rows(df, key)
        joblib.dump(fit_bundle(X, y, info.age.to_numpy(), key, spline), directory / f"{key}.pkl")


@pytest.fixture
def models_dir(knhanes_like, tmp_path, monkeypatch):
    _write_bundles(knhanes_like, tmp_path)
    monkeypatch.setattr(ps, "MODELS_DIR", tmp_path)
    ps._load_bundle.cache_clear()
    yield tmp_path
    ps._load_bundle.cache_clear()


def test_output_shape(models_dir):
    res = ps.predict_all(INBODY, PROFILE)
    assert [r["disease"] for r in res] == ["고혈압", "고콜레스테롤", "당뇨", "고중성지방"]
    for r in res:
        assert set(r) == {"disease", "probability", "risk_level", "top_factors"}
        assert 0 < r["probability"] < 1 and r["risk_level"] in {"low", "medium", "high"}
        assert 1 <= len(r["top_factors"]) <= 3
        for f in r["top_factors"]:
            assert set(f) == {"feature_kor", "value", "shap_value", "direction"}
            assert f["feature_kor"] != "성별"
            assert f["direction"] == ("up" if f["shap_value"] > 0 else "down")


def test_risk_level_matches_thresholds(models_dir):
    for r in ps.predict_all(INBODY, PROFILE):
        key = next(k for k, v in DISEASES.items() if v == r["disease"])
        b = joblib.load(models_dir / f"{key}.pkl")
        p = r["probability"]
        expected = "high" if p >= b["thresh_high"] else "medium" if p >= b["thresh_medium"] else "low"
        assert r["risk_level"] == expected


@pytest.mark.parametrize("spline", [False, True])
def test_contributions_sum_to_logit_difference(knhanes_like, tmp_path, spline):
    X, y, info = training_rows(knhanes_like, "diabetes")
    b = fit_bundle(X, y, info.age.to_numpy(), "diabetes", spline)
    x = np.array([[35.0, 1.0, 27.0, 31.0, 1.0, 0.0, 0.0]])
    contrib = ps._contributions(b, x)
    assert set(contrib) == set(b["features"])
    clf = b["model"][-1]
    p = b["model"].predict_proba(x)[0, 1]
    base_logit = clf.intercept_[0] + clf.coef_[0] @ np.asarray(b["background"])
    assert sum(contrib.values()) == pytest.approx(np.log(p / (1 - p)) - base_logit, abs=1e-9)


def test_family_history_none_or_missing_is_zero(models_dir):
    a = ps.predict_all(INBODY, PROFILE)
    b = ps.predict_all(INBODY, {**PROFILE, "fh_hypertension": None, "fh_diabetes": None, "fh_dyslipidemia": None})
    c = ps.predict_all(INBODY, {**PROFILE, "fh_diabetes": True})
    assert [r["probability"] for r in a] == [r["probability"] for r in b]
    assert [r["probability"] for r in a] != [r["probability"] for r in c]


def test_missing_bmi_returns_empty(models_dir):
    assert ps.predict_all({"body_fat_percentage": 28.0}, PROFILE) == []
    assert ps.predict_all({"bmi": 27.0, "body_fat_percentage": None}, PROFILE) == []


@pytest.mark.parametrize("profile, msg", [
    ({"birth_year": AGE35}, "성별"),
    ({"gender": "male"}, "출생연도"),
    ({"gender": "female", "birth_year": date.today().year - 18}, "성인"),
])
def test_invalid_profile_raises(models_dir, profile, msg):
    with pytest.raises(ValueError, match=msg):
        ps.predict_all(INBODY, profile)


@pytest.mark.parametrize("inbody", [
    {"bmi": 240.0, "body_fat_percentage": 28.0},   # 24.0 오독
    {"bmi": 5.0, "body_fat_percentage": 28.0},
    {"bmi": 27.0, "body_fat_percentage": 85.0},
    {"bmi": 27.0, "body_fat_percentage": 1.0},
])
def test_out_of_range_inputs_raise(models_dir, inbody):
    with pytest.raises(ValueError, match="범위"):
        ps.predict_all(inbody, PROFILE)


def test_no_models_raises(models_dir):
    for p in models_dir.glob("*.pkl"):
        p.unlink()
    with pytest.raises(ps.ModelBundleError):
        ps.load_models()
    with pytest.raises(ps.ModelBundleError):
        ps.predict_all(INBODY, PROFILE)


def test_load_models_returns_available(models_dir):
    assert [b["disease_key"] for b in ps.load_models()] == list(DISEASES)


def test_extreme_inputs_predict_finite(models_dir):
    res = ps.predict_all({"bmi": 45.0, "body_fat_percentage": 50.0},
                         {"gender": "female", "birth_year": date.today().year - 80})
    assert len(res) == 4 and all(0 < r["probability"] < 1 for r in res)


def test_integer_inputs(models_dir):
    a = ps.predict_all({"bmi": 24, "body_fat_percentage": 20}, PROFILE)
    b = ps.predict_all({"bmi": 24.0, "body_fat_percentage": 20.0}, PROFILE)
    assert [r["probability"] for r in a] == [r["probability"] for r in b]


def test_missing_model_file_skips_disease(models_dir):
    (models_dir / "cholesterol.pkl").unlink()
    res = ps.predict_all(INBODY, PROFILE)
    assert [r["disease"] for r in res] == ["고혈압", "당뇨", "고중성지방"]


def test_old_gender_files_are_ignored(models_dir):
    joblib.dump({"not": "a bundle"}, models_dir / "hypertension_male.pkl")
    assert len(ps.predict_all(INBODY, PROFILE)) == 4


def test_bad_version_raises(models_dir):
    b = joblib.load(models_dir / "hypertension.pkl")
    joblib.dump({**b, "version": 1}, models_dir / "hypertension.pkl")
    with pytest.raises(ps.ModelBundleError):
        ps.predict_all(INBODY, PROFILE)


def test_feature_mismatch_raises(models_dir):
    b = joblib.load(models_dir / "diabetes.pkl")
    joblib.dump({**b, "features": b["features"][:4]}, models_dir / "diabetes.pkl")
    with pytest.raises(ps.ModelBundleError):
        ps.predict_all(INBODY, PROFILE)


def test_real_models_if_present():
    paths = [ps.MODELS_DIR / f"{k}.pkl" for k in DISEASES]
    present = [p for p in paths if p.exists() and joblib.load(p).get("version") == 2]
    if not present:
        pytest.skip("v2 모델 파일이 아직 없음")
    ps._load_bundle.cache_clear()
    res = ps.predict_all(INBODY, {**PROFILE, "fh_hypertension": True})
    assert len(res) == len(present)
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_prediction_service.py -q`
Expected: 여러 FAIL. 예를 들어 `AttributeError: module 'app.services.prediction_service' has no attribute '_contributions'`

- [ ] **Step 3: 구현 (파일 전체 교체)**

`app/services/prediction_service.py`:

```python
"""인바디 수치 + 프로필로 성인병 4종의 현재 유병 확률·등급·기여 요인을 계산한다.
모델: training/train.py가 만든 models/{disease}.pkl(version 2).
설계: docs/superpowers/specs/2026-10-12-risk-model-redesign-design.md 6절."""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DISEASES = ["hypertension", "cholesterol", "diabetes", "triglyceride"]
FEATURES = ["age", "female", "bmi", "body_fat_percentage",
            "fh_hypertension", "fh_diabetes", "fh_dyslipidemia"]
FAMILY_HISTORY = FEATURES[4:]
MIN_AGE = 19
TOP_K = 3
# KNHANES 19–59세 실제 범위(BMI 14.5–48.0, 체지방률 6.7–54.6)에 여유를 둔 값. 벗어나면 OCR 오독으로 본다.
BMI_RANGE = (12.0, 60.0)
PBF_RANGE = (3.0, 70.0)


class ModelBundleError(RuntimeError):
    """학습 코드와 서비스 코드의 모델 형식이 맞지 않는다."""


@lru_cache(maxsize=None)
def _load_bundle(path: str) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("version") != 2:
        raise ModelBundleError(f"{path}: version 2 번들이 아닙니다")
    if bundle.get("features") != FEATURES:
        raise ModelBundleError(f"{path}: 입력 변수가 다릅니다 {bundle.get('features')}")
    return bundle


def load_models() -> list[dict]:
    """있는 모델을 모두 불러온다. 앱 시작 시 호출해 형식 오류와 '모델 없음'을 배포 단계에서 드러낸다."""
    out = []
    for key in DISEASES:
        path = MODELS_DIR / f"{key}.pkl"
        if not path.exists():
            log.warning("모델 파일이 없어 건너뜁니다: %s", path)
            continue
        out.append(_load_bundle(str(path)))
    if not out:
        raise ModelBundleError(f"{MODELS_DIR}에 사용할 모델이 없습니다")
    return out


def _check_range(name: str, value: float, bounds: tuple[float, float]) -> float:
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} {value}이(가) 허용 범위({bounds[0]:g}–{bounds[1]:g})를 벗어났습니다. "
                         "인바디 수치를 확인해 주세요.")
    return value


def _inputs(inbody: dict, user_profile: dict) -> np.ndarray | None:
    gender = user_profile.get("gender")
    if not gender:
        raise ValueError("성별 정보가 없습니다.")
    birth_year = user_profile.get("birth_year")
    if not birth_year:
        raise ValueError("출생연도 정보가 없습니다.")
    age = date.today().year - int(birth_year)
    if age < MIN_AGE:
        raise ValueError("성인(19세 이상)만 위험도를 예측할 수 있습니다.")

    bmi, pbf = inbody.get("bmi"), inbody.get("body_fat_percentage")
    if bmi is None or pbf is None:
        return None
    bmi = _check_range("BMI", float(bmi), BMI_RANGE)
    pbf = _check_range("체지방률", float(pbf), PBF_RANGE)
    family = [1.0 if user_profile.get(k) else 0.0 for k in FAMILY_HISTORY]
    return np.array([[age, 1.0 if gender == "female" else 0.0, bmi, pbf, *family]])


def _contributions(bundle: dict, x: np.ndarray) -> dict[str, float]:
    """로지스틱 회귀의 로그 오즈 기여 = 계수 × (변환값 − 학습 평균). SHAP 값과 같다."""
    pipe = bundle["model"]
    z = pipe[:-1].transform(x)[0]
    c = pipe[-1].coef_[0] * (z - np.asarray(bundle["background"]))
    return {f: float(c[idx].sum()) for f, idx in bundle["feature_groups"].items()}


def _risk_level(prob: float, bundle: dict) -> str:
    if prob >= bundle["thresh_high"]:
        return "high"
    if prob >= bundle["thresh_medium"]:
        return "medium"
    return "low"


def predict_all(inbody: dict, user_profile: dict) -> list[dict]:
    """질환별 {disease, probability, risk_level, top_factors}. 필수 수치가 없으면 빈 리스트."""
    x = _inputs(inbody, user_profile)
    if x is None:
        return []

    results = []
    for bundle in load_models():
        prob = float(bundle["model"].predict_proba(x)[0, 1])
        factors = [
            {"feature_kor": bundle["feat_kor"][f], "value": float(x[0, FEATURES.index(f)]),
             "shap_value": c, "direction": "up" if c > 0 else "down"}
            for f, c in _contributions(bundle, x).items() if f != "female"
        ]
        factors.sort(key=lambda d: abs(d["shap_value"]), reverse=True)
        results.append({"disease": bundle["target"], "probability": prob,
                        "risk_level": _risk_level(prob, bundle), "top_factors": factors[:TOP_K]})
    return results
```

- [ ] **Step 4: 앱 시작 시 모델 로드 (`main.py`)**

`main.py` 맨 위 import 블록에 추가한다.

```python
from contextlib import asynccontextmanager

from app.services.prediction_service import load_models
```

`app = FastAPI(title="Healthcare API", version="0.1.0")` 줄을 다음으로 바꾼다.

```python
@asynccontextmanager
async def lifespan(_: FastAPI):
    load_models()  # 모델 형식 오류·모델 없음은 배포(시작) 단계에서 실패시킨다
    yield


app = FastAPI(title="Healthcare API", version="0.1.0", lifespan=lifespan)
```

`main.py`는 import할 때 환경변수(`.env`)가 필요해서 단위 테스트로 검사하지 않는다. 동작 확인은 Task 9 Step 4에서 실제 모델로 한다.

- [ ] **Step 5: 서버 의존성 정리**

`requirements.txt`에서 다음을 바꾼다.
- `scikit-learn>=1.7.0` → `scikit-learn==1.8.0`
- `imbalanced-learn>=0.12.0` 줄 삭제
- `shap>=0.46.0` 줄 삭제

확인:

```bash
grep -rnE "^\s*(import|from) (shap|imblearn)" app/ main.py
```
Expected: 출력 없음. 필드 이름 `shap_value`는 출력 형식이라 그대로 남는다.

- [ ] **Step 6: 통과 확인**

Run: `.venv/bin/python -m pytest tests/risk_model/test_prediction_service.py -q`
Expected: `21 passed, 1 skipped` (`test_real_models_if_present`는 v2 모델이 생기기 전까지 skip)

- [ ] **Step 7: Commit**

```bash
git add app/services/prediction_service.py main.py requirements.txt tests/risk_model/test_prediction_service.py
git commit -m "feat(prediction): read v2 logistic bundles, validate inputs, load models at startup"
```

---

### Task 8: 실제 학습·평가 실행과 결과 확인 (사용자 확인 지점)

**Files:**
- Create(실행 산출물): `training/reports/eval-YYYY-MM-DD.md`, `training/reports/metrics.json`, `training/reports/figures/calibration_*.png`

**Interfaces:**
- Consumes: Task 0–6 전부, 실제 데이터 `training/data/`, `training/output/old_models/`
- Produces: 질환별 `spline_adopt`와 통과 여부(보고서 "요약" 표)

- [ ] **Step 1: 전체 테스트**

Run: `.venv/bin/python -m pytest tests/risk_model -q`
Expected: 전부 통과(`test_real_models_if_present`만 skip)

- [ ] **Step 2: 평가 실행**

Run: `.venv/bin/python -m training.evaluate`
Expected: `[hypertension] 평가 중…` 등 4줄, 그 뒤 `보고서: .../training/reports/eval-YYYY-MM-DD.md`. 수 분이 걸린다.

- [ ] **Step 3: 결과 확인**

보고서 "요약" 표에서 질환별 **채택 변형**과 **통과(✅/❌)**를 읽는다. 다음도 확인한다.
- `compare` 표: RF·XGBoost의 AUROC 차이 CI 하한이 0보다 큰 질환이 있는가(spec 5.3 모델 선정 규칙)
- 외부 검증이 "건너뜀"이 아닌가

- [ ] **Step 4: 사용자에게 보고하고 멈춘다**

요약 표, 실패한 질환과 실패한 기준, 모델 선정 규칙에 걸린 질환을 사용자에게 보고한다. 다음 두 가지를 확인받는다.
1. 배포할 질환 목록(통과한 질환)
2. 규칙에 걸린 질환이 있으면 그 처리

**사용자 확인 없이 Task 9로 넘어가지 않는다.**

---

### Task 9: 최종 모델 저장, 기존 모델 정리, 문서

**Files:**
- Create/overwrite: `models/{통과한 key}.pkl`
- Delete: `models/hypertension_female.pkl`, `models/hypertension_male.pkl`, `models/diabetes_female.pkl`, `models/diabetes_male.pkl`, `models/triglyceride_female.pkl`, `models/triglyceride_male.pkl`, `training/retrain_final.py`, 그리고 통과하지 못한 질환의 `models/{key}.pkl`(기존 `cholesterol.pkl` 포함)
- Modify: `training/README.md`
- Commit: `training/reports/`

**Interfaces:**
- Consumes: Task 8에서 확인한 배포 목록 `PASS`와 스플라인 채택 목록 `SPLINE`
- Produces: 서비스가 읽을 최종 `models/*.pkl`

- [ ] **Step 1: 기존 파일 삭제**

```bash
git rm -q models/hypertension_female.pkl models/hypertension_male.pkl models/diabetes_female.pkl models/diabetes_male.pkl models/triglyceride_female.pkl models/triglyceride_male.pkl models/cholesterol.pkl training/retrain_final.py
```

- [ ] **Step 2: 통과한 질환만 학습·저장**

Task 8에서 확인한 목록으로 실행한다. 예: 4개 모두 통과했고 스플라인 채택이 없으면 다음과 같다.

```bash
.venv/bin/python -m training.train --only hypertension cholesterol diabetes triglyceride
```

스플라인을 채택한 질환이 있으면 `--age-spline diabetes`처럼 덧붙인다. 출력된 기준값이 보고서와 같은 범위인지 본다.

- [ ] **Step 3: 서비스 테스트(실제 모델 포함)**

Run: `.venv/bin/python -m pytest tests/risk_model -q`
Expected: 전부 통과. `test_real_models_if_present`도 이제 실행되어 통과한다.

- [ ] **Step 4: 실제 예측 확인**

```bash
.venv/bin/python -c "
from datetime import date
from app.services.prediction_service import predict_all
for r in predict_all({'bmi': 27.0, 'body_fat_percentage': 28.0}, {'gender': 'male', 'birth_year': date.today().year - 35, 'fh_hypertension': True}):
    print(r['disease'], round(r['probability'], 3), r['risk_level'], [(f['feature_kor'], round(f['shap_value'], 2)) for f in r['top_factors']])
"
```
Expected: 통과한 질환 수만큼 줄이 나온다. 요인 이름은 `나이`, `BMI`, `체지방률`, `○○ 가족력` 중에서 나온다.

앱 시작(모델 사전 로드)도 확인한다.

```bash
.venv/bin/python -c "
from fastapi.testclient import TestClient
import main
with TestClient(main.app) as c:
    print(c.get('/health').json())
"
```
Expected: `{'status': 'ok'}`. 모델 형식이 틀리면 여기서 `ModelBundleError`가 난다.

- [ ] **Step 5: `training/README.md` 교체**

```markdown
# 성인병 위험도 모델 학습

`../models/{hypertension,cholesterol,diabetes,triglyceride}.pkl`을 만든다. 서버는 `app/services/prediction_service.py`에서 읽는다.
설계: `docs/superpowers/specs/2026-10-12-risk-model-redesign-design.md`

| 파일 | 역할 |
| --- | --- |
| `features.py` | KNHANES → 라벨·입력 7개, 등급 기준값 선택 |
| `model.py` | 로지스틱 회귀 파이프라인, v2 번들 |
| `metrics.py` | AUROC·Brier·보정·등급 지표와 신뢰구간 |
| `external.py` | NHANES 2003–04, Kaggle 국민체력100 로더 |
| `train.py` | 모델 학습·저장 |
| `evaluate.py` | 중첩 교차검증·비교·외부 검증 → `reports/` |
| `train_models.py`, `analysis/` | 이전 탐색 기록(배포와 무관) |

## 데이터

`training/data/`에 국민건강영양조사 2022–2024 원자료(`hn22_all.sas7bdat` 등)를 넣는다. 외부 검증 데이터는 `training/data/ext/`에 넣는다. 원자료는 재배포하지 않으므로 Git에 올리지 않는다.

| 파일 | 출처 |
| --- | --- |
| `ext/bodyPerformance.csv` | Kaggle Body Performance. 사본: github.com/mrinalcs/body-performance `data.csv` |
| `ext/nhanes/{bix,demo,bmx,bpx,bpq,diq,l10,l10am,l13,l13am}_c.rda` | NHANES 2003–04. 사본: github.com/vandomed/nhanesdata `data/` |

## 실행 (저장소 루트에서)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r training/requirements.txt
.venv/bin/python -m training.evaluate                        # 보고서: training/reports/
.venv/bin/python -m training.train --only <통과한 질환들> [--age-spline <질환들>]
.venv/bin/python -m pytest tests/risk_model
```

통과 기준(spec 5.4)을 넘지 못한 질환은 저장하지 않는다. 서버는 파일이 없는 질환을 건너뛴다.

## 모델 파일 형식 (version 2)

`prediction_service.py`가 아래 키를 읽는다. 바꾸면 서버도 같이 고쳐야 한다.

`version, disease_key, target, model, features, feat_kor, feature_groups, background, thresh_medium, thresh_high, meta`
```

- [ ] **Step 6: Commit**

```bash
git add models/ training/reports/ training/README.md
git commit -m "feat(models): replace risk models with redesigned logistic models

Evaluation report: training/reports/. Old gender-split SMOTE models removed."
```

- [ ] **Step 7: push 여부를 사용자에게 묻는다**

`back` 브랜치에 push하면 Railway에 배포된다. 사용자 확인 후에만 `git push origin back`을 실행한다. 함께 알릴 것: RAG 용어집 `FEATURES`에 새 요인 이름 매핑이 필요하다(spec 6.2).
