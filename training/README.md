# 성인병 위험도 모델 학습

`../models/{hypertension,cholesterol,diabetes,triglyceride}.pkl`을 만든다. 서버는 `app/services/prediction_service.py`에서 이 파일을 읽는다.
설계: `docs/superpowers/specs/2026-10-12-risk-model-redesign-design.md`
최근 평가: `reports/eval-2026-10-12.md`

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

현재 배포 모델은 다음 명령으로 만들었다(2026-10-12, 네 질환 모두 통과, 고콜레스테롤만 나이 스플라인).

```bash
.venv/bin/python -m training.train --only hypertension cholesterol diabetes triglyceride --age-spline cholesterol
```

통과 기준(spec 5.4)을 넘지 못한 질환은 저장하지 않는다. 서버는 파일이 없는 질환을 건너뛴다.

## 모델 파일 형식 (version 2)

`prediction_service.py`가 아래 키를 읽는다. 키를 바꾸면 서버 코드도 같이 고쳐야 한다.

`version, disease_key, target, model, features, feat_kor, feature_groups, background, thresh_medium, thresh_high, meta`
