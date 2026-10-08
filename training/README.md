# 성인병 위험도 모델 학습

`../models/*.pkl`(고혈압·고콜레스테롤·당뇨·고중성지방)을 만드는 스크립트입니다.
서버는 `app/services/prediction_service.py`에서 이 pkl을 읽습니다.

| 파일 | 역할 |
| --- | --- |
| `retrain_final.py` | 배포용 최종 모델 7개를 학습해 `../models/`에 저장 |
| `train_models.py` | 모델·피처셋 비교 실험 (5-fold CV, AUROC/PR-AUC/Brier). 결과는 `output/` |
| `analysis/knhanes_compare.py` | KNHANES에서 BIA 변수 vs BMI·허리둘레 예측력 비교, 현행 라벨과 비교 |
| `analysis/kaggle_check.py` | Kaggle 국민체력100 품질·고혈압 라벨 잡음, KNHANES 모델 외부검증 |
| `analysis/nhanes_check.py` | NHANES 2003–04 BIA 라벨 생성·예측력, KNHANES 모델 외부검증 |

## 데이터

국민건강영양조사(KNHANES) 2022~2024 원시자료 `hn22_all.sas7bdat`, `hn23_all.sas7bdat`, `hn24_all.sas7bdat`.
질병관리청 국민건강영양조사 홈페이지에서 이용 신청 후 받아 `training/data/`에 넣으세요.
원시자료는 재배포하지 않으므로 Git에 올리지 않습니다(`.gitignore` 처리됨).

외부검증용 데이터는 `training/data/ext/`에 둡니다.

| 파일 | 출처 |
| --- | --- |
| `ext/bodyPerformance.csv` | Kaggle Body Performance(국민체력100). 사본: github.com/mrinalcs/body-performance `data.csv` |
| `ext/nhanes/{bix,demo,bmx,bpx,bpq,diq,l10,l10am,l13,l13am}_c.rda` | NHANES 2003–04. 사본: github.com/vandomed/nhanesdata `data/` |

## 실행

```bash
pip install -r training/requirements.txt
python training/retrain_final.py     # ../models/*.pkl 덮어씀
python training/train_models.py      # 비교 실험
cd training/analysis && python knhanes_compare.py   # 분석 스크립트는 analysis/ 안에서 실행
```

## pkl 형식

`prediction_service.py`가 아래 키를 읽으므로 바꾸면 서버도 같이 고쳐야 합니다.

```python
{"model", "features", "feat_kor", "target", "thresh_binary", "thresh_high"}
```

## 알려진 문제

- 고혈압 라벨이 `HE_HP >= 3`이라 고혈압 전단계(130–139/80–89)까지 양성으로 들어갑니다. `HE_HP == 4`가 고혈압입니다.
- 당뇨 라벨이 공복혈당·당화혈색소 수치만 봐서, 약으로 조절 중인 환자는 음성이 됩니다. `HE_DM_HbA1c == 3`이 진단·치료를 포함합니다.
- `SMM`은 팔다리 4부위 BIA 근육량의 합(`BIA_LRA+LLA+LRL+LLL`)입니다. 몸통이 빠져 있어서 앱이 넣는 인바디 "골격근량"과 같은 값이 아닙니다.
- `thresh_binary`/`thresh_high`는 학습 데이터 자체의 예측값으로 정합니다. ExtraTrees 모델은 과대평가될 수 있습니다.
