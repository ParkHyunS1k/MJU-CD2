# inbody-ocr (온디바이스 규칙 파서 + 검증 레이어 프로토타입)

앱 흐름: 촬영 → ML Kit 텍스트 인식(한국어) → **파서** → **검증** → 사용자 확인 → 숫자만 서버 전송.
이미지는 기기 밖으로 나가지 않는다. 이 폴더는 굵게 표시한 두 단계를 Python으로 먼저 만들고 평가하는 곳이며,
앱 스택이 정해지면 Dart/TS로 옮긴다.

| 파일 | 역할 | 앱으로 포팅 |
|---|---|---|
| `inbody_ocr/ocr_types.py` | OCR 엔진 공통 형식 (줄 텍스트 + 0~1 좌표, 좌상단 원점) | O |
| `inbody_ocr/parser.py` | 라벨 찾기 → 같은 행 오른쪽 숫자 → 다수결 | O |
| `inbody_ocr/validate.py` | 범위 검사 + 체성분 일관성 검사 | O (서버도 사용) |
| `inbody_ocr/schema.py` | 필드·범위·소수 자릿수 정의 | O |
| `inbody_ocr/api.py` | 서버: 숫자 수신 후 재검증 | X |
| `inbody_ocr/ocr_mac.py` | 개발용 Mac Apple Vision OCR | X |
| `tests/vectors/*.json` | 공용 테스트 벡터 (OCR 줄 → 기대값). 포팅본도 이걸로 검증 | 공용 |

## 실행
```bash
uv sync
uv run pytest
uv run uvicorn inbody_ocr.api:app --reload   # POST /v1/inbody/measurements
```

## 평가셋
이미지·정답·OCR 결과는 비공개 드라이브에만 둔다. `labels.jsonl` 한 줄 = 결과지 1장:
```json
{"id": "p001", "image": "p001.jpg", "input_type": "photo",
 "fields": {"height_cm": 172.0, "weight_kg": 70.0, "total_body_water_l": 40.4, "waist_hip_ratio": null}}
```
- 정답은 결과지에 **표기된 값 그대로** (결과지에 없는 항목은 null → 채점 제외)

```bash
# 개발용: Mac Vision으로 OCR (결과는 --ocr-dir에 캐시)
uv run python -m eval.run_parse --labels <drive>/labels.jsonl --images <drive>/images --ocr-dir <drive>/ocr_vision --out eval/out/preds_vision.jsonl
# 최종: 실기기 ML Kit에서 내보낸 {id}.json으로
uv run python -m eval.run_parse --labels <drive>/labels.jsonl --ocr-dir <drive>/ocr_mlkit --out eval/out/preds_mlkit.jsonl
uv run python -m eval.score --labels <drive>/labels.jsonl --preds eval/out/preds_mlkit.jsonl
```

## 파서 개선 루프
틀린 결과지가 나오면 그 OCR JSON을 `tests/vectors/`에 기대값과 함께 추가 → 파서 수정 → 전체 벡터 통과 확인.
(벡터에는 이름 등 개인정보가 없어야 함. 실제 결과지 OCR이면 수치를 바꾸거나 식별 줄을 지우고 넣는다.)

## 현재 상태 (parser 0.2.0)
- InBody270 실물 결과지 1장(캡처): 16/16 항목 일치, 검증 4개 통과 (Mac Vision 기준)
- 같은 이미지를 변형한 강건성 점검: 1° 회전·축소는 16/16, 3° 회전은 13/16, 흐림·원근 왜곡은 15/16.
  틀린 항목은 모두 missing 또는 low로 표시돼 사용자 확인으로 넘어감 (틀렸는데 통과 0건)
- 실물 OCR 벡터는 `tests/vectors_local/`(Git 제외)에 두고 pytest가 있으면 함께 돌린다

## 미확인 사항
- 1장 기준이라 과적합 가능성 큼. 다른 기종(570/770, 앱 화면)·실제 폰 사진으로 검증 필요
- 기울기 3° 이상에서 오른쪽 열(복부지방률·내장지방레벨) 인식이 떨어짐 → 촬영 가이드 UI 또는 OCR 좌표 기반 기울기 보정 필요
- ML Kit 한국어 인식 품질, 앱 스택별 ML Kit 플러그인 상태 (Flutter `google_mlkit_text_recognition`, RN은 라이브러리 선택 필요)
- 체수분(L) 1L≈1kg 합산: InBody270 1장에서 49.3+13.4+4.51+12.8=80.01 ≈ 80.0 확인. 다른 기종은 미확인
- 범위(min/max)는 오인식 필터용 잠정값
