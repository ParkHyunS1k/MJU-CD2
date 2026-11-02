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

## 추출 항목 (parser 0.3.0)
결과지의 숫자 전부(회원번호 제외). 부위별 근육·체지방은 InBody270에서 '표준/이하/이상' 글자뿐이라 제외.
- 단일 값: 신장·나이·성별·측정일시, 체성분 14개, 인바디점수, 적정체중, 체중·지방·근육조절, 비만도, 권장섭취열량
- `ranges`: 정상범위 8개 / `exercise_kcal`: 운동소비열량 20종 / `impedance`: 5부위 × 20·100kHz
- `history`: 신체변화 그래프 (날짜·시각·체중·골격근량·체지방률). 마지막 열 = 이번 측정
- 검증식 10개: 기존 5개 + 체중조절, 조절 합계, 비만도(남성), 이력 마지막 열 = 골격근량, 골격근량/체중 타당성

## 현재 상태 (InBody270, 1인 25장, Mac Vision 타일 OCR 기준)
개발용 OCR(`ocr_mac.py`)은 페이지 전체 + 겹치는 2×3 조각으로 읽어 합친다 (작은 숫자 누락·소수점 누락 대책).
| | 캡처 25장 | 합성 사진 250장 (`eval/augment.py`) |
|---|---|---|
| 핵심 14개 정확 | 100% | 97.4% |
| 틀렸는데 통과 (핵심) | 0% | 0.23% (체중 미인식·검증식 없는 항목) |
| 새 항목 (5장 정답) | 운동·임피던스·범위·단일 값 100%, 이력 99% | 운동 98%, 단일 값 97%, 임피던스 81%, 범위 80%, 이력 63% |
- 사진에서 새 항목 오류는 대부분 작은 글씨 숫자 오독(9↔8)이라 형식으로 못 거른다. 이력·범위·임피던스는 검증식이 없으므로 참고값으로 쓴다
- 합성 사진은 실제 폰 사진을 대신하지 못한다. 실물 사진 평가 필요
- 실물 OCR 벡터는 `tests/vectors_local/`(Git 제외)에 두고 pytest가 있으면 함께 돌린다

```bash
uv run --with numpy python -m eval.augment --labels data/labels_capture.jsonl --images <캡처 폴더> --out eval/out/aug
uv run python -m eval.score_full --labels data/labels_full.jsonl --preds eval/out/preds.jsonl --show   # 새 항목 채점
```

## 미확인 사항
- 1장 기준이라 과적합 가능성 큼. 다른 기종(570/770, 앱 화면)·실제 폰 사진으로 검증 필요
- 기울기 3° 이상에서 오른쪽 열(복부지방률·내장지방레벨) 인식이 떨어짐 → 촬영 가이드 UI 또는 OCR 좌표 기반 기울기 보정 필요
- ML Kit 한국어 인식 품질, 앱 스택별 ML Kit 플러그인 상태 (Flutter `google_mlkit_text_recognition`, RN은 라이브러리 선택 필요)
- 체수분(L) 1L≈1kg 합산: InBody270 1장에서 49.3+13.4+4.51+12.8=80.01 ≈ 80.0 확인. 다른 기종은 미확인
- 범위(min/max)는 오인식 필터용 잠정값
