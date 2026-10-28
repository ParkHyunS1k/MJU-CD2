# 리포트 개인화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 백엔드가 개인 지표를 계산하고 사용자 특성으로 검색 쿼리를 더해, 같은 위험 유형이어도 사람마다 다른 근거와 수치로 리포트를 만든다.

**Architecture:** `personal_metrics.compute`가 인바디·프로필에서 지표를 계산한다. `rag_taxonomy.trait_queries`가 그 지표·위험도·추세로 특성 쿼리(최대 4개 + 영양제 1개)를 만들어 기존 질환 쿼리 앞에 붙인다. 검색은 쿼리마다 1위 근거를 보장하고, 근거에는 청크 자신의 분야를 실어 영양제 인용을 거른다. 리포트 payload에 `personal_metrics`를 넣고 프롬프트 규칙을 바꾼다.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, Supabase(Postgres + pgvector), OpenAI Responses API, pytest

**Spec:** `docs/superpowers/specs/2026-10-28-report-personalization-design.md`

## Global Constraints

- 테스트: `.venv/bin/python -m pytest tests/runtime tests/risk_model tests/test_users_router.py tests/rag/test_load.py tests/rag/test_kdca.py -q -p no:warnings`(현재 106 passed). `rag/` 전체는 `python -m pytest tests/rag -q`(anaconda python, 187 passed).
- 코드 주석·문서는 한국어, 기존 파일의 문체와 밀도를 따른다.
- 기준값(스펙 5.2): BMI 단계 <18.5 저체중, 18.5–22.9 정상, 23–24.9 비만 전단계, 25–29.9 1단계 비만, 30–34.9 2단계 비만, ≥35 3단계 비만. 체지방률(인바디 결과지 비만평가) 남 <10 낮음 / ≤20 표준 / ≤25 경도비만 / >25 비만, 여 <18 / ≤28 / ≤33 / >33. 복부지방률 남 0.90·여 0.85 이상, 내장지방레벨 10 이상.
- 특성 쿼리 문구는 스펙 5.3 표의 문구를 그대로 쓴다.
- 라이브 DB 마이그레이션 적용 전에는 사용자에게 확인을 받는다.
- 리포트 생성은 GPT API(`gpt-6.1-terra`), 개인 원자료(KNHANES·인바디 사진)와 결과물은 커밋하지 않는다.

## Review Focus

1. `height`가 없는 기존 행(대부분의 DB 행): 체중·BMI로 키를 역산해야 하고, 체중이나 BMI가 없으면 키·`kg_to_normal_bmi`를 빼야 한다(예외 없이). → Task 3 테스트
2. 가족력이 `None`(모름)인 사용자: 가족력 특성이 켜지면 안 된다(`False`와 같이 취급). → Task 4 테스트
3. DB 마이그레이션이 아직 안 된 상태(RPC 응답에 `category` 없음): 검색은 동작하고 영양제 후보만 모두 빠져야 한다. → Task 2 테스트
4. 특성 쿼리가 임계값을 못 넘어 결과가 0개: 다른 쿼리 근거 선택이 그대로여야 한다. → Task 2 테스트
5. 질환 쿼리 16개 + 특성 5개인 사람: 21개 쿼리 모두 1위 근거를 받아야 한다. → Task 2 테스트

---

### Task 1: DB 컬럼 2개와 RPC 청크 분야 반환

**Files:**
- Create: `supabase/migrations/20261028120000_inbody_height_muscle_rag_category.sql`
- Modify: `app/schemas/inbody.py` (`InBodyMetricsBase`), `app/services/inbody_service.py` (`INBODY_VALUE_FIELDS`)
- Test: `tests/runtime/test_inbody_schema_fields.py` (신규)

**Interfaces:**
- Produces: `inbody_logs.height numeric`(cm), `inbody_logs.muscle_control numeric`(kg). `InBodyMetricsBase.height: float | None = Field(None, gt=0, lt=300)`, `muscle_control: float | None = None`. RPC `match_rag_chunks`가 기존 열에 더해 `category text`(청크 자신의 분야)를 돌려준다.

- [ ] **Step 1: 실패하는 테스트**

```python
from app.schemas.inbody import InBodyCreate, InBodyUpdate
from app.services.inbody_service import INBODY_VALUE_FIELDS

def test_height_and_muscle_control_round_trip():
    rec = InBodyCreate(weight=80, height=183, muscle_control=1.5)
    assert rec.model_dump()["height"] == 183 and rec.model_dump()["muscle_control"] == 1.5
    assert InBodyUpdate(height=None).height is None

def test_upload_mapping_carries_new_fields():
    assert {"height", "muscle_control"} <= set(INBODY_VALUE_FIELDS)
```

- [ ] **Step 2:** `.venv/bin/python -m pytest tests/runtime/test_inbody_schema_fields.py -q` → FAIL(필드 없음)
- [ ] **Step 3:** 스키마 두 필드와 `INBODY_VALUE_FIELDS` 두 키 추가. 마이그레이션 SQL: `alter table public.inbody_logs add column height numeric, add column muscle_control numeric;` 그리고 `drop function public.match_rag_chunks(extensions.halfvec, int, float);` 뒤 `20261015131746_rag_tables.sql`과 같은 본문에 반환 열 `category text`와 select의 `c.category`만 더해 다시 만들고, 같은 `revoke`/`grant`(service_role만)를 다시 건다. 파일 첫 줄에 스펙 경로 주석.
- [ ] **Step 4:** 같은 테스트 → PASS, 전체 테스트 → 모두 통과
- [ ] **Step 5:** 커밋 `feat(db): add height and muscle_control, return chunk category from match_rag_chunks`
- [ ] **Step 6: 라이브 적용(사용자 확인 후)** Supabase MCP `apply_migration`(name `inbody_height_muscle_rag_category`)으로 적용하고 `execute_sql`로 확인: `inbody_logs`에 두 컬럼, `select category from match_rag_chunks(...)`가 값을 돌려줌. `get_advisors(security)`에 새 경고가 없는지 본다.

### Task 2: 근거의 청크 분야, 영양제 인용, 쿼리마다 1위 보장

**Files:**
- Modify: `app/services/retrieval_service.py` (`Evidence`, `select_evidence`, `retrieve`), `app/services/citation_service.py`
- Test: `tests/runtime/test_retrieval_service.py`, `tests/runtime/test_citation_service.py`

**Interfaces:**
- Consumes: Task 1의 RPC `category` 열(없을 수도 있음)
- Produces: `Evidence`에 끝 필드 두 개 추가: `chunk_category: str | None = None`(RPC `row.get("category")`), `trait: str | None = None`(이 태스크에서는 필드만 추가, 값은 Task 5에서 `Query.trait`로 채운다). `retrieve`는 `select_evidence(queries, results, limit=max(MAX_EVIDENCE, len(queries)))`로 부른다. `validate_citations`의 영양제 필터는 `e.chunk_category == "supplement"`.

- [ ] **Step 1: 실패하는 테스트**
  - `test_select_evidence_keeps_chunk_category_and_missing_is_none`: `category` 있는 행 → `chunk_category == "supplement"`, 없는 행 → `None`
  - `test_retrieve_gives_every_query_its_top_hit`: 쿼리 21개, 쿼리마다 서로 다른 청크 1개를 돌려주는 가짜 `match_fn` → 근거 21개
  - `test_empty_trait_results_do_not_shift_others`: 3개 쿼리 중 첫 쿼리 결과 0개 → 나머지 두 쿼리의 1위가 S1, S2
  - `test_supplement_needs_supplement_chunk`(citation): 쿼리 분야는 supplement지만 `chunk_category="diet"`인 근거만 인용한 영양제 → 제거, `chunk_category="supplement"` 인용 → 유지, `chunk_category=None` → 제거
- [ ] **Step 2:** 두 테스트 파일 실행 → 새 테스트 FAIL
- [ ] **Step 3:** 위 Interfaces대로 구현. `Evidence.category`(쿼리 분야)는 그대로 둔다.
- [ ] **Step 4:** 테스트 → PASS, 전체 테스트 통과
- [ ] **Step 5:** 커밋 `feat(retrieval): carry chunk category, gate supplements on it, one top hit per query`

### Task 3: 개인 지표 계산

**Files:**
- Create: `app/services/personal_metrics.py`
- Modify: `app/services/rag_taxonomy.py`(기준 상수를 여기서 가져다 쓰게)
- Test: `tests/runtime/test_personal_metrics.py`

**Interfaces:**
- Produces: `compute(inbody: dict, profile: dict) -> dict`. 키(값이 없으면 키 자체를 뺀다, `missing_metrics`만 항상 있음):
  - `height_cm: float`, `height_source: "measured" | "derived"`
  - `bmi_band: str`(Global Constraints의 이름 그대로)
  - `kg_to_normal_bmi: float` — BMI ≥ 23일 때 `weight − 22.9 × (height_cm/100)²`, 소수 첫째 자리
  - `weight_loss_5pct_kg`, `weight_loss_10pct_kg: float`, `weight_after_5pct`, `weight_after_10pct: float` — BMI ≥ 23일 때
  - `body_fat_status: {"status": str, "over_standard_pp": float}`(`over_standard_pp`는 표준 상한(남 20, 여 28)을 넘은 %p, 넘지 않으면 0.0)
  - `muscle_status: {"status": "표준 이하" | "표준", "increase_kg": float}` — `muscle_control > 0`이면 표준 이하
  - `abdominal: list[{"metric": "복부지방률" | "내장지방레벨", "value": float, "threshold": float}]` — 기준 이상인 것만, 비면 키를 뺀다
  - `missing_metrics: list[str]` — `ai_service._INBODY_METRIC_KEYS`(Task 5에서 `height`, `muscle_control` 추가) 중 값이 None인 것의 한국어 이름(`METRIC_NAMES`: 체중, 골격근량, 체지방량, 체지방률, 체수분, 단백질, 무기질, BMI, 기초대사량, 복부지방률, 내장지방레벨, 키, 근육조절)
  - 상수 `BMI_BANDS`, `BODY_FAT_BANDS`(성별), 기존 `ABDOMINAL_RATIO`, `VISCERAL_FAT_LEVEL`, `OBESITY_BMI`는 한 곳(이 모듈)에 두고 `rag_taxonomy`가 가져다 쓴다. 각 상수에 출처 주석(BMI: 대한비만학회, 체지방률: 인바디 결과지 비만평가).

- [ ] **Step 1: 실패하는 테스트**
  - `test_height_derived_from_weight_and_bmi`: weight 80, bmi 23.9 → `height_cm == 183.0`, `height_source == "derived"`; height 183 주면 `"measured"`
  - `test_bmi_band_boundaries`: 18.4 저체중, 18.5·22.9 정상, 23.0·24.9 비만 전단계, 25.0 1단계 비만, 30.0 2단계 비만, 35.0 3단계 비만
  - `test_weight_targets_only_from_bmi_23`: weight 74.9, bmi 24.9 → `weight_loss_5pct_kg == 3.7`, `weight_after_5pct == 71.2`, `kg_to_normal_bmi` 존재; bmi 22.0 → 세 키 모두 없음
  - `test_body_fat_status_by_sex`: 남 20.0 표준·over 0.0, 남 23.0 경도비만·over 3.0, 여 30.0 경도비만·over 2.0, 여 34.0 비만
  - `test_muscle_and_abdominal`: muscle_control 1.5 → 표준 이하·1.5, 0.0 → 표준, None → 키 없음; 남 abs_fat 0.91 → 복부지방률 항목, fat_level 10 → 내장지방레벨 항목, 0.89·9 → `abdominal` 키 없음
  - `test_missing_inputs_are_omitted_not_errors`: `{}` 입력 → `{"missing_metrics": [13개 전부]}`; bmi만 있고 weight 없음 → `height_cm` 없음, 예외 없음
- [ ] **Step 2:** `.venv/bin/python -m pytest tests/runtime/test_personal_metrics.py -q` → FAIL(모듈 없음)
- [ ] **Step 3:** 구현. `rag_taxonomy`는 상수를 import로 바꾸고 동작은 그대로.
- [ ] **Step 4:** 테스트 → PASS, `tests/runtime/test_rag_taxonomy.py` 포함 전체 통과
- [ ] **Step 5:** 커밋 `feat(report): compute personal metrics for the report`

### Task 4: 특성 쿼리

**Files:**
- Modify: `app/services/rag_taxonomy.py` (`Query`, `trait_queries` 신규)
- Test: `tests/runtime/test_rag_taxonomy.py`

**Interfaces:**
- Consumes: Task 3 `compute(...)` 결과 dict
- Produces: `Query`에 끝 필드 `trait: str | None = None`. `trait_queries(disease_risks: list[dict], profile: dict, metrics: dict, trend_analysis: dict | None) -> list[Query]` — 스펙 5.3 표의 9종. `trait` 값은 표의 특성 이름(근육 부족, 체지방 과다, 복부비만, 전단계 위험, 가족력, 여러 위험 동반, 체중 증가 추세, 비만 전단계, 영양제).
  - 선택: 표 순서대로 특성 종류마다 1개씩 먼저, 자리가 남으면 두 번째 쿼리부터 같은 순서로, 합계 4개까지. 그 뒤 영양제 쿼리 1개를 항상 붙인다(총 최대 5개).
  - 질환이 여러 개인 특성(전단계, 가족력)의 질환 순서: hypertension, diabetes, dyslipidemia. cholesterol·triglyceride는 dyslipidemia 하나로 친다.
  - 전단계 위험은 위험도 `medium`만, 여러 위험 동반은 `medium`·`high` 질환(이상지질혈증 합침) 2개 이상.
  - 가족력은 `profile[fh_*] is True`만(None·False는 아님).
  - 체중 증가 추세: `trend_analysis["feature_trends"]` 중 `feature_kor`가 "체중" 또는 "체지방률"이고 `direction == "up"`.
  - 체지방 과다: BMI < 23 그리고 `body_fat_status.status`가 경도비만·비만.

- [ ] **Step 1: 실패하는 테스트**
  - `test_traits_follow_profile`: 남 BMI 24.0, 체지방률 23, 가족력 고혈압 True·당뇨 None, 위험 cholesterol medium → 텍스트 목록에 "이상지질혈증 경계 수치 생활습관 교정", "고혈압 가족력 고위험군 조기 예방", "비만 전단계 과체중 체중 증가 예방"이 있고, 당뇨 가족력 쿼리는 없고, 마지막은 "영양제 건강기능식품 선택 주의사항"
  - `test_one_per_trait_type_first`: medium 질환 3개 + 가족력 3개 + 복부비만 → 앞 4개의 `trait`가 서로 다르고 순서가 복부비만, 전단계 위험, 가족력, 여러 위험 동반
  - `test_trait_cap_and_supplement_always`: 모든 특성이 켜진 입력 → 길이 5, 마지막 `trait == "영양제"`; 아무 특성 없음 → 영양제 1개만
  - `test_weight_trend_trait`: feature_trends에 체중 up → "체중 증가 예방 체중 유지 관리" 포함; stable → 미포함
  - `test_lean_body_fat_trait_needs_bmi_under_23`: BMI 22, 남 체지방률 26 → "체지방률 감소 근력운동 체성분 개선"; BMI 24 → 미포함
- [ ] **Step 2:** 테스트 → FAIL
- [ ] **Step 3:** 구현. `build_queries`는 바꾸지 않는다.
- [ ] **Step 4:** 테스트 → PASS, 전체 통과
- [ ] **Step 5:** 커밋 `feat(retrieval): add target-user trait queries`

### Task 5: 프롬프트와 분석 흐름 연결

**Files:**
- Modify: `app/services/ai_service.py`, `app/services/inbody_analysis_service.py`
- Test: `tests/runtime/test_report_prompt.py`, `tests/runtime/test_analysis_rag.py`

**Interfaces:**
- Consumes: Task 3 `compute`, Task 4 `trait_queries`·`Query.trait`, Task 2 `Evidence.trait`
- Produces: `generate_report(..., personal_metrics: dict | None = None)` → payload 키 `personal_metrics`. `_INBODY_METRIC_KEYS`에 `"height"`, `"muscle_control"` 추가(`muscle_control`은 소수 첫째 자리, 기존 반올림 규칙 유지). `format_evidence` 머리글은 `e.trait`가 있으면 `({trait}·{분야})`, 없으면 지금처럼 `({질환}·{분야})`. 분석 흐름: `metrics = compute(target, user)`, `queries = trait_queries(disease_risks, user, metrics, trend_analysis) + build_queries(disease_risks, target, user.get("gender"))`, `generate_report(..., personal_metrics=metrics)`.

- [ ] **Step 1: 실패하는 테스트**
  - 지시문(`_REPORT_SYSTEM_INSTRUCTION`) 검사:
    - "personal_metrics"와 "직접 계산하지"가 들어 있다.
    - "missing_metrics"와 "인바디 기준"이 들어 있다.
    - "weight_loss_5pct_kg"와 "정상 BMI까지"가 들어 있다(스펙 5.4의 감량 목표 규칙).
    - 이전 문구 "감량 비율은 이 사람의 체중으로"는 들어 있지 않다.
  - 기존 `test_rules_turn_guidelines_into_personal_targets`의 "사용자 수치에 적용" 단언은 새 문구에 맞게 바꾼다.
  - `test_payload_has_personal_metrics`: fake OpenAI로 `personal_metrics={"bmi_band": "정상"}` → 프롬프트에 `"personal_metrics": {"bmi_band": "정상"}`
  - `test_trait_header`: `Evidence(..., trait="가족력", category="lifestyle")` → 머리글이 `[S1] (가족력·생활습관) `로 시작
  - `test_analysis_puts_trait_queries_first_and_passes_metrics`(analysis): `retrieve`를 가로채 받은 쿼리를 기록하고 `generate_report`의 `personal_metrics` 인자를 기록 → 첫 쿼리의 `trait`가 None이 아니고, 마지막 특성 쿼리 뒤에 `build_queries` 결과가 이어지고, `personal_metrics`에 `missing_metrics`가 있다
- [ ] **Step 2:** 테스트 → FAIL
- [ ] **Step 3:** 구현. 지시문 변경은 스펙 5.4 목록 그대로:
  - "근거의 일반 권고는 사용자 수치에 적용해…" 두 문장을 개인 목표 수치 규칙으로 바꾼다.
  - 누락 지표 규칙, 인바디 기준 명시 규칙, 감량 목표 규칙을 추가한다.
  - "mg"를 쓰지 않는 기존 단언은 유지한다.
  - `Query.trait`가 `Evidence.trait`로 이어지도록 `select_evidence`에서 `trait=query.trait`를 넣는다.
- [ ] **Step 4:** 테스트 → PASS, 전체 통과
- [ ] **Step 5:** 커밋 `feat(report): feed personal metrics and trait evidence into the report`

### Task 6: 미리보기·비교 스크립트와 실행

**Files:**
- Create: `rag/eval/report_preview.py`
- Modify: `.gitignore`(출력 폴더 `rag/eval/preview/`)

**Interfaces:**
- Consumes: Task 1~5 전부, `training.features.build_features`·`build_labels`
- Produces: `python -m rag.eval.report_preview --out rag/eval/preview/after.json [--inbody-json 경로]`
  - 스펙 5.5의 5명 규칙으로 KNHANES 2023에서 사례를 고른다(규칙과 시드는 대화 기록의 `preview.py`와 같게).
  - `--inbody-json`이 있으면 그 기록들도 넣는다. 사용자 결과지 2장의 `height`, `muscle_control`은 손으로 채운 JSON이다.
  - 앱과 같은 순서로 리포트를 만든다: `compute` → `trait_queries` + `build_queries` → `retrieve` → `generate_report` → `validate_citations`.
  - 결과를 JSON으로 저장한다. 사례마다 특성 쿼리별 검색 결과(임계값을 넘은 청크 수, 제목·장)도 함께 남긴다.
  - `python -m rag.eval.report_preview compare before.json after.json`: 리포트를 프론트 리포트 화면의 섹션 순서(종합 건강 요약 → 세부 위험도 분석 → 인바디 해석 → 맞춤형 개선 가이드 → 보충제 후보 → 주의사항)로 출력하고, 지표 세 개를 함께 출력한다.
    - 사람들 사이 인용 청크의 쌍별 Jaccard 평균. 청크 식별은 `(title, page, heading_path)`.
    - 사용자 수치가 들어간 행동 권고 비율. 행동 문장에 `personal_metrics`나 인바디 값(소수 첫째 자리 문자열)이 있으면 센다.
    - 없는 지표를 해석한 항목 수. 인바디 해석 항목의 `metric`이 `missing_metrics`에 있으면 센다.

- [ ] **Step 1:** 스크립트 작성. 순수 계산 부분(Jaccard, 수치 포함 판정)은 `assert` 기반 `__main__` 없이 `tests/rag/test_report_preview.py`에 테스트 2개로 고정한다.
  - `test_jaccard_mean`: 3명의 청크 집합 → 손으로 계산한 값
  - `test_action_mentions_user_number`: "현재 체중 74.9 kg의 5%" → True, "주 150분" → False(사용자 값 아님)
- [ ] **Step 2:** 테스트 → FAIL 확인 후 구현, PASS
- [ ] **Step 3: 실행**
  - 바꾸기 전 결과는 대화 중에 만든 `preview_a.json`을 `rag/eval/preview/before.json`으로 복사한다.
  - `after`를 만들고 `compare`를 실행한다. GPT 비용은 약 $0.3이다.
  - Expected: 없는 지표 해석 0개, Jaccard가 before보다 낮음, 특성 쿼리 표에 임계값 미달이 몇 개인지가 나온다.
- [ ] **Step 4:** 커밋 `feat(rag-eval): report preview and personalization comparison` (출력 폴더는 git 제외)
- [ ] **Step 5:** 결과를 프론트 섹션 형식으로 사용자에게 보고하고, 임계값 미달 특성 쿼리 문구 수정이 필요한지 함께 제시한다.
