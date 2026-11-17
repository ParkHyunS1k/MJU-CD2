# 위험도 변화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기록별 질환 위험도를 계산에 쓴 입력과 함께 DB에 저장하고, 리포트에 직전 대비 변화와 그 이유를, 그래프용 API에 기록별 위험도를 내보낸다.

**Architecture:** `prediction_service.predict_all`이 측정 연도 기준 나이로 계산하고 입력별 기여를 함께 돌려준다. 새 모듈 `risk_store`가 저장값의 모양·저장·"낡았는지" 판정을 맡고, 기록·프로필이 바뀌는 곳에서 불린다. 새 모듈 `risk_change`가 두 저장값으로 변화와 요인별 몫을 계산하는 순수 함수다.

**Tech Stack:** Python 3, FastAPI, supabase-py, scikit-learn(로지스틱 회귀 번들), pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-risk-change-design.md`

## Global Constraints

- 위험 모델·`models/`·`training/`은 건드리지 않는다. 나이 입력 만드는 방식만 바뀐다.
- 위험도 계산·저장 실패는 로그만 남기고 기록 저장·프로필 저장 요청은 성공시킨다.
- 숫자는 코드가 계산하고 GPT는 해석만 한다. 변화 %p는 소수 첫째 자리로 반올림.
- 마이그레이션 원격 적용·기존 기록 채우기·푸시는 사용자 확인 뒤, 순서는 마이그레이션 → 채우기 → 푸시.
- 테스트는 합성 값만. 개인 데이터 커밋 금지.
- 전체 테스트: `.venv/bin/python -m pytest tests/runtime tests/risk_model tests/test_users_router.py tests/rag/test_load.py tests/rag/test_kdca.py tests/rag/test_report_preview.py tests/rag/test_grounding.py tests/rag/test_eval_judge.py -q -p no:warnings` (이하 `$SUITE`)
- 커밋 메시지 끝: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. `measured_at`이 없는 기록 → `created_at` 연도, 그것도 없으면 오늘 연도로 나이 계산 (Task 1 테스트).
2. 직전 기록이 범위 밖(OCR 오독)·19세 미만 측정 → 건너뛰고 그 전의 계산 가능한 기록과 비교 (Task 5 테스트).
3. 프로필 수정 뒤 다시 계산이 실패해 옛 값이 남은 기록 → 읽을 때 입력 비교로 가려내 다시 계산 (Task 2 테스트).
4. 기록 생성 뒤 위험도 저장이 예외를 내도 생성 응답은 정상 (Task 3 테스트).
5. BMI와 체지방률이 반대 방향으로 움직여 기여 합이 0 → 몫 없이 변화량만 (Task 4 테스트).

---

### Task 1: 측정 연도 기준 나이, 기여를 결과에 포함

**Files:**
- Modify: `app/services/prediction_service.py` (`_inputs`, `predict_all`)
- Test: `tests/risk_model/test_prediction_service.py`

**Interfaces:**
- Produces: `measurement_year(inbody: dict) -> int` — `measured_at` → `created_at` → 오늘 순으로 연도. ISO 문자열(`"2023-07-28T00:00:00+00:00"`)·`date`·`datetime` 모두 받는다.
- Produces: `predict_all`의 질환별 dict에 `"contributions": dict[str, float]` 추가 (`FEATURES` 7개 키 전부, `_contributions` 결과 그대로). 기존 키는 그대로.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_age_uses_measurement_year(models_dir):
    old = {**INBODY, "measured_at": f"{date.today().year - 10}-05-01T00:00:00+00:00"}
    now, past = ps.predict_all(INBODY, PROFILE), ps.predict_all(old, PROFILE)
    assert all(p["contributions"]["age"] < n["contributions"]["age"] for n, p in zip(now, past))

def test_measurement_year_fallbacks():
    assert ps.measurement_year({"measured_at": "2023-07-28T00:00:00+00:00"}) == 2023
    assert ps.measurement_year({"measured_at": None, "created_at": "2024-01-02T00:00:00Z"}) == 2024
    assert ps.measurement_year({}) == date.today().year

def test_minor_at_measurement_raises(models_dir):
    with pytest.raises(ValueError, match="성인"):
        ps.predict_all({**INBODY, "measured_at": f"{AGE35 + 18}-01-01"}, PROFILE)

def test_contributions_cover_all_features(models_dir):
    for r in ps.predict_all(INBODY, PROFILE):
        assert set(r["contributions"]) == set(ps.FEATURES)
```

- [ ] **Step 2:** `.venv/bin/python -m pytest tests/risk_model/test_prediction_service.py -q -p no:warnings` → 새 테스트 4개 FAIL (`measurement_year` 없음, `contributions` KeyError).
- [ ] **Step 3:** `_inputs`의 나이를 `measurement_year(inbody) - birth_year`로 바꾸고 주석에 "측정 연도 기준(설계 2026-11-15)"을 남긴다. `predict_all`에서 `_contributions`를 한 번만 계산해 `factors`와 `contributions`에 같이 쓴다.
- [ ] **Step 4:** 같은 명령 → 전부 PASS. 이어서 `$SUITE` → 전부 PASS (기존 테스트는 `measured_at`이 없어 오늘 연도로 떨어지므로 그대로여야 한다. 깨지면 Ruling으로 기록).
- [ ] **Step 5: 커밋** `feat(risk): age at measurement year, return all contributions`

### Task 2: 저장 모듈 `risk_store`와 마이그레이션 파일

**Files:**
- Create: `supabase/migrations/20261117120000_inbody_disease_risks.sql` — `alter table public.inbody_logs add column disease_risks jsonb;` + 내용 설명 주석(설계 문서 경로).
- Create: `app/services/risk_store.py`
- Test: `tests/runtime/test_risk_store.py`

**Interfaces:**
- Consumes: `predict_all`, `measurement_year` (Task 1).
- Produces (모두 `risk_store`):
  - `COLUMN = "disease_risks"`
  - `inputs_of(record: dict, profile: dict) -> dict` — `{"gender", "birth_year", "fh_hypertension", "fh_diabetes", "fh_dyslipidemia", "measured_year", "bmi", "body_fat_percentage"}` (bmi·체지방률은 `float` 또는 `None`).
  - `from_predictions(record: dict, profile: dict, results: list[dict]) -> dict` — `predict_all` 결과로 저장값을 만든다(`compute`와 Task 5가 같이 씀).
  - `save(sb, record_id: str, value: dict | None) -> None` — 컬럼 갱신. 예외는 `log.exception`만.
  - `compute(record: dict, profile: dict) -> dict | None` — 저장할 값 `{"inputs": inputs_of(...), "age": int, "diseases": [{"disease", "disease_key", "probability", "risk_level", "contributions"}]}`. `predict_all`이 `[]`이거나 `ValueError`면 `None`.
  - `is_current(stored: dict | None, record: dict, profile: dict) -> bool` — `stored`가 있고 `stored["inputs"] == inputs_of(record, profile)`.
  - `ensure(sb, record: dict, profile: dict) -> dict | None` — 저장값이 현재면 그대로, 아니면 `compute` 후 저장(`sb.table("inbody_logs").update({COLUMN: value}).eq("id", record["id"])`). 저장 예외는 `log.exception`만. 계산 결과(`None` 포함)를 돌려준다.
  - `refresh_record(sb, record: dict, profile: dict | None = None) -> None` — 무조건 다시 계산해 저장. `profile`이 없으면 `users` 테이블에서 `record["user_id"]`로 읽는다. 모든 예외를 잡아 로그만.
  - `refresh_user(sb, user_id: str, profile: dict) -> None` — 그 사용자 기록 전부 `refresh_record`.

테스트용 가짜 supabase: `tests/test_users_router.py`의 `FakeTable` 방식으로, `update()` 호출을 `(data, eq 값)` 목록에 기록하고 `select` 결과 행을 돌려주는 작은 클래스를 이 파일에 둔다. `predict_all`은 monkeypatch로 고정값을 돌려준다.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_compute_none_when_uncomputable(monkeypatch):  # 빈 목록, ValueError 둘 다
def test_ensure_returns_stored_when_inputs_match(fake_sb):  # update 호출 0회
def test_ensure_recomputes_when_profile_changed(fake_sb):   # 저장값 inputs.fh_diabetes=None, 지금 프로필 True → update 1회, 새 값 반환
def test_ensure_recomputes_when_record_values_changed(fake_sb):  # 행의 bmi가 inputs.bmi와 다름
def test_ensure_returns_value_even_if_save_fails(fake_sb):  # update가 예외 → 값은 반환, caplog에 기록
def test_refresh_record_swallows_errors(fake_sb):           # predict_all이 RuntimeError → 예외 없이 끝남
```
각 테스트의 assert는 반환값, `fake_sb`가 기록한 update 목록, `caplog.text`로 확인한다.

- [ ] **Step 2:** `.venv/bin/python -m pytest tests/runtime/test_risk_store.py -q -p no:warnings` → FAIL (모듈 없음).
- [ ] **Step 3:** `risk_store.py` 구현. `compute`의 `age`는 `measured_year - birth_year`.
- [ ] **Step 4:** 같은 명령 → PASS, `$SUITE` → PASS.
- [ ] **Step 5: 커밋** `feat(risk): store per-record risks with the inputs used`

### Task 3: 기록·프로필이 바뀌는 곳에서 저장

**Files:**
- Modify: `app/services/inbody_service.py` (`create_inbody_record`, `create_inbody_from_upload`, `update_inbody_record`)
- Modify: `app/routers/users.py` (`update_me`)
- Test: `tests/runtime/test_risk_write_paths.py`

**Interfaces:**
- Consumes: `risk_store.refresh_record(sb, record, profile=None)`, `risk_store.refresh_user(sb, user_id, profile)` (Task 2).
- 프로필 트리거 키: `RISK_PROFILE_FIELDS = {"gender", "birth_year", "fh_hypertension", "fh_diabetes", "fh_dyslipidemia"}` (`users.py`에 둔다).

- [ ] **Step 1: 실패하는 테스트**

```python
def test_create_record_refreshes_risks(monkeypatch):   # refresh_record가 삽입된 행으로 1회 호출
def test_update_record_refreshes_risks(monkeypatch):   # 수정된 행으로 1회 호출
def test_upload_record_refreshes_risks(monkeypatch):   # extract_inbody_from_image·upload_inbody_file 가짜로
def test_create_record_succeeds_when_refresh_raises(monkeypatch):  # refresh_record가 예외 → 응답은 정상 행
def test_profile_patch_refreshes_only_for_risk_fields(table, monkeypatch):
    # nickname만 → refresh_user 0회, fh_diabetes 포함 → 1회(user_id, 저장된 행)
```
(`refresh_record` 자체가 예외를 삼키지만, 호출부도 `try/except Exception: log.exception`으로 감싼다 — 기록 저장이 우선이라는 반목표를 호출부에서 보장. Review Focus 4.)

- [ ] **Step 2:** `.venv/bin/python -m pytest tests/runtime/test_risk_write_paths.py -q -p no:warnings` → FAIL.
- [ ] **Step 3:** 세 함수에서 insert/update 성공 직후 `refresh_record(sb, row)` 호출(감싼 채로). `update_me`는 `update_data.keys() & RISK_PROFILE_FIELDS`가 있으면 `refresh_user(sb, current.id, res.data[0])`.
- [ ] **Step 4:** 같은 명령 → PASS, `$SUITE` → PASS.
- [ ] **Step 5: 커밋** `feat(risk): refresh stored risks on record and profile changes`

### Task 4: 변화 계산 `risk_change`

**Files:**
- Create: `app/services/risk_change.py`
- Test: `tests/runtime/test_risk_change.py`

**Interfaces:**
- Produces: `SHARE_FEATURES = {"bmi": "BMI", "body_fat_percentage": "체지방률", "age": "나이"}`
- Produces: `compute(previous: dict, current: dict, previous_measured_at: str) -> list[dict]` — `previous`·`current`는 `risk_store.compute` 모양. 같은 `disease_key`끼리 짝짓고, 질환마다:
  `{"disease", "previous_measured_at": "YYYY-MM-DD", "previous_pct", "current_pct", "change_pp", "share_pp": {"BMI", "체지방률", "나이"} | None, "previous_inputs": {"bmi", "body_fat_percentage"}, "current_inputs": {...}}`. 숫자는 소수 첫째 자리.
- 몫: Δp(%p) × Δc_f / ΣΔc, ΣΔc는 7개 기여 차이의 합. `abs(ΣΔc) < 1e-9`면 `share_pp = None`. 반올림은 마지막에만.

- [ ] **Step 1: 실패하는 테스트** (저장값 dict를 직접 만든다. 모델 불필요)

```python
def test_change_and_shares_sum_to_change():
    # prev p=0.30, now p=0.25, Δc: bmi -0.10, pbf -0.20, age +0.05, 나머지 0
    r = compute(PREV, NOW, "2025-06-02T00:00:00+00:00")[0]
    assert r["change_pp"] == -5.0 and r["previous_measured_at"] == "2025-06-02"
    assert r["share_pp"] == {"BMI": -2.0, "체지방률": -4.0, "나이": 1.0}

def test_zero_logodds_change_has_no_shares():  # bmi +0.1, pbf -0.1 → share_pp is None
def test_only_common_diseases_are_paired():   # 한쪽에만 있는 질환은 결과에 없음
```
(첫 테스트의 확률은 Δp = −5.0, ΣΔc = −0.25가 되게 고른다: 몫 = −5 × (−0.10/−0.25) = −2.0 등.)

- [ ] **Step 2:** `.venv/bin/python -m pytest tests/runtime/test_risk_change.py -q -p no:warnings` → FAIL.
- [ ] **Step 3:** 구현.
- [ ] **Step 4:** 같은 명령 → PASS.
- [ ] **Step 5: 커밋** `feat(risk): risk change between two records with per-factor shares`

### Task 5: 리포트에 변화 넣기

**Files:**
- Modify: `app/services/inbody_analysis_service.py` (`_analyze_inbody_record`, 새 `_previous_risks`)
- Modify: `app/services/ai_service.py` (`generate_report`, `_REPORT_GUIDE`)
- Test: `tests/runtime/test_analysis_rag.py`, `tests/runtime/test_report_prompt.py`

**Interfaces:**
- Consumes: `risk_store.ensure`, `risk_store.is_current`, `risk_store.from_predictions`, `risk_store.save`, `risk_store.COLUMN` (Task 2), `risk_change.compute` (Task 4).
- Produces: `generate_report(..., risk_change: list[dict] | None = None)` → payload 키 `"risk_change"`.
- 대상 기록: 지금처럼 `predict_all(target, user)`로 계산(오류 메시지·400/422 경로 유지). 같은 결정적 계산이라 저장값과 같다. 저장값이 `is_current`가 아니면 `risk_store.save(sb, target["id"], risk_store.from_predictions(target, user, disease_risks))`(실패는 로그). — 설계의 "null이면 계산해 저장"을 오류 경로를 지키며 구현한 것.
- `_previous_risks(sb, user, records, target) -> tuple[dict, dict] | None`: `measured_at`이 대상보다 이른 기록을 최근 순으로 돌며 `risk_store.ensure`가 `None`이 아닌 첫 기록의 `(record, stored)`.
- 프롬프트 규칙(`_REPORT_GUIDE`의 risk_report 줄 뒤에 추가):
  `"risk_change가 있으면 risk_report의 각 message에 직전 측정 대비 변화(change_pp)와 가장 큰 share_pp 요인을 한 번 쓰세요(사용자 데이터라 인용하지 않음). |change_pp|가 1.0 미만이면 '지난번과 비슷'으로 쓰세요. '나이' 몫은 나이가 들며 자연히 늘어난 몫으로 구분해 쓰고 몸의 변화와 섞지 마세요. share_pp가 null이면 이유를 쓰지 마세요."`

- [ ] **Step 1: 실패하는 테스트**

```python
# test_analysis_rag.py (기존 wired 픽스처 사용, risk_store.ensure를 monkeypatch)
def test_report_gets_risk_change_from_previous_record(wired, monkeypatch):  # generate kw["risk_change"][0]["change_pp"] 확인
def test_previous_skips_uncomputable_records(wired, monkeypatch):  # 직전 ensure → None, 그 전 → 값 → 그 전 기록 날짜가 previous_measured_at
def test_first_record_has_no_risk_change(wired, monkeypatch):      # kw["risk_change"] is None

# test_report_prompt.py
def test_guide_has_risk_change_rule():  # "risk_change", "1.0", "나이" in _REPORT_GUIDE
def test_payload_includes_risk_change(monkeypatch):  # 가짜 OpenAI 클라이언트가 받은 input에 "change_pp" 포함
```
기존 `wired`의 `predict_all` 가짜(`RISKS`)에는 `contributions`가 없다. 픽스처의 `RISKS`에 7개 키 `contributions`를 추가한다.

- [ ] **Step 2:** `.venv/bin/python -m pytest tests/runtime/test_analysis_rag.py tests/runtime/test_report_prompt.py -q -p no:warnings` → 새 테스트 FAIL.
- [ ] **Step 3:** 구현.
- [ ] **Step 4:** 같은 명령 → PASS, `$SUITE` → PASS.
- [ ] **Step 5: 커밋** `feat(report): tell how each risk changed since the previous record`

### Task 6: 위험도 추이 API와 기존 기록 채우기

**Files:**
- Modify: `app/services/risk_store.py` (`trend`, `backfill`, `__main__`)
- Modify: `app/routers/inbody.py` (`GET /inbody/risk-trend`, `/{inbody_id}` 보다 위)
- Test: `tests/runtime/test_risk_store.py`

**Interfaces:**
- Produces: `trend(sb, user_id: str) -> list[dict]` — 사용자 행과 기록(측정일 오름차순)을 읽고 기록마다 `ensure`, `None`은 뺀다. 항목: `{"inbody_id", "measured_at", "risks": [{"disease", "disease_key", "probability", "risk_level"}]}`. 사용자 행이 없으면 빈 목록.
- Produces: `backfill(sb) -> int` — `users` 전체를 돌며 `refresh_user`, 처리한 기록 수를 돌려준다. `python -m app.services.risk_store`로 실행하면 `backfill(get_supabase_admin())` 결과를 출력.
- 라우터: `@router.get("/risk-trend")` → `trend(get_supabase_admin(), current.id)`.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_trend_orders_by_date_and_drops_uncomputable(fake_sb, monkeypatch)
def test_trend_has_no_contributions(fake_sb, monkeypatch)
def test_risk_trend_route_registered_before_record_route():
    from main import app
    paths = list(app.openapi()["paths"])
    assert paths.index("/inbody/risk-trend") < paths.index("/inbody/{inbody_id}")
```

- [ ] **Step 2:** `.venv/bin/python -m pytest tests/runtime/test_risk_store.py -q -p no:warnings` → 새 테스트 FAIL.
- [ ] **Step 3:** 구현.
- [ ] **Step 4:** 같은 명령 → PASS, `$SUITE` → PASS.
- [ ] **Step 5: 커밋** `feat(risk): risk trend endpoint and backfill`

### Task 7: 원격 적용 (사용자 확인 필요)

- [ ] **Step 1:** 사용자에게 확인: 마이그레이션 원격 적용 → 기존 기록 채우기 → `back` 푸시.
- [ ] **Step 2:** 승인되면 Supabase MCP `apply_migration`(이름 `inbody_disease_risks`, 파일 내용 그대로).
- [ ] **Step 3:** `PYTHONPATH=. .venv/bin/python -m app.services.risk_store` → 처리 기록 수 출력. 이어서 SQL `select count(*), count(disease_risks) from inbody_logs`로 확인(값 자체는 출력하지 않음).
- [ ] **Step 4:** `git push origin back`.
- [ ] **Step 5:** 프론트 담당자에게 넘길 API 형태(`GET /inbody/risk-trend` 응답 예시, 날짜 라벨 사용)를 사용자에게 정리해 준다.
