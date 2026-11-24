# 식약처 기능성 원료 영양제 근거 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 식약처 건강기능식품 기능성 원료(고시형+개별인정형) 중 앱 질환 관련 원료를 파일로 받아 두고, 위험 질환·체성분 유형 태그로 골라 영양제 근거로 프롬프트에 넣는다.

**Architecture:** 수집 스크립트 `rag/mfds.py`가 식품안전나라 API(I2710, I-0050)를 받아 태그를 단 원료 목록을 `app/data/mfds_ingredients.json`에 쓴다. 런타임 `app/services/supplement_evidence.py`가 그 파일을 읽어 태그별로 원료를 골라 `Evidence`로 만들고, 분석 서비스가 검색 근거 뒤에 붙인다. 인용 검증은 영양제 후보에 식약처 근거 인용을 요구한다.

**Tech Stack:** Python 3, FastAPI 서비스 모듈, urllib(수집), pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-mfds-supplement-ingredients-design.md`

## Global Constraints

- 런타임은 식약처 API를 부르지 않는다. `app/data/mfds_ingredients.json`만 읽는다.
- API 키(`FOODSAFETYKOREA_API_KEY`)는 코드·로그·커밋에 넣지 않는다. 오류 메시지에서 키를 가린다.
- 태그당 최대 5개(고시형 먼저, 같은 그룹은 파일 순서), 영양제 근거 전체 최대 10개, 태그별 1위부터 라운드 로빈, 중복 원료 한 번만.
- 영양제 근거는 기존 `MAX_EVIDENCE = 15` 근거와 별도로 뒤에 붙인다. 기존 근거를 빼지 않는다.
- 응답 스키마(`SupplementCandidate`)와 프론트는 바꾸지 않는다.
- 테스트는 합성 원료로. 개인 데이터 금지.
- 테스트 명령(SUITE): `.venv/bin/python -m pytest tests/runtime tests/risk_model tests/test_users_router.py tests/rag/test_load.py tests/rag/test_kdca.py tests/rag/test_mfds.py tests/rag/test_report_preview.py tests/rag/test_grounding.py tests/rag/test_eval_judge.py -q -p no:warnings`

## Review Focus

1. 위험이 모두 low이고 체성분 유형이 `해당 없음`/없음 → 영양제 근거 0개(Task 2 테스트).
2. 원료 파일이 없거나 깨짐 → 리포트는 영양제 근거 없이 계속 생성, 로그만(Task 2 테스트).
3. 검색 근거가 S1–S15를 쓰면 영양제 근거는 S16부터 이어져 번호가 겹치지 않음(Task 2·4 테스트).
4. 포털 영양제 문서와 식약처 원료를 함께 인용한 후보는 둘 다 남고, 포털만 인용한 후보는 빠짐(Task 3 테스트).
5. 칼슘·마그네슘("신경과 근육 기능 유지에 필요")·단백질은 `muscle` 태그가 아님(Task 1 테스트).

---

### Task 1: 수집 스크립트와 원료 파일

**Files:**
- Create: `rag/mfds.py`, `app/data/mfds_ingredients.json`(스크립트 실행 결과)
- Test: `tests/rag/test_mfds.py`

**Interfaces:**
- Produces: `app/data/mfds_ingredients.json` — 원료 dict 목록. 키: `name`(인정번호 뺀 원료명), `notified`(bool, 고시형이면 true),
  `recognition_no`(`"2009-39"` 형식 또는 null), `function`(주된 기능성 원문), `intake`(사람이 읽는 1일 섭취량 문자열 또는 null),
  `cautions`(섭취 시 주의사항 원문 또는 null), `tags`(list[str]), `updated`(`"YYYYMMDD"` 또는 null). 파일 순서: I2710 원본 순서, 그 뒤 I-0050에서 더한 것.
- Produces: `rag.mfds.TAGS: dict[str, str]` 태그 → 정규식. 태그 이름: `blood_glucose`, `blood_pressure`, `cholesterol`, `triglyceride`, `body_fat`, `muscle`.

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/rag/test_mfds.py`)

```python
from rag import mfds

def test_tags_from_function_text():
    assert mfds.tags_of("식후 혈당상승 억제에 도움을 줄 수 있음") == ["blood_glucose"]
    assert mfds.tags_of("혈중 중성지질 개선·혈중 콜레스테롤 개선에 도움") == ["cholesterol", "triglyceride"]
    assert mfds.tags_of("(국문) 근력개선에 도움을 줄 수 있음") == ["muscle"]
    assert mfds.tags_of("관절 건강에 도움을 줄 수 있음") == []

def test_nutrient_muscle_function_is_not_muscle_tag():
    for text in ("신경과 근육 기능 유지에 필요", "근육, 결합조직 등 신체조직의 구성성분"):
        assert "muscle" not in mfds.tags_of(text)

def test_recognition_no_and_clean_name():
    assert mfds.recognition_no("Nopal 추출물(제2009-39호)") == "2009-39"
    assert mfds.recognition_no("차조기등복합추출물(기능성원료인정New제2007-13호)") == "2007-13"
    assert mfds.recognition_no("아라비아검(아카시아검)") is None
    assert mfds.clean_name("두충우슬추출복합물(KGC08EA)(제2023-40호)") == "두충우슬추출복합물(KGC08EA)"

def test_build_keeps_relevant_adds_missing_individual_and_formats_intake():
    i2710 = [_a("바나바잎 추출물", "식후 혈당상승 억제에 도움을 줄 수 있음"),
             _a("Nopal 추출물(제2009-39호)", "식후 혈당상승 억제에 도움을 줄 수 있음"),
             _a("루테인", "노화로 인해 감소될 수 있는 황반색소밀도를 유지")]
    i0050 = [_b("Nopal 추출물(기능성원료인정제2009-39호)", "제2009-39호", "혈당"),
             _b("아마인(기능성원료인정제2007-4호)", "제2007-4호", "혈중 콜레스테롤 개선에 도움")]
    out = mfds.build_ingredients(i2710, i0050)
    assert [i["name"] for i in out] == ["바나바잎 추출물", "Nopal 추출물", "아마인"]
    assert [i["notified"] for i in out] == [True, False, False]
    assert out[0]["intake"] == "1.4–4.2 g" and out[0]["updated"] == "20200820"
    assert out[2]["recognition_no"] == "2007-4" and out[2]["updated"] is None

def test_count_check_refuses_empty_or_shrunk_lists():
    import pytest
    with pytest.raises(mfds.IncompleteFetch):
        mfds.check_counts([], previous=None)
    with pytest.raises(mfds.IncompleteFetch):
        mfds.check_counts([{}] * 70, previous=[{}] * 100)
    mfds.check_counts([{}] * 90, previous=[{}] * 100)
```

`_a(name, fn)`은 I2710 행(`PRDCT_NM`, `PRIMARY_FNCLTY`, `DAY_INTK_LOWLIMIT="1.4"`, `DAY_INTK_HIGHLIMIT="4.2"`, `INTK_UNIT="g"`,
`INTK_MEMO=""`, `IFTKN_ATNT_MATR_CN="주의"`, `LAST_UPDT_DTM="20200820"`), `_b(name, no, fn)`은 I-0050 행(`RAWMTRL_NM`,
`HF_FNCLTY_MTRAL_RCOGN_NO`, `PRIMARY_FNCLTY`, `DAY_INTK_LOWLIMIT="500 (mg/일)"`, `DAY_INTK_HIGHLIMIT="500 (mg/일)"`,
`IFTKN_ATNT_MATR_CN=""`)을 만드는 테스트 도우미.

- [ ] **Step 2: 실행해 실패 확인** — `.venv/bin/python -m pytest tests/rag/test_mfds.py -q` → `ModuleNotFoundError: rag.mfds`.

- [ ] **Step 3: `rag/mfds.py` 구현**
  - `TAGS`: blood_glucose `혈당`, blood_pressure `혈압`, cholesterol `콜레스테롤`, triglyceride `중성지[방질]`, body_fat `체지방`, muscle `근력`.
    `tags_of(function: str) -> list[str]`는 `TAGS` 순서대로 맞는 태그.
  - `recognition_no(name: str) -> str | None`: `제\s*(\d{4}-\d+)\s*호`의 숫자 부분. `clean_name(name) -> str`: 끝의 `(…제NNNN-NN호)` 괄호 하나를 뺀다.
  - `build_ingredients(i2710: list[dict], i0050: list[dict]) -> list[dict]`: 태그 없는 행은 버린다. I2710 행은 원명 번호가 없으면 고시형.
    I-0050은 `HF_FNCLTY_MTRAL_RCOGN_NO`의 번호가 I2710 관련 원료에 없을 때만 더한다. intake: 하한=상한이면 그 값, 다르면 `하한–상한`; I2710은 뒤에 ` 단위`,
    메모가 있으면 ` (메모)`; 빈 값이면 null. 빈 문자열은 null.
  - `IncompleteFetch(RuntimeError)`, `check_counts(ingredients, previous: list | None)`: 0개이거나 previous의 80% 미만이면 raise.
  - `fetch(service: str, key: str) -> list[dict]`: `http://openapi.foodsafetykorea.go.kr/api/{key}/{service}/json/{s}/{e}`를 1,000건씩 `total_count`까지.
    `RESULT.CODE`가 `INFO-000`이 아니면 키를 가린 메시지로 raise.
  - `main()`: `.env` 로드 → 두 서비스 fetch → build → 기존 파일과 check_counts → `app/data/mfds_ingredients.json`(ensure_ascii=False, indent=1) 쓰기 → 건수·태그별 개수 출력.

- [ ] **Step 4: 테스트 통과 확인** — 같은 명령 → 5 passed.

- [ ] **Step 5: 실제 수집** — `.venv/bin/python -m rag.mfds`
  Expected: 약 180개(150–220), 태그별 개수 출력. 칼슘·마그네슘이 `muscle`에 없는지 `grep`으로 확인.

- [ ] **Step 6: 커밋** — `git add rag/mfds.py tests/rag/test_mfds.py app/data/mfds_ingredients.json && git commit -m "feat(rag): collect MFDS functional ingredients for supplement evidence"`

### Task 2: 원료 고르기와 Evidence 만들기

**Files:**
- Create: `app/services/supplement_evidence.py`
- Test: `tests/runtime/test_supplement_evidence.py`

**Interfaces:**
- Consumes: Task 1의 JSON 모양과 태그 이름. `Evidence`(`app/services/retrieval_service.py`).
- Produces:
  - `MFDS_DOCUMENT_ID = "mfds"`
  - `needed_tags(disease_risks: list[dict], metrics: dict) -> list[str]`
  - `select_ingredients(ingredients: list[dict], tags: list[str], per_tag: int = 5, limit: int = 10) -> list[tuple[str, dict]]` (태그, 원료)
  - `supplement_evidence(disease_risks: list[dict], metrics: dict, start: int, ingredients: list[dict] | None = None) -> list[Evidence]` — `start`는 첫 S번호(검색 근거 수 + 1). `ingredients`가 None이면 파일을 읽는다(읽기 실패 시 로그 후 `[]`).

- [ ] **Step 1: 실패하는 테스트 작성**

```python
from app.services import supplement_evidence as se

def _ing(name, tags, notified=True, no=None):
    return {"name": name, "notified": notified, "recognition_no": no, "function": "기능", "intake": "1 g",
            "cautions": "주의", "tags": tags, "updated": "20260101"}

RISK = lambda key, level: {"disease_key": key, "risk_level": level}

def test_needed_tags_from_risks_and_body_type():
    risks = [RISK("diabetes", "high"), RISK("hypertension", "low"), RISK("triglyceride", "medium")]
    assert se.needed_tags(risks, {"body_type": {"type": "지방→근육형"}}) == \
        ["blood_glucose", "triglyceride", "body_fat", "muscle"]

def test_unknown_muscle_with_fat_excess_gets_body_fat_only():
    assert se.needed_tags([], {"body_type": {"type": "근육 판정 불가", "fat_excess": True}}) == ["body_fat"]
    assert se.needed_tags([], {"body_type": {"type": "근육 판정 불가", "fat_excess": False}}) == []

def test_all_low_and_no_type_gives_no_tags():
    assert se.needed_tags([RISK("diabetes", "low")], {"body_type": {"type": "해당 없음"}}) == []
    assert se.needed_tags([RISK("diabetes", "low")], {}) == []

def test_notified_first_cap_per_tag_round_robin_dedupe_limit():
    ings = [_ing(f"개별{i}", ["blood_glucose"], False, f"2020-{i}") for i in range(3)] + \
           [_ing(f"고시{i}", ["blood_glucose"]) for i in range(6)] + \
           [_ing("공통", ["blood_glucose", "cholesterol"]), _ing("콜1", ["cholesterol"])]
    picked = se.select_ingredients(ings, ["blood_glucose", "cholesterol"], per_tag=5, limit=10)
    names = [i["name"] for _, i in picked]
    assert names[:2] == ["고시0", "공통"]          # 태그별 1위부터 번갈아
    assert len(names) == len(set(names))            # 공통은 한 번만
    assert [n for n in names if n.startswith("고시")] == [f"고시{i}" for i in range(5)]
    assert not any(n.startswith("개별") for n in names)   # 고시형 5개가 먼저 차서 개별인정형은 밀림
    assert len(se.select_ingredients(ings * 3, ["blood_glucose", "cholesterol"], limit=4)) <= 4

def test_evidence_numbering_and_fields():
    ings = [_ing("바나바잎 추출물", ["blood_glucose"]), _ing("Nopal 추출물", ["blood_glucose"], False, "2009-39")]
    ev = se.supplement_evidence([RISK("diabetes", "high")], {}, start=16, ingredients=ings)
    assert [e.sid for e in ev] == ["S16", "S17"]
    assert {e.chunk_category for e in ev} == {"supplement"} and {e.document_id for e in ev} == {"mfds"}
    assert ev[0].heading_path == "고시형" and ev[1].heading_path == "개별인정형 제2009-39호"
    assert ev[0].publisher == "식품의약품안전처" and ev[0].year == 2026 and ev[0].page_start is None
    assert ev[0].content.split("\n", 1)[1].startswith("주된 기능성: 기능")
    assert all(e.chunk_id < 0 for e in ev)

def test_missing_file_gives_no_evidence(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(se, "DATA", tmp_path / "none.json")
    se.load_ingredients.cache_clear()
    assert se.supplement_evidence([RISK("diabetes", "high")], {}, start=1) == []
    assert "mfds" in caplog.text
```

- [ ] **Step 2: 실행해 실패 확인** — `.venv/bin/python -m pytest tests/runtime/test_supplement_evidence.py -q` → ImportError.

- [ ] **Step 3: `app/services/supplement_evidence.py` 구현**
  - `DATA = Path(__file__).resolve().parents[1] / "data" / "mfds_ingredients.json"`, `load_ingredients()`는 `functools.lru_cache`로 한 번 읽는다.
  - 질환 매핑: `hypertension→blood_pressure`, `cholesterol→cholesterol`, `diabetes→blood_glucose`, `triglyceride→triglyceride`(risk_level medium·high만, disease_risks 순서).
    유형 매핑: `지방 감량형→[body_fat]`, `지방→근육형→[body_fat, muscle]`, `근육 보강형→[muscle]`, `근육 판정 불가`+`fat_excess`→`[body_fat]`. 중복 태그 제거.
  - `select_ingredients`: 태그마다 `[고시형(파일 순서)] + [개별인정형(파일 순서)]`의 앞 `per_tag`개를 만든 뒤, 순위 0부터 태그를 돌며 이미 뽑힌 원료(이름+인정번호)는 건너뛰고 `limit`까지.
  - Evidence: `sid=f"S{start+i}"`, `chunk_id=-(i+1)`, `document_id=MFDS_DOCUMENT_ID`, `title=name`, `publisher="식품의약품안전처"`,
    `year=int(updated[:4])` 또는 None, `url=None`, `heading_path="고시형"` 또는 `f"개별인정형 제{no}호"`, page 없음,
    `content=f"{name}\n주된 기능성: {function}\n1일 섭취량: {intake or '정보 없음'}\n섭취 시 주의사항: {cautions or '없음'}"`,
    `similarity=1.0`, `condition="general"`, `category="supplement"`, `chunk_category="supplement"`, `trait=f"{TAG_NAMES[tag]} 기능성 원료"`
    (`TAG_NAMES`: 혈당·혈압·콜레스테롤·중성지방·체지방·근력).
  - 파일 읽기 실패(`OSError`, `ValueError`)는 `log.exception("failed to load mfds ingredients")` 후 `[]`.

- [ ] **Step 4: 통과 확인** — 같은 명령 → 6 passed.

- [ ] **Step 5: 커밋** — `git add app/services/supplement_evidence.py tests/runtime/test_supplement_evidence.py && git commit -m "feat(report): pick MFDS ingredients by risk and body type as supplement evidence"`

### Task 3: 인용 검증과 프롬프트 규칙

**Files:**
- Modify: `app/services/citation_service.py` (`validate_citations`), `app/services/ai_service.py` (`_REPORT_SYSTEM_INSTRUCTION` 또는 `_REPORT_GUIDE`의 영양제 규칙)
- Test: `tests/runtime/test_citation_service.py`, `tests/runtime/test_report_prompt.py`

**Interfaces:**
- Consumes: `supplement_evidence.MFDS_DOCUMENT_ID`.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# test_citation_service.py
def test_supplement_needs_an_mfds_citation_portal_kept_alongside():
    from dataclasses import replace  # Evidence는 frozen
    portal = replace(_ev("S7"), chunk_category="supplement")
    mfds = replace(_ev("S8"), chunk_category="supplement", document_id="mfds")
    c = _content()
    c["supplement_candidates"][0]["source_ids"] = ["S7"]          # 포털만
    c["supplement_candidates"][1]["source_ids"] = ["S7", "S8"]    # 포털 + 식약처
    out, sources = validate_citations(c, [portal, mfds])
    assert [s["name"] for s in out["supplement_candidates"]] == ["홍국"]
    assert out["supplement_candidates"][0]["source_ids"] == ["S7", "S8"]

# test_report_prompt.py
def test_prompt_has_mfds_supplement_rules():
    text = ai_service._REPORT_SYSTEM_INSTRUCTION + ai_service._REPORT_GUIDE
    assert "식품의약품안전처" in text and "섭취 시 주의사항" in text and "원료명" in text
```

기존 영양제 테스트가 포털(`document_id="d"`) 근거만으로 후보를 남기는 걸 기대하면, 그 근거를 `document_id="mfds"`로 바꾸고 이유를 커밋 메시지에 쓴다.

- [ ] **Step 2: 실행해 실패 확인** — `.venv/bin/python -m pytest tests/runtime/test_citation_service.py tests/runtime/test_report_prompt.py -q` → 새 테스트 2개 FAIL.

- [ ] **Step 3: 구현**
  - `validate_citations`: 영양제 후보는 supplement 근거 sid만 남긴 뒤, 그중 `document_id == MFDS_DOCUMENT_ID`인 것이 하나라도 있어야 유지. 주석에 이유(포털 문서는 일반 안내라 원료 근거가 아님).
  - 프롬프트 규칙(한국어, 기존 영양제 규칙 옆): 영양제 후보는 식품의약품안전처 기능성 원료 근거를 반드시 인용하고 `name`은 그 근거의 원료명 그대로; `cautions`·`not_recommended_if`는 그 근거의 섭취 시 주의사항에서; 기능성은 근거 문구("~에 도움을 줄 수 있음")를 넘지 말 것; 포털 영양제 문서는 주의사항 근거로만 함께 인용.

- [ ] **Step 4: 통과 확인** — SUITE 전체 통과.

- [ ] **Step 5: 커밋** — `git commit -m "feat(report): supplement candidates must cite an MFDS ingredient"`

### Task 4: 분석 흐름·미리보기 연결과 확인

**Files:**
- Modify: `app/services/inbody_analysis_service.py` (`_analyze_inbody_record`, 검색 직후), `rag/eval/report_preview.py` (`_report`)
- Test: `tests/runtime/test_analysis_rag.py`

**Interfaces:**
- Consumes: `supplement_evidence(disease_risks, metrics, start=len(evidence) + 1)`.

- [ ] **Step 1: 실패하는 테스트 작성** (`test_analysis_rag.py`, 기존 `wired` 픽스처 사용)

```python
def test_supplement_evidence_follows_retrieved_evidence(wired, monkeypatch):
    monkeypatch.setattr(svc, "retrieve", lambda queries: [EV])
    sup = Evidence(sid="S2", chunk_id=-1, document_id="mfds", title="바나바잎 추출물", publisher="식품의약품안전처",
                   year=2026, url=None, heading_path="고시형", page_start=None, page_end=None, content="x\ny",
                   similarity=1.0, condition="general", category="supplement", chunk_category="supplement")
    seen = {}
    monkeypatch.setattr(svc, "supplement_evidence",
                        lambda risks, metrics, start: seen.setdefault("start", start) and [sup])
    svc._analyze_inbody_record(None, USER, [TARGET], TARGET)
    assert seen["start"] == 2
    assert [e.sid for e in wired["generate"]["evidence"]] == ["S1", "S2"]
```

`wired` 픽스처에는 `monkeypatch.setattr(svc, "supplement_evidence", lambda risks, metrics, start: [])`를 기본으로 넣는다
(기존 테스트가 실제 원료 파일에 기대지 않게). 새 테스트는 그 위에 다시 덮어쓴다.

- [ ] **Step 2: 실행해 실패 확인** — `.venv/bin/python -m pytest tests/runtime/test_analysis_rag.py -q` → AttributeError(`supplement_evidence`).

- [ ] **Step 3: 구현** — 검색 직후 `evidence = [*evidence, *supplement_evidence(disease_risks, metrics, start=len(evidence) + 1)]`.
  `report_preview._report`도 같은 줄을 넣는다(앱과 같은 함수).

- [ ] **Step 4: 통과 확인** — SUITE 전체 통과.

- [ ] **Step 5: 커밋** — `git commit -m "feat(report): append MFDS supplement evidence after retrieval"`

- [ ] **Step 6: 미리보기로 확인** —
  `PYTHONPATH=. .venv/bin/python -m rag.eval.report_preview run --out rag/eval/preview/mfds.json --with-muscle --with-previous`
  Expected: 위험이 medium 이상인 사람은 `supplement_candidates`가 1개 이상이고, 모든 후보의 `source_ids`에 식약처 근거가 있다.
  위험 낮음(여성)은 체성분 유형에 따라 0개일 수 있다. 결과를 앱 화면(발표/report-main)으로 캡처해 사용자에게 보여 준다.
