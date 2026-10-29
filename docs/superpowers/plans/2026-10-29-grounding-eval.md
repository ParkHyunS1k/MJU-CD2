# 리포트 근거 정확도 평가 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 같은 10명에 대해 단순 프롬프트(B안)와 RAG 리포트를 만들고, 의학 주장 단위 GPT 판정으로 근거 인용률·근거 없는 주장 비율·인용 정확도를 낸다.

**Architecture:** `rag/eval/grounding.py` 한 파일. 순수 함수(프롬프트 거르기, 필드 직렬화, 판정 정리, 지표)와 외부 호출(GPT 리포트, Codex 판정, 임베딩·RPC)을 함수 인자로 분리해 가짜로 테스트한다. 단계 결과는 케이스·버전별로 `state.json`에 저장해 끊겨도 이어 간다.

**Tech Stack:** Python, pydantic, OpenAI Responses API(`ai_service.generate_report`), Codex CLI(`rag/eval/judge.codex_ask`), Gemini 임베딩 + Supabase `match_rag_chunks`.

**Spec:** `docs/superpowers/specs/2026-10-29-grounding-eval-design.md`

## Global Constraints

- 리포트 모델: 두 버전 모두 `ai_service.generate_report`(gpt-6.1-terra, reasoning low). 기준 버전을 약하게 만들지 않는다.
- 단순 프롬프트: `_REPORT_SYSTEM_INSTRUCTION`·`_REPORT_GUIDE`에서 "근거", "S번호", "source_ids", "직전 리포트"가 든 문장만 뺀다. [근거] 블록 없음, 후처리 없음, source_ids는 비운다.
- 앱 코드(`app/`)는 바꾸지 않는다. 기준 버전은 평가 스크립트 안에서 `unittest.mock.patch.object`로 프롬프트 상수와 `build_report_prompt`를 바꿔 끼운다.
- 판정기는 Codex CLI(GPT). Gemini LLM 호출 금지(임베딩만 Gemini).
- 판정 프롬프트에 버전 정보를 넣지 않는다. 코퍼스 판정 후보는 두 버전 모두 주장 문장 검색 상위 5개(유사도 하한 없음)만.
- 지표 대상은 `kind == "medical"` 주장만. 인용률 분모는 인용 칸이 있는 항목(watch_points, risk, plan, supp)의 medical 주장.
- 결과는 `rag/eval/preview/grounding/`(gitignore)에만. 키는 `.env`.
- 임베딩 무료 등급 분당 100건: 리포트 사이에 쉬어 간다(기존 `report_preview.run`처럼 15초 이상).

## Review Focus

1. Codex가 입력에 없는 claim_id·passage_id를 돌려줌 → 무시하고, 판정이 빠진 주장은 `missing`으로 남겨 지표에서 빼고 건수만 보고(Task 2 테스트).
2. medical 주장이 0개인 리포트·버전 → 비율은 `None`, 0으로 나누지 않음(Task 3 테스트).
3. 실행이 중간에 끊긴 뒤 다시 실행 → 이미 끝난 단계는 GPT·Codex를 다시 부르지 않음(Task 4 테스트).
4. 단순 프롬프트에 "[근거]"·"S번호"·"source_ids"가 한 글자라도 남음 → 실제 상수로 테스트(Task 2).
5. RAG 리포트가 존재하지 않는 S번호를 인용 → `validate_citations`가 지운 뒤 기준으로 인용 여부를 판단(Task 4는 validate 후 결과를 저장).

---

### Task 1: Codex 판정기를 스키마별로 쓰게 하기

**Files:**
- Modify: `rag/eval/judge.py` (`codex_ask`)
- Test: `tests/rag/test_judge_codex.py`(없으면 생성; 기존 codex 테스트가 있으면 그 파일에 추가)

**Interfaces:**
- Produces: `codex_ask(run=subprocess.run, model: type[BaseModel] = QueryJudgment, schema: dict = _CODEX_SCHEMA) -> Callable[[str], BaseModel]` — 기존 호출부는 그대로 동작.

- [ ] **Step 1: 실패 테스트** — `test_codex_ask_uses_given_schema_and_model`: 가짜 `run`이 `--output-schema` 파일 내용을 읽어 저장하고 `-o` 경로에 `{"x": 1}`을 쓴다. `codex_ask(run, model=_X, schema={"type":"object",...})("p")`가 `_X(x=1)`이고 저장한 스키마가 넘긴 dict와 같음을 확인.
- [ ] **Step 2: 실행** `.venv/bin/python -m pytest tests/rag/test_judge_codex.py -q` → FAIL(unexpected keyword).
- [ ] **Step 3: 구현** — 두 키워드 인자 추가, 하드코딩한 `_CODEX_SCHEMA`·`QueryJudgment`를 인자로 교체.
- [ ] **Step 4: 실행** 같은 명령 + `tests/rag` 전체 → PASS.
- [ ] **Step 5: 커밋** `feat(rag-eval): let the Codex judge take any output schema`

### Task 2: 프롬프트 거르기, 필드 직렬화, 판정 정리

**Files:**
- Create: `rag/eval/grounding.py`
- Test: `tests/rag/test_grounding.py`

**Interfaces:**
- Produces:
  - `BANNED = ("근거", "S번호", "source_ids", "직전 리포트")`
  - `plain_text(text: str) -> str` — 줄 단위로, 각 줄을 `re.split(r"(?<=\.)\s+", line)`로 문장 분리해 BANNED가 든 문장을 빼고 다시 잇는다. 빈 줄은 버린다.
  - `CITABLE = ("watch_points", "risk", "plan", "supp")`
  - `report_fields(report: dict) -> list[tuple[str, str]]` — (field_id, 문장). field_id 형식: `summary.overall_message`, `summary.key_focus[0]`, `strengths[0]`, `watch_points[0].interpretation`, `risk[0].message`, `plan[0].reason`, `plan[0].actions[1]`, `supp[0].purpose`, `supp[0].why_consider`, `supp[0].cautions[0]`. cautions 섹션은 넣지 않는다.
  - `item_of(field_id: str) -> str` — `"plan[2].actions[0]" -> "plan[2]"`, `"summary.key_focus[0]" -> "summary"`.
  - `item_sources(report: dict) -> dict[str, list[str]]` — `{"plan[2]": [...source_ids], ...}` (CITABLE 항목만).
  - pydantic 모델(`extra="forbid"`): `Claim(field_id, text, kind: Literal["medical","user_data","other"])`, `Claims(claims)`, `Verdict(claim_id, verdict: Literal["supported","contradicted","not_found"], passage_ids: list[str])`, `Verdicts(verdicts)`. Codex 스키마는 `Model.model_json_schema()`.
  - `extract_prompt(fields) -> str`, `judge_prompt(claims_with_passages: list[dict]) -> str` — 버전 단어("RAG", "단순") 없음. 주장 쪼개기 규칙(스펙 "주장 쪼개기 규칙")과 kind 정의를 프롬프트에 넣는다.
  - `clean_claims(claims: Claims, fields) -> list[dict]` — field_id가 fields에 없으면 버림. 결과에 `id`(c1, c2…) 부여.
  - `apply_verdicts(claims: list[dict], verdicts: Verdicts, passages: dict[str, list[str]]) -> dict[str, dict]` — claim id별 `{"verdict", "passage_ids"}`; 없는 claim_id 무시, passage_ids는 그 주장 후보 id로 제한, 판정이 없으면 `{"verdict": "missing", "passage_ids": []}`.

- [ ] **Step 1: 실패 테스트**
  - `test_plain_text_drops_rag_sentences_from_real_prompts`: `ai_service`의 두 상수에 `plain_text`를 적용한 결과에 BANNED 단어가 하나도 없고, "missing_metrics"·"personal_metrics"·"lifestyle_plan: 1~5개"는 남는다.
  - `test_report_fields_ids_and_item_of`: 작은 리포트 dict로 field_id 목록과 `item_of` 결과를 확인(cautions 없음).
  - `test_item_sources_only_citable`.
  - `test_apply_verdicts_ignores_unknown_ids_and_marks_missing`: c1 판정 + 없는 c9 판정 + passage_ids에 후보 밖 "p7" → c1은 후보 안 id만, c2는 `missing`.
  - `test_prompts_do_not_name_the_version`: 두 프롬프트에 "RAG"·"단순" 없음.
- [ ] **Step 2: 실행** `.venv/bin/python -m pytest tests/rag/test_grounding.py -q` → FAIL(import).
- [ ] **Step 3: 구현** 위 Interfaces 그대로.
- [ ] **Step 4: 실행** → PASS.
- [ ] **Step 5: 커밋** `feat(rag-eval): claim fields, plain prompt and verdict cleanup for grounding eval`

### Task 3: 지표와 summary.md

**Files:**
- Modify: `rag/eval/grounding.py`
- Test: `tests/rag/test_grounding.py`

**Interfaces:**
- Consumes: Task 2의 `item_of`, `CITABLE`.
- Produces:
  - 상태 항목 형식(Task 4가 저장): `{"case", "version": "plain"|"rag", "report", "evidence": [{"sid","chunk_id","title","heading_path","content"}], "claims": [{"id","field_id","text","kind"}], "corpus": {claim_id: {"verdict","passage_ids","passages":[{"pid","title","heading_path","content"}]}}, "cited": {claim_id: {"verdict","passage_ids"}}}`
  - `metrics(entries: list[dict]) -> dict` — 버전 하나의 항목들을 받아 `{"medical", "citable", "cited", "citation_rate", "judged", "unsupported", "contradicted", "unsupported_rate", "missing", "cite_judged", "cite_supported", "cite_accuracy"}`. 비율은 분모 0이면 `None`. `cited`는 claim의 `item_of`가 CITABLE이고 `item_sources`가 비어 있지 않은 것. `unsupported_rate = unsupported / judged`(missing 제외).
  - `examples(entries: list[dict], n: int = 3) -> list[tuple[dict, dict]]` — 같은 case에서 plain의 근거 없는 medical 주장과 rag의 인용·supported medical 주장 한 쌍씩, 서로 다른 case 우선.
  - `render_summary(entries: list[dict]) -> str` — 두 버전 표(건수/비율), 사례별 표, 예시 쌍, 판정기 일치율 자리(`spotcheck.json`이 있으면 그 값).

- [ ] **Step 1: 실패 테스트**
  - `test_metrics_counts_and_rates`: plain 1건(medical 3: not_found 2, supported 1; user_data 1) → citation_rate 0.0, unsupported_rate 2/3. rag 1건(medical 4 중 plan 항목 3개가 인용, summary 1개; corpus supported 3, contradicted 1; cited supported 2, not_found 1) → citation_rate 1.0(분모 3), contradicted 1, cite_accuracy 2/3.
  - `test_metrics_no_medical_claims_gives_none`.
  - `test_metrics_excludes_missing_from_rate`.
- [ ] **Step 2: 실행** → FAIL.
- [ ] **Step 3: 구현**.
- [ ] **Step 4: 실행** → PASS.
- [ ] **Step 5: 커밋** `feat(rag-eval): grounding metrics and summary`

### Task 4: 실행 파이프라인(run, summarize, spotcheck)

**Files:**
- Modify: `rag/eval/grounding.py`
- Test: `tests/rag/test_grounding.py`

**Interfaces:**
- Consumes: Task 1 `codex_ask(model=..., schema=...)`, Task 2·3 전부, `report_preview._cases()`/`_synthetic_cases()`, `report_preview._report`의 순서(predict_all → compute → trait/build_queries → retrieve → generate_report → validate_citations).
- Produces:
  - `generate(case: dict, version: str) -> tuple[dict, list[dict]]` — (report, evidence 목록). plain은 `patch.object`로 `ai_service._REPORT_SYSTEM_INSTRUCTION`·`_REPORT_GUIDE`를 `plain_text` 결과로, `ai_service.build_report_prompt`를 `[가이드, "[사용자 데이터]", payload JSON]`만 잇는 함수로 바꿔 `generate_report(evidence=None)` 호출 후 모든 source_ids를 비운다. rag는 `validate_citations` 후 결과.
  - `search(texts: list[str]) -> list[list[dict]]` — `embed(texts, "query")` 후 텍스트마다 `match_rag_chunks`(match_count 5, min_similarity -1.0).
  - `evaluate(entry: dict, ask_claims, ask_verdicts, search_fn) -> dict` — entry에 `claims`·`corpus`·(rag면) `cited`가 없을 때만 해당 단계를 수행해 채운다. medical 주장만 검색·판정. 코퍼스 후보 pid는 주장별 p1~p5, 인용 판정 후보는 그 항목 source_ids의 evidence 본문(pid = sid).
  - `run(out_dir: Path)` — 10 케이스 × 2 버전. `state.json`에 키 `f"{case}|{version}"`로 단계마다 저장. `plain_prompt.txt`에 실제 쓴 단순 프롬프트(시스템+가이드) 저장. 리포트 사이 15초.
  - `spotcheck(out_dir, n=20, seed=0)` — corpus 판정 20개를 무작위로 골라 주장·후보·판정을 `spotcheck.md`로. 사람이(이 세션의 Claude가) 읽고 `spotcheck.json`에 `{"agree": k, "total": 20}`을 적는다.
  - CLI: `python -m rag.eval.grounding run|summarize|spotcheck [--out rag/eval/preview/grounding]`.

- [ ] **Step 1: 실패 테스트**
  - `test_evaluate_fills_stages_and_skips_done_ones`: 가짜 ask/search(호출 수를 셈)로 rag entry 평가 → claims·corpus·cited 채워짐, medical만 search에 감. 같은 entry를 다시 `evaluate` → 가짜 호출 수 증가 없음.
  - `test_evaluate_plain_has_no_cited_stage`.
  - `test_generate_plain_prompt_has_no_evidence_block` : `generate_report` 내부의 OpenAI 호출을 가짜로 바꿔(`ai_service.get_openai` monkeypatch) 받은 instructions·input에 "[근거]"·"S번호"가 없고 source_ids가 모두 빈 배열인지 확인. predict_all 등은 monkeypatch로 고정값.
- [ ] **Step 2: 실행** → FAIL.
- [ ] **Step 3: 구현**.
- [ ] **Step 4: 실행** `tests/rag/test_grounding.py` + 전체 스위트 → PASS.
- [ ] **Step 5: 커밋** `feat(rag-eval): run RAG vs plain-prompt grounding evaluation`

### Task 5: 실행과 결과 정리

- [ ] **Step 1:** `PYTHONPATH=. .venv/bin/python -m rag.eval.grounding run` 백그라운드 실행. 기대: `state.json`에 20개 항목, 모두 claims·corpus 있음, rag는 cited 있음.
- [ ] **Step 2:** `spotcheck` → `spotcheck.md`를 읽고 판정 20개를 직접 재판정해 `spotcheck.json` 기록.
- [ ] **Step 3:** `summarize` → `summary.md`. 숫자와 예시 3쌍을 앱 화면 형식 대신 발표용 표로 사용자에게 보고.
- [ ] **Step 4:** 결과물은 gitignore라 커밋 없음. `rag/README.md` 평가 절에 지표 정의와 결과 요약 3줄 추가 후 커밋 `docs(rag): grounding evaluation results`.
