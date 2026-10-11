# RAG 평가 계획 A: 검색 쿼리와 정답 라벨 만들기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 검색 쿼리 약 54개를 만들고, 여러 검색 방식의 후보를 모아 Gemini로 1차 판정한 뒤, 사용자가 검수 화면에서 확인한 결과를 `rag/eval/data/labels.jsonl`(근거 문장 라벨)로 내보낸다.

**Architecture:** `rag/eval/` 아래 작은 모듈들(metrics → queries → embedders → search → pool → judge → build_pool CLI → 검수 아티팩트 → export_labels CLI). LLM·임베딩 호출은 주입 가능한 함수로 감싸 단위 테스트는 네트워크 없이 돈다. 실제 모델 실행과 사용자 검수는 마지막 태스크에서 한다.

**Tech Stack:** Python 3.13, numpy, google-genai(기존), sentence-transformers + torch(로컬 임베딩), kiwipiepy + rank_bm25(BM25), pydantic v2, pytest, claude.ai 아티팩트(`db` capability)

**Spec:** `docs/superpowers/specs/2026-10-11-rag-eval-design.md` (3·4·5절). 실험 실행(6절)과 결과물(8절)은 계획 B에서 다룬다.

## Global Constraints

- 유료 API를 쓰지 않는다. LLM은 `gemini-2.5-flash`, Gemini 임베딩은 `gemini-embedding-001`, 키는 `.env`의 `GEMINI_API_KEY`
- 근거 매칭: 공백 제거 + NFC 정규화 문자열 비교, 문자 10-gram 포함 비율, 채점 임계값 0.5. Gemini가 낸 근거는 **100% 포함**(정규화 후 부분 문자열)일 때만 채택
- 후보 모으기: 검색 방식마다 상위 **5개**
- 같은 시나리오의 키워드형·질문형 쿼리는 같은 split(tune/test)에 둔다
- 단위 테스트는 네트워크·모델 다운로드 없이 돈다(가짜 임베더·가짜 LLM 주입)
- 테스트 파일 이름은 `tests/rag/test_eval_*.py` (기존 `tests/rag/test_*.py`와 이름이 겹치지 않게)
- 지침 원문이 들어가는 산출물(`review_items.json`, 판정 캐시, 임베딩 캐시)은 Git에 올리지 않는다. `queries.jsonl`, `labels.jsonl`은 커밋한다
- 실행은 저장소 루트에서 `python -m rag.eval.<모듈>`

## Review Focus

1. **Gemini가 근거를 원문과 다르게 옮긴 경우**(따옴표, 띄어쓰기, 요약): 그 근거만 버리고 후보는 검수 화면에 남아야 한다 → Task 6 `test_eval_judge_drops_non_verbatim_evidence`
2. **Gemini 응답에 후보가 빠지거나 없는 후보 id가 있는 경우**: 빠진 후보는 "미판정"으로 남고 없는 id는 무시해야 한다 → Task 6 `test_eval_judge_missing_and_unknown_candidates`
3. **판정 도중 429/503(무료 한도)로 끊긴 경우**: 다시 실행하면 이미 판정한 쿼리는 건너뛰고 이어서 해야 한다 → Task 6 `test_eval_judge_cache_resumes`
4. **청크가 바뀐 뒤 임베딩 캐시가 낡은 값을 돌려주는 경우**: 텍스트가 다르면 캐시 키가 달라야 한다 → Task 3 `test_eval_cache_key_changes_with_texts`
5. **검수를 일부만 했거나 "관련 있음"인데 근거를 안 고른 경우**: 내보내기가 그 항목을 건너뛰고 통계에 남겨야 한다 → Task 8 `test_eval_export_skips_unreviewed_and_evidence_less`

---

## File Structure

| 파일 | 책임 |
|---|---|
| `rag/eval/__init__.py` | 패키지 표시 |
| `rag/eval/requirements.txt` | 실험 전용 의존성 |
| `rag/eval/metrics.py` | 근거 매칭(`compact`, `evidence_coverage`, `contains_evidence`). 계획 B가 순위 지표를 추가 |
| `rag/eval/llm.py` | Gemini 클라이언트, JSON 스키마 응답, 재시도 |
| `rag/eval/queries.py` | 시나리오 → 쿼리 54개, CLI |
| `rag/eval/embedders.py` | 임베더 인터페이스, Gemini·sentence-transformers 구현, 파일 캐시, 레지스트리 |
| `rag/eval/search.py` | 벡터 순위, kiwi 토큰화, BM25 |
| `rag/eval/pool.py` | 쿼리 하나의 후보 모으기 |
| `rag/eval/judge.py` | 쿼리 단위 Gemini 판정, 근거 검증, 캐시 |
| `rag/eval/build_pool.py` | CLI: 순위 → 후보 → 판정 → `review_items.json` |
| `rag/eval/review/index.html` | 검수 아티팩트 페이지 |
| `rag/eval/export_labels.py` | CLI: 검수 결과 → `labels.jsonl`, 수정된 `queries.jsonl` |
| `tests/rag/test_eval_*.py` | 단위 테스트 |

---

### Task 1: 패키지, 의존성, 근거 매칭

**Files:**
- Create: `rag/eval/__init__.py`, `rag/eval/requirements.txt`, `rag/eval/metrics.py`
- Modify: `.gitignore`
- Test: `tests/rag/test_eval_metrics.py`

**Interfaces:**
- Produces:
  - `rag.eval.metrics.compact(text: str) -> str`: NFC 정규화 후 모든 공백 제거
  - `rag.eval.metrics.evidence_coverage(evidence: str, text: str, n: int = 10) -> float`: 근거의 문자 n-gram 중 text에 있는 비율(0~1). 근거가 n자 미만이면 통째 포함 여부로 0 또는 1
  - `rag.eval.metrics.contains_evidence(text: str, evidence: str, threshold: float = 0.5) -> bool`

- [ ] **Step 1: 의존성과 gitignore**

`rag/eval/requirements.txt`:
```text
# 평가 실험 전용. 서버와 수집 파이프라인에는 필요 없음.
numpy
sentence-transformers
torch
kiwipiepy
rank_bm25
```

설치 후 실제 버전으로 고정한다:
```bash
pip install -r rag/eval/requirements.txt
python -c "import numpy, sentence_transformers, torch, kiwipiepy, rank_bm25; print(numpy.__version__, sentence_transformers.__version__, torch.__version__, kiwipiepy.__version__)"
pip show rank_bm25 | grep Version
```
출력된 버전으로 각 줄을 `패키지==버전`으로 바꾼다.

`.gitignore` 끝에 추가:
```text

# RAG 평가: 지침 원문이 들어가는 중간 산출물과 캐시
rag/eval/cache/
rag/eval/data/review_items.json
rag/eval/data/review_decisions.json
```

`rag/eval/__init__.py`는 빈 파일.

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/rag/test_eval_metrics.py`:
```python
import pytest

from rag.eval.metrics import compact, contains_evidence, evidence_coverage


def test_compact_removes_whitespace_and_normalizes():
    assert compact(" 하루 소금\n섭취량 ") == "하루소금섭취량"
    assert compact("가") == "가"


def test_full_containment_ignores_spacing():
    text = "규칙적인운동은혈압을낮추고, 심폐기능을개선하며"
    assert evidence_coverage("규칙적인 운동은 혈압을 낮추고", text) == 1.0


def test_partial_coverage_when_chunk_cuts_sentence():
    evidence = "하루 소금 섭취량을 5 g 이하로 제한한다."
    text = "...권고한다. 하루 소금 섭취량을 5 g 이하로"
    coverage = evidence_coverage(evidence, text)
    assert 0.3 < coverage < 0.8


def test_short_evidence_uses_exact_match():
    assert evidence_coverage("금연", "금연을 권고한다") == 1.0
    assert evidence_coverage("절주", "금연을 권고한다") == 0.0


def test_empty_evidence_is_zero():
    assert evidence_coverage("  ", "아무 내용") == 0.0


@pytest.mark.parametrize("text,expected", [
    ("하루 소금 섭취량을 5 g 이하로 제한한다. 체중을 줄인다.", True),
    ("체중을 줄인다. 하루 소금 섭취량을", False),
])
def test_contains_evidence_threshold(text, expected):
    assert contains_evidence(text, "하루 소금 섭취량을 5 g 이하로 제한한다.") is expected
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_eval_metrics.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.eval'`)

- [ ] **Step 4: 구현**

`rag/eval/metrics.py`:
```python
"""평가 지표. 근거 문장 매칭은 청킹·파서와 무관하게 쓰도록 공백을 지우고 비교한다."""

import re
import unicodedata


def compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def evidence_coverage(evidence: str, text: str, n: int = 10) -> float:
    e, t = compact(evidence), compact(text)
    if not e:
        return 0.0
    if len(e) < n:
        return 1.0 if e in t else 0.0
    grams = [e[i : i + n] for i in range(len(e) - n + 1)]
    return sum(gram in t for gram in grams) / len(grams)


def contains_evidence(text: str, evidence: str, threshold: float = 0.5) -> bool:
    return evidence_coverage(evidence, text) >= threshold
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_eval_metrics.py -q`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add .gitignore rag/eval/__init__.py rag/eval/requirements.txt rag/eval/metrics.py tests/rag/test_eval_metrics.py
git commit -m "feat(rag-eval): add evidence matching metrics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Gemini 호출 도우미와 쿼리 세트

**Files:**
- Create: `rag/eval/llm.py`, `rag/eval/queries.py`
- Test: `tests/rag/test_eval_queries.py`

**Interfaces:**
- Produces:
  - `rag.eval.llm.MODEL = "gemini-2.5-flash"`
  - `rag.eval.llm.get_client() -> genai.Client` (`.env`의 `GEMINI_API_KEY`)
  - `rag.eval.llm.generate_json(client, prompt: str, schema: type[BaseModel]) -> BaseModel`: 429/500/503이면 최대 4번 재시도(5, 10, 20, 40초 대기)
  - `rag.eval.queries.SCENARIOS: list[tuple[str, str]]` (condition, category) 24개
  - `rag.eval.queries.keyword_query(condition: str, category: str) -> str`
  - `rag.eval.queries.build_queries(draft_question: Callable[[str, str], str]) -> list[dict]`. 각 dict 키: `query_id, text, form, condition, category, split`. `form` ∈ `keyword|question|inbody|unanswerable`, `split` ∈ `tune|test`
  - `rag.eval.queries.main(argv: list[str] | None = None) -> int`: `--out`(기본 `rag/eval/data/queries.jsonl`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_eval_queries.py`:
```python
from collections import Counter

from rag.eval.queries import SCENARIOS, build_queries, keyword_query


def fake_draft(condition, category):
    return f"{condition}-{category} 질문?"


def test_scenarios_cover_spec_table():
    assert len(SCENARIOS) == 24
    assert ("sarcopenia", "exercise") in SCENARIOS
    assert ("general", "diet") in SCENARIOS


def test_keyword_query_templates():
    assert keyword_query("hypertension", "diet") == "고혈압 예방 식이요법 식사 관리"
    assert keyword_query("general", "exercise") == "성인 신체활동 권장량 유산소 근력운동"


def test_build_queries_shapes_and_counts():
    queries = build_queries(fake_draft)
    forms = Counter(q["form"] for q in queries)
    assert forms == {"keyword": 24, "question": 24, "inbody": 3, "unanswerable": 3}
    assert len({q["query_id"] for q in queries}) == len(queries)
    assert set(queries[0]) == {"query_id", "text", "form", "condition", "category", "split"}


def test_question_text_comes_from_drafter():
    queries = build_queries(fake_draft)
    q = next(q for q in queries if q["query_id"] == "dyslipidemia-diet-question")
    assert q["text"] == "dyslipidemia-diet 질문?"


def test_scenario_pairs_share_split_and_splits_are_balanced():
    queries = build_queries(fake_draft)
    by_id = {q["query_id"]: q for q in queries}
    for condition, category in SCENARIOS:
        a = by_id[f"{condition}-{category}-keyword"]
        b = by_id[f"{condition}-{category}-question"]
        assert a["split"] == b["split"]
    splits = Counter(q["split"] for q in queries)
    assert abs(splits["tune"] - splits["test"]) <= 2


def test_drafter_whitespace_is_stripped():
    queries = build_queries(lambda c, k: "  질문입니다?\n")
    assert all(q["text"] == "질문입니다?" for q in queries if q["form"] == "question")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_eval_queries.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.eval.queries'`)

- [ ] **Step 3: 구현**

`rag/eval/llm.py`:
```python
"""평가 스크립트용 Gemini 호출. 서버 설정(app.core.config)에 의존하지 않는다."""

import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

MODEL = "gemini-2.5-flash"
_RETRY_WAITS = (5, 10, 20, 40)
_RETRYABLE = {429, 500, 503}

log = logging.getLogger(__name__)


def get_client() -> genai.Client:
    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _status(error: Exception) -> int | None:
    return getattr(error, "code", None) or getattr(error, "status_code", None)


def generate_json(client: genai.Client, prompt: str, schema: type[BaseModel]) -> BaseModel:
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt, wait in enumerate((*_RETRY_WAITS, None)):
        try:
            res = client.models.generate_content(model=MODEL, contents=prompt, config=config)
            return schema.model_validate_json(res.text)
        except Exception as e:
            if wait is None or _status(e) not in _RETRYABLE:
                raise
            log.warning("Gemini %s, retry %d in %ss", _status(e), attempt + 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")
```

`rag/eval/queries.py`:
```python
"""평가용 검색 쿼리 세트. 저장소 루트에서 `python -m rag.eval.queries`."""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

CONDITION_NAMES = {
    "hypertension": "고혈압",
    "diabetes": "당뇨병",
    "dyslipidemia": "이상지질혈증",
    "obesity": "비만",
    "metabolic_syndrome": "대사증후군",
    "fatty_liver": "지방간",
    "sarcopenia": "근감소증",
}
CATEGORY_NAMES = {"criteria": "진단 기준", "diet": "식사", "lifestyle": "생활습관", "exercise": "운동"}

SCENARIOS: list[tuple[str, str]] = [
    ("hypertension", "criteria"), ("hypertension", "diet"), ("hypertension", "lifestyle"), ("hypertension", "exercise"),
    ("diabetes", "diet"), ("diabetes", "exercise"), ("diabetes", "lifestyle"),
    ("dyslipidemia", "criteria"), ("dyslipidemia", "diet"), ("dyslipidemia", "lifestyle"), ("dyslipidemia", "exercise"),
    ("obesity", "criteria"), ("obesity", "lifestyle"), ("obesity", "exercise"),
    ("metabolic_syndrome", "criteria"), ("metabolic_syndrome", "lifestyle"),
    ("fatty_liver", "criteria"), ("fatty_liver", "diet"), ("fatty_liver", "exercise"), ("fatty_liver", "lifestyle"),
    ("sarcopenia", "criteria"), ("sarcopenia", "exercise"),
    ("general", "diet"), ("general", "exercise"),
]

_KEYWORD_TEMPLATES = {
    "criteria": "{name} 진단 기준 분류",
    "diet": "{name} 예방 식이요법 식사 관리",
    "lifestyle": "{name} 생활습관 교정 체중 감량 금연 절주",
    "exercise": "{name} 운동요법 유산소 근력운동 권고",
}
_GENERAL_KEYWORDS = {
    "diet": "한국인 영양소 섭취기준 권장섭취량 상한섭취량",
    "exercise": "성인 신체활동 권장량 유산소 근력운동",
}
_INBODY_QUERIES = [
    "체지방률 32% 40대 여성 비만 기준",
    "골격근량 감소 노인 근력운동 방법",
    "내장지방 높고 허리둘레 큰 남성 대사증후군 위험",
]
_UNANSWERABLE_QUERIES = [
    "통풍 환자 식이요법 퓨린 제한",
    "갑상선 기능저하증 운동 주의사항",
    "임신 중 철분제 복용 시기",
]


def keyword_query(condition: str, category: str) -> str:
    if condition == "general":
        return _GENERAL_KEYWORDS[category]
    return _KEYWORD_TEMPLATES[category].format(name=CONDITION_NAMES[condition])


def _split(index: int) -> str:
    return "tune" if index % 2 == 0 else "test"


def build_queries(draft_question: Callable[[str, str], str]) -> list[dict]:
    queries = []
    for i, (condition, category) in enumerate(SCENARIOS):
        base = {"condition": condition, "category": category, "split": _split(i)}
        queries.append({"query_id": f"{condition}-{category}-keyword", "text": keyword_query(condition, category),
                        "form": "keyword", **base})
        queries.append({"query_id": f"{condition}-{category}-question",
                        "text": draft_question(condition, category).strip(), "form": "question", **base})
    for i, text in enumerate(_INBODY_QUERIES, 1):
        queries.append({"query_id": f"inbody-{i}", "text": text, "form": "inbody",
                        "condition": "mixed", "category": "mixed", "split": _split(i)})
    for i, text in enumerate(_UNANSWERABLE_QUERIES, 1):
        queries.append({"query_id": f"none-{i}", "text": text, "form": "unanswerable",
                        "condition": "none", "category": "none", "split": _split(i)})
    return queries


class _Question(BaseModel):
    question: str


def _gemini_drafter() -> Callable[[str, str], str]:
    from rag.eval.llm import generate_json, get_client

    client = get_client()

    def draft(condition: str, category: str) -> str:
        topic = "일반 성인" if condition == "general" else CONDITION_NAMES[condition]
        prompt = (
            "건강검진과 인바디 결과를 받은 일반인이 건강 앱에 물어볼 법한 질문을 한국어 한 문장으로 쓰세요. "
            "40자 이내, 의학 용어보다 일상어 위주, 질문만 출력하세요.\n"
            f"주제: {topic} / {CATEGORY_NAMES[category]}"
        )
        return generate_json(client, prompt, _Question).question

    return draft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="평가용 검색 쿼리 생성")
    parser.add_argument("--out", type=Path, default=Path("rag/eval/data/queries.jsonl"))
    args = parser.parse_args(argv)
    queries = build_queries(_gemini_drafter())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"wrote {len(queries)} queries to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_eval_queries.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add rag/eval/llm.py rag/eval/queries.py tests/rag/test_eval_queries.py
git commit -m "feat(rag-eval): add query set builder and Gemini helper

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 임베더와 캐시

**Files:**
- Create: `rag/eval/embedders.py`
- Test: `tests/rag/test_eval_embedders.py`

**Interfaces:**
- Consumes: `rag.eval.llm.get_client`
- Produces:
  - `rag.eval.embedders.Embedder` (Protocol): 속성 `name: str`, 메서드 `encode(texts: list[str], kind: str) -> np.ndarray` (`kind` ∈ `query|doc`, 행마다 L2 정규화된 float32)
  - `rag.eval.embedders.l2_normalize(m: np.ndarray) -> np.ndarray`
  - `rag.eval.embedders.CachedEmbedder(inner: Embedder, cache_dir: Path)`: 같은 인터페이스. 캐시 키 = sha256(`name`, `kind`, 텍스트 전체)
  - `rag.eval.embedders.GeminiEmbedder(dims: int, client=None)`: `name = f"gemini-{dims}"`
  - `rag.eval.embedders.STEmbedder(name, model_id, query_prefix="", doc_prefix="", query_prompt_name=None)`
  - `rag.eval.embedders.REGISTRY: dict[str, Callable[[], Embedder]]`, 키: `gemini-768, gemini-1536, gemini-3072, bge-m3, kure-v1, qwen3-0.6b, e5-large-instruct`
  - `rag.eval.embedders.load(name: str, cache_dir: Path = Path("rag/eval/cache/embeddings")) -> Embedder`: 레지스트리 + 캐시 래핑

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_eval_embedders.py`:
```python
import numpy as np

from rag.eval.embedders import REGISTRY, CachedEmbedder, l2_normalize


class CountingEmbedder:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def encode(self, texts, kind):
        self.calls += 1
        return l2_normalize(np.array([[len(t), 1.0 if kind == "query" else 2.0] for t in texts], dtype=np.float32))


def test_l2_normalize_rows():
    m = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert m.dtype == np.float32


def test_cache_hit_skips_inner(tmp_path):
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    first = cached.encode(["가", "나다"], "doc")
    second = CachedEmbedder(inner, tmp_path).encode(["가", "나다"], "doc")
    assert inner.calls == 1
    assert np.array_equal(first, second)
    assert cached.name == "fake"


def test_eval_cache_key_changes_with_texts(tmp_path):
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    cached.encode(["가"], "doc")
    cached.encode(["가 바뀜"], "doc")
    cached.encode(["가"], "query")
    assert inner.calls == 3


def test_registry_names():
    assert set(REGISTRY) == {
        "gemini-768", "gemini-1536", "gemini-3072", "bge-m3", "kure-v1", "qwen3-0.6b", "e5-large-instruct",
    }
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_eval_embedders.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.eval.embedders'`)

- [ ] **Step 3: 구현**

`rag/eval/embedders.py`:
```python
"""비교할 임베딩 모델들. 모두 encode(texts, kind) -> 행 정규화 행렬 형태로 맞춘다."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str], kind: str) -> np.ndarray: ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache_dir: Path):
        self.inner = inner
        self.name = inner.name
        self.cache_dir = cache_dir

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        key = hashlib.sha256("\x1f".join([self.name, kind, *texts]).encode()).hexdigest()
        path = self.cache_dir / f"{self.name}-{kind}-{key[:24]}.npy"
        if path.exists():
            return np.load(path)
        vectors = self.inner.encode(texts, kind)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors


class GeminiEmbedder:
    _BATCH = 100

    def __init__(self, dims: int, client=None):
        self.dims = dims
        self.name = f"gemini-{dims}"
        self._client = client

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        from google.genai import types

        from rag.eval.llm import get_client

        client = self._client or get_client()
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        rows = []
        for i in range(0, len(texts), self._BATCH):
            res = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts[i : i + self._BATCH],
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dims),
            )
            rows.extend(e.values for e in res.embeddings)
        return l2_normalize(np.array(rows))


class STEmbedder:
    def __init__(self, name: str, model_id: str, query_prefix: str = "", doc_prefix: str = "",
                 query_prompt_name: str | None = None):
        self.name = name
        self.model_id = model_id
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.query_prompt_name = query_prompt_name
        self._model = None

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.model_id)
        prefix = self.query_prefix if kind == "query" else self.doc_prefix
        kwargs = {"prompt_name": self.query_prompt_name} if kind == "query" and self.query_prompt_name else {}
        vectors = self._model.encode([prefix + t for t in texts], batch_size=8, show_progress_bar=True, **kwargs)
        return l2_normalize(vectors)


_E5_QUERY = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "

REGISTRY: dict[str, Callable[[], Embedder]] = {
    "gemini-768": lambda: GeminiEmbedder(768),
    "gemini-1536": lambda: GeminiEmbedder(1536),
    "gemini-3072": lambda: GeminiEmbedder(3072),
    "bge-m3": lambda: STEmbedder("bge-m3", "BAAI/bge-m3"),
    "kure-v1": lambda: STEmbedder("kure-v1", "nlpai-lab/KURE-v1"),
    "qwen3-0.6b": lambda: STEmbedder("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B", query_prompt_name="query"),
    "e5-large-instruct": lambda: STEmbedder("e5-large-instruct", "intfloat/multilingual-e5-large-instruct",
                                            query_prefix=_E5_QUERY),
}


def load(name: str, cache_dir: Path = Path("rag/eval/cache/embeddings")) -> Embedder:
    return CachedEmbedder(REGISTRY[name](), cache_dir)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_eval_embedders.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add rag/eval/embedders.py tests/rag/test_eval_embedders.py
git commit -m "feat(rag-eval): add embedder wrappers with file cache

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 검색 (벡터 순위, BM25)

**Files:**
- Create: `rag/eval/search.py`
- Test: `tests/rag/test_eval_search.py`

**Interfaces:**
- Produces:
  - `rag.eval.search.dense_rank(query_vecs: np.ndarray, doc_vecs: np.ndarray, k: int) -> list[list[int]]`: 쿼리마다 내적 내림차순 상위 k개 문서 인덱스(동점이면 인덱스 오름차순)
  - `rag.eval.search.kiwi_tokens(text: str) -> list[str]`: 명사·동사·형용사 어간·외국어·숫자·어근 형태소
  - `rag.eval.search.BM25Index(texts: list[str], tokenize: Callable[[str], list[str]] = kiwi_tokens)`, 메서드 `rank(query: str, k: int) -> list[int]` (점수 0인 문서 제외)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_eval_search.py`:
```python
import numpy as np

from rag.eval.search import BM25Index, dense_rank, kiwi_tokens


def test_dense_rank_orders_by_dot_product_with_stable_ties():
    docs = np.array([[1, 0], [0, 1], [0.6, 0.8], [1, 0]], dtype=np.float32)
    queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert dense_rank(queries, docs, k=3) == [[0, 3, 2], [1, 2, 0]]


def test_dense_rank_k_larger_than_corpus():
    docs = np.eye(2, dtype=np.float32)
    assert dense_rank(np.array([[1, 0]], dtype=np.float32), docs, k=10) == [[0, 1]]


def test_bm25_with_simple_tokenizer():
    texts = ["소금 섭취 제한", "운동 권고 유산소", "소금 소금 나트륨"]
    index = BM25Index(texts, tokenize=str.split)
    assert index.rank("소금", k=5) == [2, 0]
    assert index.rank("없는단어", k=5) == []


def test_kiwi_tokens_keeps_content_morphemes():
    tokens = kiwi_tokens("하루 소금 섭취량을 5 g 이하로 제한한다.")
    assert "소금" in tokens
    assert "제한" in tokens
    assert "을" not in tokens
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_eval_search.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.eval.search'`)

- [ ] **Step 3: 구현**

`rag/eval/search.py`:
```python
"""numpy 전수 벡터 검색과 kiwi 토큰 BM25."""

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

_KEEP_TAGS = ("NN", "NR", "NP", "VV", "VA", "SL", "SN", "XR", "SH")


def dense_rank(query_vecs: np.ndarray, doc_vecs: np.ndarray, k: int) -> list[list[int]]:
    scores = query_vecs @ doc_vecs.T
    order = np.lexsort((np.broadcast_to(np.arange(doc_vecs.shape[0]), scores.shape), -scores), axis=1)
    return [row[:k].tolist() for row in order]


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def kiwi_tokens(text: str) -> list[str]:
    return [t.form for t in _kiwi().tokenize(text) if t.tag.startswith(_KEEP_TAGS)]


class BM25Index:
    def __init__(self, texts: list[str], tokenize: Callable[[str], list[str]] = kiwi_tokens):
        self.tokenize = tokenize
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def rank(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(self.tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [i for i in order[:k] if scores[i] > 0]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_eval_search.py -q`
Expected: 4 passed. `test_bm25_with_simple_tokenizer`에서 IDF가 음수가 되어 순서가 다르면, rank_bm25의 `epsilon` 처리 때문이므로 `BM25Okapi(..., epsilon=0.25)`가 기본인지 확인하고 테스트 기대값이 아니라 원인을 기록한다(코퍼스 3개짜리 테스트의 특성).

- [ ] **Step 5: Commit**

```bash
git add rag/eval/search.py tests/rag/test_eval_search.py
git commit -m "feat(rag-eval): add dense ranking and kiwi BM25

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 후보 모으기

**Files:**
- Create: `rag/eval/pool.py`
- Test: `tests/rag/test_eval_pool.py`

**Interfaces:**
- Produces: `rag.eval.pool.pool_query(system_rankings: dict[str, list[str]], per_system: int) -> list[dict]`. 입력은 `{검색방식 이름: [content_hash 순위]}`. 출력은 `[{"content_hash": str, "systems": [이름...]}]`. 순서: 순위 1등을 방식 순서대로, 그다음 2등… (어느 방식 결과인지 드러나지 않게 섞음), 중복은 첫 등장 위치에 `systems`만 추가

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_eval_pool.py`:
```python
from rag.eval.pool import pool_query


def test_pool_interleaves_and_dedupes():
    rankings = {"a": ["x", "y", "z"], "b": ["y", "w", "x"]}
    assert pool_query(rankings, per_system=2) == [
        {"content_hash": "x", "systems": ["a"]},
        {"content_hash": "y", "systems": ["b", "a"]},
        {"content_hash": "w", "systems": ["b"]},
    ]


def test_pool_respects_per_system_and_short_lists():
    rankings = {"a": ["x"], "b": ["y", "z", "w"]}
    assert [c["content_hash"] for c in pool_query(rankings, per_system=2)] == ["x", "y", "z"]


def test_pool_empty():
    assert pool_query({"a": []}, per_system=5) == []
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_eval_pool.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.eval.pool'`)

- [ ] **Step 3: 구현**

`rag/eval/pool.py`:
```python
"""여러 검색 방식의 상위 결과를 합쳐 판정 후보를 만든다(TREC식 pooling)."""


def pool_query(system_rankings: dict[str, list[str]], per_system: int) -> list[dict]:
    pooled: dict[str, dict] = {}
    for rank in range(per_system):
        for system, ranking in system_rankings.items():
            if rank >= len(ranking):
                continue
            entry = pooled.setdefault(ranking[rank], {"content_hash": ranking[rank], "systems": []})
            entry["systems"].append(system)
    return list(pooled.values())
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_eval_pool.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add rag/eval/pool.py tests/rag/test_eval_pool.py
git commit -m "feat(rag-eval): add candidate pooling

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Gemini 1차 판정

**Files:**
- Create: `rag/eval/judge.py`
- Test: `tests/rag/test_eval_judge.py`

**Interfaces:**
- Consumes: `rag.eval.metrics.compact`
- Produces:
  - `rag.eval.judge.CandidateJudgment(BaseModel)`: `candidate_id: str, relevant: bool, evidence: list[str]`
  - `rag.eval.judge.QueryJudgment(BaseModel)`: `judgments: list[CandidateJudgment]`
  - `rag.eval.judge.judge_prompt(query: str, candidates: list[dict]) -> str`: 후보는 `c1..cn`으로 번호를 붙여 본문과 함께 넣음
  - `rag.eval.judge.judge_query(ask: Callable[[str], QueryJudgment], query: str, candidates: list[dict]) -> dict[str, dict]`: 입력 후보 dict는 `content_hash, content`를 가짐. 반환 `{content_hash: {"relevant": bool | None, "evidence": [str]}}`. 원문에 그대로 없는 근거는 버림, `relevant=False`면 근거 없음, 응답에 없는 후보는 `relevant=None`
  - `rag.eval.judge.JudgeCache(path: Path)`: `get(query_id) -> dict | None`, `put(query_id, result: dict)` (JSONL에 한 줄 추가, 같은 query_id는 마지막 줄 우선)
  - `rag.eval.judge.gemini_ask() -> Callable[[str], QueryJudgment]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_eval_judge.py`:
```python
from rag.eval.judge import CandidateJudgment, JudgeCache, QueryJudgment, judge_prompt, judge_query

CANDIDATES = [
    {"content_hash": "h1", "content": "고혈압 진료지침 > 소금\n하루 소금 섭취량을 5 g 이하로 제한한다. 칼륨 섭취를 늘린다."},
    {"content_hash": "h2", "content": "운동 장\n주 5회 이상 유산소 운동을 한다."},
]


def ask_returning(*judgments):
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return QueryJudgment(judgments=list(judgments))

    ask.prompts = prompts
    return ask


def test_prompt_numbers_candidates():
    prompt = judge_prompt("소금 얼마나?", CANDIDATES)
    assert "[c1]" in prompt and "[c2]" in prompt
    assert "하루 소금 섭취량" in prompt


def test_verbatim_evidence_kept_and_irrelevant_cleared():
    ask = ask_returning(
        CandidateJudgment(candidate_id="c1", relevant=True, evidence=["하루 소금 섭취량을 5 g 이하로 제한한다."]),
        CandidateJudgment(candidate_id="c2", relevant=False, evidence=["주 5회 이상 유산소 운동을 한다."]),
    )
    result = judge_query(ask, "소금 얼마나?", CANDIDATES)
    assert result == {
        "h1": {"relevant": True, "evidence": ["하루 소금 섭취량을 5 g 이하로 제한한다."]},
        "h2": {"relevant": False, "evidence": []},
    }


def test_eval_judge_drops_non_verbatim_evidence():
    ask = ask_returning(
        CandidateJudgment(candidate_id="c1", relevant=True,
                          evidence=["소금은 하루 5g 미만으로", "하루소금섭취량을5 g 이하로 제한한다."]),
        CandidateJudgment(candidate_id="c2", relevant=False, evidence=[]),
    )
    result = judge_query(ask, "q", CANDIDATES)
    assert result["h1"] == {"relevant": True, "evidence": ["하루소금섭취량을5 g 이하로 제한한다."]}


def test_eval_judge_missing_and_unknown_candidates():
    ask = ask_returning(CandidateJudgment(candidate_id="c9", relevant=True, evidence=[]))
    result = judge_query(ask, "q", CANDIDATES)
    assert result == {"h1": {"relevant": None, "evidence": []}, "h2": {"relevant": None, "evidence": []}}


def test_eval_judge_cache_resumes(tmp_path):
    cache = JudgeCache(tmp_path / "judgments.jsonl")
    assert cache.get("q1") is None
    cache.put("q1", {"h1": {"relevant": True, "evidence": ["a"]}})
    cache.put("q1", {"h1": {"relevant": False, "evidence": []}})
    reopened = JudgeCache(tmp_path / "judgments.jsonl")
    assert reopened.get("q1") == {"h1": {"relevant": False, "evidence": []}}
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_eval_judge.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.eval.judge'`)

- [ ] **Step 3: 구현**

`rag/eval/judge.py`:
```python
"""쿼리 하나의 후보 전체를 Gemini로 한 번에 판정한다(무료 한도 안에서 쿼리당 1회 호출)."""

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from rag.eval.metrics import compact


class CandidateJudgment(BaseModel):
    candidate_id: str
    relevant: bool
    evidence: list[str]


class QueryJudgment(BaseModel):
    judgments: list[CandidateJudgment]


def judge_prompt(query: str, candidates: list[dict]) -> str:
    blocks = "\n\n".join(f"[c{i}]\n{c['content']}" for i, c in enumerate(candidates, 1))
    return (
        "당신은 의학·영양 문서 검색 평가자입니다. 아래 검색 쿼리에 대해 각 후보 문단이 "
        "직접 답이 되는 내용을 담고 있는지 판정하세요.\n"
        "- relevant: 쿼리에 대한 답(기준, 권고, 방법, 수치)이 문단에 있으면 true. 주제만 비슷하면 false.\n"
        "- evidence: relevant가 true일 때 답이 되는 문장 1~3개를 문단에서 한 글자도 바꾸지 말고 그대로 복사. "
        "false면 빈 배열.\n"
        "- 모든 후보에 대해 candidate_id(c1, c2, ...)로 하나씩 답하세요.\n\n"
        f"검색 쿼리: {query}\n\n{blocks}"
    )


def judge_query(ask: Callable[[str], QueryJudgment], query: str, candidates: list[dict]) -> dict[str, dict]:
    by_id = {f"c{i}": c for i, c in enumerate(candidates, 1)}
    result = {c["content_hash"]: {"relevant": None, "evidence": []} for c in candidates}
    for judgment in ask(judge_prompt(query, candidates)).judgments:
        candidate = by_id.get(judgment.candidate_id)
        if candidate is None:
            continue
        content = compact(candidate["content"])
        evidence = [e for e in judgment.evidence if len(compact(e)) >= 5 and compact(e) in content]
        result[candidate["content_hash"]] = {
            "relevant": judgment.relevant,
            "evidence": evidence if judgment.relevant else [],
        }
    return result


class JudgeCache:
    def __init__(self, path: Path):
        self.path = path
        self._rows: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self._rows[row["query_id"]] = row["result"]

    def get(self, query_id: str) -> dict | None:
        return self._rows.get(query_id)

    def put(self, query_id: str, result: dict) -> None:
        self._rows[query_id] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"query_id": query_id, "result": result}, ensure_ascii=False) + "\n")


def gemini_ask() -> Callable[[str], QueryJudgment]:
    from rag.eval.llm import generate_json, get_client

    client = get_client()
    return lambda prompt: generate_json(client, prompt, QueryJudgment)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_eval_judge.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rag/eval/judge.py tests/rag/test_eval_judge.py
git commit -m "feat(rag-eval): add per-query Gemini judging with evidence validation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: build_pool CLI

**Files:**
- Create: `rag/eval/build_pool.py`
- Test: `tests/rag/test_eval_build_pool.py`

**Interfaces:**
- Consumes: `dense_rank`, `BM25Index`, `pool_query`, `judge_query`, `JudgeCache`, `Embedder`, `load`, `gemini_ask`
- Produces:
  - `rag.eval.build_pool.rank_all(embedders: dict[str, Embedder], doc_texts: list[str], query_texts: list[str], k: int, bm25: BM25Index | None) -> dict[str, list[list[int]]]`: 방식 이름 → 쿼리별 문서 인덱스 순위. BM25는 이름 `"bm25"`
  - `rag.eval.build_pool.build_review_items(chunks: list[dict], queries: list[dict], rankings: dict[str, list[list[int]]], per_system: int, ask, cache: JudgeCache) -> dict`: 반환 `{"queries": [query dict + "candidates": [...]]}`. 후보 키: `content_hash, doc_id, title, heading_path, page_start, page_end, content, systems, judged_relevant, judged_evidence`
  - `rag.eval.build_pool.main(argv=None) -> int`: 인자 `--chunks`(기본 `rag/data/chunks.jsonl`), `--queries`(`rag/eval/data/queries.jsonl`), `--out`(`rag/eval/data/review_items.json`), `--systems`(쉼표 구분, 기본 `gemini-1536,bge-m3,kure-v1,qwen3-0.6b,e5-large-instruct,bm25`), `--per-system`(5), `--judge-cache`(`rag/eval/cache/judgments.jsonl`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_eval_build_pool.py`:
```python
import numpy as np

from rag.eval.build_pool import build_review_items, rank_all
from rag.eval.judge import CandidateJudgment, JudgeCache, QueryJudgment
from rag.eval.search import BM25Index

CHUNKS = [
    {"content_hash": "h-salt", "doc_id": "kha", "title": "고혈압", "heading_path": "소금", "page_start": 74,
     "page_end": 74, "content": "고혈압 > 소금\n하루 소금 섭취량을 5 g 이하로 제한한다."},
    {"content_hash": "h-run", "doc_id": "kha", "title": "고혈압", "heading_path": "운동", "page_start": 77,
     "page_end": 77, "content": "고혈압 > 운동\n주 5회 유산소 운동을 한다."},
]
QUERIES = [{"query_id": "q-salt", "text": "소금 제한", "form": "keyword", "condition": "hypertension",
            "category": "diet", "split": "tune"}]


class KeywordEmbedder:
    name = "kw"

    def encode(self, texts, kind):
        return np.array([[1.0 if "소금" in t else 0.0, 1.0 if "운동" in t else 0.0] for t in texts], dtype=np.float32)


def test_rank_all_includes_embedders_and_bm25():
    docs = [c["content"] for c in CHUNKS] + ["기타 문서 내용"]  # 문서 2개면 BM25 IDF가 0이 됨
    rankings = rank_all({"kw": KeywordEmbedder()}, docs, ["소금 제한"], k=1, bm25=BM25Index(docs, tokenize=str.split))
    assert rankings == {"kw": [[0]], "bm25": [[0]]}


def test_build_review_items_judges_once_and_uses_cache(tmp_path):
    calls = []

    def ask(prompt):
        calls.append(prompt)
        return QueryJudgment(judgments=[
            CandidateJudgment(candidate_id="c1", relevant=True, evidence=["하루 소금 섭취량을 5 g 이하로 제한한다."]),
            CandidateJudgment(candidate_id="c2", relevant=False, evidence=[]),
        ])

    rankings = {"kw": [[0, 1]], "bm25": [[1]]}
    cache = JudgeCache(tmp_path / "j.jsonl")
    items = build_review_items(CHUNKS, QUERIES, rankings, per_system=2, ask=ask, cache=cache)
    again = build_review_items(CHUNKS, QUERIES, rankings, per_system=2, ask=ask, cache=JudgeCache(tmp_path / "j.jsonl"))

    assert len(calls) == 1
    assert items == again
    [query] = items["queries"]
    assert query["query_id"] == "q-salt"
    assert [c["content_hash"] for c in query["candidates"]] == ["h-salt", "h-run"]
    salt = query["candidates"][0]
    assert salt["systems"] == ["kw"]
    assert salt["judged_relevant"] is True
    assert salt["judged_evidence"] == ["하루 소금 섭취량을 5 g 이하로 제한한다."]
    assert salt["page_start"] == 74
    assert query["candidates"][1]["systems"] == ["bm25", "kw"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_eval_build_pool.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.eval.build_pool'`)

- [ ] **Step 3: 구현**

`rag/eval/build_pool.py`:
```python
"""검색 방식별 상위 결과를 모아 Gemini로 판정하고 검수용 파일을 만든다.
저장소 루트에서 `python -m rag.eval.build_pool`."""

import argparse
import json
import logging
import sys
from pathlib import Path

from rag.eval.embedders import Embedder, load
from rag.eval.judge import JudgeCache, gemini_ask, judge_query
from rag.eval.pool import pool_query
from rag.eval.search import BM25Index, dense_rank

log = logging.getLogger("rag.eval.build_pool")
_CANDIDATE_FIELDS = ("content_hash", "doc_id", "title", "heading_path", "page_start", "page_end", "content")


def rank_all(embedders: dict[str, Embedder], doc_texts: list[str], query_texts: list[str], k: int,
             bm25: BM25Index | None) -> dict[str, list[list[int]]]:
    rankings = {}
    for name, embedder in embedders.items():
        log.info("embedding with %s", name)
        rankings[name] = dense_rank(embedder.encode(query_texts, "query"), embedder.encode(doc_texts, "doc"), k)
    if bm25 is not None:
        rankings["bm25"] = [bm25.rank(q, k) for q in query_texts]
    return rankings


def build_review_items(chunks: list[dict], queries: list[dict], rankings: dict[str, list[list[int]]],
                       per_system: int, ask, cache: JudgeCache) -> dict:
    hashes = [c["content_hash"] for c in chunks]
    by_hash = {c["content_hash"]: c for c in chunks}
    out = []
    for qi, query in enumerate(queries):
        pooled = pool_query({name: [hashes[i] for i in r[qi]] for name, r in rankings.items()}, per_system)
        candidates = [{**{f: by_hash[p["content_hash"]][f] for f in _CANDIDATE_FIELDS}, "systems": p["systems"]}
                      for p in pooled]
        judged = cache.get(query["query_id"])
        if judged is None or set(judged) != {c["content_hash"] for c in candidates}:
            log.info("judging %s (%d candidates)", query["query_id"], len(candidates))
            judged = judge_query(ask, query["text"], candidates)
            cache.put(query["query_id"], judged)
        for c in candidates:
            c["judged_relevant"] = judged[c["content_hash"]]["relevant"]
            c["judged_evidence"] = judged[c["content_hash"]]["evidence"]
        out.append({**query, "candidates": candidates})
    return {"queries": out}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="평가 후보 모으기 + Gemini 1차 판정")
    parser.add_argument("--chunks", type=Path, default=Path("rag/data/chunks.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("rag/eval/data/queries.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("rag/eval/data/review_items.json"))
    parser.add_argument("--systems", default="gemini-1536,bge-m3,kure-v1,qwen3-0.6b,e5-large-instruct,bm25")
    parser.add_argument("--per-system", type=int, default=5)
    parser.add_argument("--judge-cache", type=Path, default=Path("rag/eval/cache/judgments.jsonl"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chunks, queries = _read_jsonl(args.chunks), _read_jsonl(args.queries)
    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    doc_texts = [c["content"] for c in chunks]
    embedders = {n: load(n) for n in names if n != "bm25"}
    bm25 = BM25Index(doc_texts) if "bm25" in names else None
    rankings = rank_all(embedders, doc_texts, [q["text"] for q in queries], args.per_system, bm25)
    items = build_review_items(chunks, queries, rankings, args.per_system, gemini_ask(), JudgeCache(args.judge_cache))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    total = sum(len(q["candidates"]) for q in items["queries"])
    log.info("wrote %s: %d queries, %d candidates", args.out, len(items["queries"]), total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_eval_build_pool.py -q`
Expected: 2 passed

- [ ] **Step 5: 전체 테스트**

Run: `python -m pytest tests/rag -q -W ignore::DeprecationWarning`
Expected: 51 + 31 = 82 passed

- [ ] **Step 6: Commit**

```bash
git add rag/eval/build_pool.py tests/rag/test_eval_build_pool.py
git commit -m "feat(rag-eval): add build_pool CLI producing review items

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: 검수 결과 내보내기

**Files:**
- Create: `rag/eval/export_labels.py`
- Test: `tests/rag/test_eval_export.py`

**Interfaces:**
- Consumes: `rag.eval.metrics.compact`
- Produces:
  - 검수 결과 파일 형식(`rag/eval/data/review_decisions.json`, 아티팩트 DB에서 Task 10이 만든다): `{"decisions": [{"query_id", "content_hash", "relevant": bool, "evidence": [str]}], "queries": [{"query_id", "text": str | None, "excluded": bool}]}`
  - `rag.eval.export_labels.export(items: dict, decisions: list[dict], query_edits: list[dict]) -> tuple[list[dict], list[dict], dict]`: (labels, queries, stats)
    - label 키: `query_id, doc_id, page_start, page_end, content_hash, evidence, source` (`source="human"`), (query_id, compact(evidence)) 기준 중복 제거
    - queries: review_items의 쿼리 필드(`query_id, text, form, condition, category, split`) + `excluded: bool`, 수정 문구 반영
    - stats 키: `candidates, reviewed, relevant, unreviewed, relevant_without_evidence, unknown_decisions, excluded_queries, answerable_queries_without_labels`
  - `rag.eval.export_labels.main(argv=None) -> int`: `--items`, `--decisions`, `--labels-out`(`rag/eval/data/labels.jsonl`), `--queries-out`(`rag/eval/data/queries.jsonl`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_eval_export.py`:
```python
from rag.eval.export_labels import export


def _cand(h, page):
    return {"content_hash": h, "doc_id": "kha", "title": "t", "heading_path": None, "page_start": page,
            "page_end": page, "content": "본문", "systems": ["a"], "judged_relevant": None, "judged_evidence": []}


ITEMS = {"queries": [
    {"query_id": "q1", "text": "소금", "form": "keyword", "condition": "hypertension", "category": "diet",
     "split": "tune", "candidates": [_cand("h1", 74), _cand("h2", 75), _cand("h3", 76)]},
    {"query_id": "q2", "text": "질문 초안", "form": "question", "condition": "hypertension", "category": "diet",
     "split": "tune", "candidates": [_cand("h1", 74)]},
    {"query_id": "none-1", "text": "통풍", "form": "unanswerable", "condition": "none", "category": "none",
     "split": "tune", "candidates": [_cand("h2", 75)]},
]}


def test_export_labels_and_query_edits():
    decisions = [
        {"query_id": "q1", "content_hash": "h1", "relevant": True, "evidence": ["소금을 5 g 이하로", "소금을 5g 이하로"]},
        {"query_id": "q1", "content_hash": "h2", "relevant": False, "evidence": []},
        {"query_id": "none-1", "content_hash": "h2", "relevant": False, "evidence": []},
    ]
    edits = [{"query_id": "q2", "text": "고친 질문?", "excluded": False}]
    labels, queries, stats = export(ITEMS, decisions, edits)

    assert labels == [{"query_id": "q1", "doc_id": "kha", "page_start": 74, "page_end": 74, "content_hash": "h1",
                       "evidence": "소금을 5 g 이하로", "source": "human"}]
    q2 = next(q for q in queries if q["query_id"] == "q2")
    assert q2["text"] == "고친 질문?" and q2["excluded"] is False
    assert "candidates" not in q2
    assert stats["answerable_queries_without_labels"] == ["q2"]


def test_eval_export_skips_unreviewed_and_evidence_less():
    decisions = [
        {"query_id": "q1", "content_hash": "h1", "relevant": True, "evidence": []},
        {"query_id": "q9", "content_hash": "zz", "relevant": True, "evidence": ["x"]},
    ]
    labels, _, stats = export(ITEMS, decisions, [])
    assert labels == []
    assert stats["candidates"] == 5
    assert stats["reviewed"] == 1
    assert stats["unreviewed"] == 4
    assert stats["relevant_without_evidence"] == 1
    assert stats["unknown_decisions"] == 1


def test_excluded_query_kept_with_flag_and_not_counted():
    labels, queries, stats = export(ITEMS, [], [{"query_id": "q2", "text": None, "excluded": True}])
    q2 = next(q for q in queries if q["query_id"] == "q2")
    assert q2["excluded"] is True and q2["text"] == "질문 초안"
    assert stats["excluded_queries"] == ["q2"]
    assert "q2" not in stats["answerable_queries_without_labels"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_eval_export.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.eval.export_labels'`)

- [ ] **Step 3: 구현**

`rag/eval/export_labels.py`:
```python
"""검수 결과를 근거 문장 라벨로 내보낸다. 저장소 루트에서 `python -m rag.eval.export_labels`."""

import argparse
import json
import sys
from pathlib import Path

from rag.eval.metrics import compact

_QUERY_FIELDS = ("query_id", "text", "form", "condition", "category", "split")


def export(items: dict, decisions: list[dict], query_edits: list[dict]) -> tuple[list[dict], list[dict], dict]:
    candidates = {(q["query_id"], c["content_hash"]): c for q in items["queries"] for c in q["candidates"]}
    edits = {e["query_id"]: e for e in query_edits}
    stats = {"candidates": len(candidates), "reviewed": 0, "relevant": 0, "relevant_without_evidence": 0,
             "unknown_decisions": 0}

    labels, seen = [], set()
    for d in decisions:
        candidate = candidates.get((d["query_id"], d["content_hash"]))
        if candidate is None:
            stats["unknown_decisions"] += 1
            continue
        stats["reviewed"] += 1
        if not d["relevant"]:
            continue
        stats["relevant"] += 1
        evidence = [e for e in d["evidence"] if compact(e)]
        if not evidence:
            stats["relevant_without_evidence"] += 1
            continue
        for e in evidence:
            key = (d["query_id"], compact(e))
            if key in seen:
                continue
            seen.add(key)
            labels.append({"query_id": d["query_id"], "doc_id": candidate["doc_id"],
                           "page_start": candidate["page_start"], "page_end": candidate["page_end"],
                           "content_hash": d["content_hash"], "evidence": e, "source": "human"})
    stats["unreviewed"] = stats["candidates"] - stats["reviewed"]

    queries = []
    for q in items["queries"]:
        edit = edits.get(q["query_id"], {})
        row = {f: q[f] for f in _QUERY_FIELDS}
        if edit.get("text"):
            row["text"] = edit["text"]
        row["excluded"] = bool(edit.get("excluded"))
        queries.append(row)

    labelled = {label["query_id"] for label in labels}
    stats["excluded_queries"] = [q["query_id"] for q in queries if q["excluded"]]
    stats["answerable_queries_without_labels"] = [
        q["query_id"] for q in queries
        if not q["excluded"] and q["form"] != "unanswerable" and q["query_id"] not in labelled
    ]
    return labels, queries, stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="검수 결과 → 근거 문장 라벨")
    parser.add_argument("--items", type=Path, default=Path("rag/eval/data/review_items.json"))
    parser.add_argument("--decisions", type=Path, default=Path("rag/eval/data/review_decisions.json"))
    parser.add_argument("--labels-out", type=Path, default=Path("rag/eval/data/labels.jsonl"))
    parser.add_argument("--queries-out", type=Path, default=Path("rag/eval/data/queries.jsonl"))
    args = parser.parse_args(argv)

    items = json.loads(args.items.read_text(encoding="utf-8"))
    review = json.loads(args.decisions.read_text(encoding="utf-8"))
    labels, queries, stats = export(items, review["decisions"], review["queries"])
    _write_jsonl(args.labels_out, labels)
    _write_jsonl(args.queries_out, queries)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_eval_export.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add rag/eval/export_labels.py tests/rag/test_eval_export.py
git commit -m "feat(rag-eval): export reviewed decisions as evidence labels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: 검수 아티팩트 페이지

**Files:**
- Create: `rag/eval/review/index.html`

**Interfaces:**
- Consumes: `review_items.json`(Task 7 형식, 아티팩트 보조 파일로 함께 배포)
- Produces: 아티팩트 DB 문서
  - `decisions/<query_id>__<content_hash 앞 16자>`: `{query_id, content_hash, relevant: bool, evidence: [str], reviewed_at: ISO 문자열}`
  - `queries/<query_id>`: `{query_id, text: str | null, excluded: bool}`

이 태스크는 브라우저 UI라 pytest 대상이 아니다. 대신 Step 3의 수동 확인 항목을 모두 통과해야 완료다.

- [ ] **Step 1: 스킬 로드**

`artifact-design` 스킬을 로드한다(페이지 규약: 제목, 색 토큰, 다크 모드, 모바일 폭). 아래 코드는 그 규약에 맞춰 작성되어 있으며, 스킬에 어긋나는 점이 있으면 스킬을 따르고 원장(ledger)에 Ruling으로 남긴다.

- [ ] **Step 2: 페이지 작성**

`rag/eval/review/index.html`:
```html
<title>RAG 라벨 검수</title>
<style>
:root{--bg:#fafaf9;--card:#ffffff;--text:#1c1917;--muted:#78716c;--line:#e7e5e4;--accent:#2563eb;--hit:#fef3c7;--hit-line:#f59e0b;--ok:#16a34a;--no:#dc2626}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0c0a09;--card:#1c1917;--text:#f5f5f4;--muted:#a8a29e;--line:#292524;--accent:#60a5fa;--hit:#422006;--hit-line:#f59e0b;--ok:#4ade80;--no:#f87171}}
:root[data-theme="dark"]{--bg:#0c0a09;--card:#1c1917;--text:#f5f5f4;--muted:#a8a29e;--line:#292524;--accent:#60a5fa;--hit:#422006;--hit-line:#f59e0b;--ok:#4ade80;--no:#f87171}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,-apple-system,"Apple SD Gothic Neo",sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}
header{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:1}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.muted{color:var(--muted);font-size:13px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--text);border-radius:8px;padding:6px 12px;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button:disabled{opacity:.4;cursor:default}
select,input[type=text]{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--text);max-width:100%}
.query{font-size:18px;font-weight:600;margin:12px 0 4px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin:12px 0}
.card.done-yes{border-left:4px solid var(--ok)}
.card.done-no{border-left:4px solid var(--no)}
.sent{cursor:pointer;border-radius:4px;padding:1px 2px}
.sent.on{background:var(--hit);box-shadow:inset 0 -2px 0 var(--hit-line)}
.badge{font-size:12px;padding:1px 6px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
#banner{display:none;padding:8px 12px;border-radius:8px;border:1px solid var(--no);color:var(--no);margin:8px 0}
</style>
<main>
  <header>
    <div class="row">
      <select id="qsel" aria-label="쿼리 선택"></select>
      <button id="prev">이전</button><button id="next">다음</button><button id="nextTodo">다음 미검수</button>
      <span class="muted" id="progress"></span>
    </div>
    <div id="banner"></div>
  </header>
  <div class="query" id="qtext"></div>
  <div class="row muted" id="qmeta"></div>
  <div class="row" id="qedit" style="margin-top:8px"></div>
  <div id="cards"></div>
  <p class="muted">문장을 눌러 근거를 선택·해제하세요. "관련 있음"은 근거가 하나 이상 선택돼야 저장됩니다.</p>
</main>
<script>
const $ = (id) => document.getElementById(id);
const state = { items: [], idx: 0, decisions: new Map(), queryEdits: new Map(), selected: new Map(), db: null };
const compact = (s) => s.normalize("NFC").replace(/\s+/g, "");
const decisionId = (qid, hash) => `${qid}__${hash.slice(0, 16)}`;
const sentences = (text) => text.split(/(?<=[.!?。])\s+|\n+/).map((s) => s.trim()).filter(Boolean);

function overlaps(sentence, evidence) {
  const s = compact(sentence);
  return evidence.some((e) => { const c = compact(e); return c && (s.includes(c) || c.includes(s)); });
}

function selectionFor(q, c) {
  const key = decisionId(q.query_id, c.content_hash);
  if (!state.selected.has(key)) {
    const saved = state.decisions.get(key);
    const evidence = saved ? saved.evidence : c.judged_evidence;
    const sents = sentences(c.content);
    state.selected.set(key, new Set(sents.map((s, i) => (overlaps(s, evidence) ? i : -1)).filter((i) => i >= 0)));
  }
  return state.selected.get(key);
}

function render() {
  const q = state.items[state.idx];
  if (!q) return;
  $("qsel").value = String(state.idx);
  const edit = state.queryEdits.get(q.query_id) || {};
  $("qtext").textContent = edit.text || q.text;
  $("qmeta").textContent = `${q.query_id} · ${q.form} · ${q.split}${edit.excluded ? " · 제외됨" : ""}`;
  const qedit = $("qedit");
  qedit.replaceChildren();
  if (q.form === "question") {
    const input = Object.assign(document.createElement("input"), { type: "text", value: edit.text || q.text, size: 40 });
    const save = Object.assign(document.createElement("button"), { textContent: "질문 저장" });
    save.onclick = () => saveQuery(q, { text: input.value.trim() || null, excluded: !!edit.excluded });
    qedit.append(input, save);
  }
  const ex = Object.assign(document.createElement("button"), { textContent: edit.excluded ? "제외 취소" : "이 쿼리 제외" });
  ex.onclick = () => saveQuery(q, { text: edit.text || null, excluded: !edit.excluded });
  qedit.append(ex);

  const cards = $("cards");
  cards.replaceChildren();
  q.candidates.forEach((c, n) => {
    const key = decisionId(q.query_id, c.content_hash);
    const saved = state.decisions.get(key);
    const card = document.createElement("div");
    card.className = "card" + (saved ? (saved.relevant ? " done-yes" : " done-no") : "");
    const head = document.createElement("div");
    head.className = "row muted";
    const judged = c.judged_relevant === null ? "미판정" : c.judged_relevant ? "관련 있음" : "관련 없음";
    head.textContent = `후보 ${n + 1}/${q.candidates.length} · ${c.title}${c.heading_path ? " › " + c.heading_path : ""} · p.${c.page_start}`;
    const badge = Object.assign(document.createElement("span"), { className: "badge", textContent: `Gemini: ${judged}` });
    head.append(" ", badge);
    const body = document.createElement("p");
    const sel = selectionFor(q, c);
    sentences(c.content).forEach((s, i) => {
      const span = Object.assign(document.createElement("span"), { className: "sent" + (sel.has(i) ? " on" : ""), textContent: s });
      span.onclick = () => { sel.has(i) ? sel.delete(i) : sel.add(i); render(); };
      body.append(span, " ");
    });
    const actions = document.createElement("div");
    actions.className = "row";
    const yes = Object.assign(document.createElement("button"), { className: "primary", textContent: "관련 있음 (근거 저장)" });
    yes.disabled = sel.size === 0 || !state.db;
    yes.onclick = () => saveDecision(q, c, true);
    const no = Object.assign(document.createElement("button"), { textContent: "관련 없음" });
    no.disabled = !state.db;
    no.onclick = () => saveDecision(q, c, false);
    actions.append(yes, no);
    if (saved) actions.append(Object.assign(document.createElement("span"), { className: "muted", textContent: saved.relevant ? "✓ 저장됨: 관련 있음" : "✓ 저장됨: 관련 없음" }));
    card.append(head, body, actions);
    cards.append(card);
  });
  renderProgress();
}

function renderProgress() {
  const total = state.items.reduce((n, q) => n + q.candidates.length, 0);
  $("progress").textContent = `검수 ${state.decisions.size}/${total}`;
  [...$("qsel").options].forEach((o, i) => {
    const q = state.items[i];
    const done = q.candidates.every((c) => state.decisions.has(decisionId(q.query_id, c.content_hash)));
    o.textContent = `${done ? "✓" : "·"} ${q.query_id}`;
  });
}

function showError(message) { $("banner").style.display = "block"; $("banner").textContent = message; }

async function saveDecision(q, c, relevant) {
  const key = decisionId(q.query_id, c.content_hash);
  const sents = sentences(c.content);
  const evidence = relevant ? [...selectionFor(q, c)].sort((a, b) => a - b).map((i) => sents[i]) : [];
  try {
    await state.db.doc(`decisions/${key}`).set({ query_id: q.query_id, content_hash: c.content_hash, relevant, evidence, reviewed_at: new Date().toISOString() });
  } catch (e) { showError(`저장 실패 (${e.code || e}). 잠시 후 다시 눌러 주세요.`); }
}

async function saveQuery(q, edit) {
  try {
    await state.db.doc(`queries/${q.query_id}`).set({ query_id: q.query_id, text: edit.text, excluded: edit.excluded });
  } catch (e) { showError(`저장 실패 (${e.code || e}).`); }
}

function go(i) { state.idx = Math.max(0, Math.min(state.items.length - 1, i)); render(); window.scrollTo(0, 0); }
$("prev").onclick = () => go(state.idx - 1);
$("next").onclick = () => go(state.idx + 1);
$("nextTodo").onclick = () => {
  const n = state.items.length;
  for (let k = 1; k <= n; k++) {
    const i = (state.idx + k) % n, q = state.items[i];
    if (q.candidates.some((c) => !state.decisions.has(decisionId(q.query_id, c.content_hash)))) return go(i);
  }
};
$("qsel").onchange = (e) => go(Number(e.target.value));

(async () => {
  const res = await fetch("review_items.json");
  state.items = (await res.json()).queries;
  $("qsel").replaceChildren(...state.items.map((q, i) => Object.assign(document.createElement("option"), { value: String(i), textContent: q.query_id })));
  render();
  const db = window.claude ? await window.claude.use("db") : null;
  if (!db) { showError("저장소에 연결할 수 없어 읽기 전용입니다. claude.ai에서 로그인한 상태로 열어 주세요."); render(); return; }
  state.db = db;
  db.collection("decisions").onSnapshot((snap) => {
    state.decisions = new Map(snap.docs.map((d) => [d.id, d.data()]));
    render();
  }, (e) => showError(`실시간 연결이 끊겼습니다 (${e.code}). 새로고침해 주세요.`));
  db.collection("queries").onSnapshot((snap) => {
    state.queryEdits = new Map(snap.docs.map((d) => [d.id, d.data()]));
    render();
  }, (e) => showError(`실시간 연결이 끊겼습니다 (${e.code}).`));
  render();
})();
</script>
```

- [ ] **Step 3: 소규모 데이터로 배포 확인**

실제 데이터 전에 동작만 확인한다. 아래로 2개 쿼리짜리 시험 파일을 만든다(지침 원문이 아닌 가짜 문장):
```bash
python - <<'EOF'
import json, pathlib
items = {"queries": [
  {"query_id": "demo-1", "text": "소금 얼마나 줄여요?", "form": "question", "condition": "hypertension", "category": "diet", "split": "tune",
   "candidates": [{"content_hash": "a" * 64, "doc_id": "demo", "title": "데모 문서", "heading_path": "식사", "page_start": 1, "page_end": 1,
                   "content": "데모 문장 하나입니다. 소금은 하루 5 g 이하로 줄입니다.\n다른 문장입니다.", "systems": ["x"],
                   "judged_relevant": True, "judged_evidence": ["소금은 하루 5 g 이하로 줄입니다."]}]},
  {"query_id": "demo-2", "text": "운동 권고", "form": "keyword", "condition": "hypertension", "category": "exercise", "split": "test",
   "candidates": [{"content_hash": "b" * 64, "doc_id": "demo", "title": "데모 문서", "heading_path": None, "page_start": 2, "page_end": 2,
                   "content": "주 5회 걷기를 권합니다.", "systems": ["y"], "judged_relevant": None, "judged_evidence": []}]}]}
pathlib.Path("rag/eval/cache").mkdir(parents=True, exist_ok=True)
pathlib.Path("rag/eval/cache/demo_items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
EOF
```
`Artifact` 도구로 배포한다: `file_path: rag/eval/review/index.html`, `files: {"review_items.json": "rag/eval/cache/demo_items.json"}`, `capabilities: {"db": {}}`, `icon: "checklist"`, `description: "RAG 평가용 근거 문장 라벨 검수 화면"`.

배포 직후 `ArtifactData`로 `decisions`, `queries` 컬렉션을 `list`해서 비어 있는지 확인한다. 그다음 사용자에게 데모 화면에서 아래를 눌러 보도록 요청하고, 다시 `list`로 기록을 확인한다.
- demo-1: 강조된 문장이 "소금은 하루 5 g 이하로 줄입니다."인지, "관련 있음"을 누르면 `decisions`에 `evidence`가 그 문장으로 저장되는지
- demo-2: 문장을 선택하지 않으면 "관련 있음"이 비활성인지, "관련 없음"이 저장되는지
- 질문 수정과 쿼리 제외가 `queries`에 저장되는지
- 휴대폰 폭에서 가로 스크롤 없이 보이는지

확인이 끝나면 `ArtifactData`로 데모 문서들을 삭제한다(`batch` delete).

- [ ] **Step 4: Commit**

```bash
git add rag/eval/review/index.html
git commit -m "feat(rag-eval): add label review artifact page

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: 실제 실행과 검수 (사용자 참여)

**Files:**
- Create(커밋): `rag/eval/data/queries.jsonl`, `rag/eval/data/labels.jsonl`
- Create(커밋 안 함): `rag/eval/data/review_items.json`, `rag/eval/data/review_decisions.json`, `rag/eval/cache/*`

**Interfaces:**
- Consumes: Task 2·7·8·9 전부

- [ ] **Step 1: 모델 다운로드 승인 받기**

로컬 모델 4개를 Hugging Face에서 받는다(첫 실행 시 자동). 대략 BGE-M3 2.3GB, KURE-v1 2.3GB, Qwen3-Embedding-0.6B 1.2GB, multilingual-e5-large-instruct 2.2GB, 합계 약 8GB. **사용자에게 용량을 알리고 승인을 받은 뒤 진행한다.**

- [ ] **Step 2: 쿼리 생성**

Run: `python -m rag.eval.queries`
Expected: `wrote 54 queries to rag/eval/data/queries.jsonl`. 질문형 24개를 훑어 비어 있거나 40자를 크게 넘는 문장이 없는지 본다.

- [ ] **Step 3: 청크 최신화와 후보 모으기**

Run:
```bash
python -W ignore -m rag.ingest
python -m rag.eval.build_pool
```
Expected: 마지막 로그 `wrote rag/eval/data/review_items.json: 54 queries, N candidates` (N은 약 600~900). Gemini 판정은 쿼리당 1회, 총 54회. 중간에 끊기면 같은 명령을 다시 실행하면 캐시로 이어진다.

- [ ] **Step 4: 검수 화면 배포**

Task 9 Step 3과 같은 방식으로 다시 배포하되 `files: {"review_items.json": "rag/eval/data/review_items.json"}`. 사용자에게 링크와 예상 소요(약 1.5~2시간, 나눠서 가능)를 알리고 **검수가 끝날 때까지 기다린다.**

- [ ] **Step 5: 검수 결과 가져오기**

`ArtifactData`로 `decisions`와 `queries` 컬렉션 전체를 `list`(페이지가 나뉘면 끝까지)해서 아래 형식으로 저장한다:
`rag/eval/data/review_decisions.json` = `{"decisions": [각 문서 data], "queries": [각 문서 data]}`

- [ ] **Step 6: 라벨 내보내기**

Run: `python -m rag.eval.export_labels`
Expected: stats 출력. `unreviewed`가 0이 아니거나 `relevant_without_evidence`가 0이 아니면 사용자에게 알리고 해당 항목을 다시 검수할지 묻는다. `answerable_queries_without_labels`에 나온 쿼리는 계획 B에서 Recall 계산에서 제외되므로 목록을 사용자에게 보여준다.

- [ ] **Step 7: Commit**

```bash
git add rag/eval/data/queries.jsonl rag/eval/data/labels.jsonl
git commit -m "data(rag-eval): add reviewed query set and evidence labels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
