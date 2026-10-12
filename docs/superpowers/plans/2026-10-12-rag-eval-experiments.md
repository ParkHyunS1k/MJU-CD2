# RAG 검색 구성 실험 (계획 B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 계획 A의 근거 문장 라벨로 파서 → 청킹 → 임베딩 → 하이브리드 → 리랭커 → 운영값을 한 단계씩 비교해, 단계별 결과표와 최종 설정 `rag/eval/selected.json`을 만든다.

**Architecture:** 순수 함수(지표·채점·선택)는 `metrics.py`, `evaluate.py`에 두고, `run.py`가 단계마다 구성 후보를 돌려 표(`results/0N_*.md`, `.json`)를 쓰고 1등을 `selected.json`에 누적한다. 다음 단계는 `selected.json`을 기준으로 한 변수만 바꾼다. 청크 변형은 `rag.ingest`에 파서·띄어쓰기 옵션을 추가해 `rag/data/variants/`에 캐시한다.

**Tech Stack:** Python 3.12, numpy, sentence-transformers(임베딩·CrossEncoder), kiwipiepy(BM25 토큰·띄어쓰기), rank_bm25, pdfplumber, matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-10-11-rag-eval-design.md` (3절 채점 규칙, 6~9절). 상위 설계: `docs/superpowers/specs/2026-10-08-rag-report-design.md` 10.1절.

## Global Constraints

- 채점: 검색된 청크가 같은 `doc_id`이고 근거 문장의 **50% 이상**(공백 제거 + NFC, 문자 10-gram)을 포함하면 그 근거를 찾은 것. 기존 `rag.eval.metrics.contains_evidence`를 그대로 쓴다.
- 쿼리 단위 지표: "근거 중 하나 이상 찾음"(Recall@k, MRR, nDCG@10)과 "찾은 근거 비율"(근거 Recall)을 함께 낸다.
- 글자 수 예산: 상위 결과를 누적 **3,600자**까지 담는다(첫 결과는 항상 담는다).
- 신뢰구간: 쿼리 단위 지표를 **1,000회 부트스트랩**한 95% 구간. 구간이 겹치면 더 단순·저렴한 쪽을 고른다.
- 하이브리드: 벡터 + kiwi BM25를 **RRF(k=60)** 로 합친다. 리랭커는 **상위 20개**를 재정렬한다.
- 청킹 후보: 600 / 1,200 / 2,000자 × 겹침 0 / 150자. 2단계 고정 임베딩은 **BGE-M3**.
- 임베딩 후보: gemini-embedding-001(768·1,536·3,072), BGE-M3, KURE-v1, Qwen3-Embedding-0.6B, multilingual-e5-large-instruct (이미 `embedders.REGISTRY`에 있음).
- 리랭커 후보: 없음, `BAAI/bge-reranker-v2-m3`, `dragonkue/bge-reranker-v2-m3-ko`.
- k와 임계값은 조정용(tune)으로 정하고 시험용(test)으로 보고한다.
- 결과에 **사람 검수 쿼리만의 지표**와 **영문 근거 쿼리**를 따로 싣는다(spec 5.3, 9절).
- `selected.json` 필드: `embedding_model, dimensions, parser, max_chars, overlap, hybrid, reranker, final_k, threshold` (+ 이 계획이 더하는 `spacing`, `query_form`).
- 정답 없는 쿼리(`form == "unanswerable"`)와 제외 쿼리(`excluded`)는 검색 지표에서 뺀다.
- 실험 전용 의존성은 `rag/eval/requirements.txt`에만 넣는다(서버 requirements 건드리지 않음).

**이 계획이 정하는 것 (spec이 비어 있는 곳)**
- 3~5단계 주 지표는 **nDCG@10**(k에 의존하지 않는 순위 지표; k는 6단계에서 정함). 1·2단계 주 지표는 spec대로 **Recall@예산**.
- "단순·저렴한 순서"는 각 단계 후보 목록의 순서로 표현한다(앞이 단순). 선택 규칙: 평균 1등의 하한 이하로 상한이 내려가지 않는(=구간이 겹치는) 후보 중 **가장 앞의 것**.
- 최종 k: 조정용에서 Recall@k가 Recall@10의 **90% 이상**이 되는 가장 작은 k(1~10).
- 임계값: 정답 없는 쿼리 **3개 전부**(조정용에는 1개뿐이라)의 1위 코사인 최댓값. 단, 조정용 정답 쿼리의 첫 정답 청크 코사인 중 **90%는 남도록** 그 10% 분위수를 넘지 않게 자른다. 코사인은 리랭커·RRF 점수가 아니라 임베딩 코사인(운영 pgvector와 같은 값)이다.
- 6단계 쿼리 형태: 조정용에서 키워드형 vs 질문형을 비교(키워드형이 코드로 만들어져 단순하므로 앞). 인바디형은 참고로만 표시.
- pdfplumber는 책갈피(목차)를 읽지 않으므로 목차는 PyMuPDF로 읽는다. 텍스트만 비교 대상이다.
- KURE-v2(참고 후보)는 이 계획에서 실행하지 않는다. 다중 벡터 검색이 필요해 별도 검색기가 들어가고, pgvector 운영 후보가 아니다.

## Review Focus

1. **부분 집합이 비는 경우**(예: 시험용에 사람 검수 쿼리가 없음): 지표 칸에 "–"를 표시하고 중단되지 않아야 한다 → Task 3 `test_summarize_empty_group_renders_dash`
2. **모든 후보 점수가 같은 경우**: 목록 맨 앞(가장 단순한) 후보가 선택돼야 한다 → Task 3 `test_pick_tie_prefers_first`
3. **600자 청킹에서 근거 문장이 두 청크에 반씩 걸친 경우**: 50% 이상 담은 쪽만 찾은 것으로 쳐야 한다 → Task 3 `test_find_evidence_split_across_chunks`
4. **정답 없는 쿼리·제외 쿼리**: 검색 지표 계산에 들어가면 안 된다(라벨이 없어 0점이 섞임) → Task 3 `test_load_eval_drops_unanswerable_and_excluded`
5. **`selected.json` 없이 중간 단계를 먼저 실행한 경우**: 기본값(PyMuPDF, 1,200/150, BGE-M3)으로 돌아야 한다 → Task 5 `test_stage_without_state_uses_base`

---

## File Structure

| 경로 | 역할 | 작업 |
|---|---|---|
| `rag/eval/metrics.py` | `budget_cut`, `query_metrics`, `bootstrap_ci` 추가 | 수정 (Task 1) |
| `rag/eval/search.py` | `rrf`, `Reranker` 추가 | 수정 (Task 2) |
| `rag/eval/evaluate.py` | 라벨 로드, 근거 찾기, 쿼리별 채점, 요약·선택·표 | 생성 (Task 3) |
| `rag/parse.py` | `parse_pdf_plumber`, 목차 읽기 공용화 | 수정 (Task 4) |
| `rag/ingest.py` | `parser`, `spacing` 인자와 CLI 옵션 | 수정 (Task 4) |
| `rag/eval/requirements.txt` | pdfplumber, matplotlib | 수정 (Task 4) |
| `rag/eval/run.py` | 단계 실행 CLI, 상태(`selected.json`), 결과 파일 | 생성 (Task 5~7) |
| `rag/eval/plot.py` | 단계별 막대그래프 PNG | 생성 (Task 8) |
| `rag/eval/results/` | `01_parser` … `06_operating` `.md/.json/.png`, `summary.md` | 실행 결과 (Task 9) |
| `rag/eval/selected.json` | 최종 설정 | 실행 결과 (Task 9) |
| `.gitignore` | `rag/data/variants/`는 이미 `rag/data/`로 무시됨 — 변경 없음 | — |

테스트: `tests/rag/test_eval_metrics.py`(추가), `test_eval_search.py`(추가), `test_eval_evaluate.py`, `test_parse.py`(추가), `test_ingest.py`(추가), `test_eval_run.py`, `test_eval_plot.py`.

---

### Task 1: 순위 지표, 글자 수 예산, 부트스트랩

**Files:**
- Modify: `rag/eval/metrics.py`
- Test: `tests/rag/test_eval_metrics.py` (기존 파일 끝에 추가)

**Interfaces:**
- Consumes: 없음
- Produces:
  - `budget_cut(ranked: list[int], lengths: list[int], budget: int = 3600) -> list[int]`
  - `query_metrics(ranked: list[int], found: dict[int, set[int]], n_evidence: int, lengths: list[int], ks=(1, 3, 5, 10, 20), budget: int = 3600) -> dict[str, float]` — 키: `recall@{k}`, `ev_recall@{k}`(ks마다), `mrr`, `ndcg@10`, `recall@budget`, `ev_recall@budget`, `budget_chunks`. `found`는 "청크 번호 → 그 청크가 담은 근거 번호 집합"(관련 청크만 들어 있음).
  - `bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float, float]` — (평균, 2.5%, 97.5%). 빈 목록이면 `(nan, nan, nan)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_eval_metrics.py 끝에 추가
import math

import pytest

from rag.eval.metrics import bootstrap_ci, budget_cut, query_metrics


def test_budget_cut_keeps_first_and_stops_before_overflow():
    lengths = [1000, 1000, 2000, 500, 3000]
    assert budget_cut([0, 1, 2, 3], lengths) == [0, 1]
    assert budget_cut([4, 0, 1], lengths) == [4]
    assert budget_cut([], lengths) == []


def test_query_metrics_values():
    found = {1: {0}, 3: {0, 1}}
    m = query_metrics([0, 1, 2, 3, 4], found, n_evidence=2, lengths=[1000] * 5, ks=(1, 3, 5))
    assert m["recall@1"] == 0.0 and m["recall@3"] == 1.0 and m["recall@5"] == 1.0
    assert m["ev_recall@1"] == 0.0 and m["ev_recall@3"] == 0.5 and m["ev_recall@5"] == 1.0
    assert m["mrr"] == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    assert m["ndcg@10"] == pytest.approx(dcg / (1 + 1 / math.log2(3)))
    assert m["recall@budget"] == 1.0 and m["ev_recall@budget"] == 0.5 and m["budget_chunks"] == 3.0


def test_query_metrics_nothing_found():
    m = query_metrics([0, 1], {5: {0}}, n_evidence=1, lengths=[10] * 6, ks=(1,))
    assert m["recall@1"] == 0.0 and m["mrr"] == 0.0 and m["ndcg@10"] == 0.0


def test_bootstrap_ci():
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0, 1.0)
    mean, lo, hi = bootstrap_ci([0.0, 1.0] * 25)
    assert mean == 0.5 and lo < 0.5 < hi
    assert bootstrap_ci([0.0, 1.0] * 25) == (mean, lo, hi)
    assert all(math.isnan(v) for v in bootstrap_ci([]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/rag/test_eval_metrics.py -q -W ignore::DeprecationWarning`
Expected: FAIL — `ImportError: cannot import name 'bootstrap_ci'`

- [ ] **Step 3: Implement**

`rag/eval/metrics.py` 상단 import에 `import math`, `import numpy as np` 추가, 파일 끝에:

```python
def budget_cut(ranked: list[int], lengths: list[int], budget: int = 3600) -> list[int]:
    """누적 글자 수가 budget을 넘기 전까지의 상위 결과. 첫 결과는 항상 담는다."""
    out, total = [], 0
    for i in ranked:
        if out and total + lengths[i] > budget:
            break
        out.append(i)
        total += lengths[i]
    return out


def query_metrics(ranked: list[int], found: dict[int, set[int]], n_evidence: int, lengths: list[int],
                  ks=(1, 3, 5, 10, 20), budget: int = 3600) -> dict[str, float]:
    """found: 청크 번호 → 그 청크가 담은 근거 번호. 관련 청크 = found에 있는 청크."""

    def ev(top: list[int]) -> float:
        return len(set().union(*(found[i] for i in top if i in found))) / n_evidence

    out = {}
    for k in ks:
        top = ranked[:k]
        out[f"recall@{k}"] = float(any(i in found for i in top))
        out[f"ev_recall@{k}"] = ev(top)
    first = next((r for r, i in enumerate(ranked) if i in found), None)
    out["mrr"] = 0.0 if first is None else 1 / (first + 1)
    dcg = sum(1 / math.log2(r + 2) for r, i in enumerate(ranked[:10]) if i in found)
    idcg = sum(1 / math.log2(r + 2) for r in range(min(10, len(found))))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    cut = budget_cut(ranked, lengths, budget)
    out["recall@budget"] = float(any(i in found for i in cut))
    out["ev_recall@budget"] = ev(cut)
    out["budget_chunks"] = float(len(cut))
    return out


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    if not values:
        return (math.nan, math.nan, math.nan)
    v = np.asarray(values, dtype=float)
    means = np.random.default_rng(seed).choice(v, size=(n, len(v))).mean(axis=1)
    return (float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/rag/test_eval_metrics.py -q -W ignore::DeprecationWarning`
Expected: PASS (기존 7개 + 새 4개 = 11 passed)

- [ ] **Step 5: Commit**

```bash
git add rag/eval/metrics.py tests/rag/test_eval_metrics.py
git commit -m "feat(rag-eval): add ranking metrics, char budget and bootstrap CI"
```

---

### Task 2: RRF와 리랭커

**Files:**
- Modify: `rag/eval/search.py`
- Test: `tests/rag/test_eval_search.py` (기존 파일 끝에 추가)

**Interfaces:**
- Consumes: 없음
- Produces:
  - `rrf(rankings: list[list[int]], k: int = 60) -> list[int]` — 점수 내림차순, 동점은 번호 오름차순.
  - `RERANKERS: dict[str, str]` — `{"bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3-ko": "dragonkue/bge-reranker-v2-m3-ko"}`
  - `class Reranker(name: str, model=None)`: `rerank(query: str, texts: list[str], candidates: list[int]) -> list[int]`, `ms_per_call() -> float`(모델 로드 제외, predict 시간만). `model`을 주지 않으면 생성자에서 `CrossEncoder(RERANKERS[name], max_length=512)`을 로드한다.
  - `reranker(name: str) -> Reranker` — 이름별로 한 번만 만드는 캐시(`functools.cache`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_eval_search.py 끝에 추가
from rag.eval.search import Reranker, rrf


def test_rrf_combines_and_breaks_ties_by_index():
    assert rrf([[2, 0, 1], [0, 2]]) == [0, 2, 1]
    assert rrf([[1], [0]]) == [0, 1]


class _FakeCross:
    def predict(self, pairs):
        return [len(text) for _, text in pairs]


def test_reranker_orders_by_score_and_times_calls():
    r = Reranker("bge-reranker-v2-m3", model=_FakeCross())
    assert r.rerank("q", ["a", "ccc", "bb", "dddd"], [0, 1, 2]) == [1, 2, 0]
    assert r.ms_per_call() >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/rag/test_eval_search.py -q -W ignore::DeprecationWarning`
Expected: FAIL — `ImportError: cannot import name 'Reranker'`

- [ ] **Step 3: Implement**

`rag/eval/search.py`: import에 `import time`, `from functools import cache, lru_cache`(기존 `lru_cache` import 줄을 이것으로 바꿈) 추가, 파일 끝에:

```python
def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for r, i in enumerate(ranking):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores, key=lambda i: (-scores[i], i))


RERANKERS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-v2-m3-ko": "dragonkue/bge-reranker-v2-m3-ko",
}


class Reranker:
    def __init__(self, name: str, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(RERANKERS[name], max_length=512)
        self.name = name
        self._model = model
        self._seconds = 0.0
        self._calls = 0

    def rerank(self, query: str, texts: list[str], candidates: list[int]) -> list[int]:
        start = time.perf_counter()
        scores = self._model.predict([(query, texts[i]) for i in candidates])
        self._seconds += time.perf_counter() - start
        self._calls += 1
        order = sorted(range(len(candidates)), key=lambda j: (-scores[j], j))
        return [candidates[j] for j in order]

    def ms_per_call(self) -> float:
        return self._seconds * 1000 / self._calls if self._calls else 0.0


@cache
def reranker(name: str) -> Reranker:
    return Reranker(name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/rag/test_eval_search.py -q -W ignore::DeprecationWarning`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add rag/eval/search.py tests/rag/test_eval_search.py
git commit -m "feat(rag-eval): add RRF fusion and cross-encoder reranker"
```

---

### Task 3: 채점·요약·선택 (`evaluate.py`)

**Files:**
- Create: `rag/eval/evaluate.py`
- Test: `tests/rag/test_eval_evaluate.py`

**Interfaces:**
- Consumes: `contains_evidence`, `query_metrics`, `bootstrap_ci` (Task 1)
- Produces:
  - `load_eval(queries_path: Path, labels_path: Path) -> tuple[list[dict], list[dict], dict[str, list[dict]]]` — (전체 쿼리, 채점 쿼리, 쿼리별 라벨). 채점 쿼리 = 제외 아님 + `form != "unanswerable"` + 라벨 있음.
  - `find_evidence(chunks: list[dict], labels: list[dict]) -> dict[int, set[int]]`
  - `preservation(chunks: list[dict], queries: list[dict], labels: dict[str, list[dict]]) -> float`
  - `evaluate(chunks, queries, labels, rankings: list[list[int]], ks=(1, 3, 5, 10, 20)) -> list[dict]` — 쿼리마다 `{"query_id", **query_metrics}`
  - `groups(queries, labels) -> dict[str, set[str]]` — `{"사람 검수": ..., "영문 근거": ...}`
  - `summarize(name: str, per_query: list[dict], metrics: list[str], groups: dict[str, set[str]], extra: dict | None = None) -> dict` — `{"name", metric: (mean, lo, hi) ..., f"{metrics[0]} ({그룹})": (...), **extra}`
  - `pick(rows: list[dict], metric: str) -> dict`
  - `cell(value) -> str` — 3원소 튜플/리스트는 `"0.812 [0.740, 0.880]"`, NaN이면 `"–"`, float는 소수 3자리.
  - `table(rows: list[dict], columns: list[str] | None = None) -> str` — 마크다운 표. `columns` 생략 시 모든 행의 키 합집합(첫 등장 순서)에서 `name`, `cfg`를 뺀 것. 행에 없는 칸은 빈칸.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_eval_evaluate.py
import json

from rag.eval.evaluate import (evaluate, find_evidence, groups, load_eval, pick, preservation, summarize,
                               table)

EVIDENCE = "하루 소금 섭취량을 5 g 이하로 제한한다."
CHUNKS = [
    {"doc_id": "a", "content": f"앞 내용. {EVIDENCE} 뒤 내용."},
    {"doc_id": "b", "content": EVIDENCE},
    {"doc_id": "a", "content": "운동은 주 5회 이상 한다. 관계없는 문장."},
]
LABELS = {"q1": [{"doc_id": "a", "evidence": EVIDENCE, "source": "human"}],
          "q2": [{"doc_id": "a", "evidence": "Resistance exercise twice weekly", "source": "llm"}]}


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_load_eval_drops_unanswerable_and_excluded(tmp_path):
    q = lambda qid, form="keyword", excluded=False: {"query_id": qid, "text": qid, "form": form,
                                                       "split": "tune", "excluded": excluded}
    queries = _jsonl(tmp_path / "q.jsonl", [q("q1"), q("q2", excluded=True), q("none-1", "unanswerable"), q("q3")])
    labels = _jsonl(tmp_path / "l.jsonl", [{"query_id": "q1", "doc_id": "a", "evidence": EVIDENCE, "source": "human"},
                                          {"query_id": "q2", "doc_id": "a", "evidence": EVIDENCE, "source": "llm"}])
    all_q, scored, by_q = load_eval(queries, labels)
    assert len(all_q) == 4
    assert [x["query_id"] for x in scored] == ["q1"]
    assert set(by_q) == {"q1", "q2"}


def test_find_evidence_same_doc_only():
    assert find_evidence(CHUNKS, LABELS["q1"]) == {0: {0}}


def test_find_evidence_split_across_chunks():
    # 앞 조각은 10-gram 6/9(0.67), 뒤 조각은 0/9를 담는다
    chunks = [{"doc_id": "a", "content": EVIDENCE[:-3]}, {"doc_id": "a", "content": EVIDENCE[-8:]}]
    assert find_evidence(chunks, LABELS["q1"]) == {0: {0}}


def test_preservation_and_evaluate():
    queries = [{"query_id": "q1"}, {"query_id": "q2"}]
    assert preservation(CHUNKS, queries, LABELS) == 0.5
    per_query = evaluate(CHUNKS, queries, LABELS, [[2, 0], [0, 1]], ks=(1,))
    assert [p["recall@1"] for p in per_query] == [0.0, 0.0]
    assert per_query[0]["mrr"] == 0.5 and per_query[0]["query_id"] == "q1"


def test_groups_human_and_english():
    queries = [{"query_id": "q1"}, {"query_id": "q2"}]
    assert groups(queries, LABELS) == {"사람 검수": {"q1"}, "영문 근거": {"q2"}}


def test_summarize_empty_group_renders_dash():
    per_query = [{"query_id": "q1", "ndcg@10": 1.0, "mrr": 1.0}]
    row = summarize("base", per_query, ["ndcg@10", "mrr"], {"사람 검수": set()}, {"chunks": 3})
    assert row["ndcg@10"] == (1.0, 1.0, 1.0) and row["chunks"] == 3
    md = table([row])
    assert "| 구성 | ndcg@10 | mrr | ndcg@10 (사람 검수) | chunks |" in md
    assert "| base | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | – | 3 |" in md


def test_pick_prefers_simplest_overlapping():
    rows = [{"name": "A", "m": (0.70, 0.60, 0.80)}, {"name": "B", "m": (0.75, 0.65, 0.85)},
            {"name": "C", "m": (0.90, 0.85, 0.95)}]
    assert pick(rows, "m")["name"] == "B"
    rows[2]["m"] = (0.90, 0.86, 0.95)
    assert pick(rows, "m")["name"] == "C"


def test_pick_tie_prefers_first():
    rows = [{"name": n, "m": [0.5, 0.4, 0.6]} for n in "XYZ"]
    assert pick(rows, "m")["name"] == "X"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/rag/test_eval_evaluate.py -q -W ignore::DeprecationWarning`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.eval.evaluate'`

- [ ] **Step 3: Implement**

```python
# rag/eval/evaluate.py
"""근거 문장 라벨로 검색 결과를 채점하고, 구성별 결과를 표와 선택으로 정리한다."""

import json
import math
import re
from pathlib import Path

from rag.eval.metrics import bootstrap_ci, contains_evidence, query_metrics

_HANGUL = re.compile("[가-힣]")
_KS = (1, 3, 5, 10, 20)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_eval(queries_path: Path, labels_path: Path) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    """(전체 쿼리, 채점 쿼리, 쿼리별 라벨). 제외·정답 없음 쿼리는 검색 지표에서 뺀다."""
    queries = _read_jsonl(queries_path)
    labels: dict[str, list[dict]] = {}
    for label in _read_jsonl(labels_path):
        labels.setdefault(label["query_id"], []).append(label)
    scored = [q for q in queries
              if not q["excluded"] and q["form"] != "unanswerable" and q["query_id"] in labels]
    return queries, scored, labels


def find_evidence(chunks: list[dict], labels: list[dict]) -> dict[int, set[int]]:
    """청크 번호 → 그 청크가 담은 근거 번호 (같은 doc_id + 50% 이상 포함)."""
    by_doc: dict[str, list[int]] = {}
    for ci, chunk in enumerate(chunks):
        by_doc.setdefault(chunk["doc_id"], []).append(ci)
    found: dict[int, set[int]] = {}
    for ei, label in enumerate(labels):
        for ci in by_doc.get(label["doc_id"], []):
            if contains_evidence(chunks[ci]["content"], label["evidence"]):
                found.setdefault(ci, set()).add(ei)
    return found


def preservation(chunks: list[dict], queries: list[dict], labels: dict[str, list[dict]]) -> float:
    """근거 문장 중 어떤 청크엔가 50% 이상 들어 있는 비율."""
    kept = total = 0
    for q in queries:
        qlabels = labels[q["query_id"]]
        kept += len(set().union(*find_evidence(chunks, qlabels).values()))
        total += len(qlabels)
    return kept / total


def evaluate(chunks: list[dict], queries: list[dict], labels: dict[str, list[dict]],
             rankings: list[list[int]], ks=_KS) -> list[dict]:
    lengths = [len(c["content"]) for c in chunks]
    out = []
    for q, ranked in zip(queries, rankings, strict=True):
        qlabels = labels[q["query_id"]]
        found = find_evidence(chunks, qlabels)
        out.append({"query_id": q["query_id"], **query_metrics(ranked, found, len(qlabels), lengths, ks)})
    return out


def groups(queries: list[dict], labels: dict[str, list[dict]]) -> dict[str, set[str]]:
    human = {q["query_id"] for q in queries if any(l["source"] == "human" for l in labels[q["query_id"]])}
    english = {q["query_id"] for q in queries
               if any(not _HANGUL.search(l["evidence"]) for l in labels[q["query_id"]])}
    return {"사람 검수": human, "영문 근거": english}


def summarize(name: str, per_query: list[dict], metrics: list[str], groups: dict[str, set[str]],
              extra: dict | None = None) -> dict:
    """metrics[0]이 주 지표다. 부분 집합 칸은 주 지표만 낸다."""
    row: dict = {"name": name}
    for m in metrics:
        row[m] = bootstrap_ci([p[m] for p in per_query])
    for g, ids in groups.items():
        row[f"{metrics[0]} ({g})"] = bootstrap_ci([p[metrics[0]] for p in per_query if p["query_id"] in ids])
    return row | (extra or {})


def pick(rows: list[dict], metric: str) -> dict:
    """구간이 1등과 겹치는 후보 중 목록 앞(더 단순·저렴한) 것."""
    best = max(rows, key=lambda r: r[metric][0])
    return next(r for r in rows if r[metric][2] >= best[metric][1])


def cell(value) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 3:
        mean, lo, hi = value
        return "–" if math.isnan(mean) else f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def table(rows: list[dict], columns: list[str] | None = None) -> str:
    columns = columns or list(dict.fromkeys(k for r in rows for k in r if k not in ("name", "cfg")))
    head = "| 구성 | " + " | ".join(columns) + " |"
    sep = "|---" * (len(columns) + 1) + "|"
    body = ["| " + " | ".join([r["name"], *(cell(r.get(c, "")) for c in columns)]) + " |" for r in rows]
    return "\n".join([head, sep, *body])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/rag/test_eval_evaluate.py -q -W ignore::DeprecationWarning`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add rag/eval/evaluate.py tests/rag/test_eval_evaluate.py
git commit -m "feat(rag-eval): score rankings against evidence labels with CI and selection rule"
```

---

### Task 4: 파서·띄어쓰기 변형 수집

**Files:**
- Modify: `rag/parse.py`, `rag/ingest.py`, `rag/eval/requirements.txt`
- Test: `tests/rag/test_parse.py`, `tests/rag/test_ingest.py` (끝에 추가)

**Interfaces:**
- Consumes: 기존 `parse_pdf`, `ingest(sources_path, raw_dir, max_chars, overlap)`
- Produces:
  - `parse_pdf_plumber(path: Path) -> ParsedDoc` — 줄 단위 Block, 목차는 PyMuPDF.
  - `ingest(sources_path, raw_dir, max_chars=1200, overlap=150, parser="pymupdf", spacing=False)` — `parser in {"pymupdf", "pdfplumber"}`, `spacing=True`면 정제 뒤 블록에 kiwi 띄어쓰기 적용.
  - CLI: `python -m rag.ingest --parser pdfplumber --spacing`

- [ ] **Step 1: Install dependency**

`rag/eval/requirements.txt` 끝에 두 줄 추가:

```
pdfplumber==0.11.4
matplotlib==3.10.6
```

Run: `pip install pdfplumber==0.11.4`
Expected: `Successfully installed pdfplumber-0.11.4` (pdfminer.six 등 함께)

- [ ] **Step 2: Write the failing tests**

```python
# tests/rag/test_parse.py 끝에 추가
import pymupdf

from rag.parse import parse_pdf, parse_pdf_plumber


def test_parse_pdf_plumber_matches_pymupdf_text_and_toc(tmp_path):
    path = tmp_path / "en.pdf"
    doc = pymupdf.open()
    for lines in (["Blood pressure guideline", "Reduce salt intake daily."], ["Exercise five days a week."]):
        page = doc.new_page(width=595, height=842)
        for i, text in enumerate(lines):
            page.insert_text((72, 100 + 100 * i), text, fontname="helv", fontsize=11)
    doc.set_toc([[1, "Blood pressure guideline", 1]])
    doc.save(path)
    doc.close()

    a, b = parse_pdf(path), parse_pdf_plumber(path)
    assert [x.text for x in b.blocks] == [x.text for x in a.blocks]
    assert [x.page for x in b.blocks] == [1, 1, 2]
    assert b.toc == a.toc
    assert b.page_heights == a.page_heights
```

```python
# tests/rag/test_ingest.py 끝에 추가
from rag.eval.metrics import compact


def test_ingest_spacing_restores_spaces(tmp_path, make_pdf):
    make_pdf("nospace.pdf", [[(100, "나트륨섭취를줄인다.")]])
    sources = _sources(tmp_path, [_entry("nospace", "nospace.pdf")])
    plain, _ = ingest(sources, tmp_path)
    spaced, _ = ingest(sources, tmp_path, spacing=True)
    body = lambda records: records[0]["content"].split("\n", 1)[1]
    assert " " not in body(plain)
    assert body(spaced) == "나트륨 섭취를 줄인다."
    assert compact(body(spaced)) == compact(body(plain))


def test_ingest_rejects_unknown_parser(tmp_path, make_pdf):
    make_pdf("a.pdf", [[(100, "본문")]])
    sources = _sources(tmp_path, [_entry("a", "a.pdf")])
    import pytest
    with pytest.raises(KeyError):
        ingest(sources, tmp_path, parser="nope")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/rag/test_parse.py tests/rag/test_ingest.py -q -W ignore::DeprecationWarning`
Expected: FAIL — `ImportError: cannot import name 'parse_pdf_plumber'` (parse), `TypeError: ingest() got an unexpected keyword argument 'spacing'` (ingest)

- [ ] **Step 4: Implement**

`rag/parse.py` — 목차 읽기를 함수로 빼고 pdfplumber 파서를 추가(`parse_pdf`의 목차 부분을 `toc = _toc(doc)`로 바꿈):

```python
def _toc(doc) -> list[TocEntry]:
    return [
        TocEntry(level, _nfc(title), page_no)
        for level, title, page_no, *_ in doc.get_toc(simple=True)
        if page_no >= 1
    ]


def parse_pdf_plumber(path: Path) -> ParsedDoc:
    """pdfplumber 줄 단위 추출. pdfplumber는 책갈피를 읽지 않아 목차만 PyMuPDF로 읽는다."""
    import pdfplumber

    blocks, heights, widths = [], [], []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            heights.append(float(page.height))
            widths.append(float(page.width))
            for line in page.extract_text_lines():
                text = _nfc(line["text"])
                if text:
                    blocks.append(Block(number, line["top"], line["bottom"], text, line["x0"], line["x1"]))
    with pymupdf.open(path) as doc:
        toc = _toc(doc)
    return ParsedDoc(blocks, toc, heights, widths)
```

`rag/ingest.py`:

```python
# import 추가/변경
from dataclasses import replace
from functools import cache

from rag.parse import Block, parse_pdf, parse_pdf_plumber

PARSERS = {"pymupdf": parse_pdf, "pdfplumber": parse_pdf_plumber}


@cache
def _kiwi():
    from kiwipiepy import Kiwi  # 띄어쓰기 실험에서만 필요한 의존성

    return Kiwi()


def _space(blocks: list[Block]) -> list[Block]:
    return [replace(b, text=_kiwi().space(b.text)) for b in blocks]
```

`ingest` 시그니처를 `def ingest(sources_path, raw_dir, max_chars=1200, overlap=150, parser="pymupdf", spacing=False)`로 바꾸고, 루프 앞에 `parse = PARSERS[parser]`(알 수 없는 이름이면 여기서 `KeyError`), `parsed = parse_pdf(path)` → `parsed = parse(path)`, `clean_blocks(...)` 바로 다음(빈 블록 검사 뒤)에:

```python
        if spacing:
            blocks = _space(blocks)
```

`main`에 인자 두 개를 추가하고 `ingest` 호출에 넘긴다:

```python
    parser.add_argument("--parser", choices=sorted(PARSERS), default="pymupdf")
    parser.add_argument("--spacing", action="store_true", help="kiwi로 띄어쓰기 복원")
    ...
        records, stats = ingest(args.sources, args.raw_dir, args.max_chars, args.overlap, args.parser, args.spacing)
```

(`main` 안의 지역 변수 `parser`(ArgumentParser)와 모듈 상수 `PARSERS`는 이름이 다르므로 충돌하지 않는다.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/rag -q -W ignore::DeprecationWarning`
Expected: PASS (전체; 새 테스트 3개 포함)

- [ ] **Step 6: Commit**

```bash
git add rag/parse.py rag/ingest.py rag/eval/requirements.txt tests/rag/test_parse.py tests/rag/test_ingest.py
git commit -m "feat(rag): add pdfplumber parser and kiwi spacing options to ingest"
```

---

### Task 5: 실험 실행기 뼈대 + 1·2단계 (파서, 청킹)

**Files:**
- Create: `rag/eval/run.py`
- Test: `tests/rag/test_eval_run.py`

**Interfaces:**
- Consumes: `load`(embedders), `BM25Index`, `dense_rank`, `rrf`, `reranker`(Task 2), `evaluate`, `load_eval`, `groups`, `summarize`, `pick`, `table`, `preservation`(Task 3), `ingest(..., parser, spacing)`(Task 4)
- Produces (Task 6·7이 씀):
  - 모듈 상수 `STATE`, `RESULTS`, `VARIANTS`, `QUERIES`, `LABELS`, `BASE`, `METRICS`, `DEPTH = 20`, `POOL = 100`
  - `load_state() -> dict`, `save_state(state: dict) -> None`
  - `chunk_variant(parser: str, spacing: bool, max_chars: int, overlap: int) -> list[dict]`, `chunks_for(state: dict) -> list[dict]`
  - `retrieve(state: dict, chunks: list[dict], texts: list[str]) -> tuple[list[list[int]], np.ndarray]` — (상위 20개 순위, 쿼리×청크 임베딩 코사인)
  - `class Ctx(queries_path=QUERIES, labels_path=LABELS)`: `.all_queries`, `.queries`(채점 쿼리), `.labels`, `.groups`
  - `run_stage(ctx, fname, title, variants: list[tuple[str, dict]], primary: str, extra=None, note="") -> dict` — 선택된 행 반환, `selected.json`에 `cfg`를 합쳐 저장, `results/{fname}.md`·`.json` 작성. `extra(cfg, chunks) -> dict`는 검색 뒤 호출된다.
  - `write_stage(fname, title, rows, chosen_name, primary, note="") -> None`
  - `stage1(ctx)`, `stage2(ctx)`, CLI `python -m rag.eval.run 1 2 …`

테스트용 가짜 임베더: 문자 2-gram 해시 64차원. 테스트는 `monkeypatch`로 `run.load`, `run.chunk_variant`, 경로 상수를 바꾼다.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_eval_run.py
import json
import zlib

import numpy as np
import pytest

from rag.eval import run

EV1 = "하루 소금 섭취량을 5 g 이하로 제한한다."
EV2 = "주 5회 이상 유산소 운동을 30분 한다."
CHUNKS = [
    {"doc_id": "a", "content": f"식사 요법. {EV1}"},
    {"doc_id": "a", "content": f"운동 요법. {EV2}"},
    {"doc_id": "a", "content": "고혈압의 역학과 유병률 통계를 다룬다."},
    {"doc_id": "b", "content": "통풍 환자는 퓨린 섭취를 줄인다."},
]


class FakeEmbedder:
    name = "fake"

    def encode(self, texts, kind):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for r, t in enumerate(texts):
            for i in range(len(t) - 1):
                out[r, zlib.crc32(t[i:i + 2].encode()) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    def q(qid, text, form="keyword", split="tune"):
        return {"query_id": qid, "text": text, "form": form, "condition": "hypertension",
                "category": "diet", "split": split, "excluded": False}
    queries = [q("q1", "소금 섭취량 제한"), q("q2", "유산소 운동 30분", "question"),
               q("q3", "소금 섭취 줄이기", split="test"), q("q4", "운동 몇 분", "question", "test"),
               q("none-1", "통풍 식이요법", "unanswerable", "test")]
    labels = [{"query_id": "q1", "doc_id": "a", "evidence": EV1, "source": "human"},
              {"query_id": "q2", "doc_id": "a", "evidence": EV2, "source": "llm"},
              {"query_id": "q3", "doc_id": "a", "evidence": EV1, "source": "llm"},
              {"query_id": "q4", "doc_id": "a", "evidence": EV2, "source": "llm"}]
    qp, lp = tmp_path / "queries.jsonl", tmp_path / "labels.jsonl"
    qp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in queries), encoding="utf-8")
    lp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labels), encoding="utf-8")
    monkeypatch.setattr(run, "STATE", tmp_path / "selected.json")
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run, "load", lambda name: FakeEmbedder())
    monkeypatch.setattr(run, "chunk_variant", lambda *args: CHUNKS)
    return run.Ctx(qp, lp)


def test_stage_without_state_uses_base(ctx):
    assert run.load_state() == run.BASE


def test_retrieve_ranks_matching_chunk_first(ctx):
    ranked, sims = run.retrieve(run.BASE, CHUNKS, ["소금 섭취량 제한"])
    assert ranked[0][0] == 0 and sims.shape == (1, 4)


def test_stage1_writes_results_and_state(ctx):
    chosen = run.stage1(ctx)
    assert chosen["cfg"] == {"parser": "pymupdf", "spacing": False}  # 모두 같으면 맨 앞
    state = json.loads(run.STATE.read_text())
    assert state["parser"] == "pymupdf" and state["max_chars"] == 1200
    md = (run.RESULTS / "01_parser.md").read_text()
    assert "preservation" in md and "**선택: pymupdf**" in md
    saved = json.loads((run.RESULTS / "01_parser.json").read_text())
    assert saved["primary"] == "recall@budget" and len(saved["rows"]) == 4


def test_stage2_varies_chunking(ctx):
    run.stage2(ctx)
    saved = json.loads((run.RESULTS / "02_chunking.json").read_text())
    assert [r["cfg"] for r in saved["rows"]][:2] == [{"max_chars": 2000, "overlap": 0},
                                                     {"max_chars": 2000, "overlap": 150}]
    assert json.loads(run.STATE.read_text())["max_chars"] == 2000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/rag/test_eval_run.py -q -W ignore::DeprecationWarning`
Expected: FAIL — `ImportError: cannot import name 'run' from 'rag.eval'`

- [ ] **Step 3: Implement**

```python
# rag/eval/run.py
"""계획 B 실험. 저장소 루트에서 `python -m rag.eval.run 1 2 3 4 5 6`.
단계마다 한 변수만 바꿔 비교하고, 1등을 selected.json에 누적해 다음 단계의 기준으로 쓴다."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from rag.eval.embedders import load
from rag.eval.evaluate import evaluate, groups, load_eval, pick, preservation, summarize, table
from rag.eval.search import BM25Index, dense_rank, reranker, rrf
from rag.ingest import ingest

log = logging.getLogger("rag.eval.run")

SOURCES, RAW = Path("rag/sources.json"), Path("rag/raw")
VARIANTS = Path("rag/data/variants")
QUERIES, LABELS = Path("rag/eval/data/queries.jsonl"), Path("rag/eval/data/labels.jsonl")
RESULTS = Path("rag/eval/results")
STATE = Path("rag/eval/selected.json")
DEPTH = 20  # 지표는 상위 20개까지 본다
POOL = 100  # 하이브리드로 합치기 전 각 방식에서 가져오는 수
BASE = {"parser": "pymupdf", "spacing": False, "max_chars": 1200, "overlap": 150,
        "embedding_model": "bge-m3", "dimensions": None, "hybrid": False, "reranker": None,
        "query_form": None, "final_k": None, "threshold": None}
METRICS = ["ndcg@10", "recall@3", "recall@20", "mrr", "recall@budget", "ev_recall@budget"]


def load_state() -> dict:
    return {**BASE, **(json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {})}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def chunk_variant(parser: str, spacing: bool, max_chars: int, overlap: int) -> list[dict]:
    path = VARIANTS / f"{parser}-{'spaced' if spacing else 'raw'}-{max_chars}-{overlap}.jsonl"
    if not path.exists():
        records, stats = ingest(SOURCES, RAW, max_chars, overlap, parser, spacing)
        log.info("%s: %s", path.name, json.dumps(stats, ensure_ascii=False))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def chunks_for(state: dict) -> list[dict]:
    return chunk_variant(state["parser"], state["spacing"], state["max_chars"], state["overlap"])


def retrieve(state: dict, chunks: list[dict], texts: list[str]) -> tuple[list[list[int]], np.ndarray]:
    """상위 DEPTH개 순위와 쿼리×청크 임베딩 코사인(임계값용)."""
    embedder = load(state["embedding_model"])
    docs = [c["content"] for c in chunks]
    qv, dv = embedder.encode(texts, "query"), embedder.encode(docs, "doc")
    ranked = dense_rank(qv, dv, POOL)
    if state["hybrid"]:
        bm25 = BM25Index(docs)
        ranked = [rrf([r, bm25.rank(t, POOL)]) for r, t in zip(ranked, texts)]
    ranked = [r[:DEPTH] for r in ranked]
    if state["reranker"]:
        rr = reranker(state["reranker"])
        ranked = [rr.rerank(t, docs, r) for t, r in zip(texts, ranked)]
    return ranked, qv @ dv.T


class Ctx:
    def __init__(self, queries_path: Path = QUERIES, labels_path: Path = LABELS):
        self.all_queries, self.queries, self.labels = load_eval(queries_path, labels_path)
        self.groups = groups(self.queries, self.labels)


def write_stage(fname: str, title: str, rows: list[dict], chosen_name: str, primary: str, note: str = "") -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    md = f"# {title}\n\n주 지표: `{primary}` (평균 [95% 부트스트랩 구간])\n\n{table(rows)}\n\n**선택: {chosen_name}**\n"
    if note:
        md += f"\n{note}\n"
    (RESULTS / f"{fname}.md").write_text(md, encoding="utf-8")
    (RESULTS / f"{fname}.json").write_text(json.dumps({"primary": primary, "rows": rows}, ensure_ascii=False),
                                           encoding="utf-8")


def run_stage(ctx: Ctx, fname: str, title: str, variants: list[tuple[str, dict]], primary: str,
              extra=None, note: str = "") -> dict:
    state = load_state()
    metrics = [primary, *(m for m in METRICS if m != primary)]
    texts = [q["text"] for q in ctx.queries]
    rows = []
    for name, overrides in variants:
        cfg = {**state, **overrides}
        chunks = chunks_for(cfg)
        ranked, _ = retrieve(cfg, chunks, texts)
        info = {"chunks": len(chunks), **(extra(cfg, chunks) if extra else {})}
        per_query = evaluate(chunks, ctx.queries, ctx.labels, ranked)
        rows.append(summarize(name, per_query, metrics, ctx.groups, info) | {"cfg": overrides})
        log.info("%s %s: %s=%.3f", fname, name, primary, rows[-1][primary][0])
    chosen = pick(rows, primary)
    save_state({**state, **chosen["cfg"]})
    write_stage(fname, title, rows, chosen["name"], primary, note)
    return chosen


def stage1(ctx: Ctx) -> dict:
    variants = [(f"{p}{' + 띄어쓰기' if s else ''}", {"parser": p, "spacing": s})
                for p in ("pymupdf", "pdfplumber") for s in (False, True)]
    return run_stage(
        ctx, "01_parser", "1단계: 파서", variants, "recall@budget",
        extra=lambda cfg, chunks: {"preservation": preservation(chunks, ctx.queries, ctx.labels)},
        note="라벨 근거 문장은 PyMuPDF 텍스트에서 뽑았으므로 preservation은 PyMuPDF에 유리하다. "
             "띄어쓰기는 공백을 지우고 비교하는 채점에 영향이 없고, 임베딩·BM25 결과로만 드러난다.")


def stage2(ctx: Ctx) -> dict:
    variants = [(f"{m}자 / 겹침 {o}", {"max_chars": m, "overlap": o})
                for m in (2000, 1200, 600) for o in (0, 150)]
    return run_stage(ctx, "02_chunking", "2단계: 청킹 (BGE-M3, 글자 수 예산 3,600자)", variants,
                     "recall@budget", note="앞쪽일수록 청크 수가 적다(저장·검색 비용이 작다).")


STAGES = {1: stage1, 2: stage2}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 검색 구성 실험 (계획 B)")
    parser.add_argument("stages", nargs="+", type=int, choices=sorted(STAGES))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ctx = Ctx()
    for n in args.stages:
        chosen = STAGES[n](ctx)
        log.info("stage %d chose %s", n, chosen["name"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/rag/test_eval_run.py -q -W ignore::DeprecationWarning`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add rag/eval/run.py tests/rag/test_eval_run.py
git commit -m "feat(rag-eval): add experiment runner with parser and chunking stages"
```

---

### Task 6: 3~5단계 (임베딩, 하이브리드, 리랭커)

**Files:**
- Modify: `rag/eval/run.py`
- Test: `tests/rag/test_eval_run.py` (끝에 추가)

**Interfaces:**
- Consumes: Task 5의 `run_stage`, `load_state`, `save_state`, `REGISTRY`(embedders), `reranker`(Task 2)
- Produces:
  - `EMBEDDINGS: list[str]` — 저장 용량(차원) 작은 순: `["gemini-768", "bge-m3", "kure-v1", "qwen3-0.6b", "e5-large-instruct", "gemini-1536", "gemini-3072"]`
  - `embedding_info(name: str, n_chunks: int, sample: list[str]) -> dict` — `{"dims", "storage_mb", "query_ms"}`. `storage_mb` = 차원 × 2바이트(halfvec) × 청크 수 / 1e6.
  - `stage3(ctx)` — 선택 후 `selected.json`의 `dimensions`를 선택 모델의 `dims`로 채운다.
  - `stage4(ctx)`, `stage5(ctx)` — 5단계 행에 `rerank_ms`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_eval_run.py 끝에 추가
def test_stage3_records_dims_and_storage(ctx, monkeypatch):
    monkeypatch.setattr(run, "REGISTRY", {n: FakeEmbedder for n in run.EMBEDDINGS})
    run.stage3(ctx)
    state = json.loads(run.STATE.read_text())
    assert state["embedding_model"] == "gemini-768"  # 모두 같으면 맨 앞
    assert state["dimensions"] == 64
    row = json.loads((run.RESULTS / "03_embedding.json").read_text())["rows"][0]
    assert row["dims"] == 64 and row["storage_mb"] == round(64 * 2 * 4 / 1e6, 3)


class FakeReranker:
    def rerank(self, query, texts, candidates):
        return list(reversed(candidates))

    def ms_per_call(self):
        return 1.5


def test_stage4_and_stage5(ctx, monkeypatch):
    monkeypatch.setattr(run, "reranker", lambda name: FakeReranker())
    run.stage4(ctx)
    assert [r["cfg"] for r in json.loads((run.RESULTS / "04_hybrid.json").read_text())["rows"]] == [
        {"hybrid": False}, {"hybrid": True}]
    run.stage5(ctx)
    rows = json.loads((run.RESULTS / "05_reranker.json").read_text())["rows"]
    assert rows[0]["cfg"] == {"reranker": None} and rows[1]["rerank_ms"] == 1.5
    assert "rerank_ms" in (run.RESULTS / "05_reranker.md").read_text()  # 첫 행에 없는 칸도 표에 나온다
    assert json.loads(run.STATE.read_text())["reranker"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/rag/test_eval_run.py -q -W ignore::DeprecationWarning`
Expected: FAIL — `AttributeError: module 'rag.eval.run' has no attribute 'stage3'`

- [ ] **Step 3: Implement**

`rag/eval/run.py` import를 `import time`, `from rag.eval.embedders import REGISTRY, load`, `from rag.eval.search import RERANKERS, BM25Index, dense_rank, reranker, rrf`로 바꾸고, `stage2` 다음에:

```python
EMBEDDINGS = ["gemini-768", "bge-m3", "kure-v1", "qwen3-0.6b", "e5-large-instruct", "gemini-1536", "gemini-3072"]


def embedding_info(name: str, n_chunks: int, sample: list[str]) -> dict:
    """캐시를 거치지 않고 쿼리 임베딩 지연을 잰다(로컬 모델은 로드·워밍업 뒤)."""
    inner = REGISTRY[name]()
    inner.encode(sample[:1], "query")
    start = time.perf_counter()
    vectors = inner.encode(sample, "query")
    dims = int(vectors.shape[1])
    return {"dims": dims, "storage_mb": round(dims * 2 * n_chunks / 1e6, 3),
            "query_ms": round((time.perf_counter() - start) * 1000 / len(sample), 1)}


def stage3(ctx: Ctx) -> dict:
    sample = [q["text"] for q in ctx.queries[:10]]
    chosen = run_stage(
        ctx, "03_embedding", "3단계: 임베딩 모델", [(n, {"embedding_model": n}) for n in EMBEDDINGS], "ndcg@10",
        extra=lambda cfg, chunks: embedding_info(cfg["embedding_model"], len(chunks), sample),
        note="앞쪽일수록 저장 용량(차원)이 작다. storage_mb는 halfvec(2바이트) 기준. "
             "gemini의 query_ms는 네트워크 왕복을 포함한다.")
    save_state({**load_state(), "dimensions": chosen["dims"]})
    return chosen


def stage4(ctx: Ctx) -> dict:
    return run_stage(ctx, "04_hybrid", "4단계: 하이브리드 (벡터 + kiwi BM25, RRF k=60)",
                     [("벡터만", {"hybrid": False}), ("벡터 + BM25", {"hybrid": True})], "ndcg@10")


def stage5(ctx: Ctx) -> dict:
    variants = [("없음", {"reranker": None})] + [(n, {"reranker": n}) for n in RERANKER_NAMES]
    return run_stage(
        ctx, "05_reranker", "5단계: 리랭커 (상위 20개 재정렬)", variants, "ndcg@10",
        extra=lambda cfg, chunks: {"rerank_ms": reranker(cfg["reranker"]).ms_per_call()} if cfg["reranker"] else {},
        note="rerank_ms는 쿼리 하나의 상위 20개 재정렬 시간(모델 로드 제외). 청크는 512토큰에서 잘린다.")
```

`EMBEDDINGS` 옆에 `RERANKER_NAMES = list(RERANKERS)` 상수를 두고, `STAGES`를 `{1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5}`로 바꾼다.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/rag/test_eval_run.py -q -W ignore::DeprecationWarning`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add rag/eval/run.py tests/rag/test_eval_run.py
git commit -m "feat(rag-eval): add embedding, hybrid and reranker stages"
```

---

### Task 7: 6단계 운영값, 요약, 최종 설정

**Files:**
- Modify: `rag/eval/run.py`
- Test: `tests/rag/test_eval_run.py` (끝에 추가)

**Interfaces:**
- Consumes: Task 5·6의 함수들, `evaluate(..., ks=...)`(Task 3)
- Produces:
  - `choose_k(recall: dict[int, float], ratio: float = 0.9) -> int` — Recall@k ≥ ratio × Recall@10 인 가장 작은 k.
  - `choose_threshold(none_top: list[float], hit_scores: list[float], keep: float = 0.9) -> float` — `min(max(none_top), quantile(hit_scores, 1 - keep))`, 소수 4자리 반올림.
  - `stage6(ctx)` — `results/06_operating.md`·`.json`, `results/summary.md`, `selected.json`의 `query_form`, `final_k`, `threshold`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_eval_run.py 끝에 추가
def test_choose_k():
    # 0.9 × Recall@10(0.95) = 0.855 → k=5(0.85) 미달, k=6(0.9)이 처음 넘는다
    recall = dict(zip(range(1, 11), [0.2, 0.5, 0.7, 0.8, 0.85, 0.9, 0.9, 0.95, 0.95, 0.95]))
    assert run.choose_k(recall) == 6
    assert run.choose_k(dict.fromkeys(range(1, 11), 0.0)) == 1


def test_choose_threshold_caps_to_keep_answers():
    hits = [0.70, 0.72, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.91, 0.95]
    assert run.choose_threshold([0.50, 0.60], hits) == 0.6
    assert run.choose_threshold([0.90], hits) == round(float(np.quantile(hits, 0.1)), 4)
    assert run.choose_threshold([0.55], []) == 0.55  # 조정용에서 정답을 못 찾았으면 상한 없음


def test_stage6_writes_operating_values_and_summary(ctx):
    run.stage6(ctx)
    state = json.loads(run.STATE.read_text())
    assert state["query_form"] in ("keyword", "question")
    assert 1 <= state["final_k"] <= 10
    assert isinstance(state["threshold"], float)
    md = (run.RESULTS / "06_operating.md").read_text()
    assert "쿼리 형태" in md and "최종 k" in md and "임계값" in md and "시험용" in md
    summary = (run.RESULTS / "summary.md").read_text()
    assert "selected.json" in summary and "사람 검수" in summary
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/rag/test_eval_run.py -q -W ignore::DeprecationWarning`
Expected: FAIL — `AttributeError: module 'rag.eval.run' has no attribute 'choose_k'`

- [ ] **Step 3: Implement**

`stage5` 다음에:

```python
def choose_k(recall: dict[int, float], ratio: float = 0.9) -> int:
    return next(k for k in sorted(recall) if recall[k] >= ratio * recall[10])


def choose_threshold(none_top: list[float], hit_scores: list[float], keep: float = 0.9) -> float:
    """정답 없는 쿼리의 최고 코사인. 단 정답 청크의 keep 비율은 남도록 상한을 둔다."""
    top = max(none_top)
    if hit_scores:
        top = min(top, float(np.quantile(hit_scores, 1 - keep)))
    return round(top, 4)


def _first_hit_scores(chunks, queries, labels, ranked, sims, k) -> list[float]:
    scores = []
    for qi, (q, r) in enumerate(zip(queries, ranked)):
        found = find_evidence(chunks, labels[q["query_id"]])
        hit = next((i for i in r[:k] if i in found), None)
        if hit is not None:
            scores.append(float(sims[qi, hit]))
    return scores


def stage6(ctx: Ctx) -> dict:
    state = load_state()
    chunks = chunks_for(state)
    tune = [q for q in ctx.queries if q["split"] == "tune"]
    test = [q for q in ctx.queries if q["split"] == "test"]
    texts = lambda qs: [q["text"] for q in qs]
    sections = []

    # 1) 쿼리 형태 (조정용). 키워드형이 코드로 만들어져 단순하므로 앞에 둔다.
    form_rows = []
    for form in ("keyword", "question", "inbody"):
        qs = [q for q in tune if q["form"] == form]
        if qs:
            ranked, _ = retrieve(state, chunks, texts(qs))
            per_query = evaluate(chunks, qs, ctx.labels, ranked)
            form_rows.append(summarize(form, per_query, METRICS, {}, {"queries": len(qs)}))
    query_form = pick([r for r in form_rows if r["name"] != "inbody"], "ndcg@10")["name"]
    sections.append(f"## 쿼리 형태 (조정용)\n\n{table(form_rows)}\n\n**선택: {query_form}** (inbody는 참고)")

    # 2) 최종 k (조정용)
    ranked, sims = retrieve(state, chunks, texts(tune))
    per_query = evaluate(chunks, tune, ctx.labels, ranked, ks=range(1, 11))
    recall = {k: float(np.mean([p[f"recall@{k}"] for p in per_query])) for k in range(1, 11)}
    final_k = choose_k(recall)
    k_table = "\n".join([
        "| k | " + " | ".join(str(k) for k in recall) + " |",
        "|---" * (len(recall) + 1) + "|",
        "| Recall@k | " + " | ".join(f"{v:.3f}" for v in recall.values()) + " |",
    ])
    sections.append(f"## 최종 k (조정용)\n\n{k_table}\n\n**선택: k = {final_k}** (Recall@k ≥ 0.9 × Recall@10 인 최소 k)")

    # 3) 임계값: 정답 없는 쿼리 전부(조정용엔 1개뿐) + 조정용 정답 청크 코사인
    none = [q for q in ctx.all_queries if q["form"] == "unanswerable" and not q["excluded"]]
    _, none_sims = retrieve(state, chunks, texts(none))
    none_top = [float(row.max()) for row in none_sims]
    hits = _first_hit_scores(chunks, tune, ctx.labels, ranked, sims, final_k)
    threshold = choose_threshold(none_top, hits)
    sections.append(
        f"## 임계값 (임베딩 코사인)\n\n- 정답 없는 쿼리 1위 코사인: {', '.join(f'{s:.4f}' for s in none_top)}\n"
        f"- 조정용 첫 정답 청크 코사인 {len(hits)}개 중 90%가 남도록 상한을 둠\n\n**선택: {threshold}**")

    # 4) 시험용 보고 (임계값 적용 전/후)
    ranked_t, sims_t = retrieve(state, chunks, texts(test))
    filtered = [[i for i in r if sims_t[qi, i] > threshold] for qi, r in enumerate(ranked_t)]
    ks = (final_k, 10)
    rows = [summarize("임계값 없음", evaluate(chunks, test, ctx.labels, ranked_t, ks),
                      [f"recall@{final_k}", "ndcg@10", "mrr", f"ev_recall@{final_k}"], ctx.groups),
            summarize(f"임계값 {threshold}", evaluate(chunks, test, ctx.labels, filtered, ks),
                      [f"recall@{final_k}", "ndcg@10", "mrr", f"ev_recall@{final_k}"], ctx.groups)]
    rejected = sum(float(row.max()) <= threshold for row in none_sims)
    sections.append(f"## 시험용 결과\n\n{table(rows)}\n\n정답 없는 쿼리 {len(none)}개 중 {rejected}개가 "
                    f"임계값으로 걸러진다(조정용 1개 포함).")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "06_operating.md").write_text("# 6단계: 운영값\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    (RESULTS / "06_operating.json").write_text(
        json.dumps({"primary": "ndcg@10", "rows": form_rows}, ensure_ascii=False), encoding="utf-8")
    state = {**state, "query_form": query_form, "final_k": final_k, "threshold": threshold}
    save_state(state)
    _write_summary(state, rows[1], len(none), rejected)
    return {"name": f"{query_form}, k={final_k}, threshold={threshold}", "cfg": state}


def _write_summary(state: dict, test_row: dict, n_none: int, rejected: int) -> None:
    k = state["final_k"]
    lines = [
        "# 검색 구성 실험 요약",
        "",
        "최종 설정은 `rag/eval/selected.json`에 있다(하위 프로젝트 3의 입력).",
        "",
        "| 항목 | 선택 |", "|---|---|",
        *(f"| {key} | {state[key]} |" for key in BASE),
        "",
        f"시험용 Recall@{k}: {cell(test_row[f'recall@{k}'])}, nDCG@10: {cell(test_row['ndcg@10'])}, "
        f"사람 검수 쿼리만: {cell(test_row[f'recall@{k} (사람 검수)'])}. "
        f"정답 없는 쿼리 {n_none}개 중 {rejected}개를 임계값으로 거른다.",
        "",
        "단계별 근거는 `01_parser.md` … `06_operating.md`. 신뢰구간이 겹치면 더 단순·저렴한 구성을 골랐다.",
        "",
        "## 한계",
        "- 라벨은 사람 검수 188건 + LLM(GPT) 판정. 사람 검수 쿼리는 고혈압·당뇨·이상지질혈증뿐이라 "
        "\"사람 검수\" 열을 함께 본다.",
        "- 후보 모으기(pooling)로 만든 라벨이라, 어떤 방식도 찾지 못한 정답은 빠져 있다.",
        "- 근거 문장은 PyMuPDF 텍스트에서 뽑아 1단계 preservation은 PyMuPDF에 유리하다.",
        "- 쿼리 50개 내외라 차이가 작으면 구간이 겹친다.",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
```

`run.py`의 evaluate import를 `from rag.eval.evaluate import cell, evaluate, find_evidence, groups, load_eval, pick, preservation, summarize, table`로 바꾸고, `STAGES`에 `6: stage6` 추가.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/rag -q -W ignore::DeprecationWarning`
Expected: PASS (전체)

- [ ] **Step 5: Commit**

```bash
git add rag/eval/run.py tests/rag/test_eval_run.py
git commit -m "feat(rag-eval): add operating-values stage, summary and selected config"
```

---

### Task 8: 그래프

**Files:**
- Create: `rag/eval/plot.py`
- Test: `tests/rag/test_eval_plot.py`

**Interfaces:**
- Consumes: `results/0N_*.json` 형식 `{"primary": str, "rows": [{"name", primary: [mean, lo, hi], ...}]}` (Task 5·7)
- Produces: `plot_stage(json_path: Path, out_png: Path) -> None`, CLI `python -m rag.eval.plot` → `results/0N_*.png`

- [ ] **Step 1: Write the failing test**

```python
# tests/rag/test_eval_plot.py
import json

from rag.eval.plot import plot_stage


def test_plot_stage_writes_png(tmp_path):
    src = tmp_path / "03_embedding.json"
    src.write_text(json.dumps({"primary": "ndcg@10", "rows": [
        {"name": "bge-m3", "ndcg@10": [0.6, 0.5, 0.7]},
        {"name": "kure-v1", "ndcg@10": [0.65, 0.55, 0.75]},
        {"name": "빈 값", "ndcg@10": [float("nan")] * 3},
    ]}), encoding="utf-8")
    out = tmp_path / "03_embedding.png"
    plot_stage(src, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/rag/test_eval_plot.py -q -W ignore::DeprecationWarning`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.eval.plot'`

- [ ] **Step 3: Implement**

```python
# rag/eval/plot.py
"""단계별 결과 JSON → 주 지표 막대그래프(95% 구간 오차 막대). 저장소 루트에서 `python -m rag.eval.plot`."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = ["AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_stage(json_path: Path, out_png: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    metric = data["primary"]
    rows = data["rows"]
    mean, lo, hi = (np.array([r[metric][i] for r in rows], dtype=float) for i in range(3))
    fig, ax = plt.subplots(figsize=(max(4.0, 1.3 * len(rows)), 3.2))
    ax.bar([r["name"] for r in rows], np.nan_to_num(mean),
           yerr=[np.nan_to_num(mean - lo), np.nan_to_num(hi - mean)], capsize=4, color="#4C72B0")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(json_path.stem)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(results: Path = Path("rag/eval/results")) -> int:
    for path in sorted(results.glob("0*_*.json")):
        plot_stage(path, path.with_suffix(".png"))
        print(path.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/rag/test_eval_plot.py -q -W ignore::DeprecationWarning`
Expected: PASS (1 passed; 글꼴 경고는 무시)

- [ ] **Step 5: Commit**

```bash
git add rag/eval/plot.py tests/rag/test_eval_plot.py
git commit -m "feat(rag-eval): plot stage results with CI error bars"
```

---

### Task 9: 실제 실행과 결과 커밋

**Files:**
- Create (실행 결과): `rag/eval/results/*.md`, `*.json`, `*.png`, `rag/eval/selected.json`

**Interfaces:**
- Consumes: Task 5~8 CLI
- Produces: 하위 프로젝트 3이 읽을 `rag/eval/selected.json`

모델 내려받기(BGE-M3 약 2.3GB, 리랭커 2개 각 약 2.3GB)와 CPU/MPS 임베딩으로 단계마다 수 분~수십 분 걸린다. 임베딩은 `rag/eval/cache/embeddings/`에 캐시되므로 끊겨도 다시 실행하면 이어진다. Gemini 임베딩은 무료 등급 분당 한도에 걸리면 `call_with_retry`가 기다린다.

- [ ] **Step 1: 1~2단계 실행**

Run: `python -m rag.eval.run 1 2 2>&1 | tee .superpowers/sdd/2026-10-12-rag-eval-experiments/run12.log | tail -20`
Expected: `stage 1 chose …`, `stage 2 chose …` 두 줄. `rag/data/variants/`에 jsonl 4 + 5개(1200/150 pymupdf는 1단계와 공유될 수 있음). `rag/eval/results/01_parser.md`, `02_chunking.md` 생성.

확인: `cat rag/eval/results/01_parser.md` — pdfplumber 행의 `chunks`가 0이거나 preservation이 0.5 미만이면 pdfplumber가 한글을 못 읽은 것이다. 그 경우 결과는 그대로 두고(그 사실이 결과다) 노트에 원인을 적는다.

- [ ] **Step 2: 3단계 실행**

Run: `python -m rag.eval.run 3 2>&1 | tail -20`
Expected: `stage 3 chose …`, `03_embedding.md`의 7개 행 모두 `dims`, `storage_mb`, `query_ms`가 채워짐.

- [ ] **Step 3: 4~6단계 실행**

Run: `python -m rag.eval.run 4 5 6 2>&1 | tail -20`
Expected: `stage 6 chose keyword|question, k=…, threshold=…`. `selected.json`에 `final_k`, `threshold`가 null 아님.

- [ ] **Step 4: 그래프**

Run: `python -m rag.eval.plot`
Expected: `rag/eval/results/01_parser.png` … `06_operating.png` 경로 6줄.

- [ ] **Step 5: 결과 검토**

`summary.md`와 `06_operating.md`를 읽고 다음을 확인한다. 이상하면 원인을 찾고(systematic-debugging) 결과 노트에 적는다.
- 시험용 Recall@k가 조정용 값과 크게(0.2 이상) 다르지 않은가
- 임계값 적용 후 시험용 Recall@k가 적용 전보다 0.1 넘게 떨어지지 않는가 (떨어지면 summary에 그대로 보고)
- "사람 검수" 열과 전체 열의 차이

- [ ] **Step 6: Commit**

```bash
git add rag/eval/results rag/eval/selected.json
git commit -m "docs(rag-eval): add retrieval experiment results and selected config"
```
