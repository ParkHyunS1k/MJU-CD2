# RAG 코퍼스 수집 파이프라인 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `rag/sources.json`에 등록한 PDF 가이드라인을 파싱·정제·섹션 분할·청킹·태깅해 `rag/data/chunks.jsonl`로 출력하는 오프라인 파이프라인을 만든다.

**Architecture:** `training/`처럼 서버와 분리된 `rag/` 패키지. 단계마다 순수 함수 모듈 하나(parse → clean → sections → chunk → records)를 두고 `rag/ingest.py` CLI가 엮는다. 파서·청킹은 이후 비교 실험 대상이므로 기준선(PyMuPDF, 구조 기반 + 길이 상한)으로 구현하고 파라미터는 CLI 인자로 노출한다.

**Tech Stack:** Python 3.13, PyMuPDF, pydantic v2(서버 의존성 재사용), pytest

**Spec:** `docs/superpowers/specs/2026-10-08-rag-report-design.md` (4·5·11절)

**Spec과 다른 점:** spec 11절은 하위 프로젝트 1에 "마이그레이션"을 포함하지만, `halfvec(N)`의 N과 `embedding_model`은 하위 프로젝트 2(모델 선정) 결과로 정해진다. 그래서 **임베딩과 DB 적재는 하위 프로젝트 3으로 옮긴다.** 이 계획의 산출물은 로컬 JSONL이고, 하위 프로젝트 2의 평가 실험(로컬 numpy)은 이 JSONL을 입력으로 쓴다.

## Global Constraints

- condition 키는 정확히 `hypertension`, `diabetes`, `dyslipidemia`, `obesity`, `metabolic_syndrome`, `fatty_liver`, `sarcopenia`, `general`
- category는 정확히 `lifestyle`, `diet`, `exercise`, `supplement`, `criteria`
- 청크 `content` = `문서제목 > 장 > 절` + 줄바꿈 + 본문 (spec 4절)
- 모든 텍스트는 유니코드 NFC로 정규화
- `content_hash` = `content`의 SHA-256 hex. 같은 입력이면 같은 출력(재실행 시 파일이 바이트 단위로 동일)
- 원본 PDF(`rag/raw/`)와 산출물(`rag/data/`)은 Git에 올리지 않음
- PyMuPDF는 `rag/requirements.txt`에만 추가. 서버 `requirements.txt`와 `app/`의 런타임 코드는 PyMuPDF를 import하지 않음
- 실행과 테스트는 저장소 루트에서 `python -m ...`으로 함

## Review Focus

1. **텍스트 레이어가 없는 PDF(스캔본)**: 빈 청크를 만들거나 죽지 않고, 경고와 함께 건너뛰고 통계에 남아야 한다 → Task 7 `test_ingest_skips_scanned_pdf`
2. **책갈피(TOC)가 없는 PDF**: 제목 경로 없이도 본문 전체가 쪽 번호와 함께 청크로 나와야 한다 → Task 4 `test_no_toc_single_section`, Task 7 `test_ingest_end_to_end`
3. **TOC 제목과 본문 제목 표기가 다른 경우**(띄어쓰기, "제1장" 등): 섹션 경계는 그 쪽 첫 블록으로 대체하고 본문은 하나도 잃지 않아야 한다 → Task 4 `test_unmatched_title_falls_back_to_page_start_without_losing_text`
4. **문장 구분 없이 아주 긴 문단**: 어떤 청크 본문도 `max_chars`를 넘지 않고 글자를 잃지 않아야 한다 → Task 5 `test_overlong_block_without_sentence_breaks`
5. **깨진 PDF 한 개가 섞인 배치**: 전체 실행이 중단되지 않고 해당 문서만 실패로 보고되어야 한다 → Task 7 `test_ingest_reports_corrupt_pdf_and_continues`

---

## File Structure

| 파일 | 책임 |
|---|---|
| `app/services/rag_taxonomy.py` | 공통 용어집(condition 키, category). 하위 프로젝트 3에서 매핑표가 추가됨 |
| `rag/__init__.py` | 패키지 표시 (빈 파일) |
| `rag/sources.py` | `sources.json` 스키마 검증과 파일 경로 해석 |
| `rag/parse.py` | PDF → 블록(쪽, y좌표, 텍스트) + TOC |
| `rag/clean.py` | 쪽 번호와 반복되는 머리말·꼬리말 제거 |
| `rag/sections.py` | TOC 기준 섹션 분할, 참고문헌 제외 |
| `rag/chunk.py` | 섹션 → 길이 상한·overlap 청크 |
| `rag/records.py` | 청크 → 출력 레코드(태깅, content, hash), 중복 제거 |
| `rag/ingest.py` | CLI. 위 단계를 엮어 JSONL과 통계 출력 |
| `rag/sources.json` | 문서 목록 (처음엔 빈 배열, 사용자가 채움) |
| `rag/requirements.txt` | 수집 전용 의존성 |
| `rag/README.md` | 사용법, sources.json 형식 |
| `tests/rag/conftest.py` | 테스트용 PDF 생성 fixture |
| `tests/rag/test_*.py` | 모듈별 테스트 |

---

### Task 1: 공통 용어집과 sources.json 로더

**Files:**
- Create: `app/services/rag_taxonomy.py`
- Create: `rag/__init__.py`, `rag/sources.py`, `rag/sources.json`, `rag/requirements.txt`
- Modify: `.gitignore`
- Test: `tests/rag/test_sources.py`

**Interfaces:**
- Produces:
  - `app.services.rag_taxonomy.CONDITION_KEYS: frozenset[str]`
  - `app.services.rag_taxonomy.Category = Literal["lifestyle", "diet", "exercise", "supplement", "criteria"]`
  - `rag.sources.SourcesError(Exception)`
  - `rag.sources.SectionRule(BaseModel)`: `match: str`, `category: Category`
  - `rag.sources.SourceDoc(BaseModel)`: `id, file, title, publisher, year, url, license_note, conditions: list[str], category: Category, section_rules: list[SectionRule]`
  - `rag.sources.load_sources(sources_path: Path, raw_dir: Path) -> list[tuple[SourceDoc, Path]]`

- [ ] **Step 1: 의존성 설치와 gitignore**

`rag/requirements.txt`:
```text
# 수집 전용. 서버(../requirements.txt)에는 필요 없음.
pymupdf
```

설치 후 실제 버전으로 고정한다:
```bash
pip install -r requirements.txt -r rag/requirements.txt
python -c "import pymupdf; print(pymupdf.__version__)"
```
출력된 버전으로 `pymupdf` 줄을 `pymupdf==<버전>`으로 바꾼다.

`.gitignore` 끝에 추가:
```text

# RAG 원본 문서(재배포 불가)와 수집 산출물
rag/raw/
rag/data/
```

`rag/__init__.py`는 빈 파일, `rag/sources.json`은 `[]`로 만든다.

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/rag/test_sources.py`:
```python
import json

import pytest

from rag.sources import SourcesError, load_sources


def _write(tmp_path, entries, files=("a.pdf",)):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in files:
        (raw / name).write_bytes(b"%PDF-1.4")
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return sources, raw


def _entry(**overrides):
    entry = {
        "id": "kshguide-2022",
        "file": "a.pdf",
        "title": "고혈압 진료지침",
        "publisher": "대한고혈압학회",
        "year": 2022,
        "conditions": ["hypertension"],
        "category": "lifestyle",
        "section_rules": [{"match": "운동", "category": "exercise"}],
    }
    entry.update(overrides)
    return entry


def test_load_valid_sources(tmp_path):
    sources, raw = _write(tmp_path, [_entry()])
    [(doc, path)] = load_sources(sources, raw)
    assert doc.id == "kshguide-2022"
    assert doc.section_rules[0].category == "exercise"
    assert path == raw / "a.pdf"


def test_unknown_condition_rejected(tmp_path):
    sources, raw = _write(tmp_path, [_entry(conditions=["cholesterol"])])
    with pytest.raises(SourcesError, match="cholesterol"):
        load_sources(sources, raw)


def test_unknown_category_rejected(tmp_path):
    sources, raw = _write(tmp_path, [_entry(category="sleep")])
    with pytest.raises(SourcesError):
        load_sources(sources, raw)


def test_missing_file_rejected(tmp_path):
    sources, raw = _write(tmp_path, [_entry(file="missing.pdf")])
    with pytest.raises(SourcesError, match="missing.pdf"):
        load_sources(sources, raw)


def test_duplicate_ids_rejected(tmp_path):
    sources, raw = _write(tmp_path, [_entry(), _entry()])
    with pytest.raises(SourcesError, match="duplicate"):
        load_sources(sources, raw)


def test_invalid_json_rejected(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sources = tmp_path / "sources.json"
    sources.write_text("[{", encoding="utf-8")
    with pytest.raises(SourcesError):
        load_sources(sources, raw)
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_sources.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.sources'`)

- [ ] **Step 4: 구현**

`app/services/rag_taxonomy.py`:
```python
"""RAG 공통 용어집. 코퍼스 청크 태그와 예측 모델 출력(RiskSignal)이 공유하는 키."""

from typing import Literal

CONDITION_KEYS = frozenset({
    "hypertension",
    "diabetes",
    "dyslipidemia",
    "obesity",
    "metabolic_syndrome",
    "fatty_liver",
    "sarcopenia",
    "general",
})

Category = Literal["lifestyle", "diet", "exercise", "supplement", "criteria"]
```

`rag/sources.py`:
```python
import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.services.rag_taxonomy import CONDITION_KEYS, Category


class SourcesError(Exception):
    pass


class SectionRule(BaseModel):
    match: str = Field(min_length=1)
    category: Category


class SourceDoc(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    file: str
    title: str = Field(min_length=1)
    publisher: str | None = None
    year: int | None = None
    url: str | None = None
    license_note: str | None = None
    conditions: list[str] = Field(min_length=1)
    category: Category
    section_rules: list[SectionRule] = []

    @field_validator("conditions")
    @classmethod
    def _known_conditions(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - CONDITION_KEYS)
        if unknown:
            raise ValueError(f"unknown condition keys: {unknown}")
        return value


def load_sources(sources_path: Path, raw_dir: Path) -> list[tuple[SourceDoc, Path]]:
    try:
        raw = json.loads(sources_path.read_text(encoding="utf-8"))
        docs = [SourceDoc.model_validate(entry) for entry in raw]
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as e:
        raise SourcesError(f"{sources_path}: {e}") from e

    ids = [doc.id for doc in docs]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise SourcesError(f"duplicate source ids: {duplicates}")

    resolved = []
    for doc in docs:
        path = raw_dir / doc.file
        if not path.is_file():
            raise SourcesError(f"[{doc.id}] file not found: {path}")
        resolved.append((doc, path))
    return resolved
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_sources.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add .gitignore app/services/rag_taxonomy.py rag/__init__.py rag/sources.py rag/sources.json rag/requirements.txt tests/rag/test_sources.py
git commit -m "feat(rag): add taxonomy and sources.json loader

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: PDF 파싱

**Files:**
- Create: `rag/parse.py`
- Create: `tests/rag/conftest.py`
- Test: `tests/rag/test_parse.py`

**Interfaces:**
- Produces:
  - `rag.parse.Block(page: int, y0: float, y1: float, text: str)`: frozen dataclass, `page`는 1부터 시작
  - `rag.parse.TocEntry(level: int, title: str, page: int)`: frozen dataclass
  - `rag.parse.ParsedDoc(blocks: list[Block], toc: list[TocEntry], page_heights: list[float])`
  - `rag.parse.parse_pdf(path: Path) -> ParsedDoc`
  - 테스트 fixture `make_pdf(name: str, pages: list[list[tuple[float, str]]], toc: list[list] | None = None) -> Path`: 각 쪽은 `(기준선 y, 텍스트)` 목록. A4(595×842pt)

- [ ] **Step 1: 테스트용 PDF fixture 작성**

`tests/rag/conftest.py`:
```python
import pymupdf
import pytest


@pytest.fixture
def make_pdf(tmp_path):
    def factory(name, pages, toc=None):
        path = tmp_path / name
        doc = pymupdf.open()
        for blocks in pages:
            page = doc.new_page(width=595, height=842)
            for y, text in blocks:
                page.insert_text((72, y), text, fontname="korea", fontsize=11)
        if toc:
            doc.set_toc(toc)
        doc.save(path)
        doc.close()
        return path

    return factory
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/rag/test_parse.py`:
```python
from rag.parse import TocEntry, parse_pdf


def test_parse_blocks_pages_and_toc(make_pdf):
    path = make_pdf(
        "doc.pdf",
        [
            [(100, "1. 생활요법"), (200, "나트륨 섭취를 줄인다.")],
            [(100, "규칙적인 유산소 운동을 한다.")],
        ],
        toc=[[1, "1. 생활요법", 1]],
    )
    parsed = parse_pdf(path)

    assert [(b.page, b.text) for b in parsed.blocks] == [
        (1, "1. 생활요법"),
        (1, "나트륨 섭취를 줄인다."),
        (2, "규칙적인 유산소 운동을 한다."),
    ]
    assert parsed.blocks[0].y0 < parsed.blocks[1].y0
    assert parsed.toc == [TocEntry(level=1, title="1. 생활요법", page=1)]
    assert parsed.page_heights == [842, 842]


def test_parse_pdf_without_text_layer(make_pdf):
    path = make_pdf("scan.pdf", [[]])
    parsed = parse_pdf(path)
    assert parsed.blocks == []
    assert parsed.toc == []
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_parse.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.parse'`)

- [ ] **Step 4: 구현**

`rag/parse.py`:
```python
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class Block:
    page: int  # 1부터 시작
    y0: float
    y1: float
    text: str


@dataclass(frozen=True)
class TocEntry:
    level: int
    title: str
    page: int  # 1부터 시작


@dataclass(frozen=True)
class ParsedDoc:
    blocks: list[Block]
    toc: list[TocEntry]
    page_heights: list[float]


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def parse_pdf(path: Path) -> ParsedDoc:
    blocks = []
    heights = []
    with pymupdf.open(path) as doc:
        for page in doc:
            heights.append(page.rect.height)
            for _x0, y0, _x1, y1, text, _no, kind in page.get_text("blocks", sort=True):
                text = _nfc(text)
                if kind == 0 and text:
                    blocks.append(Block(page.number + 1, y0, y1, text))
        toc = [
            TocEntry(level, _nfc(title), page_no)
            for level, title, page_no, *_ in doc.get_toc(simple=True)
            if page_no >= 1
        ]
    return ParsedDoc(blocks, toc, heights)
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_parse.py -v`
Expected: 2 passed. 한글이 깨져 `fontname="korea"` 관련 오류가 나면, PyMuPDF 문서의 내장 CJK 폰트 이름을 확인해 fixture만 고친다(파서 코드는 그대로).

- [ ] **Step 6: Commit**

```bash
git add rag/parse.py tests/rag/conftest.py tests/rag/test_parse.py
git commit -m "feat(rag): parse PDF into page blocks and TOC

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 정제 (쪽 번호, 머리말·꼬리말)

**Files:**
- Create: `rag/clean.py`
- Test: `tests/rag/test_clean.py`

**Interfaces:**
- Consumes: `rag.parse.Block`
- Produces: `rag.clean.clean_blocks(blocks: list[Block], page_heights: list[float]) -> list[Block]`

규칙:
- 숫자만 있는 블록(`12`, `- 12 -`)은 위치와 상관없이 제거한다.
- 쪽 위·아래 8% 영역에 있으면서, 숫자를 `#`으로 바꾼 형태가 `max(3, 전체 쪽 수의 30%)` 이상의 쪽에 반복되는 블록은 제거한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_clean.py`:
```python
from rag.clean import clean_blocks
from rag.parse import Block

H = 842.0


def _page(n, body):
    return [
        Block(n, 30, 43, "대한고혈압학회 진료지침 2022"),
        Block(n, 300, 313, body),
        Block(n, 810, 823, f"- {n} -"),
    ]


def test_removes_repeated_header_footer_and_page_numbers():
    blocks = [b for n in range(1, 5) for b in _page(n, f"본문 {n}쪽 내용")]
    cleaned = clean_blocks(blocks, [H] * 4)
    assert [b.text for b in cleaned] == [f"본문 {n}쪽 내용" for n in range(1, 5)]


def test_keeps_repeated_text_in_body_area():
    blocks = [Block(n, 300, 313, "나트륨 섭취를 줄인다.") for n in range(1, 5)]
    assert clean_blocks(blocks, [H] * 4) == blocks


def test_keeps_header_when_document_too_short_to_detect():
    blocks = [b for n in range(1, 3) for b in _page(n, "본문")]
    texts = [b.text for b in clean_blocks(blocks, [H] * 2)]
    assert "대한고혈압학회 진료지침 2022" in texts
    assert "- 1 -" not in texts
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_clean.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.clean'`)

- [ ] **Step 3: 구현**

`rag/clean.py`:
```python
import re

from rag.parse import Block

_PAGE_NUMBER = re.compile(r"^[-–—\s]*\d{1,4}[-–—\s]*$")
_EDGE_RATIO = 0.08


def _signature(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text)).strip()


def clean_blocks(blocks: list[Block], page_heights: list[float]) -> list[Block]:
    def at_edge(block: Block) -> bool:
        height = page_heights[block.page - 1]
        return block.y1 <= height * _EDGE_RATIO or block.y0 >= height * (1 - _EDGE_RATIO)

    pages_by_signature: dict[str, set[int]] = {}
    for block in blocks:
        if at_edge(block):
            pages_by_signature.setdefault(_signature(block.text), set()).add(block.page)

    min_pages = max(3, int(len(page_heights) * 0.3))
    repeated = {sig for sig, pages in pages_by_signature.items() if len(pages) >= min_pages}

    return [
        block
        for block in blocks
        if not _PAGE_NUMBER.match(block.text)
        and not (at_edge(block) and _signature(block.text) in repeated)
    ]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_clean.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add rag/clean.py tests/rag/test_clean.py
git commit -m "feat(rag): strip page numbers and repeated headers/footers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 섹션 분할

**Files:**
- Create: `rag/sections.py`
- Test: `tests/rag/test_sections.py`

**Interfaces:**
- Consumes: `rag.parse.Block`, `rag.parse.TocEntry`
- Produces:
  - `rag.sections.Section(heading_path: list[str], blocks: list[Block])`: frozen dataclass. `heading_path`에는 문서 제목이 들어가지 않음(TOC 경로만)
  - `rag.sections.build_sections(blocks: list[Block], toc: list[TocEntry]) -> list[Section]`

규칙:
- TOC가 없으면 블록 전체를 `heading_path=[]`인 섹션 하나로 만든다.
- TOC 항목마다 해당 쪽에서 제목 블록을 찾는다. 공백을 모두 지운 TOC 제목이, 공백을 지운 블록 텍스트의 앞부분(제목 길이 + 10자)에 포함되면 찾은 것으로 본다. 찾으면 그 블록 다음부터 섹션이 시작하고 제목 블록은 버린다. 못 찾으면 그 쪽의 첫 블록부터 시작하고 아무것도 버리지 않는다.
- 탐색은 앞 항목의 위치 이후부터 하므로 섹션 시작 위치는 줄어들지 않는다.
- 첫 TOC 항목 앞의 블록은 `heading_path=[]` 섹션이 된다.
- 경로의 어느 제목이든 앞 번호를 떼고 `참고문헌`, `References`, `Bibliography`이면 그 섹션은 버린다.
- 빈 섹션은 버린다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_sections.py`:
```python
from rag.parse import Block, TocEntry
from rag.sections import Section, build_sections


def _b(page, y, text):
    return Block(page, y, y + 12, text)


BLOCKS = [
    _b(1, 50, "머리말 문단"),
    _b(1, 100, "1. 생활요법"),
    _b(1, 200, "본문A"),
    _b(1, 300, "1.1  나트륨 제한"),
    _b(1, 400, "본문B"),
    _b(2, 100, "본문C"),
    _b(3, 100, "참고문헌"),
    _b(3, 200, "Kim et al. 2020"),
]
TOC = [
    TocEntry(1, "1. 생활요법", 1),
    TocEntry(2, "1.1 나트륨 제한", 1),
    TocEntry(1, "참고문헌", 3),
]


def test_sections_follow_toc_and_drop_references():
    sections = build_sections(BLOCKS, TOC)
    assert [(s.heading_path, [b.text for b in s.blocks]) for s in sections] == [
        ([], ["머리말 문단"]),
        (["1. 생활요법"], ["본문A"]),
        (["1. 생활요법", "1.1 나트륨 제한"], ["본문B", "본문C"]),
    ]


def test_no_toc_single_section():
    assert build_sections(BLOCKS, []) == [Section([], BLOCKS)]


def test_no_blocks_no_sections():
    assert build_sections([], TOC) == []
    assert build_sections([], []) == []


def test_unmatched_title_falls_back_to_page_start_without_losing_text():
    blocks = [_b(1, 100, "서론 내용"), _b(2, 100, "제1장 생활 요법 개요"), _b(2, 200, "본문")]
    toc = [TocEntry(1, "생활요법 총론", 2)]
    sections = build_sections(blocks, toc)
    assert [(s.heading_path, [b.text for b in s.blocks]) for s in sections] == [
        ([], ["서론 내용"]),
        (["생활요법 총론"], ["제1장 생활 요법 개요", "본문"]),
    ]


def test_numbered_references_title_dropped():
    blocks = [_b(1, 100, "1. 운동"), _b(1, 200, "본문"), _b(2, 100, "5. 참고 문헌"), _b(2, 200, "Lee 2019")]
    toc = [TocEntry(1, "1. 운동", 1), TocEntry(1, "5. 참고 문헌", 2)]
    sections = build_sections(blocks, toc)
    assert [s.heading_path for s in sections] == [["1. 운동"]]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_sections.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.sections'`)

- [ ] **Step 3: 구현**

`rag/sections.py`:
```python
import re
from dataclasses import dataclass

from rag.parse import Block, TocEntry

_REFERENCES = re.compile(r"^(참고문헌|references|bibliography)$", re.IGNORECASE)


@dataclass(frozen=True)
class Section:
    heading_path: list[str]
    blocks: list[Block]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _is_references(title: str) -> bool:
    return bool(_REFERENCES.match(_compact(re.sub(r"^[\d.\s]+", "", title))))


def _find_start(blocks: list[Block], entry: TocEntry, start: int) -> tuple[int, bool]:
    """(섹션 시작 인덱스, 제목 블록을 찾았는지)."""
    title = _compact(entry.title)
    for i in range(start, len(blocks)):
        block = blocks[i]
        if block.page > entry.page:
            break
        if block.page == entry.page and title in _compact(block.text)[: len(title) + 10]:
            return i, True
    for i in range(start, len(blocks)):
        if blocks[i].page >= entry.page:
            return i, False
    return len(blocks), False


def build_sections(blocks: list[Block], toc: list[TocEntry]) -> list[Section]:
    if not toc:
        return [Section([], blocks)] if blocks else []

    starts = []
    cursor = 0
    for entry in toc:
        index, found = _find_start(blocks, entry, cursor)
        starts.append((index, found))
        cursor = index + 1 if found else index

    sections = []
    if starts[0][0] > 0:
        sections.append(Section([], blocks[: starts[0][0]]))

    path: list[str] = []
    for k, entry in enumerate(toc):
        path = path[: entry.level - 1] + [entry.title]
        index, found = starts[k]
        end = starts[k + 1][0] if k + 1 < len(toc) else len(blocks)
        body = blocks[index + 1 if found else index : end]
        if body and not any(_is_references(title) for title in path):
            sections.append(Section(list(path), body))
    return sections
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_sections.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rag/sections.py tests/rag/test_sections.py
git commit -m "feat(rag): split documents into TOC-based sections

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 청킹

**Files:**
- Create: `rag/chunk.py`
- Test: `tests/rag/test_chunk.py`

**Interfaces:**
- Consumes: `rag.sections.Section`, `rag.parse.Block`
- Produces:
  - `rag.chunk.Chunk(heading_path: list[str], page_start: int, page_end: int, body: str)`: frozen dataclass
  - `rag.chunk.chunk_section(section: Section, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]`

규칙:
- 블록 텍스트의 연속 공백은 공백 하나로 바꾼다. 블록이 `max_chars`보다 길면 문장 끝(`.`, `!`, `?` 뒤 공백)으로 나누고, 그래도 긴 조각은 `max_chars` 단위로 자른다. 이렇게 만든 조각이 "단위"다.
- 단위를 `\n`으로 이어 붙이되, 본문 길이가 `max_chars`를 넘기 직전에 청크를 끝낸다.
- 새 청크는 직전 청크 끝의 단위들 가운데 합친 길이가 `overlap` 이하인 만큼만 이어받는다(단위 통째로만 이어받음). 이어받은 단위와 다음 단위를 합쳐 상한을 넘으면 이어받지 않는다.
- `page_start`와 `page_end`는 청크에 들어간 단위들의 쪽 번호 최소·최대다.
- `overlap`이 `0 ≤ overlap < max_chars`가 아니면 `ValueError`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_chunk.py`:
```python
import pytest

from rag.chunk import Chunk, chunk_section
from rag.parse import Block
from rag.sections import Section


def _section(*texts_with_pages, path=("1. 생활요법",)):
    blocks = [Block(page, 100 + i, 110 + i, text) for i, (page, text) in enumerate(texts_with_pages)]
    return Section(list(path), blocks)


def test_small_section_single_chunk():
    chunks = chunk_section(_section((2, "가나다"), (3, "라마바")), max_chars=100, overlap=10)
    assert chunks == [Chunk(["1. 생활요법"], 2, 3, "가나다\n라마바")]


def test_splits_with_overlap_and_respects_limit():
    texts = [(1, "가" * 40), (1, "나" * 40), (2, "다" * 40), (2, "라" * 40)]
    chunks = chunk_section(_section(*texts), max_chars=100, overlap=45)
    assert all(len(c.body) <= 100 for c in chunks)
    assert chunks[0].body == "가" * 40 + "\n" + "나" * 40
    assert chunks[1].body.startswith("나" * 40)  # 직전 마지막 단위를 이어받음
    assert (chunks[1].page_start, chunks[1].page_end) == (1, 2)
    joined = "".join(c.body for c in chunks)
    for text in ("가" * 40, "나" * 40, "다" * 40, "라" * 40):
        assert text in joined


def test_overlong_block_without_sentence_breaks():
    chunks = chunk_section(_section((1, "가" * 3000)), max_chars=1200, overlap=150)
    assert all(len(c.body) <= 1200 for c in chunks)
    assert sum(c.body.count("가") for c in chunks) >= 3000


def test_long_block_split_on_sentences():
    sentence = "나트륨 섭취를 하루 2g 이하로 줄인다. "
    chunks = chunk_section(_section((1, sentence * 60)), max_chars=300, overlap=0)
    assert len(chunks) > 1
    assert all(len(c.body) <= 300 for c in chunks)
    assert all(c.body.split("\n")[0].startswith("나트륨") for c in chunks)


def test_empty_section_no_chunks():
    assert chunk_section(Section([], []), max_chars=100, overlap=10) == []


@pytest.mark.parametrize("overlap", [-1, 100, 150])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        chunk_section(_section((1, "가")), max_chars=100, overlap=overlap)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_chunk.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.chunk'`)

- [ ] **Step 3: 구현**

`rag/chunk.py`:
```python
import re
from dataclasses import dataclass

from rag.sections import Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

Unit = tuple[str, int]  # (텍스트, 쪽 번호)


@dataclass(frozen=True)
class Chunk:
    heading_path: list[str]
    page_start: int
    page_end: int
    body: str


def _units(section: Section, max_chars: int) -> list[Unit]:
    units = []
    for block in section.blocks:
        text = re.sub(r"\s+", " ", block.text).strip()
        pieces = [text] if len(text) <= max_chars else _SENTENCE_END.split(text)
        for piece in pieces:
            for i in range(0, len(piece), max_chars):
                units.append((piece[i : i + max_chars], block.page))
    return units


def _size(units: list[Unit]) -> int:
    return sum(len(text) for text, _ in units) + max(len(units) - 1, 0)


def chunk_section(section: Section, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    chunks: list[Chunk] = []
    current: list[Unit] = []

    def emit() -> None:
        pages = [page for _, page in current]
        body = "\n".join(text for text, _ in current)
        chunks.append(Chunk(list(section.heading_path), min(pages), max(pages), body))

    for unit in _units(section, max_chars):
        if current and _size(current + [unit]) > max_chars:
            emit()
            carried: list[Unit] = []
            for previous in reversed(current):
                if _size([previous] + carried) > overlap:
                    break
                carried.insert(0, previous)
            current = carried if _size(carried + [unit]) <= max_chars else []
        current.append(unit)

    if current:
        emit()
    return chunks
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_chunk.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add rag/chunk.py tests/rag/test_chunk.py
git commit -m "feat(rag): chunk sections with size limit and overlap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: 출력 레코드 (태깅, content, hash, 중복 제거)

**Files:**
- Create: `rag/records.py`
- Test: `tests/rag/test_records.py`

**Interfaces:**
- Consumes: `rag.chunk.Chunk`, `rag.sources.SourceDoc`
- Produces:
  - `rag.records.build_records(source: SourceDoc, chunks: list[Chunk]) -> list[dict]`. 레코드 키: `doc_id, title, publisher, year, url, heading_path (str | None), page_start, page_end, conditions, category, content, content_hash`. 하위 프로젝트 2·3이 이 키에 의존함
  - `rag.records.dedupe(records: list[dict]) -> list[dict]`: `content_hash` 기준, 먼저 나온 것을 남김

규칙:
- `category`: `section_rules`를 순서대로 보며 `match`가 `" > ".join(heading_path)`에 포함되는 첫 규칙의 category. 없으면 문서 category
- `content`: `" > ".join([title, *heading_path]) + "\n" + body`를 NFC 정규화
- `heading_path`: `" > ".join(heading_path)`, 비어 있으면 `None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_records.py`:
```python
import hashlib

from rag.chunk import Chunk
from rag.records import build_records, dedupe
from rag.sources import SourceDoc

SOURCE = SourceDoc(
    id="kshguide-2022",
    file="a.pdf",
    title="고혈압 진료지침",
    publisher="대한고혈압학회",
    year=2022,
    url="https://example.org/guide.pdf",
    conditions=["hypertension"],
    category="lifestyle",
    section_rules=[{"match": "운동", "category": "exercise"}, {"match": "식이", "category": "diet"}],
)


def test_record_fields_and_content_prefix():
    [record] = build_records(SOURCE, [Chunk(["3. 생활요법", "3.2 식이요법"], 45, 46, "나트륨을 줄인다.")])
    assert record == {
        "doc_id": "kshguide-2022",
        "title": "고혈압 진료지침",
        "publisher": "대한고혈압학회",
        "year": 2022,
        "url": "https://example.org/guide.pdf",
        "heading_path": "3. 생활요법 > 3.2 식이요법",
        "page_start": 45,
        "page_end": 46,
        "conditions": ["hypertension"],
        "category": "diet",
        "content": "고혈압 진료지침 > 3. 생활요법 > 3.2 식이요법\n나트륨을 줄인다.",
        "content_hash": hashlib.sha256(
            "고혈압 진료지침 > 3. 생활요법 > 3.2 식이요법\n나트륨을 줄인다.".encode()
        ).hexdigest(),
    }


def test_first_matching_rule_wins_and_default_category():
    records = build_records(
        SOURCE,
        [Chunk(["운동과 식이"], 1, 1, "a"), Chunk([], 2, 2, "b")],
    )
    assert [r["category"] for r in records] == ["exercise", "lifestyle"]
    assert records[1]["heading_path"] is None
    assert records[1]["content"] == "고혈압 진료지침\nb"


def test_content_is_nfc_normalized():
    nfd_body = "가"  # NFD '가'
    [record] = build_records(SOURCE, [Chunk([], 1, 1, nfd_body)])
    assert record["content"].endswith("\n가")


def test_dedupe_keeps_first():
    chunks = [Chunk([], 1, 1, "같은 본문"), Chunk([], 5, 5, "같은 본문"), Chunk([], 6, 6, "다른 본문")]
    records = dedupe(build_records(SOURCE, chunks))
    assert [r["page_start"] for r in records] == [1, 6]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_records.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.records'`)

- [ ] **Step 3: 구현**

`rag/records.py`:
```python
import hashlib
import unicodedata

from rag.chunk import Chunk
from rag.sources import SourceDoc


def _category(source: SourceDoc, heading_path: list[str]) -> str:
    joined = " > ".join(heading_path)
    for rule in source.section_rules:
        if rule.match in joined:
            return rule.category
    return source.category


def build_records(source: SourceDoc, chunks: list[Chunk]) -> list[dict]:
    records = []
    for chunk in chunks:
        prefix = " > ".join([source.title, *chunk.heading_path])
        content = unicodedata.normalize("NFC", f"{prefix}\n{chunk.body}")
        records.append({
            "doc_id": source.id,
            "title": source.title,
            "publisher": source.publisher,
            "year": source.year,
            "url": source.url,
            "heading_path": " > ".join(chunk.heading_path) or None,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "conditions": source.conditions,
            "category": _category(source, chunk.heading_path),
            "content": content,
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        })
    return records


def dedupe(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for record in records:
        if record["content_hash"] not in seen:
            seen.add(record["content_hash"])
            unique.append(record)
    return unique
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_records.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add rag/records.py tests/rag/test_records.py
git commit -m "feat(rag): build tagged chunk records with content hash

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: ingest CLI와 README

**Files:**
- Create: `rag/ingest.py`, `rag/README.md`
- Test: `tests/rag/test_ingest.py`

**Interfaces:**
- Consumes: `load_sources`, `SourcesError`, `parse_pdf`, `clean_blocks`, `build_sections`, `chunk_section`, `build_records`, `dedupe` (Task 1~6)
- Produces:
  - `rag.ingest.ingest(sources_path: Path, raw_dir: Path, max_chars: int = 1200, overlap: int = 150) -> tuple[list[dict], dict]`. 통계 키: `documents: int`, `chunks: int`, `no_toc: list[str]`, `skipped_no_text: list[str]`, `failed: list[str]`
  - `rag.ingest.main(argv: list[str] | None = None) -> int`. 인자 `--sources`(기본 `rag/sources.json`), `--raw-dir`(기본 `rag/raw`), `--out`(기본 `rag/data/chunks.jsonl`), `--max-chars`(1200), `--overlap`(150). 성공 0, `SourcesError`면 1
  - 출력 파일: 레코드 하나당 한 줄의 JSON(`ensure_ascii=False`). 하위 프로젝트 2의 입력

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/rag/test_ingest.py`:
```python
import json

from rag.ingest import ingest, main


def _sources(tmp_path, entries):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return path


def _entry(doc_id, file, **overrides):
    entry = {
        "id": doc_id,
        "file": file,
        "title": f"{doc_id} 제목",
        "conditions": ["hypertension"],
        "category": "lifestyle",
    }
    entry.update(overrides)
    return entry


def _guide_pdf(make_pdf):
    return make_pdf(
        "guide.pdf",
        [
            [(40, "고혈압 진료지침"), (100, "1. 식이요법"), (200, "나트륨 섭취를 줄인다."), (820, "1")],
            [(40, "고혈압 진료지침"), (100, "2. 운동요법"), (200, "주 5회 유산소 운동을 한다."), (820, "2")],
            [(40, "고혈압 진료지침"), (100, "참고문헌"), (200, "Kim 2020"), (820, "3")],
        ],
        toc=[[1, "1. 식이요법", 1], [1, "2. 운동요법", 2], [1, "참고문헌", 3]],
    )


def test_ingest_end_to_end(tmp_path, make_pdf):
    _guide_pdf(make_pdf)
    make_pdf("notoc.pdf", [[(100, "목차 없는 문서 본문입니다.")], [(100, "둘째 쪽 본문입니다.")]])
    sources = _sources(tmp_path, [
        _entry("guide", "guide.pdf", section_rules=[{"match": "운동", "category": "exercise"}]),
        _entry("notoc", "notoc.pdf"),
    ])

    records, stats = ingest(sources, tmp_path)

    assert [(r["doc_id"], r["heading_path"], r["category"], r["page_start"]) for r in records] == [
        ("guide", "1. 식이요법", "lifestyle", 1),
        ("guide", "2. 운동요법", "exercise", 2),
        ("notoc", None, "lifestyle", 1),
    ]
    assert records[0]["content"] == "guide 제목 > 1. 식이요법\n나트륨 섭취를 줄인다."
    assert records[2]["page_end"] == 2
    assert all("고혈압 진료지침" not in r["content"].split("\n", 1)[1] for r in records)
    assert all("Kim 2020" not in r["content"] for r in records)
    assert stats == {"documents": 2, "chunks": 3, "no_toc": ["notoc"], "skipped_no_text": [], "failed": []}


def test_ingest_skips_scanned_pdf(tmp_path, make_pdf):
    make_pdf("scan.pdf", [[], []])
    records, stats = ingest(_sources(tmp_path, [_entry("scan", "scan.pdf")]), tmp_path)
    assert records == []
    assert stats["skipped_no_text"] == ["scan"]
    assert stats["documents"] == 0


def test_ingest_reports_corrupt_pdf_and_continues(tmp_path, make_pdf):
    _guide_pdf(make_pdf)
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    sources = _sources(tmp_path, [_entry("broken", "broken.pdf"), _entry("guide", "guide.pdf")])
    records, stats = ingest(sources, tmp_path)
    assert stats["failed"] == ["broken"]
    assert {r["doc_id"] for r in records} == {"guide"}


def test_main_writes_jsonl_and_is_deterministic(tmp_path, make_pdf):
    _guide_pdf(make_pdf)
    sources = _sources(tmp_path, [_entry("guide", "guide.pdf")])
    out = tmp_path / "out" / "chunks.jsonl"
    argv = ["--sources", str(sources), "--raw-dir", str(tmp_path), "--out", str(out)]

    assert main(argv) == 0
    first = out.read_bytes()
    lines = first.decode("utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["doc_id"] == "guide"
    assert "나트륨" in lines[0]  # ensure_ascii=False

    assert main(argv) == 0
    assert out.read_bytes() == first


def test_main_returns_1_on_bad_sources(tmp_path):
    sources = _sources(tmp_path, [_entry("guide", "missing.pdf")])
    out = tmp_path / "chunks.jsonl"
    assert main(["--sources", str(sources), "--raw-dir", str(tmp_path), "--out", str(out)]) == 1
    assert not out.exists()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/rag/test_ingest.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'rag.ingest'`)

- [ ] **Step 3: 구현**

`rag/ingest.py`:
```python
"""sources.json의 PDF를 청크 JSONL로 변환한다. 저장소 루트에서 `python -m rag.ingest`."""

import argparse
import json
import logging
import sys
from pathlib import Path

from rag.chunk import chunk_section
from rag.clean import clean_blocks
from rag.parse import parse_pdf
from rag.records import build_records, dedupe
from rag.sections import build_sections
from rag.sources import SourcesError, load_sources

log = logging.getLogger("rag.ingest")


def ingest(
    sources_path: Path,
    raw_dir: Path,
    max_chars: int = 1200,
    overlap: int = 150,
) -> tuple[list[dict], dict]:
    stats = {"documents": 0, "chunks": 0, "no_toc": [], "skipped_no_text": [], "failed": []}
    records: list[dict] = []
    for source, path in load_sources(sources_path, raw_dir):
        try:
            parsed = parse_pdf(path)
        except Exception:
            log.exception("[%s] failed to parse %s", source.id, path)
            stats["failed"].append(source.id)
            continue
        if not parsed.blocks:
            log.warning("[%s] no text layer (scanned PDF?); skipped", source.id)
            stats["skipped_no_text"].append(source.id)
            continue
        if not parsed.toc:
            stats["no_toc"].append(source.id)

        blocks = clean_blocks(parsed.blocks, parsed.page_heights)
        chunks = [
            chunk
            for section in build_sections(blocks, parsed.toc)
            for chunk in chunk_section(section, max_chars, overlap)
        ]
        records.extend(build_records(source, chunks))
        stats["documents"] += 1

    records = dedupe(records)
    stats["chunks"] = len(records)
    return records, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 코퍼스 수집: PDF → 청크 JSONL")
    parser.add_argument("--sources", type=Path, default=Path("rag/sources.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("rag/raw"))
    parser.add_argument("--out", type=Path, default=Path("rag/data/chunks.jsonl"))
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        records, stats = ingest(args.sources, args.raw_dir, args.max_chars, args.overlap)
    except SourcesError as e:
        log.error("%s", e)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("wrote %s", args.out)
    log.info("stats: %s", json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/rag/test_ingest.py -v`
Expected: 5 passed

- [ ] **Step 5: README 작성**

`rag/README.md`:
````markdown
# RAG 코퍼스 수집

리포트 근거로 쓸 의학·영양 가이드라인 PDF를 청크 JSONL로 변환합니다.
설계: `docs/superpowers/specs/2026-10-08-rag-report-design.md`

| 파일 | 역할 |
| --- | --- |
| `sources.json` | 수집할 문서 목록과 인용 메타데이터 (직접 관리) |
| `ingest.py` | 파싱 → 정제 → 섹션 분할 → 청킹 → 태깅 → `data/chunks.jsonl` |
| `parse.py` `clean.py` `sections.py` `chunk.py` `records.py` | 단계별 모듈 |

임베딩과 DB 적재는 모델 선정(평가 실험) 후에 추가합니다.

## 문서 준비

PDF를 `rag/raw/`에 넣고 `sources.json`에 등록합니다. 원본 PDF와 산출물(`rag/data/`)은 재배포 문제 때문에 Git에 올리지 않습니다.

```json
[
  {
    "id": "kshguide-2022",
    "file": "고혈압진료지침_2022.pdf",
    "title": "2022 고혈압 진료지침",
    "publisher": "대한고혈압학회",
    "year": 2022,
    "url": "https://...",
    "license_note": "학회 홈페이지 공개본. 재배포 불가",
    "conditions": ["hypertension"],
    "category": "lifestyle",
    "section_rules": [
      {"match": "운동", "category": "exercise"},
      {"match": "식이", "category": "diet"}
    ]
  }
]
```

- `conditions`: `hypertension`, `diabetes`, `dyslipidemia`, `obesity`, `metabolic_syndrome`, `fatty_liver`, `sarcopenia`, `general`
- `category`: `lifestyle`, `diet`, `exercise`, `supplement`, `criteria`
- `section_rules`: 제목 경로에 `match`가 포함된 섹션은 그 category로 태깅합니다(먼저 맞는 규칙 우선).

## 실행

저장소 루트에서 실행합니다.

```bash
pip install -r requirements.txt -r rag/requirements.txt
python -m rag.ingest
python -m rag.ingest --max-chars 800 --overlap 100 --out rag/data/chunks_800.jsonl
```

로그의 `stats`를 확인하세요.
- `no_toc`: 책갈피가 없어 제목 경로 없이 청킹된 문서. 인용에 장·절이 표시되지 않습니다.
- `skipped_no_text`: 텍스트 레이어가 없는 스캔본. OCR이 필요합니다.
- `failed`: 열 수 없는 PDF.
````

- [ ] **Step 6: 전체 테스트 실행**

Run: `python -m pytest tests/rag -v`
Expected: 33 passed

- [ ] **Step 7: Commit**

```bash
git add rag/ingest.py rag/README.md tests/rag/test_ingest.py
git commit -m "feat(rag): add ingest CLI producing chunk JSONL

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
