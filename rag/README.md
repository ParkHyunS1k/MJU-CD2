# RAG 코퍼스 수집과 검색 평가

리포트 근거로 쓸 의학·영양 가이드라인 PDF를 청크 JSONL로 변환하고(`rag/`), 검색 구성을 평가해 고릅니다(`rag/eval/`).
설계: `docs/superpowers/specs/2026-10-08-rag-report-design.md`

| 파일 | 역할 |
| --- | --- |
| `sources.json` | 수집할 문서 목록과 인용 메타데이터 (직접 관리) |
| `ingest.py` | 파싱 → 정제 → 섹션 분할 → 청킹 → 태깅 → `data/chunks.jsonl` |
| `parse.py` `clean.py` `sections.py` `paragraphs.py` `chunk.py` `records.py` | 단계별 모듈 |
| `corpus-candidates.md` | 넣지 않은 후보 문서와 공개 API 조사 (코퍼스 확장 시 참고) |

임베딩과 DB 적재는 `load.py`가 합니다(아래 DB 적재).

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

### 책갈피가 없는 PDF: `sections`로 넣을 장을 직접 지정

학회 지침 PDF는 대부분 책갈피(TOC)가 없습니다. 이때는 넣을 장의 제목과 **PDF 쪽 번호**(뷰어에 표시되는 번호, 인쇄된 번호 아님)를 적습니다.

```json
"sections": [
  {"title": "2형당뇨병의 예방", "pages": [17, 24], "category": "lifestyle"},
  {"title": "운동요법", "pages": [59, 64], "category": "exercise"}
]
```

- `sections`가 있으면 **적은 쪽 범위만** 넣고 나머지 쪽은 버립니다. 책갈피는 무시합니다.
- `title`이 청크의 장·절 경로가 되고, `category`가 있으면 그 장의 태그가 됩니다(없으면 `section_rules` → 문서 `category` 순).
- 쪽 범위가 겹치거나 제목이 중복되면 오류입니다. 범위에 글자가 없으면 경고만 남깁니다.

## 실행

저장소 루트에서 실행합니다.

```bash
pip install -r requirements.txt -r rag/requirements.txt
python -m rag.ingest --chunker paragraph --overlap 0
```

선정된 구성(`rag/eval/selected.json`)은 PyMuPDF, 문단 묶기 1,200자, 겹침 0입니다. 인자 기본값(`--chunker chars --overlap 150`)은 평가 초기의 기준 구성이라 다릅니다.

- `--parser {pymupdf,pdfplumber}`, `--spacing {off,on,auto}`(kiwi 띄어쓰기 복원, auto는 공백이 거의 없는 문서만)
- `--chunker {chars,paragraph,paragraph-single}`: 글자 수 / 문단 묶기 / 문단 하나씩. 문단 청커는 표·그림 캡션, 숫자 위주의 표, 문장 없는 조각을 버립니다(`rag.paragraphs.is_table_or_caption`).
- 문단 형식이 아닌 문서(표·그래프 위주)는 `sources.json`에 넣지 않습니다. 뺀 문서와 이유는 `corpus-candidates.md`에 있습니다.

로그의 `stats`를 확인하세요.
- `no_toc`: 책갈피가 없어 제목 경로 없이 청킹된 문서. 인용에 장·절이 표시되지 않습니다.
- `skipped_no_text`: 텍스트 레이어가 없는 스캔본. OCR이 필요합니다.
- `failed`: 열 수 없는 PDF.

## DB 적재 (`rag/load.py`)

```bash
python -m rag.load
```

- `rag/eval/selected.json`의 구성으로 청크를 만들고 Supabase `rag_documents`, `rag_chunks`와 맞춥니다. 새 `content_hash`만 Gemini 768차원으로 임베딩하고, 사라진 청크·문서는 지웁니다.
- 바뀐 것이 없으면 임베딩 API를 호출하지 않습니다. Gemini 임베딩 무료 한도는 하루 1,000건입니다.
- 테이블과 검색 RPC는 `supabase/migrations/20261015000000_rag_tables.sql`입니다.

## 검색 평가 (`rag/eval/`)

설계는 `docs/superpowers/specs/2026-10-08-rag-report-design.md` 10.1절, 결과는 `rag/eval/results/REPORT.md`에 있습니다.

```bash
pip install -r rag/eval/requirements.txt
python -m rag.eval.run 1 2 3 4 5 6
python -m rag.eval.plot rag/eval/results
```

| 경로 | 내용 |
| --- | --- |
| `data/queries.jsonl`, `data/labels.jsonl` | 검색 쿼리 54개와 근거 문장 라벨(커밋됨) |
| `results/` | 단계별 결과 표·그래프, `REPORT.md`(종합 보고서) |
| `runs/{paragraph,single}/selected.json` | 청킹별로 3~6단계를 따로 돌린 상태 파일 |
| `selected.json` | 최종 구성(하위 프로젝트 3 입력) |
| `cache/` | 임베딩 캐시(텍스트별, gitignore) |

- 같은 단계를 다른 기준으로 돌릴 때: `--state rag/eval/runs/single/selected.json --results rag/eval/results/single`
- 3단계에서 일부 모델만: `--embeddings gemini-768,bge-m3`
- 라벨: `build_pool.py`(후보 모으기와 1차 판정) → `review/index.html`(사람 검수) → `export_labels.py`. 청킹을 바꾸면 `augment.py collect|judge|apply`로 보강합니다. 판정은 GPT(Codex CLI, `build_pool.py --judge codex`)로 합니다.
- API 키(`GEMINI_API_KEY`, `OPENAI_API_KEY`)는 루트 `.env`에 둡니다. Gemini 임베딩 무료 등급은 하루 1,000건, 분당 한도가 있어 429가 나면 재시도합니다.
