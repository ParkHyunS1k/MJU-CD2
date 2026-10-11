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
python -m rag.ingest
python -m rag.ingest --max-chars 800 --overlap 100 --out rag/data/chunks_800.jsonl
```

로그의 `stats`를 확인하세요.
- `no_toc`: 책갈피가 없어 제목 경로 없이 청킹된 문서. 인용에 장·절이 표시되지 않습니다.
- `skipped_no_text`: 텍스트 레이어가 없는 스캔본. OCR이 필요합니다.
- `failed`: 열 수 없는 PDF.
