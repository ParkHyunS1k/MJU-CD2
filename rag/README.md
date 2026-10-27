# RAG: 리포트 근거 검색

AI 리포트가 진료지침·공공 건강정보를 근거로 쓰도록, 문서를 청크로 나눠 벡터 DB에 넣고 검색합니다.
수집·평가는 `rag/`, 앱의 검색·인용은 `app/services/`(retrieval, embedding, citation)에 있습니다.

## 한눈에 보기

| 항목 | 선택 | 근거 (비교 결과) |
|---|---|---|
| 문서 | 27개, 청크 498개 | 지침·논문 16개와 포털 33개를 검수해 문단형이 아니거나 중복·충돌하는 문서를 뺌 |
| PDF 파서 | PyMuPDF | pdfplumber, 띄어쓰기 복원과 차이 없음 |
| 청킹 | 문단 묶기 1,200자, 겹침 0 (잠정) | 문단 경계를 지킴. 표·캡션·참고문헌 제거 |
| 임베딩 | gemini-embedding-001, 768차원 | 8개 모델 중 1위(nDCG 0.810). BGE-M3(0.791)와 차이는 작지만 서버에 모델을 올리지 않아도 됨 |
| 벡터 DB | Supabase Postgres + pgvector | 앱이 이미 쓰는 Supabase라 별도 벡터 서비스·비용이 없고, 청크 수백 개 규모에서는 전용 DB가 필요 없음. `halfvec(768)`(저장 공간 절반), HNSW 코사인, RPC `match_rag_chunks`(서버 키로만 호출) |
| 하이브리드·리랭커 | 사용 안 함 | BM25를 더하면 nDCG 0.877 → 0.757, 리랭커는 0.788에 쿼리당 약 3초 추가 |
| 검색 | 키워드형 쿼리, 쿼리당 상위 3개, 코사인 0.7028 이상 | 정답 없는 쿼리 3개를 모두 거름 |

**성능 (시험용 쿼리, 앱과 같은 키워드형):** Recall@3 1.000, nDCG@10 0.903. 평가 데이터는 쿼리 54개, 근거 문장 라벨 1,226개(사람 검수 163개, 나머지는 GPT 판정)입니다. 자세한 내용은 [`eval/results/REPORT.md`](eval/results/REPORT.md)에 있습니다.

## DB에 들어간 문서

**진료지침·논문 9개 (청크 228개)**

| 질환 | 문서 |
|---|---|
| 고혈압 | 대한고혈압학회 고혈압 진료지침 2026 |
| 당뇨병 | 대한당뇨병학회 당뇨병 진료지침 2025 |
| 이상지질혈증 | 한국지질·동맥경화학회 이상지질혈증 진료지침 2026 |
| 대사증후군 | 대한심장대사증후군학회 대사증후군 진료지침 2021 요약본 |
| 지방간 | 대한간학회 비알코올 지방간질환 가이드라인 2021, 대사이상지방간질환 가이드라인 2025 |
| 비만 | 대한비만학회 행동치료 합의문 2026(영문), 비만 팩트시트 2025(영문) |
| 운동 | 보건복지부 한국인을 위한 신체활동 지침서 2023 |

**국가건강정보포털(질병관리청) 18개 (청크 270개)**

- 질환: 고혈압, 당뇨병, 이상지질혈증, 대사증후군, 대사이상지방간질환, 비만, 근 손실
- 식사: 고혈압 환자의 식이요법, 당뇨환자의 식이요법, 건강한 체중조절을 위한 식사, 식이영양, 탄수화물
- 운동·생활: 운동, 당뇨환자의 운동요법, 음주, 흡연
- 영양제: 영양제, 건강기능식품 (리포트의 영양제 후보는 이 두 문서만 근거로 쓸 수 있습니다)

**뺀 것:** 표 위주 문서, 중복 문서, 주제가 맞지 않는 문서와 영문 논문 3편(한국어 쿼리로는 검색되지 않음). 포털 문서에서 진료지침과 수치가 다른 문단도 뺐습니다(예: 소금 6 g, 혈압 목표 140/85). 남은 차이는 프롬프트 규칙으로 처리합니다(진료지침 우선). 목록과 이유는 [`corpus-candidates.md`](corpus-candidates.md)에 있습니다.

## 앱에서 쓰는 흐름

1. 위험도가 medium 이상인 질환과 인바디 기준을 넘는 항목으로 키워드 쿼리를 만듭니다(`rag_taxonomy.build_queries`).
2. 쿼리를 한 번에 임베딩하고, 쿼리마다 RPC를 병렬로 호출해 상위 3개를 받습니다. 10초를 넘기면 503을 돌려줍니다.
3. 쿼리별 1위부터 돌아가며 최대 15개를 골라 `[S1]…` 근거로 프롬프트에 넣습니다.
4. 리포트가 인용한 S번호를 검증하고, 출처(문서·장·쪽)를 붙입니다.

## 실행

저장소 루트에서 실행합니다. 키는 루트 `.env`에 둡니다(`GEMINI_API_KEY`, `KDCA_API_TOKEN`, Supabase 키).

```bash
pip install -r requirements.txt -r rag/requirements.txt
python -m rag.kdca                                     # 포털 문서 받기 (이미 받은 것은 건너뜀)
python -m rag.ingest --chunker paragraph --overlap 0   # 청크 확인 → data/chunks.jsonl
python -m rag.load                                     # 임베딩 후 DB와 맞추기 (바뀐 청크만)
python -m rag.eval.run 4 5 6 --results rag/eval/results/final   # 검색 재평가
```

- PDF 원본(`raw/`)과 산출물(`data/`)은 재배포 문제로 Git에 올리지 않습니다.
- Gemini 임베딩 무료 등급은 하루 1,000건이고 분당 한도가 있습니다. 429가 나면 재시도하고, 적재가 중간에 멈추면 다시 실행하면 이어서 진행됩니다.

## 문서 추가하기 (`sources.json`)

```json
{"id": "example-2026", "file": "example_2026.pdf", "title": "...", "publisher": "...", "year": 2026, "url": "...",
 "conditions": ["diabetes"], "category": "lifestyle",
 "section_rules": [{"match": "운동", "category": "exercise"}],
 "sections": [{"title": "2형당뇨병의 예방", "pages": [17, 24]}]}
```

- `conditions`: hypertension, diabetes, dyslipidemia, obesity, metabolic_syndrome, fatty_liver, sarcopenia, general
- `category`: lifestyle, diet, exercise, supplement, criteria. `section_rules`는 장 제목에 맞춰 덮어씁니다.
- `sections`: 책갈피가 없는 PDF에서 넣을 장을 PDF 쪽 번호로 지정합니다.
- 포털 문서는 `"kind": "kdca", "cntnts_sn": 번호`로 등록합니다. `exclude_sections`, `exclude_paragraphs`로 충돌 부분을 빼고, 원문이 바뀌어 문구가 사라지면 수집이 멈춥니다.
- 넣은 뒤에는 라벨을 보강하고(`eval/augment.py`, GPT 판정) 4~6단계를 다시 돌린 다음 적재합니다.

## 파일

| 경로 | 내용 |
|---|---|
| `sources.json` | 문서 목록과 인용 메타데이터 |
| `ingest.py` ← `parse` `clean` `sections` `paragraphs` `chunk` `records` | PDF·포털 JSON → 청크 |
| `kdca.py` / `load.py` | 포털 수집 / 임베딩·DB 동기화 |
| `eval/` | 검색 평가: 쿼리·라벨(`data/`), 단계별 결과(`results/`), 최종 설정(`selected.json`) |
| `corpus-candidates.md` | 후보 문서 조사와 넣고 뺀 이유 |

설계: [`docs/superpowers/specs/2026-10-08-rag-report-design.md`](../docs/superpowers/specs/2026-10-08-rag-report-design.md), [`2026-10-15-rag-runtime-design.md`](../docs/superpowers/specs/2026-10-15-rag-runtime-design.md)
