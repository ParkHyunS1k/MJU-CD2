# RAG 코퍼스 후보 조사

코퍼스를 넓힐 때 참고하는 조사 기록이다. 실제로 넣은 문서는 `sources.json`이 기준이다(21개: 지침·논문 9개, 국가건강정보포털 12개). 영양제·식품·통계 같은 API 데이터는 아직 넣지 않았다.

## 검토 후 뺀 문서 (2026-10-15)

처음 넣은 16개를 문단 청킹 적합도(문단 비율)와 문서 간 충돌로 검수했다. 아래 7개는 `sources.json`에서 뺐다. 쪽 범위 등 원래 설정은 git 기록(커밋 b345054의 `rag/sources.json`)에 있다.

| 문서 | 결정 | 이유 |
|---|---|---|
| 2024 비만병 팩트시트 (대한비만학회, 국문) | 제외 | 그래프·인포그래픽 문서. 문단 17% |
| 2025 한국인 영양소 섭취기준 요약표 | 제외 | 표 문서. 문단 1%. 주요 기준은 당뇨·지질 지침이 인용 |
| Evidence-Based Exercise Guidelines for Sarcopenia (KWGS 2025) | 제외 | 문단 42%, 연구 요약표 위주. 노인 대상 |
| 2024 Obesity Fact Sheet in Korea (JOMES) | 제외 | 2025판과 거의 같은 청크 127쌍, 연도별 통계 충돌 |
| 2024 KSSO 비만 진단·약물치료 지침 (JOMES) | 제외 | 주제가 약물치료. 앱은 약을 권하지 않음 |
| KWGS 근감소증 선별·진단 합의문 2023 | 보류 | 문단 53%(Q&A·흐름도). 노인 대상. 넣는다면 p.8 제외 |
| Exercise Guidelines for Severe Obesity (JOMES 2024) | 보류 | 형식은 좋으나 중증(BMI ≥35) 대상, 문서 안에서 BMI 기준이 엇갈림 |

### 국가건강정보포털 (2026-10-27)

질병관리청 국가건강정보포털 건강정보 25개를 API로 받아(`python -m rag.kdca`) 검수했다. 12개를 넣고 13개는 뺐다.

- 넣음: 고혈압, 당뇨병, 이상지질혈증, 비만, 대사증후군, 대사이상지방간질환, 근 손실, 운동, 영양제, 고혈압 환자의 식이요법, 당뇨환자의 식이요법, 당뇨환자의 운동요법
- 뺌: 고혈압 환자의 운동요법(문단 34%, 표 위주), 이상지질혈증 식사요법(2026 지질 지침 식사요법 장과 유사 청크 9쌍), 당뇨환자의 약물요법(앱은 약을 권하지 않음), 합병증 10개(고혈압심장질환, 고혈압성 콩팥병, 고혈압과 눈, 당뇨망막병증, 당뇨병성 족부병증, 소변이상, 당뇨병 합병증 4종. 앱 대상과 거리가 있고 당뇨병 본문과 겹침)

남긴 문서에서도 지방간 2021의 '정의' 장(MASLD 2025와 용어 충돌), MASLD 2025의 p.22 비교표, 신체활동 지침서의 목차·부록(p.2–3, p.66 이후)은 뺐다. 표·그림 캡션과 문장이 없는 조각은 청킹 단계에서 거른다(`rag.paragraphs.is_table_or_caption`).

## A. 문서 (PDF·웹)

- 확인일: 2026-10-09
- 방법: 공식 사이트 직접 접속, 파일은 HTTP 헤더(크기·파일명)로 확인. 작은 PDF 4개(KASL 2021·2025, 당뇨 2025 전문, KDRI 요약표)만 내려받아 쪽수와 텍스트 레이어 확인. 로그인·회원가입·약관 동의는 하지 않음.
- 표기: **[확인]** 직접 접속 / **[검색결과]** 검색 결과에서 URL만 봄 / **[미확인]** 추정 또는 확인 실패

### 1. 후보 문서

| 우선순위 | 문서 | 발행처 | 최신판 | 접근 | 형식·분량 | 이용조건 | condition | 리포트 섹션 | URL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 고혈압 진료지침 제6판(수정본) | 대한고혈압학회 | 2026 (2026-05-22) | 무료, 구글드라이브 [확인] | PDF 22.2MB, 쪽수·텍스트 여부 미확인 | "All rights reserved"만 표기 | hypertension | 위험 설명, 생활계획 | https://www.koreanhypertension.org/reference/guide?mode=read&idno=10446 |
| 2 | 제6판 하이라이트 논문(영문) | 대한고혈압학회 | 2026 | 무료 [확인] | PDF | 위와 같음 | hypertension | 위험 설명 | https://www.koreanhypertension.org/reference/guide?mode=read&idno=10447 |
| 1 | 당뇨병 진료지침 전문(제9판) | 대한당뇨병학회 | 2025 (2025-07-09) | 무료 [확인] | PDF 5.4MB, 185 PDF쪽(펼침면, 인쇄 약 300쪽), 텍스트 PDF, 2단 편집 [확인] | "All rights reserved", ISBN 979-11-980274-5-0 | diabetes (+hypertension, dyslipidemia, obesity, fatty_liver) | 위험 설명, 생활계획 | https://diabetes.or.kr/bbs/download.php?code=guide&number=1522 |
| 2 | 당뇨병 진료지침 요약본(26.2.12 수정본) | 대한당뇨병학회 | 2025 | 무료 [확인] | PDF 14.5MB, 63쪽 | 위와 같음 | diabetes | 생활계획 | https://diabetes.or.kr/bbs/download.php?code=guide&number=1596 |
| 1 | 이상지질혈증 진료지침 전체본 | 한국지질·동맥경화학회 | 2026 (2026-09-10) | 무료 [확인] | PDF 12.0MB, 252쪽, 텍스트 여부 미확인 | 상업적 사용은 사무국 문의 명시 [확인] | dyslipidemia | 위험 설명, 생활계획 | https://www.lipid.or.kr/reference/guideline.php?boardid=guideline&mode=view&idx=1456 |
| 1 | 비만병 팩트시트 | 대한비만학회 | 2024 | 무료 [확인] | PDF 4.3MB, 56쪽 | 미기재 | obesity, metabolic_syndrome | 위험 설명, 체성분 | https://general.kosso.or.kr/html/user/core/view/reaction/main/kosso/inc/data/2024_Obesity_Fact_sheet_web_kor1223.pdf |
| 1 | 비만진료지침 8판 요약본 | 대한비만학회 | 2022 (9판 존재 [미확인]) | 무료 [확인: 게시물] | PDF, 분량 미확인 | 미기재 | obesity | 위험 설명, 체성분, 생활계획 | https://general.kosso.or.kr/html/?pmode=BBBS0001300003 |
| 2 | KSSO 2022 Update (JOMES 2023;32(1):1-24, 영문) | 대한비만학회 | 2023 | 오픈액세스 [검색결과] | 24쪽 | 미확인 | obesity, metabolic_syndrome | 생활계획 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10088549 |
| 1 | 대사증후군 진료지침 요약본 | 심장대사증후군학회 | 2021 | 무료 [확인] | PDF 1.8MB | 미기재 | metabolic_syndrome | 위험 설명, 생활계획 | https://www.kscms.org/Data |
| 2 | 대사증후군 팩트시트 | 심장대사증후군학회 | 2026 | 무료 [확인: 목록] | PDF | 미기재 | metabolic_syndrome | 위험 설명 | https://www.kscms.org/Fact-sheet |
| 1 | 비알코올 지방간질환 진료 가이드라인 | 대한간학회 | 2021 | 무료 [확인] | PDF 3.2MB, 152쪽, 텍스트 PDF [확인] | 미기재 | fatty_liver | 생활계획, 위험 설명 | https://www.kasl.org/bbs/?number=4630&mode=view&code=guide |
| 1 | 대사이상지방간질환(MASLD) 진료 가이드라인 | 대한간학회 | 2025 | 무료 [확인] | PDF 1.4MB, 82쪽, 텍스트 PDF, **생활습관 장 없음**(2021판 참조) [확인] | 허락 없는 수정·변형·전재 금지 [확인] | fatty_liver | 위험 설명(정의·진단) | https://www.kasl.org/bbs/?number=6193&mode=view&code=guide |
| 1 | KWGS 근감소증 선별·진단 합의문 (AGMR 2023, 영문) | 대한근감소증학회 외 | 2023 | 오픈액세스 [검색결과] | 약 13쪽 | 미확인 | sarcopenia | 체성분, 위험 설명 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10073972 |
| 1 | KWGS 근감소증 운동 지침 (AGMR 2025, 영문) | KWGS | 2025 | 오픈액세스 [확인] | HTML 17쪽 | CC BY-NC 4.0 [확인] | sarcopenia | 생활계획(운동) | https://www.e-agmr.org/journal/view.php?doi=10.4235/agmr.25.0052 |
| 3 | AWGS 2025 Consensus Update (Nature Aging) | AWGS | 2025 | 접근 여부 미확인 | 영문 | 미확인 | sarcopenia | 체성분 기준 | https://pubmed.ncbi.nlm.nih.gov/41188603/ |
| 1 | 한국인 영양소 섭취기준(KDRI) 1~3권, 국문·영문 요약본 | 보건복지부·한국영양학회 | 2025 (2025-12-31, 정오표 '26.3.16 반영) | 무료 [확인] | ZIP 17.8MB | 영문요약본에 공공누리 표기, 유형 미확인 | 영양 일반 | 식사계획, 영양제 상한 | https://www.kns.or.kr/FileRoom/FileRoom.asp?BoardID=Kdr |
| 1 | KDRI 요약표(국문) | 보건복지부 | 2025 | 무료 [확인] | PDF 11쪽, 텍스트 PDF, 표 위주 [확인] | 미기재 | 영양 일반 | 식사계획, 영양제 상한 | 위 게시판 |
| 2 | 한국인을 위한 식생활지침 | 복지부·농식품부·식약처 | 2021 (개정 공청회 2026-11-05 예정) | 보도자료만 확인 | PDF/HWP 보도자료 | 공공누리 1유형 [확인] | 영양 일반 | 식사·습관 | https://www.mafra.go.kr/bbs/mafra/68/326885/artclView.do |
| 1 | 한국인을 위한 신체활동 지침서 개정판 | 보건복지부·한국건강증진개발원 | 2023 | 무료 [확인] | PDF 9.5MB, 약 81쪽 | 공공누리 표기, 유형 미확인 | 운동 일반 | 생활계획(운동) | https://www.mohw.go.kr/board.es?mid=a10411010100&bid=0019&act=view&list_no=1479208 |
| 2 | WHO guidelines on physical activity and sedentary behaviour | WHO | 2020 | 무료 [확인] | PDF 3.9MB, 영문 | CC BY-NC-SA 3.0 IGO [확인] | 운동 일반 | 생활계획(운동) | https://www.who.int/publications/i/item/9789240015128 |
| 1 | 건강기능식품 기능성 원료인정 현황 (OpenAPI) | 식품의약품안전처 | 2025-08-26 | API, 식품안전나라 인증키 필요 [확인] | JSON/XML | 이용허락 제한 없음 [확인] | supplement | 영양제 후보·주의 | https://www.data.go.kr/data/15058359/openapi.do |
| 1 | 건강기능식품 개별인정형 정보 (OpenAPI) | 식품의약품안전처 | 2025-08-26 | API, 인증키 필요 [확인] | JSON/XML | 제한 없음 [확인] | supplement | 영양제 후보·주의 | https://www.data.go.kr/data/15074311/openapi.do |
| 2 | 건강기능식품의 기준 및 규격 (고시형 원료) | 식약처 고시 | 제2026-43호 (2026-06-11) [확인] | 무료 | 고시 PDF / 법령정보센터 HTML | 미기재 | supplement | 영양제 후보·주의 | https://www.mfds.go.kr/brd/m_207/view.do?seq=15173 |
| 2 | 국가건강정보포털 질환 페이지 | 질병관리청 | 비만 2026-07-29 수정 [확인] | 무료 웹페이지 | HTML | 공공누리 4유형(출처표시·상업이용금지·변경금지) [확인] | 전 condition | 위험 설명, 생활계획 | https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoView.do?cntnts_sn=6694 |

### 2. 추천 코퍼스

#### 1차 (약 450~550쪽)

전문을 통째로 넣지 않고, 생활요법과 정의·진단 기준 장만 쪽 범위로 골라 넣는다.

| condition | 문서 | 넣을 범위 |
|---|---|---|
| hypertension | 고혈압 진료지침 2026 | 진단·분류, 생활요법 장 (PDF 확인 후 쪽 범위 결정) |
| diabetes | 당뇨병 진료지침 2025 전문 | 2장 예방, 5-3 의학영양요법, 5-4 운동요법, 9-1 비만 (약 40~60쪽) 또는 요약본 63쪽 |
| dyslipidemia | 이상지질혈증 진료지침 2026 | 진단기준·위험도 분류, 생활요법 장 (약 30~50쪽) |
| obesity | 비만병 팩트시트 2024 + 비만진료지침 8판 요약본 | BMI 25, 허리둘레 남 90 / 여 85cm, 단계표 |
| metabolic_syndrome | 대사증후군 진료지침 2021 요약본 | 전체 |
| fatty_liver | KASL 2021 NAFLD(생활습관 장) + KASL 2025 MASLD(정의·진단·예후) | 각 20~30쪽 |
| sarcopenia | KWGS 2023 진단 + KWGS 2025 운동 (영문) | 약 30쪽 |
| 영양 일반 | KDRI 2025 요약표 + 국문 요약본 | 전체 |
| 운동 일반 | 신체활동 지침서 2023 | 성인·노인·만성질환자 부분 약 40쪽 |
| supplement | 식약처 기능성 원료인정 현황 + 개별인정형 API | 혈압·혈당·콜레스테롤·중성지방·체지방 기능성만, 원료별 1청크 |

#### 2차 확장

- 당뇨 2025의 7-1 고혈압, 7-2 지질, 9-2 MASLD 장 (condition 간 교차 근거)
- 이상지질혈증 2026 요약슬라이드, 대사증후군 팩트시트 2026
- 고혈압 2026 하이라이트(영문), KSSO 2022 Update(영문)
- KDRI 본권 중 단백질, 나트륨·칼륨, 비타민D, 칼슘 상한 장
- WHO 신체활동 지침 2020, AWGS 2025(접근 확인 후)
- 건강기능식품의 기준 및 규격 통합본(고시형 원료: 홍국, 오메가-3, 녹차추출물, 가르시니아 등)
- 국가건강정보포털의 나머지 질환 페이지(URL 수집 필요)
- 식생활지침 개정본(2026-11 공청회 이후)

### 3. 위험과 공백

1. **학회 지침은 무료지만 개방 라이선스가 아님.** 지질학회는 상업적 사용 시 문의, 간학회는 무단 변형·전재 금지, 고혈압·당뇨는 "All rights reserved". 앱에서 청크 원문을 사용자에게 보여주려면 학회에 이용 허락을 받아 두거나, 짧은 인용 + 출처 + 원문 링크로 제한한다.
2. **질병관리청 포털은 공공누리 4유형(변경금지).** 요약·재작성은 변경으로 볼 여지가 있어 원문 인용·출처 표시 용도로만 쓴다.
3. **한국어 근감소증 지침이 없음.** KWGS·AWGS 모두 영문. 영문 청크로 검색해 한국어로 생성하거나 번역 후 색인해야 한다. BIA 기반 근육량 기준은 AWGS 원문 확인 필요.
4. **MASLD 2025에는 생활습관 장이 없음.** 2021 NAFLD판과 함께 넣어야 한다.
5. **비만 9판(2024) 존재 미확인.** 기준값은 2024 팩트시트로 충분하다.
6. **영양제 데이터는 PDF가 아니라 API.** 식품안전나라 인증키는 사용자가 직접 가입해 발급해야 한다. 원료별로 "원료명 / 주된 기능성 / 1일 섭취량 / 주의사항 / 인정번호" 텍스트를 만들어 1청크로 넣고, 출처는 "식약처, 조회일"로 남긴다. 고시형 원료(오메가-3, 홍국 등)는 API에 없을 수 있어 기준·규격 고시에서 따로 추출한다. 식약처 재평가로 내용이 자주 바뀌므로 주기적으로 다시 수집한다.
7. **스캔본 위험은 낮지만 레이아웃 위험이 있음.** 확인한 4개는 모두 텍스트 PDF. 당뇨 전문은 2단 편집 + 펼침면(PDF 한 쪽에 인쇄 두 쪽)이다. 고혈압 2026, 지질 2026, 신체활동 지침서는 텍스트 레이어 미확인.
8. **인용 쪽번호.** 당뇨 전문은 PDF 쪽(185)과 인쇄 쪽(약 300)이 다르다. 인쇄 쪽을 보여주려면 매핑이 필요하다.
9. **식생활지침 원문 책자 위치 미확인.**
10. **국가건강검진 판정기준 해설 원문 못 찾음.** 우선순위 낮음.
11. **인바디 체성분(체지방률, 골격근량) 정상범위에 대한 국내 공식 지침은 없음.** 비만은 BMI·허리둘레, 근육은 KWGS/AWGS의 ASM/키² 기준이다. 제조사 기준을 쓰려면 별도 출처 표기가 필요하다.

## B. 공개 API·오픈데이터

- 확인일: 2026-10-11
- 방법: data.go.kr 상세페이지, 식품안전나라 OpenAPI 상세, 농식품올바로, NIH·NLM·USDA·WHO 문서를 직접 열람. 키가 필요한 호출, 가입, 로그인은 하지 않음.
- condition 약어: HTN 고혈압, DM 당뇨, DL 이상지질혈증, OB 비만, MS 대사증후군, FL 지방간, SP 근감소증, GEN 일반

### 1. 후보 목록

| 우선순위 | API/데이터셋 | 제공기관 | 주요 데이터 | 형식 | 인증 | 이용조건 | condition | 리포트 섹션 | RAG 유용성 | URL |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 건강기능식품 품목분류정보 (I2710, 고시형 원료로 추정) | 식약처 식품안전나라 | 품목명, 주된 기능성, 1일 섭취량 하한·상한, 단위, 섭취 시 주의사항. 페이지 최종수정 2020-06-17 | XML/JSON | 식품안전나라 키 | 미기재 | 전체 | 영양제 | 높음: 원료 하나에 기능성·섭취량·주의사항이 구조화 | https://www.foodsafetykorea.go.kr/api/openApiInfo.do?menu_grp=MENU_GRP31&menu_no=661&show_cnt=10&start_idx=1&svc_no=I2710&svc_type_cd=API_TYPE06 |
| 2 | 건강기능식품 기능성 원료인정현황 (15058359 = I-0040) | 식약처 | 인정번호, 신청원료명, 기능성, 1일 섭취량, 주의사항 | JSON/XML | 키 (자동승인) | 이용허락 제한 없음 | 전체 | 영양제 | 높음: 개별인정 원료의 기능성 문구와 주의사항 | https://www.data.go.kr/data/15058359/openapi.do |
| 3 | 건강기능식품 개별인정형 정보 (15074311 = I-0050) | 식약처 | 인정번호, 섭취량 상·하한, 원료명, 주의사항, 주된 기능성 | JSON/XML | 키 (개발 자동, 운영 심의) | 제한 없음 | 전체 | 영양제 | 높음: 섭취량 범위가 수치로 있음 | https://www.data.go.kr/data/15074311/openapi.do |
| 4 | 국가건강정보포털 건강정보 (15087442) | 질병관리청 | 전문가 집필·감수 질환·건강정보 콘텐츠 | XML | 키 (개발 자동, 운영 심의) | **공공누리 4유형 (출처표시·상업이용금지·변경금지)** | 전체 | 위험 설명, 생활습관 | 내용 높음, 라이선스 주의 | https://www.data.go.kr/data/15087442/openapi.do |
| 5 | 국가표준식품성분 DB 10.4 | 농촌진흥청 국립식량과학원 | 식품 3,366점 × 130개 성분 | Excel, OpenAPI | Excel은 가입 없이 다운로드, API는 신청 | **DB Excel 공공누리 1유형**(책자 PDF는 2유형) | OB, DM, HTN, DL, FL, SP | 식단 | 높음: 식품별 나트륨·당·포화지방·단백질 | https://www.nics.go.kr/food/kfi/fct/fctIntro/list?menuId=PS03562 |
| 6 | 국민건강영양조사 통계 (15166328) | 질병관리청 | 만성질환 유병률, 관리수준, 건강행태, 영양섭취 통계 | JSON/XML | 키 (개발 자동, 운영 심의) | 공공누리 4유형(페이지 표기 일부 충돌) | HTN, DM, DL, OB, MS | 위험 설명 | 중상: "같은 연령대 한국인 대비" 근거. 2026-09 신규 | https://www.data.go.kr/data/15166328/openapi.do |
| 7 | 2024 Compendium of Physical Activities | Compendium 연구팀 | 활동 1,114개의 코드·MET 값 | 웹, Excel(이메일 요청) | 없음 | 상업 이용 가능, MET 값 변경 금지, 인용 요청 | OB, DM, HTN, MS, FL, SP | 운동 | 높음: 운동 강도·소모 칼로리 근거 | https://pacompendium.com/ |
| 8 | 의약품개요정보 e약은요 (15075057) | 식약처 | 효능, 사용법, 주의사항, **상호작용**, 부작용 (일반의약품) | JSON/XML | 키 (개발 자동, 운영 심의) | 제한 없음 | GEN | 영양제 주의 | 중: 약 쪽 상호작용 서술. 건기식 원료와 자동 매칭은 어려움 | https://www.data.go.kr/data/15075057/openapi.do |
| 9 | 질병정보서비스 (15119055) | 건강보험심사평가원 | 질병명·코드, 성·연령·지역별 진료 통계 | XML | 키 (자동승인) | 공공누리 1유형 | 전체 | 위험 설명, 용어 정규화 | 중 | https://www.data.go.kr/data/15119055/openapi.do |
| 10 | MedlinePlus Web Service | 미국 NLM | 건강 주제 요약 (영문) | XML | 없음 (분당 85회) | 주제 요약은 public domain, 출처 표기 요청 | 전체 | 위험 설명, 생활습관 | 중: 영문이지만 저작권 부담이 적음 | https://medlineplus.gov/about/developers/webservices/ |
| 11 | 식품영양성분DB정보 (15127578) | 식약처 | 식품명, 기준량, 에너지, 탄수화물, 단백질, 지방 등 (가공식품 포함). 2026-09-16 수정 | JSON/XML | 키 (개발 자동, 운영 심의) | 제한 없음 | OB, DM, HTN, DL | 식단 | 중: 수치 조회용 | https://www.data.go.kr/data/15127578/openapi.do |
| 12 | 국민체력100 동영상 정보 (15108846) | 국민체육진흥공단 | 운동처방 가이드, 생애주기별 표준운동, 목적별 루틴 메타데이터 | JSON/XML | 키 (자동승인) | 공공누리 1유형 | SP, OB, MS, GEN | 운동 | 중: 한국어 운동처방과 영상 링크 | https://www.data.go.kr/data/15108846/openapi.do |
| 13 | 상병마스터 (15067467) | 건강보험심사평가원 | 상병코드, 한글명, 영문명 등 47,798행 | CSV | 가입 없이 다운로드 | 공공누리 1유형 | GEN | 용어 정규화 | 중 | https://www.data.go.kr/data/15067467/fileData.do |
| 14 | PubMed E-utilities | 미국 NCBI | 문헌 검색, 초록 | XML/JSON | 선택 (없으면 초당 3회) | 초록 저작권 조건 미확인 | 전체 | 근거 검증 | 중: 코퍼스보다 검증용 | https://www.ncbi.nlm.nih.gov/books/NBK25497/ |
| 15 | MedlinePlus Connect | 미국 NLM | ICD-10 등 코드 → 관련 MedlinePlus 주제 | XML/JSON | 없음 | 링크·표시 허용, 복제 금지 | 전체 | 위험 설명 | 중하 | https://medlineplus.gov/medlineplus-connect/web-service/ |
| 16 | 직역별 성별 연령별 건강검진정보 (15144521) | 국민건강보험공단 | 2022–2023 검진 인원, 비만·혈압·지질·간·당뇨 의심 인원 | CSV | 가입 없이 다운로드 | 제한 없음 | OB, HTN, DL, FL, DM | 위험 설명 | 중하: 비율 계산 필요 | https://www.data.go.kr/data/15144521/fileData.do |
| 17 | 국민체력100 체력인증센터 측정결과 (15108938) | 국민체육진흥공단 | 연령, 신장, 체중, **체지방률**, 허리둘레, 혈압, 악력 등 | JSON/XML | 키 (자동승인) | 제한 없음 | SP, OB | 체성분 | 하중: 원자료. 연령대별 기준 분포 계산용 | https://www.data.go.kr/data/15108938/openapi.do |
| 18 | 건강기능식품 품목제조 신고사항 (I0030) | 식약처 식품안전나라 | 제품 단위 기능성, 섭취방법, 주의사항 | XML/JSON | 식품안전나라 키 | 미기재 | GEN | 영양제 | 낮음: 제품 단위라 노이즈, 브랜드 노출 위험 | https://www.foodsafetykorea.go.kr/api/openApiInfo.do?menu_grp=MENU_GRP31&menu_no=661&show_cnt=10&start_idx=1&svc_no=I0030&svc_type_cd=API_TYPE06 |
| 19 | DUR 성분정보 (15056780) / 품목정보 (15059486) | 식약처 | 병용금기, 연령·임부 금기, 노인주의 | JSON/XML | 키 (자동승인) | 제한 없음 | GEN | 영양제 주의 | 낮음: **약물 간 정보만 있고 건기식 상호작용은 없음** | https://www.data.go.kr/data/15056780/openapi.do |
| 20 | 건강기능식품의 기준 및 규격 (15074321) | 식약처 | 시험항목, 규격값 | JSON/XML | 키 | 제한 없음 | GEN | 영양제 | 낮음: 품질 규격이라 리포트와 무관 | https://www.data.go.kr/data/15074321/openapi.do |
| 21 | USDA FoodData Central | USDA | 미국 식품 영양성분 | JSON | api.data.gov 키 | public domain / CC0 | OB, DM, HTN, DL | 식단 | 하중: 한국 식품 커버리지 약함 | https://fdc.nal.usda.gov/api-guide/ |
| 22 | DSLD Label API | NIH ODS | 미국 건기식 라벨 | JSON | 없음 | CC0 | GEN | 영양제 | 낮음: 미국 제품 | https://dsldapi.od.nih.gov/ |
| 23 | NIH ODS Fact Sheets API | NIH ODS | 원료별 팩트시트 (상호작용 정보 풍부) | 미확인 | 미확인 | 미확인 | GEN | 영양제 | 내용 높음, API 존재 미확인(봇 차단) | https://ods.od.nih.gov/api/ |
| 24 | KOSIS 공유서비스 OpenAPI | 국가데이터처 | 통계 목록·자료 | JSON/XML 등 | 미확인 | 미기재 | HTN, DM, OB | 위험 설명 | 하중: 6번과 겹침 | https://kosis.kr/serviceInfo/openAPIGuide.do |
| 25 | WHO GHO OData API | WHO | 국가별 지표 | OData | 언급 없음 | 미기재 | OB, HTN, DM | 위험 설명 | 낮음: 국가 단위 | https://www.who.int/data/gho/info/gho-odata-api |

**찾지 못한 것**: KDRI의 기계판독용 파일·API, 건강기능식품 이상사례 API, 식품안전나라 "의약품 병용 섭취 주의사항" API, NHIS 100만 명 표본(상세페이지 404), 국민체력100 운동처방 원자료.

### 2. 추천 shortlist와 청크 설계

1. **영양제: I2710 + I-0040(15058359) + I-0050(15074311)**
   - 원료 하나 = 청크 하나: `원료명 | 고시형/개별인정 | 인정번호 | 주된 기능성 | 1일 섭취량(하한–상한, 단위) | 섭취 시 주의사항 | 출처: 식약처 식품안전나라 {서비스ID}, 최종수정일, 조회일`
   - condition 태그는 기능성 문구 키워드로 매핑: 중성지방 → dyslipidemia, 혈당 → diabetes, 체지방 → obesity, 간 → fatty_liver, 근력 → sarcopenia, 혈압 → hypertension
   - I-0040과 I-0050은 인정번호로 조인
2. **국가건강정보포털(15087442)**: 질환별 소단원(정의, 위험요인, 생활수칙) 하나 = 청크 하나. **변경금지 조건 때문에 원문 그대로 저장하고 인용**
3. **국가표준식품성분 DB Excel**: 식품 하나 = 청크 하나(100g당 에너지·나트륨·당류·포화지방·식이섬유·단백질). 질환별 "고나트륨 식품 상위 N" 같은 요약 청크 추가
4. **국민건강영양조사 통계(15166328)**: 지표 × 연도 × 성·연령대 = 청크 하나. "같은 연령대 한국인 대비" 문장의 근거
5. **Compendium(MET)**: 활동 하나 = 청크 하나(코드, 활동명, MET, 강도). MET 값은 바꾸지 않음
6. **e약은요(15075057)**: 의약품 하나 = 청크 하나. 사용자가 복용약을 입력했을 때만 검색
7. **HIRA 질병정보(15119055) + 상병마스터(15067467)**: KCD 코드와 condition 키 매핑 사전(E11 → diabetes, I10 → hypertension, E78 → dyslipidemia, K76.0 → fatty_liver, E66 → obesity 등. KCD 원표로 검증 필요)
8. (선택) **MedlinePlus**: 근감소증처럼 한국어 자료가 부족한 주제의 영문 보조 근거

### 3. 가입이 필요한 곳

| 가입처 | 대상 | 비고 |
|---|---|---|
| 공공데이터포털 data.go.kr | 15058359, 15074311, 15087442, 15166328, 15075057, 15119055, 15108846, 15108938, 15127578 등 | 데이터셋마다 "활용신청". 개발단계 대부분 자동승인, 일부는 운영단계 심의 |
| 식품안전나라 foodsafetykorea.go.kr | I2710, I-0040, I-0050, I0030 | OpenAPI 인증키 신청. 요청 형식 `openapi.foodsafetykorea.go.kr/api/{키}/{서비스ID}/{json}/{start}/{end}` |
| 농식품올바로 nics.go.kr | 국가표준식품성분 OpenAPI | Excel은 가입 없이 다운로드 가능 |
| 가입 불필요 | MedlinePlus, WHO GHO, DSLD, data.go.kr 파일데이터(15067467, 15144521) | Compendium Excel은 이메일 요청 |

### 4. 위험과 공백

1. **라이선스**: 국가건강정보포털과 국민건강영양조사 통계는 공공누리 4유형(상업이용금지·변경금지)이다. 상용화 시 쓸 수 없고, Gemini가 원문을 바꿔 쓰는 것도 "변경"으로 볼 수 있다. 원문 인용과 링크 위주로 설계한다. 식품성분표는 Excel(1유형)을 쓴다.
2. **갱신 주기**: I2710(2020-06)과 I0030(2021-07)은 페이지 수정일이 오래됨. 실제 데이터의 최종수정일로 확인 필요. 국민건강영양조사 통계(2026-09), 식품영양성분DB(2026-09), HIRA 질병정보(2026-04)는 최신.
3. **건기식-의약품 상호작용 데이터 없음**: 국내 공공 API 중 건기식 원료와 의약품 상호작용을 구조화해 주는 것은 없다. DUR은 약물 간 정보, e약은요는 일반의약품 서술뿐. 영양제 주의사항은 원료 API의 "섭취 시 주의사항"과 "복용 중인 약이 있으면 의사·약사 상담" 템플릿으로 보완한다.
4. **한국어 커버리지**: 근감소증·지방간은 한국어 구조화 데이터가 드물다. PDF 지침과 MedlinePlus 영문으로 보완한다.
5. **필드 미확인**: 15087442, 15166328, 15119055, 15108846은 응답 필드가 첨부 문서(hwp/docx)에만 있어 확인하지 못했다.
6. **접근 차단**: NIH ODS, NCBI 문서는 봇 확인에 막혀 읽지 못했다(우회하지 않음).
