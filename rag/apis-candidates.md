# RAG 코퍼스 후보 공개 API·오픈데이터 조사

- 확인일: 2026-10-11
- 방법: data.go.kr 상세페이지, 식품안전나라 OpenAPI 상세, 농식품올바로, NIH·NLM·USDA·WHO 문서를 직접 열람. 키가 필요한 호출, 가입, 로그인은 하지 않음.
- condition 약어: HTN 고혈압, DM 당뇨, DL 이상지질혈증, OB 비만, MS 대사증후군, FL 지방간, SP 근감소증, GEN 일반

## 1. 후보 목록

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

## 2. 추천 shortlist와 청크 설계

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

## 3. 가입이 필요한 곳

| 가입처 | 대상 | 비고 |
|---|---|---|
| 공공데이터포털 data.go.kr | 15058359, 15074311, 15087442, 15166328, 15075057, 15119055, 15108846, 15108938, 15127578 등 | 데이터셋마다 "활용신청". 개발단계 대부분 자동승인, 일부는 운영단계 심의 |
| 식품안전나라 foodsafetykorea.go.kr | I2710, I-0040, I-0050, I0030 | OpenAPI 인증키 신청. 요청 형식 `openapi.foodsafetykorea.go.kr/api/{키}/{서비스ID}/{json}/{start}/{end}` |
| 농식품올바로 nics.go.kr | 국가표준식품성분 OpenAPI | Excel은 가입 없이 다운로드 가능 |
| 가입 불필요 | MedlinePlus, WHO GHO, DSLD, data.go.kr 파일데이터(15067467, 15144521) | Compendium Excel은 이메일 요청 |

## 4. 위험과 공백

1. **라이선스**: 국가건강정보포털과 국민건강영양조사 통계는 공공누리 4유형(상업이용금지·변경금지)이다. 상용화 시 쓸 수 없고, Gemini가 원문을 바꿔 쓰는 것도 "변경"으로 볼 수 있다. 원문 인용과 링크 위주로 설계한다. 식품성분표는 Excel(1유형)을 쓴다.
2. **갱신 주기**: I2710(2020-06)과 I0030(2021-07)은 페이지 수정일이 오래됨. 실제 데이터의 최종수정일로 확인 필요. 국민건강영양조사 통계(2026-09), 식품영양성분DB(2026-09), HIRA 질병정보(2026-04)는 최신.
3. **건기식-의약품 상호작용 데이터 없음**: 국내 공공 API 중 건기식 원료와 의약품 상호작용을 구조화해 주는 것은 없다. DUR은 약물 간 정보, e약은요는 일반의약품 서술뿐. 영양제 주의사항은 원료 API의 "섭취 시 주의사항"과 "복용 중인 약이 있으면 의사·약사 상담" 템플릿으로 보완한다.
4. **한국어 커버리지**: 근감소증·지방간은 한국어 구조화 데이터가 드물다. PDF 지침과 MedlinePlus 영문으로 보완한다.
5. **필드 미확인**: 15087442, 15166328, 15119055, 15108846은 응답 필드가 첨부 문서(hwp/docx)에만 있어 확인하지 못했다.
6. **접근 차단**: NIH ODS, NCBI 문서는 봇 확인에 막혀 읽지 못했다(우회하지 않음).
