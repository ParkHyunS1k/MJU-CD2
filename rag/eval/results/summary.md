# 검색 구성 실험 요약

최종 설정은 `rag/eval/selected.json`에 있다(하위 프로젝트 3의 입력).

| 항목 | 선택 |
|---|---|
| parser | pymupdf |
| spacing | False |
| max_chars | 1200 |
| overlap | 0 |
| embedding_model | gemini-768 |
| dimensions | 768 |
| hybrid | False |
| reranker | None |
| query_form | keyword |
| final_k | 2 |
| threshold | 0.7069 |

시험용 Recall@2: 0.923 [0.808, 1.000], nDCG@10: 0.721 [0.613, 0.813], 사람 검수 쿼리만: 1.000 [1.000, 1.000]. 정답 없는 쿼리 3개 중 3개를 임계값으로 거른다(임계값을 이 쿼리들로 정했으므로 검증된 값은 아니다).

단계별 근거는 `01_parser.md` … `06_operating.md`. 1등과의 쿼리별 차이(쌍체 부트스트랩 95% 구간)가 0을 포함하면 더 단순·저렴한 구성을 골랐다.

## 한계
- 라벨은 사람 검수 188건 + LLM(GPT) 판정. 사람 검수 쿼리는 고혈압·당뇨·이상지질혈증뿐이라 "사람 검수" 열을 함께 본다.
- 후보 모으기(pooling)로 만든 라벨이라, 어떤 방식도 찾지 못한 정답은 빠져 있다.
- 근거 문장은 PyMuPDF 텍스트에서 뽑아 1단계 preservation은 PyMuPDF에 유리하다.
- 쿼리 50개 내외라 차이가 작으면 구간이 겹친다.
