"""RAG 공통 용어집과 검색 쿼리 규칙. 코퍼스 청크 태그, 평가 쿼리, 런타임 쿼리가 같은 키와 문구를 쓴다.
설계: docs/superpowers/specs/2026-10-15-rag-runtime-design.md 5.1절."""

import logging
from dataclasses import dataclass
from typing import Literal

from app.services.personal_metrics import ABDOMINAL_RATIO, OBESITY_BMI, VISCERAL_FAT_LEVEL

log = logging.getLogger(__name__)

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

CONDITION_NAMES = {
    "hypertension": "고혈압",
    "diabetes": "당뇨병",
    "dyslipidemia": "이상지질혈증",
    "obesity": "비만",
    "metabolic_syndrome": "대사증후군",
    "fatty_liver": "지방간",
    "sarcopenia": "근감소증",
}

# 평가 쿼리 세트의 (질환, 카테고리) 조합. 런타임 쿼리도 이 조합만 쓴다(임계값을 이 문구로 조정했다).
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

# 예측 모델 질환 키 → 용어집 키. 콜레스테롤과 중성지방은 지침 체계에서 이상지질혈증 하나다.
RISK_CONDITIONS = {"hypertension": "hypertension", "diabetes": "diabetes",
                   "cholesterol": "dyslipidemia", "triglyceride": "dyslipidemia"}


@dataclass(frozen=True)
class Query:
    condition: str
    category: str
    text: str
    trait: str | None = None  # 특성 쿼리면 특성 이름(근거 머리글에 쓴다)


def keyword_query(condition: str, category: str) -> str:
    if condition == "general":
        return _GENERAL_KEYWORDS[category]
    return _KEYWORD_TEMPLATES[category].format(name=CONDITION_NAMES[condition])


def _at_least(value, threshold) -> bool:
    return value is not None and threshold is not None and float(value) >= threshold


def build_queries(disease_risks: list[dict], inbody: dict, gender: str | None) -> list[Query]:
    """위험도 medium 이상 질환과 인바디 기준 초과 항목마다 평가에서 쓴 키워드 쿼리를 만든다.
    해당하는 것이 없으면 일반 식사·운동 쿼리 2개."""
    conditions: list[str] = []
    for risk in disease_risks:
        if risk.get("risk_level") not in ("medium", "high"):
            continue
        condition = RISK_CONDITIONS.get(risk.get("disease_key"))
        if condition is None:
            log.warning("매핑에 없는 질환 키 %r는 쿼리에서 뺍니다", risk.get("disease_key"))
        elif condition not in conditions:
            conditions.append(condition)
    if _at_least(inbody.get("bmi"), OBESITY_BMI):
        conditions.append("obesity")
    if (_at_least(inbody.get("abs_fat"), ABDOMINAL_RATIO.get(gender or ""))
            or _at_least(inbody.get("fat_level"), VISCERAL_FAT_LEVEL)):
        conditions.append("metabolic_syndrome")
    if not conditions:
        conditions = ["general"]
    return [Query(condition, category, keyword_query(condition, category))
            for condition in conditions for c, category in SCENARIOS if c == condition]


# 타겟(30대 전후 성인병 초기 위험군)의 개인 특성 쿼리. 설계: 2026-10-28-report-personalization-design.md 5.3절
_PRE_STAGE = {"hypertension": "고혈압 전단계 주의혈압 생활습관 교정", "diabetes": "당뇨병 전단계 공복혈당장애 예방",
              "dyslipidemia": "이상지질혈증 경계 수치 생활습관 교정"}
_FAMILY = {"hypertension": "fh_hypertension", "diabetes": "fh_diabetes", "dyslipidemia": "fh_dyslipidemia"}
_TRAIT_LIMIT = 4
SUPPLEMENT_QUERY = Query("general", "supplement", "영양제 건강기능식품 선택 주의사항", "영양제")


def _risk_conditions(disease_risks: list[dict], levels: set[str]) -> list[str]:
    found = {RISK_CONDITIONS.get(r.get("disease_key")) for r in disease_risks if r.get("risk_level") in levels}
    return [c for c in _FAMILY if c in found]  # 고혈압, 당뇨병, 이상지질혈증 순서


def trait_queries(disease_risks: list[dict], profile: dict, metrics: dict,
                  trend_analysis: dict | None) -> list[Query]:
    """특성 종류마다 1개씩 먼저(표 순서), 남는 자리는 같은 순서로 두 번째 쿼리부터 채워 최대 4개 + 영양제 1개."""
    lean = metrics.get("bmi_band") in ("저체중", "정상")
    trends = (trend_analysis or {}).get("feature_trends", [])
    segmental = metrics.get("segmental", {})
    weak_parts = segmental.get("근육 표준이하", [])
    # 평가한 키워드형(근감소증 운동요법) 문구라야 임계값을 넘는다. "스쿼트 런지"처럼 구체적으로 쓰면 0.66까지 떨어졌다
    part_query = ("근감소증 운동요법 하체 근력운동 권고" if any("다리" in p for p in weak_parts)
                  else "근감소증 운동요법 상체 근력운동 권고" if weak_parts else None)
    groups: list[list[Query]] = [
        [Query("sarcopenia", "exercise", "근육량 증가 저항운동 단백질 섭취", "근육 부족")]
        if metrics.get("muscle_status", {}).get("status") == "표준 이하" else [],
        [Query("sarcopenia", "exercise", part_query, "부위 근육 부족")] if part_query else [],
        [Query("obesity", "exercise", "체지방률 감소 근력운동 체성분 개선", "체지방 과다")]
        if lean and metrics.get("body_fat_status", {}).get("status") in ("경도비만", "비만") else [],
        [Query("metabolic_syndrome", "lifestyle", "복부비만 내장지방 감소 운동 식사", "복부비만")]
        if metrics.get("abdominal") or "몸통" in segmental.get("체지방 표준이상", []) else [],
        [Query(c, "lifestyle", _PRE_STAGE[c], "전단계 위험") for c in _risk_conditions(disease_risks, {"medium"})],
        [Query(c, "lifestyle", f"{CONDITION_NAMES[c]} 가족력 고위험군 조기 예방", "가족력")
         for c, key in _FAMILY.items() if profile.get(key) is True],
        [Query("metabolic_syndrome", "lifestyle", "대사 위험인자 동반 생활습관 관리", "여러 위험 동반")]
        if len(_risk_conditions(disease_risks, {"medium", "high"})) >= 2 else [],
        [Query("obesity", "lifestyle", "체중 증가 예방 체중 유지 관리", "체중 증가 추세")]
        if any(t.get("feature_kor") in ("체중", "체지방률") and t.get("direction") == "up" for t in trends) else [],
        [Query("obesity", "lifestyle", "비만 전단계 과체중 체중 증가 예방", "비만 전단계")]
        if metrics.get("bmi_band") == "비만 전단계" else [],
    ]
    picked: list[Query] = []
    for rank in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if rank < len(group) and len(picked) < _TRAIT_LIMIT:
                picked.append(group[rank])
    return [*picked, SUPPLEMENT_QUERY]
