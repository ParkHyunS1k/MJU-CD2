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
