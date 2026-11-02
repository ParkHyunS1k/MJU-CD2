"""인바디 결과지 추출 필드 정의.

필드 범위(min/max)는 성인 대상 '명백한 오인식'을 거르기 위한 잠정값이다.
의학적 정상범위가 아니며, 테스트셋 결과를 보고 조정한다.
이름(성명) 필드는 개인정보 최소수집 원칙에 따라 의도적으로 두지 않는다.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class FieldSpec:
    label: str  # 결과지 표기 (LLM 프롬프트에도 사용)
    unit: str
    min: float
    max: float
    decimals: int  # 결과지 표기 소수 자릿수. exact match 비교 기준


FIELD_SPECS: dict[str, FieldSpec] = {
    "height_cm": FieldSpec("신장", "cm", 130, 210, 1),
    "age": FieldSpec("나이", "세", 19, 100, 0),  # 미성년 제외 정책 반영
    "weight_kg": FieldSpec("체중", "kg", 30, 200, 1),
    "total_body_water_l": FieldSpec("체수분", "L", 15, 80, 1),
    "protein_kg": FieldSpec("단백질", "kg", 4, 25, 1),
    "minerals_kg": FieldSpec("무기질", "kg", 1.5, 7, 2),
    "body_fat_mass_kg": FieldSpec("체지방량", "kg", 2, 100, 1),
    "fat_free_mass_kg": FieldSpec("제지방량", "kg", 20, 120, 1),
    "skeletal_muscle_mass_kg": FieldSpec("골격근량", "kg", 10, 60, 1),
    "bmi": FieldSpec("BMI", "kg/m²", 12, 60, 1),
    "percent_body_fat": FieldSpec("체지방률", "%", 3, 65, 1),
    "waist_hip_ratio": FieldSpec("복부지방률(WHR)", "", 0.6, 1.2, 2),
    "visceral_fat_level": FieldSpec("내장지방레벨", "", 1, 20, 0),
    "basal_metabolic_rate_kcal": FieldSpec("기초대사량", "kcal", 800, 3000, 0),
    "inbody_score": FieldSpec("인바디점수", "점", 30, 120, 0),
    "ideal_weight_kg": FieldSpec("적정체중", "kg", 30, 150, 1),
    "weight_control_kg": FieldSpec("체중조절", "kg", -60, 60, 1),
    "fat_control_kg": FieldSpec("지방조절", "kg", -60, 60, 1),
    "muscle_control_kg": FieldSpec("근육조절", "kg", -30, 30, 1),
    "obesity_degree_pct": FieldSpec("비만도", "%", 50, 250, 0),
    "recommended_calorie_kcal": FieldSpec("권장섭취열량", "kcal", 500, 5000, 0),
}

NUMERIC_FIELDS = list(FIELD_SPECS)

# 정상범위 "(41.4~50.6)"가 함께 표기되는 항목 (InBody270)
RANGE_FIELDS = [
    "total_body_water_l", "protein_kg", "minerals_kg", "body_fat_mass_kg", "weight_kg",
    "fat_free_mass_kg", "basal_metabolic_rate_kcal", "obesity_degree_pct",
]

# 운동소비열량 (현재 체중, 30분 기준). 결과지 순서
EXERCISE_SPECS: dict[str, FieldSpec] = {
    key: FieldSpec(label, "kcal", 20, 1500, 0)
    for key, label in [
        ("golf", "골프"), ("gateball", "게이트볼"), ("walking", "걷기"), ("yoga", "요가"),
        ("badminton", "배드민턴"), ("table_tennis", "탁구"), ("tennis", "테니스"), ("cycling", "자전거"),
        ("boxing", "복싱"), ("basketball", "농구"), ("hiking", "등산"), ("jump_rope", "줄넘기"),
        ("aerobics", "에어로빅"), ("jogging", "조깅"), ("soccer", "축구"), ("swimming", "수영"),
        ("kendo", "검도"), ("racquetball", "라켓볼"), ("squash", "스쿼시"), ("taekwondo", "태권도"),
    ]
}

# 임피던스 (Ω): 오른팔·왼팔·몸통·오른다리·왼다리 × 20kHz·100kHz
IMPEDANCE_SEGMENTS = ["RA", "LA", "TR", "RL", "LL"]
IMPEDANCE_FREQS = ["20kHz", "100kHz"]


class HistoryPoint(BaseModel):
    """신체변화(이력) 그래프의 한 열. 마지막 열이 이번 측정."""

    date: Optional[str] = None  # YYYY-MM-DD
    time: Optional[str] = None  # HH:MM
    weight_kg: Optional[float] = None
    skeletal_muscle_mass_kg: Optional[float] = None
    percent_body_fat: Optional[float] = None


class InBodyExtraction(BaseModel):
    """LLM(경로 A)이나 규칙 파서(경로 B)가 공통으로 내놓는 원시 추출 결과."""

    is_inbody_sheet: bool
    sex: Optional[Literal["M", "F"]] = None
    measured_at: Optional[str] = None  # YYYY-MM-DD
    measured_time: Optional[str] = None  # HH:MM
    height_cm: Optional[float] = None
    age: Optional[float] = None
    weight_kg: Optional[float] = None
    total_body_water_l: Optional[float] = None
    protein_kg: Optional[float] = None
    minerals_kg: Optional[float] = None
    body_fat_mass_kg: Optional[float] = None
    fat_free_mass_kg: Optional[float] = None
    skeletal_muscle_mass_kg: Optional[float] = None
    bmi: Optional[float] = None
    percent_body_fat: Optional[float] = None
    waist_hip_ratio: Optional[float] = None
    visceral_fat_level: Optional[float] = None
    basal_metabolic_rate_kcal: Optional[float] = None
    inbody_score: Optional[float] = None
    ideal_weight_kg: Optional[float] = None
    weight_control_kg: Optional[float] = None
    fat_control_kg: Optional[float] = None
    muscle_control_kg: Optional[float] = None
    obesity_degree_pct: Optional[float] = None
    recommended_calorie_kcal: Optional[float] = None
    ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)  # 항목 → (하한, 상한)
    exercise_kcal: dict[str, float] = Field(default_factory=dict)  # EXERCISE_SPECS 키 → kcal
    impedance: dict[str, dict[str, float]] = Field(default_factory=dict)  # 주파수 → 부위 → Ω
    history: list[HistoryPoint] = Field(default_factory=list)


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def llm_json_schema() -> dict:
    """structured outputs(output_config.format)용 JSON 스키마."""
    props: dict[str, dict] = {
        "is_inbody_sheet": {
            "type": "boolean",
            "description": "이미지가 인바디 체성분 결과지(종이 사진 또는 앱 화면)인지",
        },
        "sex": _nullable({"type": "string", "enum": ["M", "F"]}),
        "measured_at": _nullable({"type": "string", "description": "측정일 YYYY-MM-DD"}),
    }
    for key, spec in FIELD_SPECS.items():
        unit = f" ({spec.unit})" if spec.unit else ""
        props[key] = _nullable({"type": "number", "description": f"{spec.label}{unit}"})
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }
