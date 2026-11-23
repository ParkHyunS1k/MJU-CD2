from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ShapFactor(BaseModel):
    feature_kor: str
    value: float
    shap_value: float
    direction: Literal["up", "down"]


class DiseaseRisk(BaseModel):
    disease: str
    probability: float
    risk_level: RiskLevel
    top_factors: list[ShapFactor]


class FeatureTrend(BaseModel):
    feature_kor: str
    dates: list[str]
    values: list[float]
    direction: Literal["up", "down", "stable"]
    total_change: float
    slope_per_90_days: float | None = None
    is_shap_important: bool
    shap_directions: list[Literal["up", "down"]] = Field(default_factory=list)
    is_risk_worsening: bool = False


class TrendAnalysis(BaseModel):
    record_count: int
    span_days: int
    feature_trends: list[FeatureTrend]


class ReportSummary(BaseModel):
    title: str
    overall_message: str
    key_focus: list[str]


class MetricInsight(BaseModel):
    metric: str
    message: str


class BodyWatchPoint(BaseModel):
    metric: str
    current_value: str
    interpretation: str
    priority: Literal["low", "medium", "high"]
    source_ids: list[str] = Field(default_factory=list)


class BodyCompositionReport(BaseModel):
    strengths: list[MetricInsight]
    watch_points: list[BodyWatchPoint]


class RiskReportItem(BaseModel):
    disease: str
    risk_level: RiskLevel
    probability: float
    main_factors: list[str]
    message: str
    source_ids: list[str] = Field(default_factory=list)


class LifestylePlanItem(BaseModel):
    category: Literal["nutrition", "exercise", "sleep", "habit"]
    title: str
    reason: str
    actions: list[str]
    source_ids: list[str] = Field(default_factory=list)


class SupplementCandidate(BaseModel):
    name: str
    purpose: str
    related_metrics: list[str]
    evidence_level: Literal["general", "moderate", "limited"]
    why_consider: str
    cautions: list[str]
    not_recommended_if: list[str]
    source_ids: list[str] = Field(default_factory=list)


class SourceRef(BaseModel):
    """리포트가 인용한 근거. 서버가 채우며 LLM 응답 스키마(AIReportContent)에는 없다."""

    sid: str
    title: str
    publisher: str | None = None
    year: int | None = None
    page: str | None = None
    heading_path: str | None = None
    url: str | None = None


class ReportCautions(BaseModel):
    medical_disclaimer: str
    consult_doctor_if: list[str]


class AIReportContent(BaseModel):
    summary: ReportSummary
    body_composition_report: BodyCompositionReport
    risk_report: list[RiskReportItem]
    lifestyle_plan: list[LifestylePlanItem]
    supplement_candidates: list[SupplementCandidate]
    cautions: ReportCautions


class HealthReport(AIReportContent):
    disease_risks: list[DiseaseRisk]
    trend_analysis: TrendAnalysis | None = None
    sources: list[SourceRef] = Field(default_factory=list)


class InBodyMetricsBase(BaseModel):
    measured_at: datetime | None = None
    weight: float | None = Field(None, gt=0, lt=500)
    skeletal_muscle_mass: float | None = Field(None, ge=0)
    body_fat_mass: float | None = Field(None, ge=0)
    body_fat_percentage: float | None = Field(None, ge=0, le=100)
    body_water: float | None = Field(None, ge=0)
    protein: float | None = Field(None, ge=0)
    minerals: float | None = Field(None, ge=0)
    bmi: float | None = Field(None, gt=0)
    basal_metabolic_rate: float | None = Field(None, ge=0)
    abs_fat: float | None = Field(None, ge=0)
    fat_level: float | None = Field(None, ge=0)
    height: float | None = Field(None, gt=0, lt=300)  # 신장(cm)
    muscle_control: float | None = None  # 결과지 체중조절의 근육조절(kg). 양수면 그만큼 늘려야 표준
    # 결과지의 나머지 수치(InBody270 기준). 표준 범위·운동소비열량은 키·성별·체중으로 계산되는 참고값이라 저장하지 않는다
    inbody_score: float | None = Field(None, ge=0)
    target_weight: float | None = Field(None, gt=0)  # 적정체중(kg)
    weight_control: float | None = None  # 체중조절(kg)
    fat_control: float | None = None  # 지방조절(kg)
    fat_free_mass: float | None = Field(None, ge=0)  # 제지방량(kg)
    obesity_degree: float | None = Field(None, ge=0)  # 비만도(%)
    recommended_calories: float | None = Field(None, ge=0)  # 권장섭취열량(kcal)
    bmi_evaluation: str | None = None  # 비만평가 BMI: 표준·저체중·과체중·심한과체중
    body_fat_evaluation: str | None = None  # 비만평가 체지방률: 표준·경도비만·비만
    # 부위별 근육·체지방: {"left_arm"|"right_arm"|"trunk"|"left_leg"|"right_leg": "표준"|"표준이하"|"표준이상"}
    segmental_lean: dict[str, str] | None = None
    segmental_fat: dict[str, str] | None = None
    impedance: dict[str, dict[str, float]] | None = None  # {"20kHz"|"100kHz": {"RA","LA","TR","RL","LL": Ω}}


class InBodyCreate(InBodyMetricsBase):
    weight: float = Field(..., gt=0, lt=500)
    image_url: str | None = None


class InBodyUpdate(InBodyMetricsBase):
    pass


class InBodyResponse(InBodyCreate):
    id: str
    user_id: str
    ai_report: HealthReport | None = None


Segment = Literal["표준", "표준이하", "표준이상"]


class SegmentalExtraction(BaseModel):
    """결과지 부위별 분석의 판정. 응답 스키마에 자유 키 dict를 쓸 수 없어 부위를 고정한다."""

    left_arm: Segment | None = Field(None, description="왼팔")
    right_arm: Segment | None = Field(None, description="오른팔")
    trunk: Segment | None = Field(None, description="몸통")
    left_leg: Segment | None = Field(None, description="왼다리")
    right_leg: Segment | None = Field(None, description="오른다리")


class InBodyExtraction(BaseModel):
    """Gemini Vision이 이미지/PDF에서 추출하는 구조화 결과. 제약 없이 단순 타입만 사용."""

    is_inbody: bool = Field(description="인바디 결과지 여부")
    confidence: float = Field(description="추출 신뢰도 0~1")
    notes: str | None = Field(None, description="누락/애매한 값 메모")
    measured_at: datetime | None = Field(None, description="측정일시")
    weight: float | None = Field(None, description="체중(kg)")
    skeletal_muscle_mass: float | None = Field(None, description="골격근량(kg)")
    body_fat_mass: float | None = Field(None, description="체지방량(kg)")
    body_fat_percentage: float | None = Field(None, description="체지방률(%)")
    body_water: float | None = Field(None, description="체수분(kg 또는 L)")
    protein: float | None = Field(None, description="단백질(kg)")
    minerals: float | None = Field(None, description="무기질(kg)")
    bmi: float | None = Field(None, description="BMI(체질량지수)")
    basal_metabolic_rate: float | None = Field(None, description="기초대사량(kcal)")
    abs_fat: float | None = Field(None, description="복부지방률 또는 복부비만율. 결과지에서 '복부지방률', 'WHR', '복부비만율' 등의 항목으로 표시되며 보통 0.7~1.0 사이의 소수값")
    fat_level: float | None = Field(None, description="내장지방레벨(1~20 정수)")
    # 리포트 개인화에 쓰는 결과지 항목(InBody270 기준). 임피던스는 리포트에 쓰지 않아 추출하지 않는다
    height: float | None = Field(None, description="신장(cm)")
    muscle_control: float | None = Field(None, description="체중조절의 근육조절(kg). 적정이면 결과지에 0.0으로 표시")
    fat_control: float | None = Field(None, description="체중조절의 지방조절(kg). 감량은 음수, 결과지 부호 그대로")
    weight_control: float | None = Field(None, description="체중조절(kg). 감량은 음수, 결과지 부호 그대로")
    target_weight: float | None = Field(None, description="적정체중(kg)")
    inbody_score: float | None = Field(None, description="인바디점수")
    fat_free_mass: float | None = Field(None, description="제지방량(kg)")
    obesity_degree: float | None = Field(None, description="비만도(%)")
    recommended_calories: float | None = Field(None, description="권장섭취열량(kcal)")
    bmi_evaluation: str | None = Field(None, description="비만평가의 BMI 판정(표준·저체중·과체중·심한과체중)")
    body_fat_evaluation: str | None = Field(None, description="비만평가의 체지방률 판정(표준·경도비만·비만)")
    segmental_lean: SegmentalExtraction | None = Field(None, description="부위별 근육분석 판정")
    segmental_fat: SegmentalExtraction | None = Field(None, description="부위별 체지방분석 판정")


class InBodyUploadResponse(BaseModel):
    """업로드+추출 직후 응답. 분석은 아직 안 됨."""

    record: InBodyResponse
    extraction_confidence: float
    extraction_notes: str | None = None
