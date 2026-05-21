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


class HealthReport(BaseModel):
    disease_risks: list[DiseaseRisk]
    lifestyle_tips: list[str]
    supplement_tips: list[str]
    cautions: str | None
    trend_analysis: TrendAnalysis | None = None


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


class InBodyCreate(InBodyMetricsBase):
    weight: float = Field(..., gt=0, lt=500)
    image_url: str | None = None


class InBodyUpdate(InBodyMetricsBase):
    pass


class InBodyResponse(InBodyCreate):
    id: str
    user_id: str
    ai_report: HealthReport | None = None


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


class InBodyUploadResponse(BaseModel):
    """업로드+추출 직후 응답. 분석은 아직 안 됨."""

    record: InBodyResponse
    extraction_confidence: float
    extraction_notes: str | None = None
