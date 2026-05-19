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
    is_shap_important: bool


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
    basal_metabolic_rate: int | None = Field(None, ge=0)
    abs_fat: float | None = Field(None, ge=0)
    fat_level: int | None = Field(None, ge=0)


class InBodyCreate(InBodyMetricsBase):
    weight: float = Field(..., gt=0, lt=500)
    image_url: str | None = None


class InBodyUpdate(InBodyMetricsBase):
    pass


class InBodyResponse(InBodyCreate):
    id: str
    user_id: str
    ai_report: HealthReport | None = None


class InBodyExtraction(InBodyMetricsBase):
    """Gemini Vision이 이미지/PDF에서 추출하는 구조화 결과."""

    is_inbody: bool = Field(..., description="이 이미지가 인바디 결과지로 보이는지")
    confidence: float = Field(..., ge=0, le=1, description="전체 추출 신뢰도 0~1")
    notes: str | None = Field(None, description="누락 또는 애매한 값에 대한 메모")


class InBodyUploadResponse(BaseModel):
    """업로드+추출 직후 응답. 분석은 아직 안 됨."""

    record: InBodyResponse
    extraction_confidence: float
    extraction_notes: str | None = None
