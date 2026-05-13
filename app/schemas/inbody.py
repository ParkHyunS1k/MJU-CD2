from datetime import datetime

from pydantic import BaseModel, Field


class AIReport(BaseModel):
    summary: str = Field(..., description="현재 측정의 한 문단 요약")
    comparison: str = Field(
        ...,
        description="직전 측정 대비 변화. 직전 데이터가 없으면 '첫 측정이라 비교 데이터 없음' 으로 명시.",
    )
    exercise_guide: list[str] = Field(..., description="다음 2주 운동 가이드 항목들")
    diet_guide: list[str] = Field(..., description="다음 2주 식단 가이드 항목들")
    cautions: str | None = Field(None, description="건강상 주의사항 (없으면 null)")


class InBodyCreate(BaseModel):
    measured_at: datetime | None = None
    weight: float = Field(..., gt=0, lt=500)
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
    image_url: str | None = None


class InBodyUpdate(BaseModel):
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


class InBodyResponse(InBodyCreate):
    id: str
    user_id: str
    ai_report: AIReport | None = None


class InBodyExtraction(BaseModel):
    """Gemini Vision이 이미지/PDF에서 추출하는 구조화 결과."""

    is_inbody: bool = Field(..., description="이 이미지가 인바디 결과지로 보이는지")
    measured_at: datetime | None = Field(None, description="결과지에 적힌 측정일시(있으면)")
    weight: float | None = Field(None, description="체중 kg")
    skeletal_muscle_mass: float | None = Field(None, description="골격근량 kg")
    body_fat_mass: float | None = Field(None, description="체지방량 kg")
    body_fat_percentage: float | None = Field(None, description="체지방률 %")
    body_water: float | None = Field(None, description="체수분 L")
    protein: float | None = Field(None, description="단백질 kg")
    minerals: float | None = Field(None, description="무기질 kg")
    bmi: float | None = Field(None, description="BMI")
    basal_metabolic_rate: int | None = Field(None, description="기초대사량 kcal")
    abs_fat: float | None = Field(None, description="복부지방률")
    fat_level: int | None = Field(None, description="내장지방 레벨")
    confidence: float = Field(..., ge=0, le=1, description="전체 추출 신뢰도 0~1")
    notes: str | None = Field(None, description="누락 또는 애매한 값에 대한 메모")


class InBodyUploadResponse(BaseModel):
    """업로드+추출 직후 응답. 분석은 아직 안 됨."""

    record: InBodyResponse
    extraction_confidence: float
    extraction_notes: str | None = None
