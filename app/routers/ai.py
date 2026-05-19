from fastapi import APIRouter, Depends, HTTPException

from app.deps.auth import CurrentUser, get_current_user
from app.schemas.inbody import HealthReport
from app.services.inbody_analysis_service import (
    AnalysisServiceError,
    analyze_latest_inbody,
)

router = APIRouter(prefix="/ai", tags=["ai"])


@router.post("/inbody/analyze", response_model=HealthReport)
def analyze_my_inbody(current_user: CurrentUser = Depends(get_current_user)):
    """최신 인바디 측정을 ML 모델로 분석하고 생활습관 추천을 반환한다."""
    try:
        return analyze_latest_inbody(current_user.id)
    except AnalysisServiceError as e:
        raise HTTPException(e.status_code, e.detail) from e
