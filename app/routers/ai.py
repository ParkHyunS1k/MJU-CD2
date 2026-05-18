import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.supabase import get_supabase_admin
from app.deps.auth import CurrentUser, get_current_user
from app.schemas.inbody import HealthReport
from app.services.ai_service import generate_report
from app.services.prediction_service import predict_all
from app.services.storage_service import delete_inbody_file

router = APIRouter(prefix="/ai", tags=["ai"])
log = logging.getLogger(__name__)


@router.post("/inbody/analyze", response_model=HealthReport)
def analyze_my_inbody(current_user: CurrentUser = Depends(get_current_user)):
    """최신 인바디 측정을 ML 모델로 분석해 4개 성인병 위험도와 생활습관 추천을 반환한다.
    결과는 해당 row의 ai_report 컬럼에 저장되고, 원본 이미지는 삭제된다.
    성별 정보가 없으면 400을 반환한다.
    """
    sb = get_supabase_admin()

    user_res = (
        sb.table("users").select("*").eq("id", current_user.id).limit(1).execute()
    )
    if not user_res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    user = user_res.data[0]

    if not user.get("gender"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "성별 정보를 먼저 입력해주세요. PATCH /users/me 로 gender를 설정하세요.",
        )

    log_res = (
        sb.table("inbody_logs")
        .select("*")
        .eq("user_id", current_user.id)
        .order("measured_at", desc=True)
        .limit(1)
        .execute()
    )
    if not log_res.data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No InBody records yet")

    latest = log_res.data[0]

    try:
        disease_risks = predict_all(inbody=latest, user_profile=user)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    if not disease_risks:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "인바디 수치가 부족해 위험도를 계산할 수 없습니다. 수치를 보완 후 다시 시도하세요.",
        )

    gemini_result = generate_report(disease_risks=disease_risks, user_profile=user)

    report = HealthReport(
        disease_risks=disease_risks,
        lifestyle_tips=gemini_result["lifestyle_tips"],
        supplement_tips=gemini_result["supplement_tips"],
        cautions=gemini_result.get("cautions"),
    )

    update_payload: dict = {"ai_report": report.model_dump()}
    image_path = latest.get("image_url")
    if image_path:
        try:
            delete_inbody_file(image_path)
            update_payload["image_url"] = None
        except Exception:
            log.exception("failed to delete inbody image %s", image_path)

    sb.table("inbody_logs").update(update_payload).eq("id", latest["id"]).execute()

    return report
