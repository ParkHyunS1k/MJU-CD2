import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.supabase import get_supabase_admin
from app.deps.auth import CurrentUser, get_current_user
from app.schemas.inbody import AIReport
from app.services.ai_service import analyze_inbody
from app.services.storage_service import delete_inbody_file

router = APIRouter(prefix="/ai", tags=["ai"])
log = logging.getLogger(__name__)


@router.post("/inbody/analyze", response_model=AIReport)
def analyze_my_inbody(current_user: CurrentUser = Depends(get_current_user)):
    """가장 최근 인바디 측정에 대한 AI 분석을 생성해 해당 row의 ai_report 컬럼에 저장한다.
    직전 측정이 있으면 함께 전달해 변화 추이를 분석한다.
    분석 완료 후 image_url 이 있으면 Storage의 파일을 삭제하고 컬럼은 null 로 비운다.
    """
    sb = get_supabase_admin()

    user_res = (
        sb.table("users").select("*").eq("id", current_user.id).limit(1).execute()
    )
    if not user_res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    user = user_res.data[0]

    logs_res = (
        sb.table("inbody_logs")
        .select("*")
        .eq("user_id", current_user.id)
        .order("measured_at", desc=True)
        .limit(2)
        .execute()
    )
    if not logs_res.data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No InBody records yet")

    latest = logs_res.data[0]
    previous = logs_res.data[1] if len(logs_res.data) > 1 else None

    report = analyze_inbody(current=latest, previous=previous, user_profile=user)

    update_payload: dict = {"ai_report": report}
    image_path = latest.get("image_url")
    if image_path:
        try:
            delete_inbody_file(image_path)
            update_payload["image_url"] = None
        except Exception:
            log.exception("failed to delete inbody image %s", image_path)

    sb.table("inbody_logs").update(update_payload).eq("id", latest["id"]).execute()

    return report
