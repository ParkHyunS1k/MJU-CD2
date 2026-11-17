import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.supabase import get_supabase_admin
from app.deps.auth import CurrentUser, get_current_user
from app.schemas.user import UserResponse, UserUpdate
from app.services.risk_store import refresh_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])
# 위험도 계산에 쓰는 프로필 항목. 바뀌면 그 사용자의 기록별 위험도를 다시 계산한다
RISK_PROFILE_FIELDS = {"gender", "birth_year", "fh_hypertension", "fh_diabetes", "fh_dyslipidemia"}


@router.get("/me", response_model=UserResponse)
def get_me(current: CurrentUser = Depends(get_current_user)):
    sb = get_supabase_admin()
    res = sb.table("users").select("*").eq("id", current.id).limit(1).execute()
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return res.data[0]


@router.patch("/me", response_model=UserResponse)
def update_me(
    payload: UserUpdate,
    current: CurrentUser = Depends(get_current_user),
):
    # 보낸 필드만 저장한다. null로 보낸 값도 저장한다(가족력 '모름' 등).
    update_data = payload.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No fields to update")
    sb = get_supabase_admin()
    res = sb.table("users").update(update_data).eq("id", current.id).execute()
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if update_data.keys() & RISK_PROFILE_FIELDS:
        try:
            refresh_user(sb, current.id, res.data[0])
        except Exception:  # 프로필 저장이 우선. 낡은 위험도는 읽을 때 입력 비교로 바로잡힌다
            log.exception("failed to refresh risks for user %s", current.id)
    return res.data[0]
