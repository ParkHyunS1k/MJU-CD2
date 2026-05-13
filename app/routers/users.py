from fastapi import APIRouter, Depends, HTTPException, status

from app.core.supabase import get_supabase_admin
from app.deps.auth import CurrentUser, get_current_user
from app.schemas.user import UserResponse, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


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
    sb = get_supabase_admin()
    update_data = payload.model_dump(exclude_none=True)
    res = sb.table("users").update(update_data).eq("id", current.id).execute()
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return res.data[0]
