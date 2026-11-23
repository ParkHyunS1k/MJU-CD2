from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.core.supabase import get_supabase_admin
from app.deps.auth import CurrentUser, get_current_user
from app.schemas.inbody import (
    InBodyCreate,
    InBodyResponse,
    InBodyUpdate,
    InBodyUploadResponse,
)
from app.services.inbody_service import (
    InBodyServiceError,
    create_inbody_from_upload,
    create_inbody_record,
    delete_inbody_record,
    get_inbody_record,
    list_inbody_records,
    update_inbody_record,
    validate_upload,
)
from app.services.risk_store import trend as risk_trend
from app.services.trend_analysis_service import analyze_trend

router = APIRouter(prefix="/inbody", tags=["inbody"])


def _raise_service_error(error: InBodyServiceError) -> None:
    raise HTTPException(error.status_code, error.detail)


@router.post("", response_model=InBodyResponse, status_code=status.HTTP_201_CREATED)
def create_inbody(
    payload: InBodyCreate,
    current: CurrentUser = Depends(get_current_user),
):
    try:
        return create_inbody_record(current.id, payload)
    except InBodyServiceError as e:
        _raise_service_error(e)


@router.get("", response_model=list[InBodyResponse])
def list_inbody(
    limit: int = Query(20, ge=1, le=100),
    current: CurrentUser = Depends(get_current_user),
):
    return list_inbody_records(current.id, limit)


@router.get("/trend")
def get_inbody_trend(
    current: CurrentUser = Depends(get_current_user),
):
    """로그인한 사용자의 전체 측정 기록으로 변화 추세를 계산한다."""
    return analyze_trend(current.id)


@router.get("/risk-trend")
def get_risk_trend(
    current: CurrentUser = Depends(get_current_user),
):
    """기록별 질환 위험도(측정일 오름차순). 위험도 추이 그래프용."""
    return risk_trend(get_supabase_admin(), current.id)


@router.get("/{inbody_id}", response_model=InBodyResponse)
def get_inbody(
    inbody_id: str,
    current: CurrentUser = Depends(get_current_user),
):
    try:
        return get_inbody_record(current.id, inbody_id)
    except InBodyServiceError as e:
        _raise_service_error(e)


@router.patch("/{inbody_id}", response_model=InBodyResponse)
def update_inbody(
    inbody_id: str,
    payload: InBodyUpdate,
    current: CurrentUser = Depends(get_current_user),
):
    try:
        return update_inbody_record(current.id, inbody_id, payload)
    except InBodyServiceError as e:
        _raise_service_error(e)


@router.delete("/{inbody_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_inbody(
    inbody_id: str,
    current: CurrentUser = Depends(get_current_user),
):
    try:
        delete_inbody_record(current.id, inbody_id)
    except InBodyServiceError as e:
        _raise_service_error(e)


@router.post(
    "/upload",
    response_model=InBodyUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_inbody(
    file: UploadFile = File(...),
    current: CurrentUser = Depends(get_current_user),
):
    content_type = (file.content_type or "").lower()
    file_bytes = await file.read()

    try:
        validate_upload(content_type, file_bytes)
        return await run_in_threadpool(
            create_inbody_from_upload,
            current.id,
            file_bytes,
            content_type,
        )
    except InBodyServiceError as e:
        _raise_service_error(e)
    except ValueError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e)) from e
