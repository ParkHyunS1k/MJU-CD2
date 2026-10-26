from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from app.core.supabase import get_supabase_admin
from app.deps.auth import CurrentUser, get_current_user
from app.schemas.inbody import (
    InBodyCreate,
    InBodyResponse,
    InBodyUpdate,
    InBodyUploadResponse,
)
from app.services.ai_service import extract_inbody_from_image
from app.services.storage_service import EXT_BY_MIME, upload_inbody_file
from app.services.trend_service import analyze_trend

router = APIRouter(prefix="/inbody", tags=["inbody"])

TABLE = "inbody_logs"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


@router.post("", response_model=InBodyResponse, status_code=status.HTTP_201_CREATED)
def create_inbody(
    payload: InBodyCreate,
    current: CurrentUser = Depends(get_current_user),
):
    sb = get_supabase_admin()
    data = payload.model_dump(mode="json", exclude_none=True)
    data["user_id"] = current.id
    res = sb.table(TABLE).insert(data).execute()
    return res.data[0]


@router.get("", response_model=list[InBodyResponse])
def list_inbody(
    limit: int = Query(20, ge=1, le=100),
    current: CurrentUser = Depends(get_current_user),
):
    sb = get_supabase_admin()
    res = (
        sb.table(TABLE)
        .select("*")
        .eq("user_id", current.id)
        .order("measured_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data


@router.get("/trend")
def get_inbody_trend(
    current: CurrentUser = Depends(get_current_user),
):
    """로그인한 사용자의 전체 측정 기록으로 변화 추세를 계산한다."""
    return analyze_trend(current.id)


@router.get("/{inbody_id}", response_model=InBodyResponse)
def get_inbody(
    inbody_id: str,
    current: CurrentUser = Depends(get_current_user),
):
    sb = get_supabase_admin()
    res = (
        sb.table(TABLE)
        .select("*")
        .eq("id", inbody_id)
        .eq("user_id", current.id)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "InBody record not found")
    return res.data[0]


@router.patch("/{inbody_id}", response_model=InBodyResponse)
def update_inbody(
    inbody_id: str,
    payload: InBodyUpdate,
    current: CurrentUser = Depends(get_current_user),
):
    """추출된 수치를 사용자가 수동 보정할 때 사용."""
    sb = get_supabase_admin()
    data = payload.model_dump(mode="json", exclude_none=True)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No fields to update")
    res = (
        sb.table(TABLE)
        .update(data)
        .eq("id", inbody_id)
        .eq("user_id", current.id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "InBody record not found")
    return res.data[0]


@router.delete("/{inbody_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_inbody(
    inbody_id: str,
    current: CurrentUser = Depends(get_current_user),
):
    sb = get_supabase_admin()
    sb.table(TABLE).delete().eq("id", inbody_id).eq("user_id", current.id).execute()


@router.post(
    "/upload",
    response_model=InBodyUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_inbody(
    file: UploadFile = File(...),
    current: CurrentUser = Depends(get_current_user),
):
    """인바디 결과지(이미지/PDF)를 업로드하면 Gemini Vision으로 수치를 추출하고 새 row를 만든다.
    분석은 별도로 POST /ai/inbody/analyze 를 호출해야 한다.
    """
    content_type = (file.content_type or "").lower()
    if content_type not in EXT_BY_MIME:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type: {content_type}. Allowed: {list(EXT_BY_MIME)}",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File too large (max {MAX_UPLOAD_BYTES // 1024 // 1024} MB)",
        )

    extraction = extract_inbody_from_image(file_bytes, content_type)

    if not extraction.get("is_inbody"):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "인바디 결과지로 보이지 않습니다. 다른 파일을 업로드해주세요.",
        )
    if not extraction.get("weight"):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "체중 값을 추출하지 못했습니다. 다른 사진으로 시도하거나 직접 입력해주세요.",
        )

    image_path = upload_inbody_file(current.id, file_bytes, content_type)

    record_data = {
        "user_id": current.id,
        "image_url": image_path,
    }
    for key in (
        "measured_at",
        "weight",
        "skeletal_muscle_mass",
        "body_fat_mass",
        "body_fat_percentage",
        "body_water",
        "protein",
        "minerals",
        "bmi",
        "basal_metabolic_rate",
        "abs_fat",
        "fat_level",
    ):
        value = extraction.get(key)
        if value is not None:
            record_data[key] = value

    sb = get_supabase_admin()
    res = sb.table(TABLE).insert(record_data).execute()

    return InBodyUploadResponse(
        record=res.data[0],
        extraction_confidence=extraction["confidence"],
        extraction_notes=extraction.get("notes"),
    )
