import logging

from app.core.supabase import get_supabase_admin
from app.schemas.inbody import InBodyCreate, InBodyUpdate, InBodyUploadResponse
from app.services.ai_service import extract_inbody_from_image
from app.services.risk_store import refresh_record
from app.services.storage_service import EXT_BY_MIME, delete_inbody_file, upload_inbody_file

TABLE = "inbody_logs"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

INBODY_VALUE_FIELDS = (
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
    "height",
    "muscle_control",
    "inbody_score",
    "target_weight",
    "weight_control",
    "fat_control",
    "fat_free_mass",
    "obesity_degree",
    "recommended_calories",
    "bmi_evaluation",
    "body_fat_evaluation",
    "segmental_lean",
    "segmental_fat",
    "impedance",
)


log = logging.getLogger(__name__)


class InBodyServiceError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def create_inbody_record(user_id: str, payload: InBodyCreate) -> dict:
    sb = get_supabase_admin()
    data = payload.model_dump(mode="json", exclude_none=True)
    data["user_id"] = user_id
    res = sb.table(TABLE).insert(data).execute()
    if not res.data:
        raise InBodyServiceError(500, "Failed to create InBody record")
    _refresh_risks(sb, res.data[0])
    return _normalize_record(res.data[0])


def list_inbody_records(user_id: str, limit: int) -> list[dict]:
    sb = get_supabase_admin()
    res = (
        sb.table(TABLE)
        .select("*")
        .eq("user_id", user_id)
        .order("measured_at", desc=True)
        .limit(limit)
        .execute()
    )
    return [_normalize_record(record) for record in res.data]


def get_inbody_record(user_id: str, inbody_id: str) -> dict:
    sb = get_supabase_admin()
    res = (
        sb.table(TABLE)
        .select("*")
        .eq("id", inbody_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise InBodyServiceError(404, "InBody record not found")
    return _normalize_record(res.data[0])


def update_inbody_record(user_id: str, inbody_id: str, payload: InBodyUpdate) -> dict:
    sb = get_supabase_admin()
    data = payload.model_dump(mode="json", exclude_none=True)
    if not data:
        raise InBodyServiceError(400, "No fields to update")

    res = (
        sb.table(TABLE)
        .update(data)
        .eq("id", inbody_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not res.data:
        raise InBodyServiceError(404, "InBody record not found")
    _refresh_risks(sb, res.data[0])
    return _normalize_record(res.data[0])


def delete_inbody_record(user_id: str, inbody_id: str) -> None:
    sb = get_supabase_admin()
    existing = (
        sb.table(TABLE)
        .select("image_url")
        .eq("id", inbody_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not existing.data:
        raise InBodyServiceError(404, "InBody record not found")

    sb.table(TABLE).delete().eq("id", inbody_id).eq("user_id", user_id).execute()
    image_path = existing.data[0].get("image_url")
    if image_path:
        delete_inbody_file(image_path)


def validate_upload(content_type: str, file_bytes: bytes) -> None:
    if content_type not in EXT_BY_MIME:
        raise InBodyServiceError(
            415,
            f"Unsupported file type: {content_type}. Allowed: {list(EXT_BY_MIME)}",
        )
    if len(file_bytes) == 0:
        raise InBodyServiceError(400, "Empty file")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise InBodyServiceError(
            413,
            f"File too large (max {MAX_UPLOAD_BYTES // 1024 // 1024} MB)",
        )


def create_inbody_from_upload(
    user_id: str,
    file_bytes: bytes,
    content_type: str,
) -> InBodyUploadResponse:
    extraction = extract_inbody_from_image(file_bytes, content_type)

    if not extraction.get("is_inbody"):
        raise InBodyServiceError(
            422,
            "인바디 결과지로 보이지 않습니다. 다른 파일을 업로드해주세요.",
        )
    if not extraction.get("weight"):
        raise InBodyServiceError(
            422,
            "체중 값을 추출하지 못했습니다. 다른 사진으로 시도하거나 직접 입력해주세요.",
        )

    image_path = upload_inbody_file(user_id, file_bytes, content_type)
    record_data = _record_data_from_extraction(user_id, image_path, extraction)

    sb = get_supabase_admin()
    try:
        res = sb.table(TABLE).insert(record_data).execute()
    except Exception:
        delete_inbody_file(image_path)
        raise

    if not res.data:
        delete_inbody_file(image_path)
        raise InBodyServiceError(500, "Failed to create InBody record")

    _refresh_risks(sb, res.data[0])
    return InBodyUploadResponse(
        record=_normalize_record(res.data[0]),
        extraction_confidence=extraction["confidence"],
        extraction_notes=extraction.get("notes"),
    )


def _refresh_risks(sb, record: dict) -> None:
    """기록 저장이 우선이라, 위험도 저장이 실패해도 요청은 성공시킨다."""
    try:
        refresh_record(sb, record)
    except Exception:
        log.exception("failed to refresh risks for record %s", record.get("id"))


def _normalize_record(record: dict) -> dict:
    normalized = dict(record)
    normalized["ai_report"] = _normalize_ai_report(normalized.get("ai_report"))
    return normalized


def _normalize_ai_report(report: dict | None) -> dict | None:
    if not report or "summary" in report:
        return report

    disease_risks = report.get("disease_risks", [])
    lifestyle_tips = report.get("lifestyle_tips", [])
    supplement_tips = report.get("supplement_tips", [])
    cautions = report.get("cautions")

    return {
        "summary": {
            "title": "이전 AI 건강 리포트",
            "overall_message": "이전 형식으로 저장된 리포트를 새 화면 구조에 맞춰 표시합니다.",
            "key_focus": lifestyle_tips[:3],
        },
        "body_composition_report": {
            "strengths": [],
            "watch_points": [],
        },
        "risk_report": [
            {
                "disease": risk.get("disease", "unknown"),
                "risk_level": risk.get("risk_level", "low"),
                "probability": risk.get("probability", 0),
                "main_factors": [
                    factor.get("feature_kor", "")
                    for factor in risk.get("top_factors", [])
                    if factor.get("feature_kor")
                ],
                "message": "이전 리포트의 성인병 위험도 결과입니다.",
            }
            for risk in disease_risks
        ],
        "lifestyle_plan": [
            {
                "category": "habit",
                "title": "생활습관 추천",
                "reason": "이전 리포트의 생활습관 추천 항목입니다.",
                "actions": lifestyle_tips,
            }
        ] if lifestyle_tips else [],
        "supplement_candidates": [
            {
                "name": f"보충제 후보 {idx + 1}",
                "purpose": tip,
                "related_metrics": [],
                "evidence_level": "limited",
                "why_consider": tip,
                "cautions": [],
                "not_recommended_if": [],
            }
            for idx, tip in enumerate(supplement_tips)
        ],
        "cautions": {
            "medical_disclaimer": cautions or "의학적 진단이 아니며 필요 시 전문가와 상담하세요.",
            "consult_doctor_if": [],
        },
        "disease_risks": disease_risks,
        "trend_analysis": report.get("trend_analysis"),
    }


def _record_data_from_extraction(
    user_id: str,
    image_path: str,
    extraction: dict,
) -> dict:
    record_data = {
        "user_id": user_id,
        "image_url": image_path,
    }
    for key in INBODY_VALUE_FIELDS:
        value = extraction.get(key)
        if value is not None:
            record_data[key] = value
    return record_data
