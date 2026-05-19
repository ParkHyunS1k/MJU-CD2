import logging

from app.core.supabase import get_supabase_admin
from app.schemas.inbody import HealthReport
from app.services.ai_service import generate_report
from app.services.prediction_service import predict_all
from app.services.storage_service import delete_inbody_file
from app.services.trend_service import compute_trends, is_eligible

log = logging.getLogger(__name__)


class AnalysisServiceError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def analyze_latest_inbody(user_id: str) -> HealthReport:
    sb = get_supabase_admin()
    user = _get_user(sb, user_id)
    if not user.get("gender"):
        raise AnalysisServiceError(
            400,
            "성별 정보를 먼저 입력해주세요. PATCH /users/me 로 gender를 설정하세요.",
        )

    records = _list_inbody_records(sb, user_id)
    if not records:
        raise AnalysisServiceError(400, "No InBody records yet")

    latest = records[0]
    try:
        disease_risks = predict_all(inbody=latest, user_profile=user)
    except ValueError as e:
        raise AnalysisServiceError(400, str(e)) from e

    if not disease_risks:
        raise AnalysisServiceError(
            422,
            "인바디 수치가 부족해 위험도를 계산할 수 없습니다. 수치를 보완 후 다시 시도하세요.",
        )

    trend_analysis = _build_trend_analysis(records, disease_risks)

    try:
        gemini_result = generate_report(
            disease_risks=disease_risks,
            user_profile=user,
            trend_analysis=trend_analysis,
        )
    except ValueError as e:
        raise AnalysisServiceError(502, str(e)) from e

    report = HealthReport(
        disease_risks=disease_risks,
        lifestyle_tips=gemini_result["lifestyle_tips"],
        supplement_tips=gemini_result["supplement_tips"],
        cautions=gemini_result.get("cautions"),
        trend_analysis=trend_analysis,
    )

    _save_report_and_cleanup_image(sb, latest, report)
    return report


def _get_user(sb, user_id: str) -> dict:
    res = sb.table("users").select("*").eq("id", user_id).limit(1).execute()
    if not res.data:
        raise AnalysisServiceError(404, "User not found")
    return res.data[0]


def _list_inbody_records(sb, user_id: str) -> list[dict]:
    res = (
        sb.table("inbody_logs")
        .select("*")
        .eq("user_id", user_id)
        .order("measured_at", desc=True)
        .execute()
    )
    return res.data


def _build_trend_analysis(records: list[dict], disease_risks: list[dict]) -> dict | None:
    if not is_eligible(records):
        return None

    shap_feature_directions: dict[str, set[str]] = {}
    for risk in disease_risks:
        for factor in risk["top_factors"]:
            shap_feature_directions.setdefault(factor["feature_kor"], set()).add(
                factor["direction"]
            )

    return compute_trends(records, shap_feature_directions)


def _save_report_and_cleanup_image(sb, latest: dict, report: HealthReport) -> None:
    update_payload: dict = {"ai_report": report.model_dump(mode="json")}
    image_path = latest.get("image_url")
    if image_path:
        try:
            delete_inbody_file(image_path)
            update_payload["image_url"] = None
        except Exception:
            log.exception("failed to delete inbody image %s", image_path)

    sb.table("inbody_logs").update(update_payload).eq("id", latest["id"]).execute()
