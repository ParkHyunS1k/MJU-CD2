import logging

from app.core.supabase import get_supabase_admin
from app.services import risk_change, risk_store
from app.schemas.inbody import HealthReport
from app.services.ai_service import generate_report
from app.services.citation_service import validate_citations
from app.services.personal_metrics import compute as compute_personal_metrics
from app.services.prediction_service import predict_all
from app.services.rag_taxonomy import build_queries, trait_queries
from app.services.retrieval_service import RetrievalError, retrieve
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

    return _analyze_inbody_record(sb, user, records, records[0])


def analyze_inbody_record(user_id: str, inbody_id: str) -> HealthReport:
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

    target = next((record for record in records if record["id"] == inbody_id), None)
    if not target:
        raise AnalysisServiceError(404, "InBody record not found")

    return _analyze_inbody_record(sb, user, records, target)


def _analyze_inbody_record(
    sb,
    user: dict,
    records: list[dict],
    target: dict,
) -> HealthReport:
    try:
        disease_risks = predict_all(inbody=target, user_profile=user)
    except ValueError as e:
        raise AnalysisServiceError(400, str(e)) from e

    if not disease_risks:
        raise AnalysisServiceError(
            422,
            "인바디 수치가 부족해 위험도를 계산할 수 없습니다. 수치를 보완 후 다시 시도하세요.",
        )

    # 대상 기록은 오류 메시지를 지키려고 직접 계산한다. 같은 결정적 계산이라 저장값과 같고, 낡았으면 이 값으로 바꿔 둔다
    current = risk_store.from_predictions(target, user, disease_risks)
    if not risk_store.is_current(target.get(risk_store.COLUMN), target, user):
        risk_store.save(sb, target["id"], current)
    previous = _previous_risks(sb, user, records, target)
    changes = risk_change.compute(previous[1], current, previous[0]["measured_at"]) if previous else None

    trend_analysis = _build_trend_analysis(records, disease_risks)

    metrics = compute_personal_metrics(target, user)
    # 상한이 쿼리 수 이상이라 모든 쿼리의 1위는 순서와 관계없이 들어간다. 질환 쿼리를 앞에 둬
    # 특성의 2위 근거가 질환의 2·3위 근거를 밀어내지 않게 한다
    queries = [*build_queries(disease_risks, target, user.get("gender")),
               *trait_queries(disease_risks, user, metrics, trend_analysis)]
    try:
        evidence = retrieve(queries)
    except RetrievalError as e:
        log.exception("failed to retrieve RAG evidence")
        raise AnalysisServiceError(503, "참고 자료를 불러오지 못했습니다. 잠시 후 다시 분석해주세요.") from e
    if not evidence:
        log.warning("RAG 근거 0개: %s", [q.text for q in queries])

    try:
        gemini_result = generate_report(
            disease_risks=disease_risks,
            user_profile=user,
            trend_analysis=trend_analysis,
            inbody_metrics=target,
            evidence=evidence,
            previous_report=_previous_report_summary(records, target),
            personal_metrics=metrics,
            risk_change=changes,
        )
    except ValueError as e:
        raise AnalysisServiceError(502, str(e)) from e
    except Exception as e:
        log.exception("failed to generate AI report")
        raise AnalysisServiceError(502, "AI 리포트 생성 중 오류가 발생했습니다.") from e

    content, sources = validate_citations(gemini_result, evidence)
    try:
        report = HealthReport(
            disease_risks=disease_risks,
            trend_analysis=trend_analysis,
            sources=sources,
            **content,
        )
    except Exception as e:
        log.exception("failed to build health report")
        raise AnalysisServiceError(502, "AI 리포트 응답 형식이 올바르지 않습니다.") from e

    try:
        _save_report_and_cleanup_image(sb, target, report)
    except Exception as e:
        log.exception("failed to save AI report")
        raise AnalysisServiceError(502, "AI 리포트 저장 중 오류가 발생했습니다.") from e
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


def _previous_risks(sb, user: dict, records: list[dict], target: dict) -> tuple[dict, dict] | None:
    """대상보다 먼저 측정했고 위험도를 계산할 수 있는 가장 최근 기록과 그 저장값. 리포트 유무는 보지 않는다."""
    measured = target.get("measured_at") or ""
    earlier = sorted((r for r in records if r.get("measured_at") and r["measured_at"] < measured),
                     key=lambda r: r["measured_at"], reverse=True)
    for record in earlier:
        stored = risk_store.ensure(sb, record, user)
        if stored:
            return record, stored
    return None


def _previous_report_summary(records: list[dict], target: dict) -> dict | None:
    """대상보다 먼저 측정했고 리포트가 있는 가장 최근 기록의 요약. 옛 스키마도 읽을 수 있게 get으로 꺼낸다."""
    measured = target.get("measured_at") or ""
    earlier = [r for r in records if r.get("ai_report") and r.get("measured_at") and r["measured_at"] < measured]
    if not earlier:
        return None
    report = max(earlier, key=lambda r: r["measured_at"])["ai_report"]
    return {
        "key_focus": (report.get("summary") or {}).get("key_focus", []),
        "lifestyle_titles": [p.get("title") for p in report.get("lifestyle_plan") or []],
        "supplements": [s.get("name") for s in report.get("supplement_candidates") or []],
    }


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
