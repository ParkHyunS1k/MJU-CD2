"""기록별 질환 위험도를 계산에 쓴 입력과 함께 inbody_logs에 저장한다.
저장값은 입력이 지금 프로필·기록 값과 같을 때만 믿는다. 다르면 다시 계산해 저장한다(계산 실패 뒤 남은 옛 값도 여기서 바로잡힌다).
설계: docs/superpowers/specs/2026-11-15-risk-change-design.md"""
import logging

from app.services.prediction_service import FAMILY_HISTORY, measurement_year, predict_all

log = logging.getLogger(__name__)

TABLE = "inbody_logs"
COLUMN = "disease_risks"


def _num(value) -> float | None:
    return None if value is None else float(value)


def inputs_of(record: dict, profile: dict) -> dict:
    return {
        "gender": profile.get("gender"),
        "birth_year": profile.get("birth_year"),
        **{k: profile.get(k) for k in FAMILY_HISTORY},
        "measured_year": measurement_year(record),
        "bmi": _num(record.get("bmi")),
        "body_fat_percentage": _num(record.get("body_fat_percentage")),
    }


def from_predictions(record: dict, profile: dict, results: list[dict]) -> dict:
    inputs = inputs_of(record, profile)
    return {
        "inputs": inputs,
        "age": inputs["measured_year"] - int(inputs["birth_year"]),
        "diseases": [{k: r[k] for k in ("disease", "disease_key", "probability", "risk_level", "contributions")}
                     for r in results],
    }


def compute(record: dict, profile: dict) -> dict | None:
    """계산할 수 없으면(수치 없음, 범위 밖, 프로필 부족, 측정 당시 미성년) None."""
    try:
        results = predict_all(record, profile)
    except ValueError:
        return None
    return from_predictions(record, profile, results) if results else None


def is_current(stored: dict | None, record: dict, profile: dict) -> bool:
    return bool(stored) and stored.get("inputs") == inputs_of(record, profile)


def save(sb, record_id: str, value: dict | None) -> None:
    try:
        sb.table(TABLE).update({COLUMN: value}).eq("id", record_id).execute()
    except Exception:
        log.exception("failed to save risks for record %s", record_id)


def ensure(sb, record: dict, profile: dict) -> dict | None:
    stored = record.get(COLUMN)
    if is_current(stored, record, profile):
        return stored
    value = compute(record, profile)
    save(sb, record["id"], value)
    return value


def refresh_record(sb, record: dict, profile: dict | None = None) -> None:
    """기록·프로필이 바뀐 뒤 부른다. 실패해도 원래 요청을 막지 않는다(읽는 쪽이 입력 비교로 바로잡음)."""
    try:
        if profile is None:
            res = sb.table("users").select("*").eq("id", record["user_id"]).limit(1).execute()
            profile = res.data[0] if res.data else {}
        save(sb, record["id"], compute(record, profile))
    except Exception:
        log.exception("failed to refresh risks for record %s", record.get("id"))


def refresh_user(sb, user_id: str, profile: dict) -> int:
    """그 사용자의 기록 수를 돌려준다."""
    res = sb.table(TABLE).select("*").eq("user_id", user_id).execute()
    for record in res.data:
        refresh_record(sb, record, profile)
    return len(res.data)


def trend(sb, user_id: str) -> list[dict]:
    """측정일 오름차순으로 기록별 위험도. 계산할 수 없는 기록은 뺀다."""
    users = sb.table("users").select("*").eq("id", user_id).limit(1).execute().data
    if not users:
        return []
    records = sb.table(TABLE).select("*").eq("user_id", user_id).order("measured_at").execute().data
    out = []
    for record in sorted(records, key=lambda r: r.get("measured_at") or ""):
        stored = ensure(sb, record, users[0])
        if stored:
            out.append({"inbody_id": record["id"], "measured_at": record.get("measured_at"),
                        "risks": [{k: d[k] for k in ("disease", "disease_key", "probability", "risk_level")}
                                  for d in stored["diseases"]]})
    return out


def backfill(sb) -> int:
    """모든 사용자의 기록을 다시 계산해 저장한다. 처리한 기록 수."""
    return sum(refresh_user(sb, user["id"], user) for user in sb.table("users").select("*").execute().data)


if __name__ == "__main__":
    from app.core.supabase import get_supabase_admin

    print(f"{backfill(get_supabase_admin())} records refreshed")
