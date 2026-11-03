"""인바디 측정 기록의 변화 추세를 분석한다.

- 지표별 처음 대비 변화량, 월별 변화 속도(추세선 기울기)
- 측정 오차를 고려한 개선/유지/악화 판정
숫자 계산은 여기서 하고, 설명은 ai_service(Gemini)가 담당한다.
"""

from datetime import date

import numpy as np

from app.core.supabase import get_supabase_admin

# TODO: routers/inbody.py 에서 쓰는 실제 테이블 이름으로 바꾸기
TABLE = "inbody_logs"  # routers/inbody.py 와 같은 테이블

DATE_FIELD = "measured_at"          # schemas/inbody.py 의 측정일 필드
FALLBACK_DATE_FIELD = "created_at"  # 측정일이 비어 있으면 등록 시각으로 대체

# 지표별 '좋아지는 방향'
#  -1: 감소가 좋음 / +1: 증가가 좋음 / 0: 좋고 나쁨을 판단하지 않음(증가·감소만 표시)
DIRECTION = {
    "body_fat_percentage": -1,   # 체지방률 (%)
    "body_fat_mass": -1,         # 체지방량 (kg)
    "skeletal_muscle_mass": +1,  # 골격근량 (kg)
    "abs_fat": -1,               # 복부지방률
    "fat_level": -1,             # 내장지방 레벨
    "weight": 0,                 # 체중 (kg) — 근육 증가로도 늘 수 있어 판정하지 않음
}

# 이 값보다 작은 변화는 측정 오차로 보고 '유지'로 판정
# TODO: 임시값 — InBody 측정 재현성 근거로 팀에서 확정
THRESHOLD = {
    "body_fat_percentage": 1.0,
    "body_fat_mass": 0.5,
    "skeletal_muscle_mass": 0.5,
    "abs_fat": 0.02,
    "fat_level": 1,
    "weight": 1.0,
}

MIN_POINTS_FOR_SLOPE = 3  # 추세선 기울기는 3회 이상일 때만 계산


def list_inbody_records(user_id: str) -> list[dict]:
    """사용자의 측정 기록을 모두 가져온다. (정렬은 analyze_records 에서 처리)"""
    sb = get_supabase_admin()
    res = sb.table(TABLE).select("*").eq("user_id", user_id).execute()
    return res.data or []


def _to_date(value) -> date:
    """'YYYY-MM-DD' 또는 'YYYY-MM-DDTHH:MM:SS...' 를 date로 변환한다."""
    if isinstance(value, date):
        return value if not hasattr(value, "date") else value.date()
    return date.fromisoformat(str(value)[:10])


def _record_date(record: dict):
    """측정일을 우선 사용하고, 없으면 등록 시각을 쓴다."""
    value = record.get(DATE_FIELD) or record.get(FALLBACK_DATE_FIELD)
    return _to_date(value) if value else None


def _monthly_slope(dates: list[date], values: list[float]) -> float:
    """전체 기록에 직선을 맞춰 한 달(30일) 기준 변화량을 구한다."""
    days = np.array([(d - dates[0]).days for d in dates], dtype=float)
    if np.all(days == days[0]):  # 모든 측정일이 같으면 기울기 계산 불가
        return 0.0
    slope_per_day = np.polyfit(days, np.array(values, dtype=float), 1)[0]
    return float(slope_per_day * 30)


def _judge(key: str, delta: float) -> str:
    """변화량과 좋아지는 방향을 보고 판정한다."""
    if abs(delta) < THRESHOLD[key]:
        return "유지"
    direction = DIRECTION[key]
    if direction == 0:
        return "증가" if delta > 0 else "감소"
    return "개선" if delta * direction > 0 else "악화"


def analyze_records(records: list[dict]) -> dict:
    """측정 기록 리스트로 추세를 계산한다. (DB 없이 테스트할 때도 사용)"""
    dated = [(_record_date(r), r) for r in records]
    dated = sorted([(d, r) for d, r in dated if d is not None], key=lambda x: x[0])

    if len(dated) < 2:
        return {
            "status": "insufficient",
            "message": "변화를 보려면 측정 기록이 2회 이상 필요해요.",
            "count": len(dated),
        }

    dates = [d for d, _ in dated]
    result = {
        "status": "ok",
        "count": len(dated),
        "first_date": dates[0].isoformat(),
        "latest_date": dates[-1].isoformat(),
        "period_days": (dates[-1] - dates[0]).days,
        "metrics": {},
    }

    for key in DIRECTION:
        pairs = [(d, r[key]) for d, r in dated if r.get(key) is not None]
        if len(pairs) < 2:
            continue

        ds = [p[0] for p in pairs]
        vs = [float(p[1]) for p in pairs]
        delta = vs[-1] - vs[0]
        slope = _monthly_slope(ds, vs) if len(pairs) >= MIN_POINTS_FOR_SLOPE else None

        result["metrics"][key] = {
            "values": [{"date": d.isoformat(), "value": v} for d, v in zip(ds, vs)],
            "first": vs[0],
            "latest": vs[-1],
            "delta": round(delta, 2),
            "delta_pct": round(delta / vs[0] * 100, 1) if vs[0] else None,
            "monthly_slope": round(slope, 2) if slope is not None else None,
            "judgement": _judge(key, delta),
        }

    judgements = [m["judgement"] for m in result["metrics"].values()]
    result["summary"] = {
        "improved": judgements.count("개선"),
        "maintained": judgements.count("유지"),
        "worsened": judgements.count("악화"),
    }
    return result


def analyze_trend(user_id: str) -> dict:
    """사용자 ID로 DB에서 기록을 불러와 추세를 분석한다."""
    return analyze_records(list_inbody_records(user_id))
