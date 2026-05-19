from __future__ import annotations

from datetime import datetime

# 인바디 DB 컬럼 → 한국어 레이블 매핑 (모델 feature와 동일한 항목만)
_FEATURE_MAP = {
    "weight": "체중",
    "bmi": "BMI",
    "body_fat_percentage": "체지방률",
    "body_fat_mass": "체지방량",
    "skeletal_muscle_mass": "골격근량",
    "body_water": "체수분",
}

_MIN_RECORDS = 3
_MIN_SPAN_DAYS = 90


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _records_with_measured_at(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("measured_at")]


def is_eligible(records: list[dict]) -> bool:
    """시계열 분석 제공 조건: 기록 3개 이상 + 첫·마지막 측정 간격 90일 이상."""
    dated_records = _records_with_measured_at(records)
    if len(dated_records) < _MIN_RECORDS:
        return False
    sorted_r = sorted(dated_records, key=lambda r: r["measured_at"])
    span = (_parse_dt(sorted_r[-1]["measured_at"]) - _parse_dt(sorted_r[0]["measured_at"])).days
    return span >= _MIN_SPAN_DAYS


def compute_trends(records: list[dict], shap_feature_kors: set[str]) -> dict:
    """시계열 추세를 계산해 TrendAnalysis 딕셔너리로 반환한다."""
    sorted_r = sorted(_records_with_measured_at(records), key=lambda r: r["measured_at"])
    span_days = (_parse_dt(sorted_r[-1]["measured_at"]) - _parse_dt(sorted_r[0]["measured_at"])).days

    feature_trends = []
    for col, kor in _FEATURE_MAP.items():
        pairs = [
            (r["measured_at"], r[col])
            for r in sorted_r
            if r.get(col) is not None
        ]
        if len(pairs) < 2:
            continue

        dates = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        total_change = round(values[-1] - values[0], 2)

        if total_change > 0:
            direction = "up"
        elif total_change < 0:
            direction = "down"
        else:
            direction = "stable"

        feature_trends.append({
            "feature_kor": kor,
            "dates": dates,
            "values": values,
            "direction": direction,
            "total_change": total_change,
            "is_shap_important": kor in shap_feature_kors,
        })

    return {
        "record_count": len(sorted_r),
        "span_days": span_days,
        "feature_trends": feature_trends,
    }
