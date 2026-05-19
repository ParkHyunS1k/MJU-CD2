from __future__ import annotations

from datetime import datetime

# 모델 핵심 피처 + 사용자가 직관적으로 확인하는 보조 지표
_FEATURE_MAP = {
    "weight": "체중",
    "bmi": "BMI",
    "body_fat_percentage": "체지방률",
    "skeletal_muscle_mass": "골격근량",
}

_MIN_CHANGE_PER_90_DAYS = {
    "weight": 1.0,
    "bmi": 0.3,
    "body_fat_percentage": 1.0,
    "skeletal_muscle_mass": 0.5,
}

_MIN_RECORDS = 3
_MIN_SPAN_DAYS = 90


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _records_with_measured_at(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("measured_at")]


def _sort_by_measured_at(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: _parse_dt(r["measured_at"]))


def _slope_per_90_days(pairs: list[tuple[str, float]]) -> float:
    start = _parse_dt(pairs[0][0])
    xs = [(_parse_dt(dt) - start).total_seconds() / 86400 for dt, _ in pairs]
    ys = [float(value) for _, value in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    slope_per_day = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    return slope_per_day * 90


def _trend_direction(col: str, slope_per_90_days: float) -> str:
    threshold = _MIN_CHANGE_PER_90_DAYS[col]
    if slope_per_90_days > threshold:
        return "up"
    if slope_per_90_days < -threshold:
        return "down"
    return "stable"


def _is_risk_worsening(direction: str, shap_directions: set[str]) -> bool:
    return direction != "stable" and direction in shap_directions


def is_eligible(records: list[dict]) -> bool:
    """시계열 분석 제공 조건: 기록 3개 이상 + 첫·마지막 측정 간격 90일 이상."""
    dated_records = _records_with_measured_at(records)
    if len(dated_records) < _MIN_RECORDS:
        return False
    sorted_r = _sort_by_measured_at(dated_records)
    span = (_parse_dt(sorted_r[-1]["measured_at"]) - _parse_dt(sorted_r[0]["measured_at"])).days
    return span >= _MIN_SPAN_DAYS


def compute_trends(records: list[dict], shap_feature_directions: dict[str, set[str]]) -> dict:
    """시계열 추세를 계산해 TrendAnalysis 딕셔너리로 반환한다."""
    sorted_r = _sort_by_measured_at(_records_with_measured_at(records))
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
        slope_per_90_days = round(_slope_per_90_days(pairs), 2)
        direction = _trend_direction(col, slope_per_90_days)
        shap_directions = shap_feature_directions.get(kor, set())

        feature_trends.append({
            "feature_kor": kor,
            "dates": dates,
            "values": values,
            "direction": direction,
            "total_change": total_change,
            "slope_per_90_days": slope_per_90_days,
            "is_shap_important": bool(shap_directions),
            "shap_directions": sorted(shap_directions),
            "is_risk_worsening": _is_risk_worsening(direction, shap_directions),
        })

    return {
        "record_count": len(sorted_r),
        "span_days": span_days,
        "feature_trends": feature_trends,
    }
