"""두 기록의 저장된 위험도(risk_store 모양)로 질환별 변화와 요인별 몫을 계산한다.
로지스틱 회귀라 입력별 로그 오즈 기여의 합이 로그 오즈 전체다. 두 기록의 기여 차이가 로그 오즈 변화를 남김없이 나누므로,
확률 변화를 그 비율대로 나눈다. 프로필은 두 기록에 같은 값이라(risk_store가 보장) 차이는 BMI·체지방률·나이에서만 나온다.
설계: docs/superpowers/specs/2026-11-15-risk-change-design.md"""

SHARE_FEATURES = {"bmi": "BMI", "body_fat_percentage": "체지방률", "age": "나이"}
_INPUTS = ("bmi", "body_fat_percentage")


def _shares(change_pp: float, prev: dict, curr: dict) -> dict | None:
    deltas = {f: curr[f] - prev[f] for f in curr}
    total = sum(deltas.values())
    if abs(total) < 1e-9:  # 로그 오즈가 그대로면 나눌 몫이 없다
        return None
    # 두 요인이 반대 방향이면 몫 하나가 change_pp보다 클 수 있다. 합은 맞으므로 그대로 둔다
    return {name: round(change_pp * deltas[f] / total, 1) for f, name in SHARE_FEATURES.items()}


def compute(previous: dict, current: dict, previous_measured_at: str) -> list[dict]:
    before = {d["disease_key"]: d for d in previous["diseases"]}
    out = []
    for now in current["diseases"]:
        prev = before.get(now["disease_key"])
        if prev is None:
            continue
        change_pp = (now["probability"] - prev["probability"]) * 100
        out.append({
            "disease": now["disease"],
            "previous_measured_at": str(previous_measured_at)[:10],
            "previous_pct": round(prev["probability"] * 100, 1),
            "current_pct": round(now["probability"] * 100, 1),
            "change_pp": round(change_pp, 1),
            "share_pp": _shares(change_pp, prev["contributions"], now["contributions"]),
            "previous_inputs": {k: previous["inputs"][k] for k in _INPUTS},
            "current_inputs": {k: current["inputs"][k] for k in _INPUTS},
        })
    return out
