"""리포트에 넣을 개인 지표. 숫자는 여기서 계산하고 GPT는 해석만 한다.
설계: docs/superpowers/specs/2026-10-28-report-personalization-design.md 5.2절."""

import math

# 대한비만학회 BMI 기준(코퍼스: 포털 비만, 대한비만학회 문서). (하한, 이름) 오름차순
BMI_BANDS = [(35.0, "3단계 비만"), (30.0, "2단계 비만"), (25.0, "1단계 비만"), (23.0, "비만 전단계"),
             (18.5, "정상"), (0.0, "저체중")]
OBESITY_BMI = 25.0
PRE_OBESITY_BMI = 23.0
NORMAL_BMI_UPPER = 22.9
# 인바디 결과지 "비만평가" 체지방률 기준(의학 지침이 아니라 기기 기준). (낮음 미만, 표준 이하, 경도비만 이하)
BODY_FAT_BANDS = {"male": (10.0, 20.0, 25.0), "female": (18.0, 28.0, 33.0)}
ABDOMINAL_RATIO = {"male": 0.90, "female": 0.85}  # 복부지방률(허리/엉덩이) 복부비만 기준
VISCERAL_FAT_LEVEL = 10

# 리포트에 넣는 인바디 항목과 한국어 이름. ai_service._INBODY_METRIC_KEYS와 같은 순서·같은 키
METRIC_NAMES = {
    "weight": "체중", "skeletal_muscle_mass": "골격근량", "body_fat_mass": "체지방량", "body_fat_percentage": "체지방률",
    "body_water": "체수분", "protein": "단백질", "minerals": "무기질", "bmi": "BMI", "basal_metabolic_rate": "기초대사량",
    "abs_fat": "복부지방률", "fat_level": "내장지방레벨", "height": "키", "muscle_control": "근육조절",
}


# 결과지 부위별 분석의 부위 키와 이름. 좌우 쌍은 판정이 다르면 "좌우 차이"로 표시한다
SEGMENT_NAMES = {"left_arm": "왼팔", "right_arm": "오른팔", "trunk": "몸통", "left_leg": "왼다리", "right_leg": "오른다리"}
_SEGMENT_PAIRS = {"팔": ("left_arm", "right_arm"), "다리": ("left_leg", "right_leg")}


def _segmental(lean: dict | None, fat: dict | None) -> dict:
    """표준이 아닌 부위만 모은다. 인바디 판정 그대로이며 우리가 다시 판정하지 않는다."""
    out: dict[str, list[str]] = {}
    for kind, parts in (("근육", lean or {}), ("체지방", fat or {})):
        for status in ("표준이하", "표준이상"):
            names = [name for key, name in SEGMENT_NAMES.items() if parts.get(key) == status]
            if names:
                out[f"{kind} {status}"] = names
        uneven = [pair for pair, (left, right) in _SEGMENT_PAIRS.items()
                  if parts.get(left) and parts.get(right) and parts[left] != parts[right]]
        if uneven:
            out[f"{kind} 좌우 차이"] = uneven
    return out


def _num(value) -> float | None:
    return None if value is None else float(value)


def _bmi_band(bmi: float) -> str:
    return next(name for lower, name in BMI_BANDS if bmi >= lower)


def _body_fat_status(pbf: float, gender: str) -> dict:
    low, standard, mild = BODY_FAT_BANDS[gender]
    status = "낮음" if pbf < low else "표준" if pbf <= standard else "경도비만" if pbf <= mild else "비만"
    return {"status": status, "over_standard_pp": round(max(pbf - standard, 0.0), 1)}


# 체지방 과다 × 근육 부족 → 인바디 기준 체성분 유형. 권고 방향(유산소·식사 / 저항운동·단백질)을 가른다
_BODY_TYPES = {(True, False): "지방 감량형", (True, True): "지방→근육형", (False, True): "근육 보강형",
               (False, False): "해당 없음"}


def _body_type(fat: dict | None, muscle: dict | None, fat_control: float | None) -> dict | None:
    """체지방 판정이 없으면 None. 근육조절이 없으면 '근육 판정 불가': 결과지에서 적정은 0.0으로 찍히므로
    값이 없다는 건 읽지 않았다는 뜻이고, 근육이 부족한지 모른 채 감량만 권하면 근육까지 줄 수 있다."""
    if fat is None:
        return None
    fat_excess = fat["status"] in ("경도비만", "비만")
    if muscle is None:
        return {"type": "근육 판정 불가", "fat_excess": fat_excess}
    out = {"type": _BODY_TYPES[(fat_excess, muscle["status"] == "표준 이하")]}
    if fat_control is not None and fat_control < 0:  # 결과지 지방조절은 음수가 감량
        out["fat_loss_kg"] = -fat_control
    return out


def compute(inbody: dict, profile: dict) -> dict:
    """값이 없어 계산할 수 없는 지표는 키를 뺀다. missing_metrics만 항상 있다."""
    out: dict = {}
    weight, bmi, height = _num(inbody.get("weight")), _num(inbody.get("bmi")), _num(inbody.get("height"))
    gender = profile.get("gender")

    if height:
        out["height_cm"], out["height_source"] = height, "measured"
    elif weight and bmi:
        out["height_cm"], out["height_source"] = round(math.sqrt(weight / bmi) * 100, 1), "derived"

    if bmi is not None:
        out["bmi_band"] = _bmi_band(bmi)
        if bmi >= PRE_OBESITY_BMI and weight:
            if "height_cm" in out:
                out["kg_to_normal_bmi"] = round(weight - NORMAL_BMI_UPPER * (out["height_cm"] / 100) ** 2, 1)
            for pct in (5, 10):
                loss = weight * pct / 100
                out[f"weight_loss_{pct}pct_kg"] = round(loss, 1)
                out[f"weight_after_{pct}pct"] = round(weight - loss, 1)

    pbf = _num(inbody.get("body_fat_percentage"))
    if pbf is not None and gender in BODY_FAT_BANDS:
        out["body_fat_status"] = _body_fat_status(pbf, gender)

    muscle = _num(inbody.get("muscle_control"))
    if muscle is not None:
        out["muscle_status"] = ({"status": "표준 이하", "increase_kg": muscle} if muscle > 0
                                else {"status": "표준", "increase_kg": 0.0})

    body_type = _body_type(out.get("body_fat_status"), out.get("muscle_status"), _num(inbody.get("fat_control")))
    if body_type:
        out["body_type"] = body_type

    abdominal = []
    abs_fat, fat_level = _num(inbody.get("abs_fat")), _num(inbody.get("fat_level"))
    if abs_fat is not None and gender in ABDOMINAL_RATIO and abs_fat >= ABDOMINAL_RATIO[gender]:
        abdominal.append({"metric": "복부지방률", "value": abs_fat, "threshold": ABDOMINAL_RATIO[gender]})
    if fat_level is not None and fat_level >= VISCERAL_FAT_LEVEL:
        abdominal.append({"metric": "내장지방레벨", "value": fat_level, "threshold": VISCERAL_FAT_LEVEL})
    if abdominal:
        out["abdominal"] = abdominal

    segmental = _segmental(inbody.get("segmental_lean"), inbody.get("segmental_fat"))
    if segmental:
        out["segmental"] = segmental

    out["missing_metrics"] = [name for key, name in METRIC_NAMES.items() if inbody.get(key) is None]
    return out
