"""검증 레이어: 범위 검사 + 체성분 간 일관성 검사.

필드 상태
- ok:        범위 통과 + 자신이 포함된 일관성 검사를 1개 이상 통과 (실패 없음)
- unchecked: 범위는 통과했지만 교차검증할 수 있는 식이 없음
- low:       범위 실패, 또는 실패한 일관성 검사에 포함됨 → 사용자 확인 필요
- missing:   값을 읽지 못함

일관성 검사가 실패하면 어느 필드가 틀렸는지 특정할 수 없으므로 관련 필드를 모두 low로 둔다.
'틀렸는데 통과'를 줄이는 쪽으로 보수적으로 설계했다.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .schema import FIELD_SPECS, InBodyExtraction

Status = Literal["ok", "unchecked", "low", "missing"]


@dataclass(frozen=True)
class Check:
    name: str
    fields: tuple[str, ...]
    expected: Callable[[dict], float]  # 좌변 계산
    actual: Callable[[dict], float]  # 우변 계산
    tol: float  # 반올림 오차를 감안한 허용치
    description: str
    confirms: bool = True  # False: 타당성 검사. 실패하면 low지만 통과해도 ok로 올리지 않는다


# 허용치 근거: 결과지 값은 대부분 소수 1자리 반올림이라 항목 4개 합산 시 최대 ±0.2 오차.
# 체수분은 L 단위지만 인바디 결과지는 1L≈1kg로 합산해 체중을 구성한다(실기기 결과지로 재확인 필요).
CHECKS: list[Check] = [
    Check(
        "mass_balance",
        ("weight_kg", "total_body_water_l", "protein_kg", "minerals_kg", "body_fat_mass_kg"),
        lambda v: v["total_body_water_l"] + v["protein_kg"] + v["minerals_kg"] + v["body_fat_mass_kg"],
        lambda v: v["weight_kg"],
        0.3,
        "체중 ≈ 체수분 + 단백질 + 무기질 + 체지방량",
    ),
    Check(
        "fat_free_mass",
        ("weight_kg", "body_fat_mass_kg", "fat_free_mass_kg"),
        lambda v: v["weight_kg"] - v["body_fat_mass_kg"],
        lambda v: v["fat_free_mass_kg"],
        0.2,
        "제지방량 = 체중 - 체지방량",
    ),
    Check(
        "bmi",
        ("weight_kg", "height_cm", "bmi"),
        lambda v: v["weight_kg"] / (v["height_cm"] / 100) ** 2,
        lambda v: v["bmi"],
        0.15,
        "BMI = 체중 / 신장(m)²",
    ),
    Check(
        "percent_body_fat",
        ("body_fat_mass_kg", "weight_kg", "percent_body_fat"),
        lambda v: v["body_fat_mass_kg"] / v["weight_kg"] * 100,
        lambda v: v["percent_body_fat"],
        0.3,
        "체지방률 = 체지방량 / 체중 × 100",
    ),
    # Katch-McArdle 식. InBody270 결과지 25장(1인)에서 오차 ±1.3 이내 확인. 다른 기종은 미확인
    Check(
        "basal_metabolic_rate",
        ("fat_free_mass_kg", "basal_metabolic_rate_kcal"),
        lambda v: 370 + 21.6 * v["fat_free_mass_kg"],
        lambda v: v["basal_metabolic_rate_kcal"],
        2.0,
        "기초대사량 = 370 + 21.6 × 제지방량",
    ),
    Check(
        "weight_control",
        ("ideal_weight_kg", "weight_kg", "weight_control_kg"),
        lambda v: v["ideal_weight_kg"] - v["weight_kg"],
        lambda v: v["weight_control_kg"],
        0.15,
        "체중조절 = 적정체중 - 체중",
    ),
    Check(
        "control_sum",
        ("fat_control_kg", "muscle_control_kg", "weight_control_kg"),
        lambda v: v["fat_control_kg"] + v["muscle_control_kg"],
        lambda v: v["weight_control_kg"],
        0.15,
        "체중조절 = 지방조절 + 근육조절",
    ),
    # 표준체중 = 22 × 신장². InBody270 남성 1인 결과지 25장에서 확인. 여성 기준은 미확인이라 남성만 검사
    Check(
        "obesity_degree",
        ("_male", "weight_kg", "height_cm", "obesity_degree_pct"),
        lambda v: v["weight_kg"] / (22 * (v["height_cm"] / 100) ** 2) * 100,
        lambda v: v["obesity_degree_pct"],
        1.0,
        "비만도 = 체중 / 표준체중(22 × 신장²) × 100 (남성)",
    ),
    # 신체변화 그래프 마지막 열 = 이번 측정. 골격근량은 다른 검증식이 없어 이것으로 확인한다
    Check(
        "history_smm",
        ("_history_smm", "skeletal_muscle_mass_kg"),
        lambda v: v["_history_smm"],
        lambda v: v["skeletal_muscle_mass_kg"],
        0.05,
        "골격근량 = 신체변화 그래프의 이번 측정값",
    ),
    # 잠정 범위 (근거: 이 데이터 남성 1인 0.46~0.49, 여성·고도비만은 더 낮을 수 있어 넓게 잡음).
    # 그래프 눈금 '100'을 10.0으로 읽는 등 자릿수가 틀린 값을 거른다
    Check(
        "smm_ratio",
        ("skeletal_muscle_mass_kg", "weight_kg"),
        lambda v: v["skeletal_muscle_mass_kg"] / v["weight_kg"],
        lambda v: 0.425,
        0.225,
        "골격근량 / 체중 = 0.2~0.65 (타당성)",
        confirms=False,
    ),
]


@dataclass
class FieldResult:
    value: Optional[float]
    status: Status
    reasons: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
    name: str
    description: str
    passed: Optional[bool]  # None = 필요한 값이 없어 검사 불가
    expected: Optional[float] = None
    actual: Optional[float] = None


@dataclass
class ValidationReport:
    fields: dict[str, FieldResult]
    checks: list[CheckResult]

    @property
    def needs_review(self) -> list[str]:
        """값은 읽었지만 검증 실패 → 사용자가 반드시 확인."""
        return [k for k, f in self.fields.items() if f.status == "low"]

    @property
    def missing(self) -> list[str]:
        """읽지 못함 (결과지 기종에 따라 원래 없는 항목일 수 있음) → 직접 입력 유도."""
        return [k for k, f in self.fields.items() if f.status == "missing"]


def _aux_values(ext: InBodyExtraction) -> dict[str, Optional[float]]:
    """검사에만 쓰는 보조 값 (밑줄로 시작, 필드 상태는 만들지 않는다)."""
    last = ext.history[-1] if ext.history else None
    is_current = (
        last is not None and last.date == ext.measured_at
        and (last.time is None or ext.measured_time is None or last.time == ext.measured_time)
    )
    return {
        "_male": 1.0 if ext.sex == "M" else None,
        "_history_smm": last.skeletal_muscle_mass_kg if is_current else None,
    }


def validate(ext: InBodyExtraction) -> ValidationReport:
    values = {k: getattr(ext, k) for k in FIELD_SPECS} | _aux_values(ext)
    results: dict[str, FieldResult] = {}

    for key, spec in FIELD_SPECS.items():
        v = values[key]
        if v is None:
            results[key] = FieldResult(None, "missing", ["값을 읽지 못함"])
        elif not spec.min <= v <= spec.max:
            results[key] = FieldResult(v, "low", [f"범위 밖 ({spec.min}~{spec.max})"])
        else:
            results[key] = FieldResult(v, "unchecked")

    check_results: list[CheckResult] = []
    for chk in CHECKS:
        if any(values[f] is None for f in chk.fields):
            check_results.append(CheckResult(chk.name, chk.description, None))
            continue
        exp, act = chk.expected(values), chk.actual(values)
        passed = abs(exp - act) <= chk.tol
        check_results.append(CheckResult(chk.name, chk.description, passed, round(exp, 2), act))
        for f in chk.fields:
            if f not in results:  # 보조 값
                continue
            r = results[f]
            if not passed:
                r.status = "low"
                r.reasons.append(f"불일치: {chk.description} (계산 {exp:.2f}, 표기 {act})")
            elif r.status == "unchecked" and chk.confirms:
                r.status = "ok"

    return ValidationReport(results, check_results)
