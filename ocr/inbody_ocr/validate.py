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


def validate(ext: InBodyExtraction) -> ValidationReport:
    values = {k: getattr(ext, k) for k in FIELD_SPECS}
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
            r = results[f]
            if not passed:
                r.status = "low"
                r.reasons.append(f"불일치: {chk.description} (계산 {exp:.2f}, 표기 {act})")
            elif r.status == "unchecked":
                r.status = "ok"

    return ValidationReport(results, check_results)
