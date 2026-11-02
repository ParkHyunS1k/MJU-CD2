from inbody_ocr.schema import InBodyExtraction
from inbody_ocr.validate import validate


def consistent_sample() -> dict:
    """내부적으로 일관된 합성 값 (실제 결과지 아님)."""
    return dict(
        is_inbody_sheet=True, sex="M", height_cm=172.0, age=25, weight_kg=70.0,
        total_body_water_l=40.4, protein_kg=10.9, minerals_kg=3.50, body_fat_mass_kg=15.2,
        fat_free_mass_kg=54.8, skeletal_muscle_mass_kg=31.0, bmi=23.7, percent_body_fat=21.7,
        visceral_fat_level=6, basal_metabolic_rate_kcal=1554,
    )


def _ext(**overrides) -> InBodyExtraction:
    return InBodyExtraction(**{**consistent_sample(), **overrides})


def test_consistent_sample_passes_all_checks():
    r = validate(_ext())
    # 샘플에 없는 값(적정체중·이력 등)이 필요한 검사는 계산 불가(None)
    computed = {c.name: c.passed for c in r.checks if c.passed is not None}
    assert {"mass_balance", "fat_free_mass", "bmi", "percent_body_fat", "basal_metabolic_rate"} <= set(computed)
    assert all(computed.values())
    assert r.fields["weight_kg"].status == "ok"
    # 교차검증 식이 없는 필드는 unchecked
    assert r.fields["skeletal_muscle_mass_kg"].status == "unchecked"
    assert r.needs_review == []


def test_digit_misread_flags_involved_fields():
    # 체지방량 15.2 → 18.2 오인식 (3↔8 혼동 가정)
    r = validate(_ext(body_fat_mass_kg=18.2))
    failed = {c.name for c in r.checks if c.passed is False}
    assert {"mass_balance", "fat_free_mass", "percent_body_fat"} <= failed
    assert r.fields["body_fat_mass_kg"].status == "low"
    assert "body_fat_mass_kg" in r.needs_review


def test_decimal_shift_caught_by_range():
    r = validate(_ext(height_cm=17.25))
    assert r.fields["height_cm"].status == "low"


def test_missing_value_skips_check():
    r = validate(_ext(bmi=None))
    assert r.fields["bmi"].status == "missing"
    bmi_check = next(c for c in r.checks if c.name == "bmi")
    assert bmi_check.passed is None


def test_history_last_point_verifies_smm():
    # 신체변화 그래프 마지막 열(이번 측정)과 골격근량이 같으면 ok, 다르면 low
    hist = [{"date": "2026-10-01", "time": "10:00", "skeletal_muscle_mass_kg": 31.0}]
    same = dict(measured_at="2026-10-01", measured_time="10:00", history=hist)
    assert validate(_ext(**same)).fields["skeletal_muscle_mass_kg"].status == "ok"
    assert validate(_ext(**same, skeletal_muscle_mass_kg=37.0)).fields["skeletal_muscle_mass_kg"].status == "low"


def test_minor_age_flagged():
    assert validate(_ext(age=16)).fields["age"].status == "low"
