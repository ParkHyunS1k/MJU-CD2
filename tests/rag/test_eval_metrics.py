import pytest

from rag.eval.metrics import compact, contains_evidence, evidence_coverage


def test_compact_removes_whitespace_and_normalizes():
    assert compact(" 하루 소금\n섭취량 ") == "하루소금섭취량"
    assert compact("가") == "가"


def test_full_containment_ignores_spacing():
    text = "규칙적인운동은혈압을낮추고, 심폐기능을개선하며"
    assert evidence_coverage("규칙적인 운동은 혈압을 낮추고", text) == 1.0


def test_partial_coverage_when_chunk_cuts_sentence():
    evidence = "하루 소금 섭취량을 5 g 이하로 제한한다."
    text = "...권고한다. 하루 소금 섭취량을 5 g 이하로"
    coverage = evidence_coverage(evidence, text)
    assert 0.3 < coverage < 0.8


def test_short_evidence_uses_exact_match():
    assert evidence_coverage("금연", "금연을 권고한다") == 1.0
    assert evidence_coverage("절주", "금연을 권고한다") == 0.0


def test_empty_evidence_is_zero():
    assert evidence_coverage("  ", "아무 내용") == 0.0


@pytest.mark.parametrize("text,expected", [
    ("하루 소금 섭취량을 5 g 이하로 제한한다. 체중을 줄인다.", True),
    ("체중을 줄인다. 하루 소금 섭취량을", False),
])
def test_contains_evidence_threshold(text, expected):
    assert contains_evidence(text, "하루 소금 섭취량을 5 g 이하로 제한한다.") is expected
