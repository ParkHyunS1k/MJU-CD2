import math

import pytest

from rag.eval.metrics import (bootstrap_ci, compact, contains_evidence, evidence_coverage,
                              query_metrics)


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


def test_query_metrics_values():
    found = {1: {0}, 3: {0, 1}}
    m = query_metrics([0, 1, 2, 3, 4], found, n_evidence=2, ks=(1, 3, 5))
    assert m["recall@1"] == 0.0 and m["recall@3"] == 1.0 and m["recall@5"] == 1.0
    assert m["ev_recall@1"] == 0.0 and m["ev_recall@3"] == 0.5 and m["ev_recall@5"] == 1.0
    assert m["mrr"] == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    assert m["ndcg@10"] == pytest.approx(dcg / (1 + 1 / math.log2(3)))
    assert "recall@budget" not in m  # 예산 지표는 청크 본문이 필요해 evaluate()가 계산한다


def test_query_metrics_nothing_found():
    m = query_metrics([0, 1], {5: {0}}, n_evidence=1, ks=(1,))
    assert m["recall@1"] == 0.0 and m["mrr"] == 0.0 and m["ndcg@10"] == 0.0


def test_bootstrap_ci():
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0, 1.0)
    mean, lo, hi = bootstrap_ci([0.0, 1.0] * 25)
    assert mean == 0.5 and lo < 0.5 < hi
    assert bootstrap_ci([0.0, 1.0] * 25) == (mean, lo, hi)
    assert all(math.isnan(v) for v in bootstrap_ci([]))


def test_compact_drops_soft_hyphens():
    assert compact("per­ formed") == compact("performed")
