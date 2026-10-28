import pytest

from rag.eval.report_preview import jaccard_mean, mentions_user_number


def test_jaccard_mean():
    a, b, c = {1, 2}, {2, 3}, {4}
    # (a,b)=1/3, (a,c)=0, (b,c)=0
    assert jaccard_mean([a, b, c]) == pytest.approx(1 / 9)
    assert jaccard_mean([a]) == 0.0 and jaccard_mean([set(), set()]) == 0.0


def test_action_mentions_user_number():
    values = [{"weight": 74.9, "bmi": 24.9}, {"weight_loss_5pct_kg": 3.7, "bmi_band": "비만 전단계"}]
    assert mentions_user_number("현재 체중 74.9 kg의 5%인 3.7 kg 감량", values)
    assert not mentions_user_number("중강도 유산소 운동을 주 150분 하세요", values)
    assert mentions_user_number("체중 80 kg 기준", [{"weight": 80.0}])
