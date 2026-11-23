import pytest

from training.analysis.common import EXT_DIR
from training.features import BASE_FEATURES

LABELS = ["hypertension", "diabetes", "cholesterol", "triglyceride"]


@pytest.mark.skipif(not (EXT_DIR / "nhanes" / "bix_c.rda").exists(), reason="NHANES 파일 없음")
def test_load_nhanes():
    from training.external import load_nhanes
    d = load_nhanes()
    assert list(d.columns) == BASE_FEATURES + LABELS
    assert 1500 < len(d) <= 1668 and d.age.between(20, 49).all()
    for c in LABELS:
        assert set(d[c].dropna().unique()) <= {0.0, 1.0}
    assert d.diabetes.sum() >= 70


@pytest.mark.skipif(not (EXT_DIR / "bodyPerformance.csv").exists(), reason="Kaggle 파일 없음")
def test_load_kaggle():
    from training.external import load_kaggle
    d = load_kaggle()
    assert list(d.columns) == BASE_FEATURES + ["hypertension"]
    assert 13300 < len(d) < 13393
    assert d.bmi.between(10, 60).all()
