import numpy as np
import pandas as pd
import pytest

FH_COLS = [f"HE_{d}fh{i}" for d in ("HP", "DM", "HL") for i in (1, 2, 3)]


@pytest.fixture
def knhanes_like():
    """KNHANES 열 이름을 가진 합성 데이터. 나이·BMI가 높을수록 질환 확률이 오른다."""
    rng = np.random.default_rng(0)
    n = 3000
    age = rng.integers(19, 60, n).astype(float)
    sex = rng.integers(1, 3, n).astype(float)
    bmi = rng.normal(24, 3.5, n)
    pbf = rng.normal(27, 7, n)

    def label(shift):
        lin = -9 + 0.07 * age + 0.18 * bmi + 0.02 * pbf + shift
        return (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)

    fh = {c: rng.choice([0.0, 1.0, 9.0], n, p=[0.7, 0.2, 0.1]) for c in FH_COLS}
    return pd.DataFrame({
        "age": age, "sex": sex, "HE_BMI": bmi, "BIA_PBF": pbf, **fh,
        "HE_HP": np.where(label(0.0) == 1, 4.0, rng.choice([1.0, 2.0, 3.0], n)),
        "HE_DM_HbA1c": np.where(label(-1.0) == 1, 3.0, rng.choice([1.0, 2.0], n)),
        "HE_HCHOL": label(0.5),
        "HE_HTG": label(-0.5),
        "yr": rng.choice([2022.0, 2023.0, 2024.0], n),
    })
