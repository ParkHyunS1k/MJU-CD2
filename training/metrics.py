"""평가 지표와 신뢰구간. 설계: spec 5.1."""
import numpy as np
from sklearn.linear_model import LogisticRegression

LEVELS = ("low", "medium", "high")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def _resample(y, n_boot, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():  # 한 클래스만 뽑히면 AUROC를 못 구한다
            yield i


def bootstrap_ci(y, p, metric, n_boot=1000, seed=42) -> tuple[float, float, float]:
    vals = [metric(y[i], p[i]) for i in _resample(y, n_boot, seed)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(metric(y, p)), float(lo), float(hi))


def paired_diff_ci(y, p_new, p_ref, metric, n_boot=1000, seed=42) -> tuple[float, float, float]:
    vals = [metric(y[i], p_new[i]) - metric(y[i], p_ref[i]) for i in _resample(y, n_boot, seed)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(metric(y, p_new) - metric(y, p_ref)), float(lo), float(hi))


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """logit(p)에 대한 무규제 로지스틱 회귀. 기울기 1, 절편 0이 이상적이다."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e10, max_iter=1000)  # 사실상 무규제(sklearn 1.8은 penalty=None·C=inf 모두 경고).fit(lp, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def risk_levels(p, medium, high) -> np.ndarray:
    return np.where(p >= high, "high", np.where(p >= medium, "medium", "low")).astype(object)


def grade_table(y, level) -> list[dict]:
    rows = []
    for lv in LEVELS:
        m = level == lv
        k, n = int(y[m].sum()), int(m.sum())
        rows.append({"level": lv, "n": n, "share": n / len(y),
                     "prevalence": k / n if n else None,
                     "prevalence_ci": wilson_ci(k, n) if n else None})
    return rows


def sens_spec(y, level) -> dict:
    """민감도 = 양성 중 보통 이상, 특이도 = 음성 중 낮음, high_ppv = 높음 중 양성 비율."""
    pos, neg = y == 1, y == 0
    tp, tn = int((level[pos] != "low").sum()), int((level[neg] == "low").sum())
    hi = level == "high"
    return {"sensitivity": tp / int(pos.sum()), "sensitivity_ci": wilson_ci(tp, int(pos.sum())),
            "specificity": tn / int(neg.sum()), "specificity_ci": wilson_ci(tn, int(neg.sum())),
            "high_ppv": float(y[hi].mean()) if hi.any() else None}
