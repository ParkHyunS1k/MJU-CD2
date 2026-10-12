"""로지스틱 회귀 파이프라인과 v2 모델 번들. 설계: spec 4.1–4.4."""
from datetime import datetime, timezone

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from training.features import DISEASES, FEAT_KOR, THRESHOLD_AGE, TRAIN_AGE, choose_thresholds

SEED = 42


def build_pipeline(age_spline: bool) -> Pipeline:
    """age_spline이면 0번 열(age)에 3차 스플라인(매듭 4개)을 적용한다."""
    steps = []
    if age_spline:
        steps.append(("spline", ColumnTransformer(
            [("age", SplineTransformer(n_knots=4, degree=3), [0])], remainder="passthrough")))
    steps += [("scaler", StandardScaler()),
              ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))]
    return Pipeline(steps)


def feature_groups(pipe: Pipeline, features: list[str]) -> dict[str, list[int]]:
    """변환 후 열 번호 → 원래 변수. ColumnTransformer는 스플라인 열을 먼저, 나머지를 뒤에 둔다."""
    if "spline" not in pipe.named_steps:
        return {f: [i] for i, f in enumerate(features)}
    n_age = pipe.named_steps["spline"].named_transformers_["age"].n_features_out_
    groups = {features[0]: list(range(n_age))}
    for i, f in enumerate(features[1:]):
        groups[f] = [n_age + i]
    return groups


def fit_bundle(X, y, age, disease_key, age_spline) -> dict:
    """OOF 예측으로 25–44세 기준값을 정한 뒤 전체로 다시 학습해 번들을 만든다."""
    features = list(X.columns)
    Xa = X.to_numpy(dtype=float)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(build_pipeline(age_spline), Xa, y, cv=cv, method="predict_proba")[:, 1]
    tgt = (age >= THRESHOLD_AGE[0]) & (age <= THRESHOLD_AGE[1])
    medium, high = choose_thresholds(y[tgt], oof[tgt])
    pipe = build_pipeline(age_spline).fit(Xa, y)
    return {
        "version": 2,
        "disease_key": disease_key,
        "target": DISEASES[disease_key],
        "model": pipe,
        "features": features,
        "feat_kor": {f: FEAT_KOR[f] for f in features},
        "feature_groups": feature_groups(pipe, features),
        "background": pipe[:-1].transform(Xa).mean(axis=0).tolist(),
        "thresh_medium": medium,
        "thresh_high": high,
        "meta": {
            "data": "KNHANES 2022-2024",
            "train_age": list(TRAIN_AGE),
            "threshold_age": list(THRESHOLD_AGE),
            "n_train": int(len(y)),
            "n_pos": int(np.sum(y)),
            "age_spline": bool(age_spline),
            "sklearn": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
