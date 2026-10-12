"""질병 위험 모델을 학습해 models/{disease}.pkl로 저장한다.
실행: python -m training.train [--age-spline KEY ...] [--only KEY ...]
설계: docs/superpowers/specs/2026-10-12-risk-model-redesign-design.md 4절."""
import argparse
from pathlib import Path

import joblib

from training.analysis.common import load_knhanes
from training.features import DISEASES, training_rows
from training.model import fit_bundle

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--age-spline", nargs="*", default=[], choices=list(DISEASES),
                    help="나이 스플라인을 쓸 질환(evaluate.py 결과로 정한다)")
    ap.add_argument("--only", nargs="*", choices=list(DISEASES),
                    help="저장할 질환. 생략하면 전부. 통과 기준(spec 5.4)을 넘은 질환만 넣는다")
    args = ap.parse_args(argv)

    df = load_knhanes()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for key in args.only or list(DISEASES):
        X, y, info = training_rows(df, key)
        b = fit_bundle(X, y, info.age.to_numpy(), key, key in args.age_spline)
        joblib.dump(b, MODELS_DIR / f"{key}.pkl")
        print(f"{key:13s} n={b['meta']['n_train']:5d} 양성={b['meta']['n_pos']:5d} "
              f"spline={b['meta']['age_spline']} 보통≥{b['thresh_medium']:.3f} 높음≥{b['thresh_high']:.3f}")


if __name__ == "__main__":
    main()
