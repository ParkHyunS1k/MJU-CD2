"""단계별 결과 JSON → 주 지표 막대그래프(95% 구간 오차 막대). 저장소 루트에서 `python -m rag.eval.plot`."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_INSTALLED = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
# 한글 글꼴 중 설치된 것만 넣는다(없는 이름은 막대마다 경고를 낸다)
plt.rcParams["font.family"] = [f for f in ("AppleGothic", "Malgun Gothic", "NanumGothic") if f in _INSTALLED] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_stage(json_path: Path, out_png: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    metric = data["primary"]
    rows = data["rows"]
    mean, lo, hi = (np.array([r[metric][i] for r in rows], dtype=float) for i in range(3))
    fig, ax = plt.subplots(figsize=(max(4.0, 1.3 * len(rows)), 3.2))
    ax.bar([r["name"] for r in rows], np.nan_to_num(mean),
           yerr=[np.nan_to_num(mean - lo), np.nan_to_num(hi - mean)], capsize=4, color="#4C72B0")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(json_path.stem)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(results: Path = Path("rag/eval/results")) -> int:
    for path in sorted(results.glob("0*_*.json")):
        plot_stage(path, path.with_suffix(".png"))
        print(path.with_suffix(".png"))
    return 0


def cli() -> int:
    """`python -m rag.eval.plot [결과 폴더]` — 폴더를 주지 않으면 rag/eval/results."""
    return main(Path(sys.argv[1])) if len(sys.argv) > 1 else main()


if __name__ == "__main__":
    sys.exit(cli())
