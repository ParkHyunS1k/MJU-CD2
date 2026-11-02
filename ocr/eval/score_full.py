"""새 항목(정상범위·운동소비열량·임피던스·이력·측정 시각 등) 채점.

정답 JSONL의 "full" 객체와 예측 extraction을 평탄화해 항목별로 비교한다.
이력은 날짜로 열을 맞추고, 예측 날짜가 없으면 시각으로 맞춘다.

사용법:
  uv run python -m eval.score_full --labels data/labels_full.jsonl --preds eval/out/preds_capture.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _flat(full: dict) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in full.items():
        if k == "ranges":
            for f, (lo, hi) in v.items():
                out[f"ranges.{f}"] = (float(lo), float(hi))
        elif k == "exercise_kcal":
            out.update({f"exercise.{e}": float(x) for e, x in v.items()})
        elif k == "impedance":
            out.update({f"impedance.{fr}.{s}": float(x) for fr, segs in v.items() for s, x in segs.items()})
        elif k != "history":
            out[k] = float(v) if isinstance(v, (int, float)) else v
    return out


def _history(gold: list[dict], pred: list[dict]) -> dict[str, object]:
    out = {}
    for g in gold:
        p = next((p for p in pred if p.get("date") == g["date"]), None) or next(
            (p for p in pred if p.get("date") is None and p.get("time") == g["time"]), {}
        )
        for f in ("date", "weight_kg", "skeletal_muscle_mass_kg", "percent_body_fat"):
            out[f"history.{g['time']}.{f}"] = (g[f], p.get(f))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--show", action="store_true", help="틀린 항목 출력")
    args = ap.parse_args()
    preds = {json.loads(l)["id"]: json.loads(l) for l in args.preds.read_text().splitlines() if l.strip()}

    c: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in (json.loads(l) for l in args.labels.read_text().splitlines() if l.strip()):
        ext = preds.get(row["id"], {}).get("extraction") or {}
        pairs = {k: (g, None) for k, g in _flat(row["full"]).items()}
        got = _flat({k: v for k, v in ext.items() if k in row["full"] and v is not None})
        pairs.update({k: (g, got.get(k)) for k, (g, _) in pairs.items()})
        pairs.update(_history(row["full"].get("history", []), ext.get("history", [])))
        for k, (g, p) in pairs.items():
            group = k.split(".")[0] if "." in k else "scalar"
            if isinstance(g, tuple) and isinstance(p, (list, tuple)):
                p = tuple(float(x) for x in p)
            status = "missing" if p is None else ("correct" if p == g else "wrong")
            c[group]["n"] += 1
            c[group][status] += 1
            if args.show and status != "correct":
                print(f"{row['id']} {k}: want {g} got {p}")
    for group, k in sorted(c.items()):
        print(f"{group:10} n={k['n']:4}  correct={k['correct'] / k['n']:.3f}  missing={k['missing']:3}  wrong={k['wrong']:3}")


if __name__ == "__main__":
    main()
