"""예측 JSONL vs 정답 JSONL 채점.

지표 (입력 유형 photo/capture 별):
- exact_match: 필드 단위 정확 일치율 (결과지 표기 소수 자릿수로 반올림 후 비교)
- flag_rate: 사용자 확인으로 넘긴 비율 (status low/missing)
- wrong_but_passed: 틀렸는데 검증을 통과(ok/unchecked)한 비율 ← 핵심 지표
    * per_field: 전체 필드 중 비율 (사용자에게 실제로 새는 오류율)
    * per_wrong: 틀린 필드 중 비율 (검증 레이어가 놓친 비율)

정답에 null인 필드(결과지에 없는 항목)는 채점에서 제외한다.

사용법:
  uv run python -m eval.score --labels labels.jsonl --preds eval/out/preds_A.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from inbody_ocr.schema import FIELD_SPECS, InBodyExtraction
from inbody_ocr.validate import validate


def _eq(key: str, pred, gold) -> bool:
    if pred is None:
        return False
    d = FIELD_SPECS[key].decimals
    return round(float(pred), d) == round(float(gold), d)


def score(labels: list[dict], preds: dict[str, dict]) -> dict:
    # (input_type, field) -> counters
    c: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in labels:
        pred = preds.get(row["id"], {})
        ext = pred.get("extraction")
        report = validate(InBodyExtraction.model_validate(ext)) if ext else None
        for key, gold in row["fields"].items():
            if gold is None or key not in FIELD_SPECS:
                continue
            for group in (row["input_type"], "all"):
                k = c[(group, key)]
                k["n"] += 1
                if report is None:
                    k["wrong"] += 1
                    k["flagged"] += 1  # 추출 실패는 사용자 수동 입력으로 넘어감
                    continue
                fr = report.fields[key]
                correct = _eq(key, fr.value, gold)
                flagged = fr.status in ("low", "missing")
                k["correct"] += correct
                k["wrong"] += not correct
                k["flagged"] += flagged
                k["wrong_passed"] += (not correct) and not flagged
    return c


def summarize(c) -> dict:
    out: dict[str, dict] = {}
    for group in sorted({g for g, _ in c}):
        tot = defaultdict(int)
        per_field = {}
        for (g, key), k in c.items():
            if g != group:
                continue
            for m, v in k.items():
                tot[m] += v
            per_field[key] = {"n": k["n"], "exact_match": round(k["correct"] / k["n"], 3),
                              "wrong_but_passed": k["wrong_passed"]}
        n, wrong = tot["n"], tot["wrong"]
        out[group] = {
            "n_fields": n,
            "exact_match": round(tot["correct"] / n, 3),
            "flag_rate": round(tot["flagged"] / n, 3),
            "wrong_but_passed_per_field": round(tot["wrong_passed"] / n, 4),
            "wrong_but_passed_per_wrong": round(tot["wrong_passed"] / wrong, 3) if wrong else None,
            "by_field": per_field,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--preds", required=True, type=Path)
    args = ap.parse_args()
    labels = [json.loads(l) for l in args.labels.read_text().splitlines() if l.strip()]
    preds = {}
    for l in args.preds.read_text().splitlines():
        if l.strip():
            p = json.loads(l)
            preds[p["id"]] = p
    print(json.dumps(summarize(score(labels, preds)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
