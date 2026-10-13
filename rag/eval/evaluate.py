"""근거 문장 라벨로 검색 결과를 채점하고, 구성별 결과를 표와 선택으로 정리한다."""

import json
import math
import re
from pathlib import Path

from rag.eval.metrics import bootstrap_ci, contains_evidence, query_metrics

_HANGUL = re.compile("[가-힣]")
_KS = (1, 3, 5, 10, 20)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_eval(queries_path: Path, labels_path: Path) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    """(전체 쿼리, 채점 쿼리, 쿼리별 라벨). 제외·정답 없음 쿼리는 검색 지표에서 뺀다."""
    queries = _read_jsonl(queries_path)
    labels: dict[str, list[dict]] = {}
    for label in _read_jsonl(labels_path):
        labels.setdefault(label["query_id"], []).append(label)
    scored = [q for q in queries
              if not q["excluded"] and q["form"] != "unanswerable" and q["query_id"] in labels]
    return queries, scored, labels


def find_evidence(chunks: list[dict], labels: list[dict]) -> dict[int, set[int]]:
    """청크 번호 → 그 청크가 담은 근거 번호 (같은 doc_id + 50% 이상 포함)."""
    by_doc: dict[str, list[int]] = {}
    for ci, chunk in enumerate(chunks):
        by_doc.setdefault(chunk["doc_id"], []).append(ci)
    found: dict[int, set[int]] = {}
    for ei, label in enumerate(labels):
        for ci in by_doc.get(label["doc_id"], []):
            if contains_evidence(chunks[ci]["content"], label["evidence"]):
                found.setdefault(ci, set()).add(ei)
    return found


def preservation(chunks: list[dict], queries: list[dict], labels: dict[str, list[dict]]) -> float:
    """근거 문장 중 어떤 청크엔가 50% 이상 들어 있는 비율."""
    kept = total = 0
    for q in queries:
        qlabels = labels[q["query_id"]]
        kept += len(set().union(*find_evidence(chunks, qlabels).values()))
        total += len(qlabels)
    return kept / total


def evaluate(chunks: list[dict], queries: list[dict], labels: dict[str, list[dict]],
             rankings: list[list[int]], ks=_KS) -> list[dict]:
    lengths = [len(c["content"]) for c in chunks]
    out = []
    for q, ranked in zip(queries, rankings, strict=True):
        qlabels = labels[q["query_id"]]
        found = find_evidence(chunks, qlabels)
        out.append({"query_id": q["query_id"], **query_metrics(ranked, found, len(qlabels), lengths, ks)})
    return out


def groups(queries: list[dict], labels: dict[str, list[dict]]) -> dict[str, set[str]]:
    human = {q["query_id"] for q in queries if any(l["source"] == "human" for l in labels[q["query_id"]])}
    english = {q["query_id"] for q in queries
               if any(not _HANGUL.search(l["evidence"]) for l in labels[q["query_id"]])}
    return {"사람 검수": human, "영문 근거": english}


def summarize(name: str, per_query: list[dict], metrics: list[str], groups: dict[str, set[str]],
              extra: dict | None = None) -> dict:
    """metrics[0]이 주 지표다. 부분 집합 칸은 주 지표만 낸다."""
    row: dict = {"name": name}
    for m in metrics:
        row[m] = bootstrap_ci([p[m] for p in per_query])
    for g, ids in groups.items():
        row[f"{metrics[0]} ({g})"] = bootstrap_ci([p[metrics[0]] for p in per_query if p["query_id"] in ids])
    return row | (extra or {})


def pick(rows: list[dict], metric: str) -> dict:
    """구간이 1등과 겹치는 후보 중 목록 앞(더 단순·저렴한) 것."""
    best = max(rows, key=lambda r: r[metric][0])
    return next(r for r in rows if r[metric][2] >= best[metric][1])


def cell(value) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 3:
        mean, lo, hi = value
        return "–" if math.isnan(mean) else f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def table(rows: list[dict], columns: list[str] | None = None) -> str:
    columns = columns or list(dict.fromkeys(k for r in rows for k in r if k not in ("name", "cfg")))
    head = "| 구성 | " + " | ".join(columns) + " |"
    sep = "|---" * (len(columns) + 1) + "|"
    body = ["| " + " | ".join([r["name"], *(cell(r.get(c, "")) for c in columns)]) + " |" for r in rows]
    return "\n".join([head, sep, *body])
