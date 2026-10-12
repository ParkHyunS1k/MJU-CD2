"""검수 결과를 근거 문장 라벨로 내보낸다. 저장소 루트에서 `python -m rag.eval.export_labels`."""

import argparse
import json
import sys
from pathlib import Path

from rag.eval.metrics import compact

_QUERY_FIELDS = ("query_id", "text", "form", "condition", "category", "split")
_MIN_EVIDENCE_CHARS = 10  # 10-gram 매칭 단위보다 짧은 조각은 아무 청크에나 걸린다


def _spans(fragments: list[str], content: str) -> list[str]:
    """청크 안에서 바로 이어지는 조각을 한 구간으로 합치고, 너무 짧은 구간은 버린다."""
    text, runs, end = compact(content), [], -1
    for f in fragments:
        cf = compact(f)
        if not cf:
            continue
        pos = text.find(cf, max(end, 0))
        if runs and pos >= 0 and pos == end:
            runs[-1] += " " + f
        else:
            runs.append(f)
        end = pos + len(cf) if pos >= 0 else -1
    return [r for r in runs if len(compact(r)) >= _MIN_EVIDENCE_CHARS]


def export(items: dict, decisions: list[dict], query_edits: list[dict]) -> tuple[list[dict], list[dict], dict]:
    candidates = {(q["query_id"], c["content_hash"]): c for q in items["queries"] for c in q["candidates"]}
    edits = {e["query_id"]: e for e in query_edits}
    stats = {"candidates": len(candidates), "reviewed": 0, "llm_filled": 0, "relevant": 0,
             "relevant_without_evidence": 0, "unknown_decisions": 0}

    decided = []
    for d in decisions:
        if (d["query_id"], d["content_hash"]) not in candidates:
            stats["unknown_decisions"] += 1
            continue
        stats["reviewed"] += 1
        decided.append(d | {"source": "human"})
    # 사람이 안 본 후보는 LLM 1차 판정을 그대로 쓴다 (source="llm"으로 구분).
    done = {(d["query_id"], d["content_hash"]) for d in decided}
    for (qid, h), c in candidates.items():
        if (qid, h) not in done and c["judged_relevant"] is not None:
            stats["llm_filled"] += 1
            decided.append({"query_id": qid, "content_hash": h, "relevant": c["judged_relevant"],
                            "evidence": c["judged_evidence"], "source": "llm"})

    labels, seen = [], set()
    for d in decided:
        candidate = candidates[(d["query_id"], d["content_hash"])]
        if not d["relevant"]:
            continue
        stats["relevant"] += 1
        evidence = _spans(d["evidence"], candidate["content"])
        if not evidence:
            stats["relevant_without_evidence"] += 1
            continue
        for e in evidence:
            key = (d["query_id"], compact(e))
            if key in seen:
                continue
            seen.add(key)
            labels.append({"query_id": d["query_id"], "doc_id": candidate["doc_id"],
                           "page_start": candidate["page_start"], "page_end": candidate["page_end"],
                           "content_hash": d["content_hash"], "evidence": e, "source": d["source"]})
    stats["unreviewed"] = stats["candidates"] - stats["reviewed"]

    queries = []
    for q in items["queries"]:
        edit = edits.get(q["query_id"], {})
        row = {f: q[f] for f in _QUERY_FIELDS}
        if edit.get("text"):
            row["text"] = edit["text"]
        row["excluded"] = bool(edit.get("excluded"))
        queries.append(row)

    labelled = {label["query_id"] for label in labels}
    stats["excluded_queries"] = [q["query_id"] for q in queries if q["excluded"]]
    stats["answerable_queries_without_labels"] = [
        q["query_id"] for q in queries
        if not q["excluded"] and q["form"] != "unanswerable" and q["query_id"] not in labelled
    ]
    return labels, queries, stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="검수 결과 → 근거 문장 라벨")
    parser.add_argument("--items", type=Path, default=Path("rag/eval/data/review_items.json"))
    parser.add_argument("--decisions", type=Path, default=Path("rag/eval/data/review_decisions.json"))
    parser.add_argument("--labels-out", type=Path, default=Path("rag/eval/data/labels.jsonl"))
    parser.add_argument("--queries-out", type=Path, default=Path("rag/eval/data/queries.jsonl"))
    args = parser.parse_args(argv)

    items = json.loads(args.items.read_text(encoding="utf-8"))
    review = json.loads(args.decisions.read_text(encoding="utf-8"))
    labels, queries, stats = export(items, review["decisions"], review["queries"])
    _write_jsonl(args.labels_out, labels)
    _write_jsonl(args.queries_out, queries)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
