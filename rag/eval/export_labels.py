"""검수 결과를 근거 문장 라벨로 내보낸다. 저장소 루트에서 `python -m rag.eval.export_labels`."""

import argparse
import json
import sys
from pathlib import Path

from rag.eval.metrics import compact

_QUERY_FIELDS = ("query_id", "text", "form", "condition", "category", "split")


def export(items: dict, decisions: list[dict], query_edits: list[dict]) -> tuple[list[dict], list[dict], dict]:
    candidates = {(q["query_id"], c["content_hash"]): c for q in items["queries"] for c in q["candidates"]}
    edits = {e["query_id"]: e for e in query_edits}
    stats = {"candidates": len(candidates), "reviewed": 0, "relevant": 0, "relevant_without_evidence": 0,
             "unknown_decisions": 0}

    labels, seen = [], set()
    for d in decisions:
        candidate = candidates.get((d["query_id"], d["content_hash"]))
        if candidate is None:
            stats["unknown_decisions"] += 1
            continue
        stats["reviewed"] += 1
        if not d["relevant"]:
            continue
        stats["relevant"] += 1
        evidence = [e for e in d["evidence"] if compact(e)]
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
                           "content_hash": d["content_hash"], "evidence": e, "source": "human"})
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
