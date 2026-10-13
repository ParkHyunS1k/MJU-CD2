"""라벨 보강(spec 5.4절). 새 청킹이 상위에 올린 청크 중 기존 근거가 없는 것만 모아 GPT(Codex CLI)가 판정하고,
검증을 통과한 근거를 labels.jsonl에 더한다. 저장소 루트에서

    python -m rag.eval.augment collect --chunker paragraph --out rag/eval/data/augment_paragraph.json
    python -m rag.eval.augment judge --chunker paragraph --items ... --judgments ...   (GPT, Codex CLI)
    python -m rag.eval.augment apply --chunker paragraph --items ... --judgments ...
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from rag.eval.evaluate import find_evidence
from rag.eval.export_labels import _MIN_EVIDENCE_CHARS, _nonspecific
from rag.eval.judge import judge_query
from rag.eval.metrics import compact

_FIELDS = ("content_hash", "doc_id", "page_start", "page_end", "content")
SOURCE = "llm-gpt"


def collect(chunks: list[dict], queries: list[dict], labels: dict[str, list[dict]], rankings: list[list[int]],
            k: int = 5) -> list[dict]:
    """쿼리마다 상위 k개 중 기존 근거를 하나도 담지 않은 청크를 판정 후보로 모은다."""
    items = []
    for q, ranked in zip(queries, rankings, strict=True):
        found = find_evidence(chunks, labels[q["query_id"]])
        new = [i for i in ranked[:k] if i not in found]
        if new:
            items.append({"query_id": q["query_id"], "text": q["text"], "candidates": [
                {"candidate_id": f"c{n}", **{f: chunks[i][f] for f in _FIELDS}} for n, i in enumerate(new, 1)]})
    return items


def judge_items(items: list[dict], ask: Callable, cache: Path) -> dict[str, list[dict]]:
    """쿼리마다 한 번 판정(rag.eval.judge.judge_query: 원문에 없는 근거는 버림). 쿼리마다 저장해 끊겨도 이어서 한다."""
    judgments = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    for item in items:
        if item["query_id"] in judgments:
            continue
        result = judge_query(ask, item["text"], item["candidates"])
        judgments[item["query_id"]] = [{"candidate_id": c["candidate_id"], **result[c["content_hash"]]}
                                       for c in item["candidates"] if result[c["content_hash"]]["relevant"] is not None]
        cache.write_text(json.dumps(judgments, ensure_ascii=False, indent=1), encoding="utf-8")
    return judgments


def apply_judgments(items: list[dict], judgments: dict[str, list[dict]], labels: dict[str, list[dict]],
                    corpus: list[str]) -> tuple[list[dict], dict]:
    """원문 그대로이고, 10자 이상이고, 상투 문구가 아니고, 기존 라벨과 겹치지 않는 근거만 라벨로 만든다."""
    stats = {"candidates": sum(len(item["candidates"]) for item in items), "judged": 0, "relevant": 0, "added": 0,
             "dropped_evidence": 0}
    rows = []
    for item in items:
        by_id = {c["candidate_id"]: c for c in item["candidates"]}
        seen = {compact(label["evidence"]) for label in labels.get(item["query_id"], [])}
        for judgment in judgments.get(item["query_id"], []):
            candidate = by_id.get(judgment["candidate_id"])
            if candidate is None:
                continue
            stats["judged"] += 1
            if not judgment["relevant"]:
                continue
            stats["relevant"] += 1
            content = compact(candidate["content"])
            for evidence in judgment["evidence"]:
                key = compact(evidence)
                if (len(key) < _MIN_EVIDENCE_CHARS or key not in content or key in seen
                        or _nonspecific(evidence, corpus)):
                    stats["dropped_evidence"] += 1
                    continue
                seen.add(key)
                rows.append({"query_id": item["query_id"], **{f: candidate[f] for f in _FIELDS if f != "content"},
                             "evidence": evidence, "source": SOURCE})
                stats["added"] += 1
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    from rag.eval import run

    parser = argparse.ArgumentParser(description="라벨 보강: 새 청킹의 상위 결과 중 라벨 없는 청크 판정")
    parser.add_argument("action", choices=["collect", "judge", "apply"])
    parser.add_argument("--chunker", required=True)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--items", type=Path)
    parser.add_argument("--judgments", type=Path)
    args = parser.parse_args(argv)

    ctx = run.Ctx()
    state = {**run.BASE, "chunker": args.chunker, "max_chars": args.max_chars, "overlap": args.overlap,
             "embedding_model": "bge-m3"}  # 2단계 비교와 같은 임베딩
    chunks = run.chunks_for(state)
    if args.action == "collect":
        ranked, _ = run.retrieve(state, chunks, [q["text"] for q in ctx.queries])
        items = collect(chunks, ctx.queries, ctx.labels, ranked, args.k)
        args.out.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{args.out}: {len(items)} queries, {sum(len(i['candidates']) for i in items)} candidates")
        return 0

    items = json.loads(args.items.read_text(encoding="utf-8"))
    if args.action == "judge":
        from rag.eval.judge import codex_ask

        judgments = judge_items(items, codex_ask(), args.judgments)
        print(f"{args.judgments}: {len(judgments)}/{len(items)} queries judged")
        return 0
    judgments = json.loads(args.judgments.read_text(encoding="utf-8"))
    rows, stats = apply_judgments(items, judgments, ctx.labels, [c["content"] for c in chunks])
    with run.LABELS.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
