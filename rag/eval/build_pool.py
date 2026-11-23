"""검색 방식별 상위 결과를 모아 Gemini로 판정하고 검수용 파일을 만든다.
저장소 루트에서 `python -m rag.eval.build_pool`."""

import argparse
import json
import logging
import sys
from pathlib import Path

from rag.eval.embedders import Embedder, load
from rag.eval.judge import JudgeCache, codex_ask, gemini_ask, judge_query
from rag.eval.pool import pool_query
from rag.eval.search import BM25Index, dense_rank

log = logging.getLogger("rag.eval.build_pool")
_CANDIDATE_FIELDS = ("content_hash", "doc_id", "title", "heading_path", "page_start", "page_end", "content")


def rank_all(embedders: dict[str, Embedder], doc_texts: list[str], query_texts: list[str], k: int,
             bm25: BM25Index | None) -> dict[str, list[list[int]]]:
    rankings = {}
    for name, embedder in embedders.items():
        log.info("embedding with %s", name)
        rankings[name] = dense_rank(embedder.encode(query_texts, "query"), embedder.encode(doc_texts, "doc"), k)
    if bm25 is not None:
        rankings["bm25"] = [bm25.rank(q, k) for q in query_texts]
    return rankings


def build_review_items(chunks: list[dict], queries: list[dict], rankings: dict[str, list[list[int]]],
                       per_system: int, ask, cache: JudgeCache) -> dict:
    hashes = [c["content_hash"] for c in chunks]
    by_hash = {c["content_hash"]: c for c in chunks}
    out = []
    for qi, query in enumerate(queries):
        pooled = pool_query({name: [hashes[i] for i in r[qi]] for name, r in rankings.items()}, per_system)
        candidates = [{**{f: by_hash[p["content_hash"]][f] for f in _CANDIDATE_FIELDS}, "systems": p["systems"]}
                      for p in pooled]
        judged = cache.get(query["query_id"])
        if judged is None or set(judged) != {c["content_hash"] for c in candidates}:
            log.info("judging %s (%d candidates)", query["query_id"], len(candidates))
            judged = judge_query(ask, query["text"], candidates)
            cache.put(query["query_id"], judged)
        for c in candidates:
            c["judged_relevant"] = judged[c["content_hash"]]["relevant"]
            c["judged_evidence"] = judged[c["content_hash"]]["evidence"]
        out.append({**query, "candidates": candidates})
    return {"queries": out}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="평가 후보 모으기 + Gemini 1차 판정")
    parser.add_argument("--chunks", type=Path, default=Path("rag/data/chunks.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("rag/eval/data/queries.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("rag/eval/data/review_items.json"))
    parser.add_argument("--systems", default="gemini-1536,bge-m3,kure-v1,qwen3-0.6b,e5-large-instruct,bm25")
    parser.add_argument("--per-system", type=int, default=5)
    parser.add_argument("--judge-cache", type=Path, default=Path("rag/eval/cache/judgments.jsonl"))
    parser.add_argument("--judge", choices=["gemini", "codex"], default="gemini")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chunks, queries = _read_jsonl(args.chunks), _read_jsonl(args.queries)
    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    doc_texts = [c["content"] for c in chunks]
    embedders = {n: load(n) for n in names if n != "bm25"}
    bm25 = BM25Index(doc_texts) if "bm25" in names else None
    rankings = rank_all(embedders, doc_texts, [q["text"] for q in queries], args.per_system, bm25)
    items = build_review_items(chunks, queries, rankings, args.per_system, codex_ask() if args.judge == "codex" else gemini_ask(),
                               JudgeCache(args.judge_cache))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    total = sum(len(q["candidates"]) for q in items["queries"])
    log.info("wrote %s: %d queries, %d candidates", args.out, len(items["queries"]), total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
