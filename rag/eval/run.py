"""계획 B 실험. 저장소 루트에서 `python -m rag.eval.run 1 2 3 4 5 6`.
단계마다 한 변수만 바꿔 비교하고, 1등을 selected.json에 누적해 다음 단계의 기준으로 쓴다."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from rag.eval.embedders import REGISTRY, load
from rag.eval.evaluate import evaluate, groups, load_eval, pick, preservation, summarize, table
from rag.eval.search import RERANKERS, BM25Index, dense_rank, reranker, rrf
from rag.ingest import ingest

log = logging.getLogger("rag.eval.run")

SOURCES, RAW = Path("rag/sources.json"), Path("rag/raw")
VARIANTS = Path("rag/data/variants")
QUERIES, LABELS = Path("rag/eval/data/queries.jsonl"), Path("rag/eval/data/labels.jsonl")
RESULTS = Path("rag/eval/results")
STATE = Path("rag/eval/selected.json")
DEPTH = 20  # 지표는 상위 20개까지 본다
POOL = 100  # 하이브리드로 합치기 전 각 방식에서 가져오는 수
BASE = {"parser": "pymupdf", "spacing": False, "max_chars": 1200, "overlap": 150,
        "embedding_model": "bge-m3", "dimensions": None, "hybrid": False, "reranker": None,
        "query_form": None, "final_k": None, "threshold": None}
METRICS = ["ndcg@10", "recall@3", "recall@20", "mrr", "recall@budget", "ev_recall@budget"]


def load_state() -> dict:
    return {**BASE, **(json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {})}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def chunk_variant(parser: str, spacing: bool, max_chars: int, overlap: int) -> list[dict]:
    path = VARIANTS / f"{parser}-{'spaced' if spacing else 'raw'}-{max_chars}-{overlap}.jsonl"
    if not path.exists():
        records, stats = ingest(SOURCES, RAW, max_chars, overlap, parser, spacing)
        log.info("%s: %s", path.name, json.dumps(stats, ensure_ascii=False))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def chunks_for(state: dict) -> list[dict]:
    return chunk_variant(state["parser"], state["spacing"], state["max_chars"], state["overlap"])


def retrieve(state: dict, chunks: list[dict], texts: list[str]) -> tuple[list[list[int]], np.ndarray]:
    """상위 DEPTH개 순위와 쿼리×청크 임베딩 코사인(임계값용)."""
    embedder = load(state["embedding_model"])
    docs = [c["content"] for c in chunks]
    qv, dv = embedder.encode(texts, "query"), embedder.encode(docs, "doc")
    ranked = dense_rank(qv, dv, POOL)
    if state["hybrid"]:
        bm25 = BM25Index(docs)
        ranked = [rrf([r, bm25.rank(t, POOL)]) for r, t in zip(ranked, texts)]
    ranked = [r[:DEPTH] for r in ranked]
    if state["reranker"]:
        rr = reranker(state["reranker"])
        ranked = [rr.rerank(t, docs, r) for t, r in zip(texts, ranked)]
    return ranked, qv @ dv.T


class Ctx:
    def __init__(self, queries_path: Path = QUERIES, labels_path: Path = LABELS):
        self.all_queries, self.queries, self.labels = load_eval(queries_path, labels_path)
        self.groups = groups(self.queries, self.labels)


def write_stage(fname: str, title: str, rows: list[dict], chosen_name: str, primary: str, note: str = "") -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    md = f"# {title}\n\n주 지표: `{primary}` (평균 [95% 부트스트랩 구간])\n\n{table(rows)}\n\n**선택: {chosen_name}**\n"
    if note:
        md += f"\n{note}\n"
    (RESULTS / f"{fname}.md").write_text(md, encoding="utf-8")
    (RESULTS / f"{fname}.json").write_text(json.dumps({"primary": primary, "rows": rows}, ensure_ascii=False),
                                           encoding="utf-8")


def run_stage(ctx: Ctx, fname: str, title: str, variants: list[tuple[str, dict]], primary: str,
              extra=None, note: str = "") -> dict:
    state = load_state()
    metrics = [primary, *(m for m in METRICS if m != primary)]
    texts = [q["text"] for q in ctx.queries]
    rows = []
    for name, overrides in variants:
        cfg = {**state, **overrides}
        chunks = chunks_for(cfg)
        ranked, _ = retrieve(cfg, chunks, texts)
        info = {"chunks": len(chunks), **(extra(cfg, chunks) if extra else {})}
        per_query = evaluate(chunks, ctx.queries, ctx.labels, ranked)
        rows.append(summarize(name, per_query, metrics, ctx.groups, info) | {"cfg": overrides})
        log.info("%s %s: %s=%.3f", fname, name, primary, rows[-1][primary][0])
    chosen = pick(rows, primary)
    save_state({**state, **chosen["cfg"]})
    write_stage(fname, title, rows, chosen["name"], primary, note)
    return chosen


def stage1(ctx: Ctx) -> dict:
    variants = [(f"{p}{' + 띄어쓰기' if s else ''}", {"parser": p, "spacing": s})
                for p in ("pymupdf", "pdfplumber") for s in (False, True)]
    return run_stage(
        ctx, "01_parser", "1단계: 파서", variants, "recall@budget",
        extra=lambda cfg, chunks: {"preservation": preservation(chunks, ctx.queries, ctx.labels)},
        note="라벨 근거 문장은 PyMuPDF 텍스트에서 뽑았으므로 preservation은 PyMuPDF에 유리하다. "
             "띄어쓰기는 공백을 지우고 비교하는 채점에 영향이 없고, 임베딩·BM25 결과로만 드러난다.")


def stage2(ctx: Ctx) -> dict:
    variants = [(f"{m}자 / 겹침 {o}", {"max_chars": m, "overlap": o})
                for m in (2000, 1200, 600) for o in (0, 150)]
    return run_stage(ctx, "02_chunking", "2단계: 청킹 (BGE-M3, 글자 수 예산 3,600자)", variants,
                     "recall@budget", note="앞쪽일수록 청크 수가 적다(저장·검색 비용이 작다).")


RERANKER_NAMES = list(RERANKERS)
EMBEDDINGS = ["gemini-768", "bge-m3", "kure-v1", "qwen3-0.6b", "e5-large-instruct", "gemini-1536", "gemini-3072"]


def embedding_info(name: str, n_chunks: int, sample: list[str]) -> dict:
    """캐시를 거치지 않고 쿼리 임베딩 지연을 잰다(로컬 모델은 로드·워밍업 뒤)."""
    inner = REGISTRY[name]()
    inner.encode(sample[:1], "query")
    start = time.perf_counter()
    vectors = inner.encode(sample, "query")
    dims = int(vectors.shape[1])
    return {"dims": dims, "storage_mb": round(dims * 2 * n_chunks / 1e6, 3),
            "query_ms": round((time.perf_counter() - start) * 1000 / len(sample), 1)}


def stage3(ctx: Ctx) -> dict:
    sample = [q["text"] for q in ctx.queries[:10]]
    chosen = run_stage(
        ctx, "03_embedding", "3단계: 임베딩 모델", [(n, {"embedding_model": n}) for n in EMBEDDINGS], "ndcg@10",
        extra=lambda cfg, chunks: embedding_info(cfg["embedding_model"], len(chunks), sample),
        note="앞쪽일수록 저장 용량(차원)이 작다. storage_mb는 halfvec(2바이트) 기준. "
             "gemini의 query_ms는 네트워크 왕복을 포함한다.")
    save_state({**load_state(), "dimensions": chosen["dims"]})
    return chosen


def stage4(ctx: Ctx) -> dict:
    return run_stage(ctx, "04_hybrid", "4단계: 하이브리드 (벡터 + kiwi BM25, RRF k=60)",
                     [("벡터만", {"hybrid": False}), ("벡터 + BM25", {"hybrid": True})], "ndcg@10")


def stage5(ctx: Ctx) -> dict:
    variants = [("없음", {"reranker": None})] + [(n, {"reranker": n}) for n in RERANKER_NAMES]
    return run_stage(
        ctx, "05_reranker", "5단계: 리랭커 (상위 20개 재정렬)", variants, "ndcg@10",
        extra=lambda cfg, chunks: {"rerank_ms": reranker(cfg["reranker"]).ms_per_call()} if cfg["reranker"] else {},
        note="rerank_ms는 쿼리 하나의 상위 20개 재정렬 시간(모델 로드 제외). 청크는 512토큰에서 잘린다.")


STAGES = {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 검색 구성 실험 (계획 B)")
    parser.add_argument("stages", nargs="+", type=int, choices=sorted(STAGES))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ctx = Ctx()
    for n in args.stages:
        chosen = STAGES[n](ctx)
        log.info("stage %d chose %s", n, chosen["name"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
