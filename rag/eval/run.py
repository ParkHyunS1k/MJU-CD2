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
from rag.eval.evaluate import cell, evaluate, find_evidence, groups, load_eval, pick, preservation, summarize, table
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
    md = f"# {title}\n\n주 지표: `{primary}` (평균 [95% 부트스트랩 구간]). 선택: 1등과의 쿼리별 차이 구간이 0을 포함하는 후보 중 표의 가장 위\n\n{table(rows)}\n\n**선택: {chosen_name}**\n"
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
    """차원은 캐시된 벡터로, 쿼리 임베딩 지연은 캐시 없이 잰다(로컬 모델은 로드·워밍업 뒤)."""
    dims = int(load(name).encode(sample, "query").shape[1])
    try:
        inner = REGISTRY[name]()
        inner.encode(sample[:1], "query")
        start = time.perf_counter()
        inner.encode(sample, "query")
        query_ms = round((time.perf_counter() - start) * 1000 / len(sample), 1)
    except Exception as e:  # API 한도 등으로 못 재면 결과표에 표시만 한다
        log.warning("%s query latency not measured: %s", name, e)
        query_ms = "n/a"
    return {"dims": dims, "storage_mb": round(dims * 2 * n_chunks / 1e6, 3), "query_ms": query_ms}


def stage3(ctx: Ctx, names: list[str] = EMBEDDINGS) -> dict:
    sample = [q["text"] for q in ctx.queries[:10]]
    chosen = run_stage(
        ctx, "03_embedding", "3단계: 임베딩 모델", [(n, {"embedding_model": n}) for n in names], "ndcg@10",
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


def choose_k(recall: dict[int, float], ratio: float = 0.9) -> int:
    return next(k for k in sorted(recall) if recall[k] >= ratio * recall[10])


def choose_threshold(none_top: list[float], hit_scores: list[float], keep: float = 0.9) -> float:
    """정답 없는 쿼리의 최고 코사인. 단 정답 청크의 keep 비율은 남도록 상한을 둔다."""
    top = max(none_top)
    if hit_scores:
        top = min(top, float(np.quantile(hit_scores, 1 - keep)))
    return round(top, 4)


def _first_hit_scores(chunks, queries, labels, ranked, sims, k) -> list[float]:
    scores = []
    for qi, (q, r) in enumerate(zip(queries, ranked)):
        found = find_evidence(chunks, labels[q["query_id"]])
        hit = next((i for i in r[:k] if i in found), None)
        if hit is not None:
            scores.append(float(sims[qi, hit]))
    return scores


def stage6(ctx: Ctx) -> dict:
    state = load_state()
    chunks = chunks_for(state)
    tune = [q for q in ctx.queries if q["split"] == "tune"]
    test = [q for q in ctx.queries if q["split"] == "test"]
    texts = lambda qs: [q["text"] for q in qs]
    sections = []

    # 1) 쿼리 형태 (조정용). 키워드형이 코드로 만들어져 단순하므로 앞에 둔다.
    form_rows = []
    for form in ("keyword", "question", "inbody"):
        qs = [q for q in tune if q["form"] == form]
        if qs:
            ranked, _ = retrieve(state, chunks, texts(qs))
            per_query = evaluate(chunks, qs, ctx.labels, ranked)
            form_rows.append(summarize(form, per_query, METRICS, {}, {"queries": len(qs)}))
    query_form = pick([r for r in form_rows if r["name"] != "inbody"], "ndcg@10")["name"]
    sections.append(f"## 쿼리 형태 (조정용)\n\n{table(form_rows)}\n\n**선택: {query_form}** (inbody는 참고)")

    # 2) 최종 k (조정용)
    ranked, sims = retrieve(state, chunks, texts(tune))
    per_query = evaluate(chunks, tune, ctx.labels, ranked, ks=range(1, 11))
    recall = {k: float(np.mean([p[f"recall@{k}"] for p in per_query])) for k in range(1, 11)}
    final_k = choose_k(recall)
    k_table = "\n".join([
        "| k | " + " | ".join(str(k) for k in recall) + " |",
        "|---" * (len(recall) + 1) + "|",
        "| Recall@k | " + " | ".join(f"{v:.3f}" for v in recall.values()) + " |",
    ])
    sections.append(f"## 최종 k (조정용)\n\n{k_table}\n\n**선택: k = {final_k}** (Recall@k ≥ 0.9 × Recall@10 인 최소 k)")

    # 3) 임계값: 정답 없는 쿼리 전부(조정용엔 1개뿐) + 조정용 정답 청크 코사인
    none = [q for q in ctx.all_queries if q["form"] == "unanswerable" and not q["excluded"]]
    _, none_sims = retrieve(state, chunks, texts(none))
    none_top = [float(row.max()) for row in none_sims]
    hits = _first_hit_scores(chunks, tune, ctx.labels, ranked, sims, final_k)
    threshold = choose_threshold(none_top, hits)
    sections.append(
        f"## 임계값 (임베딩 코사인)\n\n- 정답 없는 쿼리 1위 코사인: {', '.join(f'{s:.4f}' for s in none_top)}\n"
        f"- 조정용 첫 정답 청크 코사인 {len(hits)}개 중 90%가 남도록 상한을 둠\n\n**선택: {threshold}**")

    # 4) 시험용 보고 (임계값 적용 전/후)
    ranked_t, sims_t = retrieve(state, chunks, texts(test))
    filtered = [[i for i in r if sims_t[qi, i] > threshold] for qi, r in enumerate(ranked_t)]
    ks = (final_k, 10)
    rows = [summarize("임계값 없음", evaluate(chunks, test, ctx.labels, ranked_t, ks),
                      [f"recall@{final_k}", "ndcg@10", "mrr", f"ev_recall@{final_k}"], ctx.groups),
            summarize(f"임계값 {threshold}", evaluate(chunks, test, ctx.labels, filtered, ks),
                      [f"recall@{final_k}", "ndcg@10", "mrr", f"ev_recall@{final_k}"], ctx.groups)]
    rejected = sum(float(row.max()) <= threshold for row in none_sims)
    sections.append(f"## 시험용 결과\n\n{table(rows)}\n\n정답 없는 쿼리 {len(none)}개 중 {rejected}개가 "
                    f"임계값으로 걸러진다. 임계값을 정할 때 이 {len(none)}개를 모두 썼으므로 시험용 검증이 아니다.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "06_operating.md").write_text("# 6단계: 운영값\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    (RESULTS / "06_operating.json").write_text(
        json.dumps({"primary": "ndcg@10", "rows": form_rows}, ensure_ascii=False), encoding="utf-8")
    state = {**state, "query_form": query_form, "final_k": final_k, "threshold": threshold}
    save_state(state)
    _write_summary(state, rows[1], len(none), rejected)
    return {"name": f"{query_form}, k={final_k}, threshold={threshold}", "cfg": state}


def _write_summary(state: dict, test_row: dict, n_none: int, rejected: int) -> None:
    k = state["final_k"]
    lines = [
        "# 검색 구성 실험 요약",
        "",
        "최종 설정은 `rag/eval/selected.json`에 있다(하위 프로젝트 3의 입력).",
        "",
        "| 항목 | 선택 |", "|---|---|",
        *(f"| {key} | {state[key]} |" for key in BASE),
        "",
        f"시험용 Recall@{k}: {cell(test_row[f'recall@{k}'])}, nDCG@10: {cell(test_row['ndcg@10'])}, "
        f"사람 검수 쿼리만: {cell(test_row[f'recall@{k} (사람 검수)'])}. "
        f"정답 없는 쿼리 {n_none}개 중 {rejected}개를 임계값으로 거른다(임계값을 이 쿼리들로 정했으므로 검증된 값은 아니다).",
        "",
        "단계별 근거는 `01_parser.md` … `06_operating.md`. 1등과의 쿼리별 차이(쌍체 부트스트랩 95% 구간)가 0을 포함하면 "
        "더 단순·저렴한 구성을 골랐다.",
        "",
        "## 한계",
        "- 라벨은 사람 검수 188건 + LLM(GPT) 판정. 사람 검수 쿼리는 고혈압·당뇨·이상지질혈증뿐이라 "
        "\"사람 검수\" 열을 함께 본다.",
        "- 후보 모으기(pooling)로 만든 라벨이라, 어떤 방식도 찾지 못한 정답은 빠져 있다.",
        "- 근거 문장은 PyMuPDF 텍스트에서 뽑아 1단계 preservation은 PyMuPDF에 유리하다.",
        "- 쿼리 50개 내외라 차이가 작으면 구간이 겹친다.",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


STAGES = {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 검색 구성 실험 (계획 B)")
    parser.add_argument("stages", nargs="+", type=int, choices=sorted(STAGES))
    parser.add_argument("--embeddings", help="3단계에서 비교할 모델(쉼표 구분). 기본은 EMBEDDINGS 전부")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ctx = Ctx()
    for n in args.stages:
        if n == 3 and args.embeddings:
            chosen = STAGES[n](ctx, [m.strip() for m in args.embeddings.split(",")])
        else:
            chosen = STAGES[n](ctx)
        log.info("stage %d chose %s", n, chosen["name"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
