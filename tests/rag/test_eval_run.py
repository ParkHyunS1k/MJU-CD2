import json
import zlib

import numpy as np
import pytest

from rag.eval import run

EV1 = "하루 소금 섭취량을 5 g 이하로 제한한다."
EV2 = "주 5회 이상 유산소 운동을 30분 한다."
CHUNKS = [
    {"doc_id": "a", "content": f"식사 요법. {EV1}"},
    {"doc_id": "a", "content": f"운동 요법. {EV2}"},
    {"doc_id": "a", "content": "고혈압의 역학과 유병률 통계를 다룬다."},
    {"doc_id": "b", "content": "통풍 환자는 퓨린 섭취를 줄인다."},
]


class FakeEmbedder:
    name = "fake"

    def encode(self, texts, kind):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for r, t in enumerate(texts):
            for i in range(len(t) - 1):
                out[r, zlib.crc32(t[i:i + 2].encode()) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    def q(qid, text, form="keyword", split="tune"):
        return {"query_id": qid, "text": text, "form": form, "condition": "hypertension",
                "category": "diet", "split": split, "excluded": False}
    queries = [q("q1", "소금 섭취량 제한"), q("q2", "유산소 운동 30분", "question"),
               q("q3", "소금 섭취 줄이기", split="test"), q("q4", "운동 몇 분", "question", "test"),
               q("none-1", "통풍 식이요법", "unanswerable", "test")]
    labels = [{"query_id": "q1", "doc_id": "a", "evidence": EV1, "source": "human"},
              {"query_id": "q2", "doc_id": "a", "evidence": EV2, "source": "llm"},
              {"query_id": "q3", "doc_id": "a", "evidence": EV1, "source": "llm"},
              {"query_id": "q4", "doc_id": "a", "evidence": EV2, "source": "llm"}]
    qp, lp = tmp_path / "queries.jsonl", tmp_path / "labels.jsonl"
    qp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in queries), encoding="utf-8")
    lp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labels), encoding="utf-8")
    monkeypatch.setattr(run, "STATE", tmp_path / "selected.json")
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run, "load", lambda name: FakeEmbedder())
    monkeypatch.setattr(run, "chunk_variant", lambda *args: CHUNKS)
    sp = tmp_path / "sources.json"
    sp.write_text(json.dumps([{"id": "a"}]), encoding="utf-8")
    monkeypatch.setattr(run, "SOURCES", sp)
    return run.Ctx(qp, lp)


def test_stage_without_state_uses_base(ctx):
    assert run.load_state() == run.BASE


def test_retrieve_ranks_matching_chunk_first(ctx):
    ranked, sims = run.retrieve(run.BASE, CHUNKS, ["소금 섭취량 제한"])
    assert ranked[0][0] == 0 and sims.shape == (1, 4)


def test_stage1_writes_results_and_state(ctx):
    chosen = run.stage1(ctx)
    assert chosen["cfg"] == {"parser": "pymupdf", "spacing": False}  # 모두 같으면 맨 앞
    state = json.loads(run.STATE.read_text())
    assert state["parser"] == "pymupdf" and state["max_chars"] == 1200
    md = (run.RESULTS / "01_parser.md").read_text()
    assert "preservation" in md and "**선택: pymupdf**" in md
    saved = json.loads((run.RESULTS / "01_parser.json").read_text())
    assert saved["primary"] == "ev_recall@budget" and len(saved["rows"]) == 5
    assert saved["rows"][1]["cfg"] == {"parser": "pymupdf", "spacing": "auto"}


def test_stage2_varies_chunking(ctx):
    run.stage2(ctx)
    saved = json.loads((run.RESULTS / "02_chunking.json").read_text())
    assert [r["cfg"] for r in saved["rows"]][:2] == [
        {"chunker": "chars", "max_chars": 2000, "overlap": 0, "embedding_model": "bge-m3"},
        {"chunker": "chars", "max_chars": 2000, "overlap": 150, "embedding_model": "bge-m3"}]
    assert json.loads(run.STATE.read_text())["max_chars"] == 2000


def test_stage3_records_dims_and_storage(ctx, monkeypatch):
    monkeypatch.setattr(run, "REGISTRY", {n: FakeEmbedder for n in run.EMBEDDINGS})
    run.stage3(ctx)
    state = json.loads(run.STATE.read_text())
    assert state["embedding_model"] == "gemini-768"  # 모두 같으면 맨 앞
    assert state["dimensions"] == 64
    row = json.loads((run.RESULTS / "03_embedding.json").read_text())["rows"][0]
    assert row["dims"] == 64 and row["storage_mb"] == round(64 * 2 * 4 / 1e6, 3)


class FakeReranker:
    def rerank(self, query, texts, candidates):
        return list(reversed(candidates))

    def ms_per_call(self):
        return 1.5


def test_stage4_and_stage5(ctx, monkeypatch):
    monkeypatch.setattr(run, "reranker", lambda name: FakeReranker())
    run.stage4(ctx)
    assert [r["cfg"] for r in json.loads((run.RESULTS / "04_hybrid.json").read_text())["rows"]] == [
        {"hybrid": False}, {"hybrid": True}]
    run.stage5(ctx)
    rows = json.loads((run.RESULTS / "05_reranker.json").read_text())["rows"]
    assert rows[0]["cfg"] == {"reranker": None} and rows[1]["rerank_ms"] == 1.5
    assert "rerank_ms" in (run.RESULTS / "05_reranker.md").read_text()  # 첫 행에 없는 칸도 표에 나온다
    assert json.loads(run.STATE.read_text())["reranker"] is None


def test_choose_k_fills_char_budget():
    # k는 청크 길이 중앙값으로 3,600자 예산을 채우는 개수(최소 1)
    assert run.choose_k([1000, 1100, 1200]) == 3
    assert run.choose_k([180] * 10) == 20
    assert run.choose_k([5000, 6000]) == 1
    assert run.choose_k([10] * 3) == run.DEPTH  # 검색은 상위 DEPTH개까지만 한다


def test_choose_threshold_caps_to_keep_answers():
    hits = [0.70, 0.72, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.91, 0.95]
    assert run.choose_threshold([0.50, 0.60], hits) == 0.6
    assert run.choose_threshold([0.90], hits) == round(float(np.quantile(hits, 0.1)), 4)
    assert run.choose_threshold([0.55], []) == 0.55  # 조정용에서 정답을 못 찾았으면 상한 없음


def test_stage6_writes_operating_values_and_summary(ctx):
    run.stage6(ctx)
    state = json.loads(run.STATE.read_text())
    assert state["query_form"] in ("keyword", "question")
    assert 1 <= state["final_k"] <= run.DEPTH
    assert isinstance(state["threshold"], float)
    md = (run.RESULTS / "06_operating.md").read_text()
    assert "쿼리 형태" in md and "최종 k" in md and "임계값" in md and "시험용" in md
    summary = (run.RESULTS / "summary.md").read_text()
    assert "selected.json" in summary and "사람 검수" in summary
    assert "영문 근거" in summary and "| keyword · 임계값" in summary
    assert "시험용 쿼리까지" in summary


def test_stage3_runs_only_named_models(ctx, monkeypatch):
    monkeypatch.setattr(run, "REGISTRY", {n: FakeEmbedder for n in run.EMBEDDINGS})
    run.stage3(ctx, ["bge-m3", "kure-v1"])
    rows = json.loads((run.RESULTS / "03_embedding.json").read_text())["rows"]
    assert [r["name"] for r in rows] == ["bge-m3", "kure-v1"]
    md = (run.RESULTS / "03_embedding.md").read_text()
    assert "미실행" in md and "gemini-3072" in md


def test_cli_passes_embedding_subset(monkeypatch):
    seen = {}
    monkeypatch.setattr(run, "Ctx", lambda: None)
    monkeypatch.setitem(run.STAGES, 3, lambda ctx, names=run.EMBEDDINGS: seen.setdefault("names", names) and {"name": "x"})
    run.main(["3", "--embeddings", "gemini-768,bge-m3"])
    assert seen["names"] == ["gemini-768", "bge-m3"]


class QuotaEmbedder(FakeEmbedder):
    def encode(self, texts, kind):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


def test_embedding_info_survives_unmeasurable_latency(ctx, monkeypatch):
    monkeypatch.setattr(run, "REGISTRY", {"gemini-768": QuotaEmbedder})
    info = run.embedding_info("gemini-768", 4, ["소금"])
    assert info["dims"] == 64 and info["query_ms"] == "n/a"


def test_choose_threshold_never_rounds_below_unanswerable_score():
    assert run.choose_threshold([0.70684], []) >= 0.70684


def test_chunk_variant_name_includes_chunker(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(run, "VARIANTS", tmp_path)
    monkeypatch.setattr(run, "ingest", lambda *a, **k: (seen.setdefault("args", (a, k)) and [{"content": "x"}], {}))
    run.chunk_variant("pymupdf", False, 1200, 0, "paragraph")
    assert (tmp_path / "pymupdf-raw-1200-0-paragraph.jsonl").exists()
    assert seen["args"][1]["chunker"] == "paragraph"
    assert run.BASE["chunker"] == "chars"


def test_stage2_includes_paragraph_chunkers(ctx):
    run.stage2(ctx)
    saved = json.loads((run.RESULTS / "02_chunking.json").read_text())
    assert saved["primary"] == "ev_recall@budget"
    cfgs = [r["cfg"] for r in saved["rows"]]
    assert {"chunker": "paragraph", "max_chars": 1200, "overlap": 0, "embedding_model": "bge-m3"} in cfgs
    assert cfgs[-1]["chunker"] == "paragraph-single"


def test_cli_state_and_results_paths(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(run, "STATE", run.STATE)  # main()이 바꾼 경로를 테스트 뒤 되돌린다
    monkeypatch.setattr(run, "RESULTS", run.RESULTS)
    monkeypatch.setattr(run, "Ctx", lambda: None)
    monkeypatch.setitem(run.STAGES, 4, lambda ctx: seen.update(state=run.STATE, results=run.RESULTS) or {"name": "x"})
    run.main(["4", "--state", str(tmp_path / "s.json"), "--results", str(tmp_path / "r")])
    assert seen == {"state": tmp_path / "s.json", "results": tmp_path / "r"}


def test_chunk_variant_auto_spacing_has_own_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "VARIANTS", tmp_path)
    monkeypatch.setattr(run, "ingest", lambda *a, **k: ([{"content": "x"}], {}))
    run.chunk_variant("pymupdf", "auto", 1200, 0, "paragraph")
    assert (tmp_path / "pymupdf-autospaced-1200-0-paragraph.jsonl").exists()
