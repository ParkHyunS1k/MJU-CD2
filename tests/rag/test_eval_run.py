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
    assert saved["primary"] == "recall@budget" and len(saved["rows"]) == 4


def test_stage2_varies_chunking(ctx):
    run.stage2(ctx)
    saved = json.loads((run.RESULTS / "02_chunking.json").read_text())
    assert [r["cfg"] for r in saved["rows"]][:2] == [{"max_chars": 2000, "overlap": 0},
                                                     {"max_chars": 2000, "overlap": 150}]
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
