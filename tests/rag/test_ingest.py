import json

from rag.ingest import ingest, main


def _sources(tmp_path, entries):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return path


def _entry(doc_id, file, **overrides):
    entry = {
        "id": doc_id,
        "file": file,
        "title": f"{doc_id} 제목",
        "conditions": ["hypertension"],
        "category": "lifestyle",
    }
    entry.update(overrides)
    return entry


def _guide_pdf(make_pdf):
    return make_pdf(
        "guide.pdf",
        [
            [(40, "고혈압 진료지침"), (100, "1. 식이요법"), (200, "나트륨 섭취를 줄인다."), (820, "1")],
            [(40, "고혈압 진료지침"), (100, "2. 운동요법"), (200, "주 5회 유산소 운동을 한다."), (820, "2")],
            [(40, "고혈압 진료지침"), (100, "참고문헌"), (200, "Kim 2020"), (820, "3")],
        ],
        toc=[[1, "1. 식이요법", 1], [1, "2. 운동요법", 2], [1, "참고문헌", 3]],
    )


def test_ingest_end_to_end(tmp_path, make_pdf):
    _guide_pdf(make_pdf)
    make_pdf("notoc.pdf", [[(100, "목차 없는 문서 본문입니다.")], [(100, "둘째 쪽 본문입니다.")]])
    sources = _sources(tmp_path, [
        _entry("guide", "guide.pdf", section_rules=[{"match": "운동", "category": "exercise"}]),
        _entry("notoc", "notoc.pdf"),
    ])

    records, stats = ingest(sources, tmp_path)

    assert [(r["doc_id"], r["heading_path"], r["category"], r["page_start"]) for r in records] == [
        ("guide", "1. 식이요법", "lifestyle", 1),
        ("guide", "2. 운동요법", "exercise", 2),
        ("notoc", None, "lifestyle", 1),
    ]
    assert records[0]["content"] == "guide 제목 > 1. 식이요법\n나트륨 섭취를 줄인다."
    assert records[2]["page_end"] == 2
    assert all("고혈압 진료지침" not in r["content"].split("\n", 1)[1] for r in records)
    assert all("Kim 2020" not in r["content"] for r in records)
    assert stats == {"documents": 2, "chunks": 3, "no_toc": ["notoc"], "skipped_no_text": [], "failed": []}


def test_ingest_skips_scanned_pdf(tmp_path, make_pdf):
    make_pdf("scan.pdf", [[], []])
    records, stats = ingest(_sources(tmp_path, [_entry("scan", "scan.pdf")]), tmp_path)
    assert records == []
    assert stats["skipped_no_text"] == ["scan"]
    assert stats["documents"] == 0


def test_ingest_reports_corrupt_pdf_and_continues(tmp_path, make_pdf):
    _guide_pdf(make_pdf)
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    sources = _sources(tmp_path, [_entry("broken", "broken.pdf"), _entry("guide", "guide.pdf")])
    records, stats = ingest(sources, tmp_path)
    assert stats["failed"] == ["broken"]
    assert {r["doc_id"] for r in records} == {"guide"}


def test_main_writes_jsonl_and_is_deterministic(tmp_path, make_pdf):
    _guide_pdf(make_pdf)
    sources = _sources(tmp_path, [_entry("guide", "guide.pdf")])
    out = tmp_path / "out" / "chunks.jsonl"
    argv = ["--sources", str(sources), "--raw-dir", str(tmp_path), "--out", str(out)]

    assert main(argv) == 0
    first = out.read_bytes()
    lines = first.decode("utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["doc_id"] == "guide"
    assert "나트륨" in lines[0]  # ensure_ascii=False

    assert main(argv) == 0
    assert out.read_bytes() == first


def test_main_returns_1_on_bad_sources(tmp_path):
    sources = _sources(tmp_path, [_entry("guide", "missing.pdf")])
    out = tmp_path / "chunks.jsonl"
    assert main(["--sources", str(sources), "--raw-dir", str(tmp_path), "--out", str(out)]) == 1
    assert not out.exists()
