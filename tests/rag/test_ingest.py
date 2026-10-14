import json

import pytest

from rag.eval.metrics import compact
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


def test_ingest_skips_pdf_with_only_page_numbers(tmp_path, make_pdf):
    make_pdf("scan2.pdf", [[(820, f"- {n} -")] for n in range(1, 6)])
    records, stats = ingest(_sources(tmp_path, [_entry("scan2", "scan2.pdf")]), tmp_path)
    assert records == []
    assert stats["skipped_no_text"] == ["scan2"]
    assert stats["documents"] == 0


def test_ingest_uses_section_spec_and_ignores_other_pages(tmp_path, make_pdf):
    make_pdf("dm.pdf", [
        [(100, "표지 문구")],
        [(100, "주 150분 이상 운동한다.")],
        [(100, "근력운동을 병행한다.")],
        [(100, "약물치료 내용")],
    ])
    sources = _sources(tmp_path, [_entry(
        "dm", "dm.pdf", conditions=["diabetes"],
        sections=[{"title": "운동요법", "pages": [2, 3], "category": "exercise"}],
    )])
    records, stats = ingest(sources, tmp_path)
    assert [(r["heading_path"], r["category"], r["page_start"], r["page_end"]) for r in records] == [
        ("운동요법", "exercise", 2, 3),
    ]
    assert "약물치료" not in records[0]["content"]
    assert stats["no_toc"] == []


def test_ingest_spacing_restores_spaces(tmp_path, make_pdf):
    make_pdf("nospace.pdf", [[(100, "나트륨섭취를줄인다.")]])
    sources = _sources(tmp_path, [_entry("nospace", "nospace.pdf")])
    plain, _ = ingest(sources, tmp_path)
    spaced, _ = ingest(sources, tmp_path, spacing=True)
    body = lambda records: records[0]["content"].split("\n", 1)[1]
    assert " " not in body(plain)
    assert body(spaced) == "나트륨 섭취를 줄인다."
    assert compact(body(spaced)) == compact(body(plain))


def test_ingest_rejects_unknown_parser(tmp_path, make_pdf):
    make_pdf("a.pdf", [[(100, "본문")]])
    sources = _sources(tmp_path, [_entry("a", "a.pdf")])
    with pytest.raises(KeyError):
        ingest(sources, tmp_path, parser="nope")


def test_ingest_drops_reference_list(tmp_path, make_pdf):
    make_pdf("refs.pdf", [[(100, "포화지방 섭취를 줄인다."), (200, "참고문헌"),
                           (260, "1. Ference BA, Ginsberg HN, et al. Eur Heart J 2017;38:2459-2472.")]])
    records, _ = ingest(_sources(tmp_path, [_entry("refs", "refs.pdf")]), tmp_path)
    assert "Ference" not in records[0]["content"] and "참고문헌" not in records[0]["content"]


def test_ingest_paragraph_chunkers(tmp_path, make_pdf):
    make_pdf("p.pdf", [[(100, "운동은 혈압을 낮추고"), (115, "체중 감량에도 도움이 된다."), (200, "금연도 혈압 관리에 중요하다.")]])
    sources = _sources(tmp_path, [_entry("p", "p.pdf")])
    packed, _ = ingest(sources, tmp_path, chunker="paragraph")
    single, _ = ingest(sources, tmp_path, chunker="paragraph-single")
    assert len(packed) == 1 and "낮추고 체중" in packed[0]["content"]
    assert len(single) == 1  # 두 문단 모두 80자 미만이라 하나로 붙는다
    with pytest.raises(KeyError):
        ingest(sources, tmp_path, chunker="nope")


def test_ingest_auto_spacing_only_for_documents_without_spaces(tmp_path, make_pdf):
    make_pdf("nospace.pdf", [[(100, "나트륨섭취를줄이고규칙적으로운동한다.")]])
    make_pdf("spaced.pdf", [[(100, "나트륨 섭취를줄이고 규칙적으로 운동한다.")]])
    sources = _sources(tmp_path, [_entry("nospace", "nospace.pdf"), _entry("spaced", "spaced.pdf")])
    records, _ = ingest(sources, tmp_path, spacing="auto")
    body = {r["doc_id"]: r["content"].split("\n", 1)[1] for r in records}
    assert body["nospace"] == "나트륨 섭취를 줄이고 규칙적으로 운동한다."
    assert body["spaced"] == "나트륨 섭취를줄이고 규칙적으로 운동한다."  # 공백이 있는 문서는 그대로
