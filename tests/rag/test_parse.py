import pymupdf

from rag.parse import TocEntry, parse_pdf, parse_pdf_plumber


def test_parse_blocks_pages_and_toc(make_pdf):
    path = make_pdf(
        "doc.pdf",
        [
            [(100, "1. 생활요법"), (200, "나트륨 섭취를 줄인다.")],
            [(100, "규칙적인 유산소 운동을 한다.")],
        ],
        toc=[[1, "1. 생활요법", 1]],
    )
    parsed = parse_pdf(path)

    assert [(b.page, b.text) for b in parsed.blocks] == [
        (1, "1. 생활요법"),
        (1, "나트륨 섭취를 줄인다."),
        (2, "규칙적인 유산소 운동을 한다."),
    ]
    assert parsed.blocks[0].y0 < parsed.blocks[1].y0
    assert parsed.toc == [TocEntry(level=1, title="1. 생활요법", page=1)]
    assert parsed.page_heights == [842, 842]


def test_parse_pdf_without_text_layer(make_pdf):
    path = make_pdf("scan.pdf", [[]])
    parsed = parse_pdf(path)
    assert parsed.blocks == []
    assert parsed.toc == []


def test_two_column_page_keeps_column_order(tmp_path):
    import pymupdf

    path = tmp_path / "cols.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i in range(3):
        page.insert_text((72, 100 + 200 * i), f"왼쪽 {i}", fontname="korea", fontsize=11)
    for i in range(3):
        page.insert_text((320, 130 + 200 * i), f"오른쪽 {i}", fontname="korea", fontsize=11)
    doc.save(path)
    doc.close()

    texts = [b.text for b in parse_pdf(path).blocks]
    assert texts == ["왼쪽 0", "왼쪽 1", "왼쪽 2", "오른쪽 0", "오른쪽 1", "오른쪽 2"]


def test_parse_pdf_plumber_matches_pymupdf_text_and_toc(tmp_path):
    path = tmp_path / "en.pdf"
    doc = pymupdf.open()
    for lines in (["Blood pressure guideline", "Reduce salt intake daily."], ["Exercise five days a week."]):
        page = doc.new_page(width=595, height=842)
        for i, text in enumerate(lines):
            page.insert_text((72, 100 + 100 * i), text, fontname="helv", fontsize=11)
    doc.set_toc([[1, "Blood pressure guideline", 1]])
    doc.save(path)
    doc.close()

    a, b = parse_pdf(path), parse_pdf_plumber(path)
    assert [x.text for x in b.blocks] == [x.text for x in a.blocks]
    assert [x.page for x in b.blocks] == [1, 1, 2]
    assert b.toc == a.toc
    assert b.page_heights == a.page_heights
