from rag.parse import TocEntry, parse_pdf


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
