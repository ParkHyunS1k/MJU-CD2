from rag.clean import clean_blocks
from rag.parse import Block

H = 842.0
W = 595.0


def _blk(page, y0, y1, text, x0=72, x1=500):
    return Block(page, y0, y1, text, x0=x0, x1=x1)


def _page(n, body):
    return [
        _blk(n, 30, 43, "대한고혈압학회 진료지침 2022"),
        _blk(n, 300, 313, body),
        _blk(n, 810, 823, f"- {n} -"),
    ]


def test_removes_repeated_header_footer_and_page_numbers():
    blocks = [b for n in range(1, 5) for b in _page(n, f"본문 {n}쪽 내용")]
    cleaned = clean_blocks(blocks, [H] * 4, [W] * 4)
    assert [b.text for b in cleaned] == [f"본문 {n}쪽 내용" for n in range(1, 5)]


def test_keeps_repeated_text_in_body_area():
    blocks = [_blk(n, 300, 313, "나트륨 섭취를 줄인다.") for n in range(1, 5)]
    assert clean_blocks(blocks, [H] * 4, [W] * 4) == blocks


def test_keeps_header_when_document_too_short_to_detect():
    blocks = [b for n in range(1, 3) for b in _page(n, "본문")]
    texts = [b.text for b in clean_blocks(blocks, [H] * 2, [W] * 2)]
    assert "대한고혈압학회 진료지침 2022" in texts
    assert "- 1 -" not in texts


def test_keeps_numeric_cells_in_body_area():
    blocks = [_blk(1, 300, 313, "수축기"), _blk(1, 320, 333, "140"), _blk(1, 340, 353, "이완기"), _blk(1, 360, 373, "90")]
    assert clean_blocks(blocks, [H], [W]) == blocks


def test_removes_dot_leader_toc_listing():
    listing = _blk(1, 300, 400, "목차\n1. 생활요법 ········ 13\n1.1 나트륨 제한 ...... 15\n2. 운동 …… 21")
    body = _blk(2, 300, 313, "나트륨 섭취를 줄인다.")
    assert clean_blocks([listing, body], [H, H], [W, W]) == [body]


def test_removes_page_numbers_in_side_margins():
    left = _blk(1, 360, 380, "32", x0=15, x1=30)
    right = _blk(1, 360, 380, "33", x0=565, x1=580)
    body = _blk(1, 360, 380, "140", x0=300, x1=320)
    assert clean_blocks([left, body, right], [H], [W]) == [body]


def test_removes_footer_repeated_in_one_chapter_of_long_document():
    blocks = [_blk(n, 300, 313, f"본문 {n}") for n in range(1, 21)]
    blocks += [_blk(n, 810, 823, "약물치료") for n in range(5, 9)]
    texts = [b.text for b in clean_blocks(blocks, [H] * 20, [W] * 20)]
    assert "약물치료" not in texts
    assert len(texts) == 20
