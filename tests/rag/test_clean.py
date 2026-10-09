from rag.clean import clean_blocks
from rag.parse import Block

H = 842.0


def _page(n, body):
    return [
        Block(n, 30, 43, "대한고혈압학회 진료지침 2022"),
        Block(n, 300, 313, body),
        Block(n, 810, 823, f"- {n} -"),
    ]


def test_removes_repeated_header_footer_and_page_numbers():
    blocks = [b for n in range(1, 5) for b in _page(n, f"본문 {n}쪽 내용")]
    cleaned = clean_blocks(blocks, [H] * 4)
    assert [b.text for b in cleaned] == [f"본문 {n}쪽 내용" for n in range(1, 5)]


def test_keeps_repeated_text_in_body_area():
    blocks = [Block(n, 300, 313, "나트륨 섭취를 줄인다.") for n in range(1, 5)]
    assert clean_blocks(blocks, [H] * 4) == blocks


def test_keeps_header_when_document_too_short_to_detect():
    blocks = [b for n in range(1, 3) for b in _page(n, "본문")]
    texts = [b.text for b in clean_blocks(blocks, [H] * 2)]
    assert "대한고혈압학회 진료지침 2022" in texts
    assert "- 1 -" not in texts
