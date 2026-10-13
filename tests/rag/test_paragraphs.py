from rag.chunk import chunk_paragraphs, chunk_single
from rag.paragraphs import merge_paragraphs
from rag.parse import Block
from rag.sections import Section


def _line(page, y0, text, x0=72, h=12):
    return Block(page, y0, y0 + h, text, x0=x0, x1=x0 + 230)


def _texts(blocks):
    return [b.text for b in merge_paragraphs(blocks)]


def test_lines_without_sentence_end_merge_until_sentence_ends():
    lines = [_line(1, 100, "운동은 혈압을 낮추고"), _line(1, 115, "체중 감량에도 도움이 된다."), _line(1, 130, "금연도 중요하다.")]
    assert _texts(lines) == ["운동은 혈압을 낮추고 체중 감량에도 도움이 된다.", "금연도 중요하다."]


def test_large_gap_or_other_column_starts_new_paragraph():
    lines = [_line(1, 100, "식사요법의 원칙"), _line(1, 160, "열량을 줄인다."),
             _line(1, 175, "오른쪽 단의 문장이다.", x0=320)]
    assert _texts(lines) == ["식사요법의 원칙", "열량을 줄인다.", "오른쪽 단의 문장이다."]


def test_paragraph_continues_to_next_column_and_next_page():
    lines = [_line(1, 700, "왼쪽 단 끝에서 문장이"), _line(1, 100, "오른쪽 단으로 이어진다.", x0=320),
             _line(1, 720, "쪽 끝에서 문장이", x0=320), _line(2, 90, "다음 쪽으로 이어진다.")]
    merged = merge_paragraphs(lines)
    assert [b.text for b in merged] == ["왼쪽 단 끝에서 문장이 오른쪽 단으로 이어진다.", "쪽 끝에서 문장이 다음 쪽으로 이어진다."]
    assert (merged[1].page, merged[1].y0) == (1, 720)


def test_soft_hyphen_line_break_joins_word():
    lines = [_line(1, 100, "training should be per­"), _line(1, 115, "formed twice weekly.")]
    assert _texts(lines) == ["training should be performed twice weekly."]


def test_citation_numbers_after_sentence_end_still_end_paragraph():
    lines = [_line(1, 100, "효과가 알려져 있다353,354"), _line(1, 115, "운동 강도는 중강도가 적절하다[12]"),
             _line(1, 130, "다음 문단이다.")]
    assert len(_texts(lines)) == 3


def _section(paragraph_lengths):
    blocks, y = [], 100
    for i, n in enumerate(paragraph_lengths):
        blocks.append(_line(1 + i // 3, y, f"{i}" + "가" * (n - 2) + "."))
        y += 40
    return Section(["장"], blocks)


def test_chunk_paragraphs_packs_whole_paragraphs():
    chunks = chunk_paragraphs(_section([500, 500, 500]), max_chars=1200)
    assert [len(c.body) for c in chunks] == [1001, 500]
    assert chunks[0].body.split("\n")[1].startswith("1")


def test_chunk_paragraphs_splits_only_oversized_paragraph():
    long = "가" * 700 + ". " + "나" * 700 + "."
    chunks = chunk_paragraphs(Section(["장"], [_line(1, 100, long)]), max_chars=1200)
    assert len(chunks) == 2 and chunks[0].body.endswith(".")


def test_chunk_single_one_paragraph_each_and_tiny_ones_join_next():
    blocks = [_line(1, 100, "1. 식사요법"), _line(1, 160, "가" * 150), _line(1, 175, "가" * 150 + "."),
              _line(1, 220, "나" * 150), _line(1, 235, "나" * 150 + ".")]
    chunks = chunk_single(Section(["장"], blocks), max_chars=1200)
    assert len(chunks) == 2
    assert chunks[0].body.startswith("1. 식사요법\n가") and chunks[1].body.startswith("나")
