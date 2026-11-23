import pytest

from rag.chunk import chunk_paragraphs, chunk_single
from rag.paragraphs import is_table_or_caption, merge_paragraphs
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
    blocks = [_line(1, 100, "식사요법을 지킨다."), _line(1, 160, "가" * 150), _line(1, 175, "가" * 150 + "."),
              _line(1, 220, "나" * 150), _line(1, 235, "나" * 150 + ".")]
    chunks = chunk_single(Section(["장"], blocks), max_chars=1200)
    assert len(chunks) == 2
    assert chunks[0].body.startswith("식사요법을 지킨다.\n가") and chunks[1].body.startswith("나")


@pytest.mark.parametrize("text", [
    "표 3-1. 지질개선 효과가 입증된 식사패턴 식사패턴 권고 내용",
    "그림1. 주요고혈압진료지침의혈압분류비교",
    "Table 1. Consensus process for development of the strategies Stage Procedure Output",
    "2007 2008 2009 2010 2011 2012 prevalence(%) 전체 남자 여자",
    "Regular eating and nutritional balance Maintain three regular meals; include protein",
    "➌ 음주 관리",
    "Department of Statistics, Soongsil University, 369 Sangdo-ro, Seoul 06978, Korea Tel",
])
def test_tables_captions_and_labels_are_dropped(text):
    assert is_table_or_caption(text)


@pytest.mark.parametrize("text", [
    "유산소 운동과 저항성 운동을 병행하는 복합운동을 권고한다. Ⅰ B",
    "권고 8. 저항운동은 일주일에 2회 이상 한다. [무작위대조군연구, 일반적권고]",
    "표 1에 제시된 5개 항목들 중에서 3개 이상을 가지고 있는 경우를 대사증후군으로 정의한다.",
    "식이섬유 섭취 수준이 높은 식사 패턴이 중성지방과 역의 상관관계를 보이는 것이 확인되었다.67,81,82",
    "저항성운동은근량과근력을증가시키고긍정적영향을미친다363-365. 일부메타분석연구에서는",
    "생활요법은혈압감소효과가뚜렷하기때문에모든사람에서적극적으로권장된다<표14>309-313. 바람직한생활습관의실천은",
    "Korean eating culture is distinguished by shared meals. Families often share a high",
])
def test_sentences_and_recommendations_are_kept(text):
    assert not is_table_or_caption(text)


def test_paragraph_chunkers_drop_tables_and_captions():
    blocks = [_line(1, 100, "표 2. 운동 강도 빈도 주 5일 이상"), _line(1, 160, "운동은 혈압을 낮춘다."),
              _line(1, 220, "2019 2020 2021 2022 38.4 37.1")]
    assert chunk_paragraphs(Section(["장"], blocks))[0].body == "운동은 혈압을 낮춘다."
    assert [c.body for c in chunk_single(Section(["장"], blocks))] == ["운동은 혈압을 낮춘다."]
