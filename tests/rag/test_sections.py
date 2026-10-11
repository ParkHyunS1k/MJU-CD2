from rag.parse import Block, TocEntry
from rag.sections import Section, build_sections, sections_from_spec
from rag.sources import SectionSpec


def _b(page, y, text):
    return Block(page, y, y + 12, text, x0=72, x1=300)


BLOCKS = [
    _b(1, 50, "머리말 문단"),
    _b(1, 100, "1. 생활요법"),
    _b(1, 200, "본문A"),
    _b(1, 300, "1.1  나트륨 제한"),
    _b(1, 400, "본문B"),
    _b(2, 100, "본문C"),
    _b(3, 100, "참고문헌"),
    _b(3, 200, "Kim et al. 2020"),
]
TOC = [
    TocEntry(1, "1. 생활요법", 1),
    TocEntry(2, "1.1 나트륨 제한", 1),
    TocEntry(1, "참고문헌", 3),
]


def test_sections_follow_toc_and_drop_references():
    sections = build_sections(BLOCKS, TOC)
    assert [(s.heading_path, [b.text for b in s.blocks]) for s in sections] == [
        ([], ["머리말 문단"]),
        (["1. 생활요법"], ["본문A"]),
        (["1. 생활요법", "1.1 나트륨 제한"], ["본문B", "본문C"]),
    ]


def test_no_toc_single_section():
    assert build_sections(BLOCKS, []) == [Section([], BLOCKS)]


def test_no_blocks_no_sections():
    assert build_sections([], TOC) == []
    assert build_sections([], []) == []


def test_unmatched_title_falls_back_to_page_start_without_losing_text():
    blocks = [_b(1, 100, "서론 내용"), _b(2, 100, "제1장 생활 요법 개요"), _b(2, 200, "본문")]
    toc = [TocEntry(1, "생활요법 총론", 2)]
    sections = build_sections(blocks, toc)
    assert [(s.heading_path, [b.text for b in s.blocks]) for s in sections] == [
        ([], ["서론 내용"]),
        (["생활요법 총론"], ["제1장 생활 요법 개요", "본문"]),
    ]


def test_numbered_references_title_dropped():
    blocks = [_b(1, 100, "1. 운동"), _b(1, 200, "본문"), _b(2, 100, "5. 참고 문헌"), _b(2, 200, "Lee 2019")]
    toc = [TocEntry(1, "1. 운동", 1), TocEntry(1, "5. 참고 문헌", 2)]
    sections = build_sections(blocks, toc)
    assert [s.heading_path for s in sections] == [["1. 운동"]]


def test_heading_merged_with_body_keeps_body():
    block = _b(1, 100, "1. 생활요법\n나트륨 섭취를 하루 2g 이하로 줄인다.\n체중을 감량한다.")
    sections = build_sections([block], [TocEntry(1, "1. 생활요법", 1)])
    assert [(s.heading_path, s.blocks) for s in sections] == [(["1. 생활요법"], [block])]


def test_unmatched_entry_on_same_page_does_not_steal_previous_section():
    blocks = [_b(5, 100, "2.1 나트륨"), _b(5, 200, "나트륨 본문"), _b(5, 300, "2.2. 칼륨 섭취"), _b(5, 400, "칼륨 본문")]
    toc = [TocEntry(2, "2.1 나트륨", 5), TocEntry(2, "2.2 칼륨", 5)]
    sections = build_sections(blocks, toc)
    assert [(s.heading_path, [b.text for b in s.blocks]) for s in sections] == [
        (["2.1 나트륨"], ["나트륨 본문", "2.2. 칼륨 섭취", "칼륨 본문"]),
    ]


def test_sections_from_spec_keeps_only_listed_pages():
    blocks = [_b(1, 100, "표지"), _b(2, 100, "운동 본문1"), _b(3, 100, "운동 본문2"), _b(4, 100, "약물치료"), _b(5, 100, "식사 본문")]
    specs = [
        SectionSpec(title="운동요법", pages=(2, 3), category="exercise"),
        SectionSpec(title="의학영양요법", pages=(5, 5), category="diet"),
    ]
    sections = sections_from_spec(blocks, specs)
    assert [(s.heading_path, [b.text for b in s.blocks]) for s in sections] == [
        (["운동요법"], ["운동 본문1", "운동 본문2"]),
        (["의학영양요법"], ["식사 본문"]),
    ]


def test_sections_from_spec_skips_ranges_without_text():
    specs = [SectionSpec(title="없는 장", pages=(9, 10))]
    assert sections_from_spec([_b(1, 100, "본문")], specs) == []
