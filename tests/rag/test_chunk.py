import pytest

from rag.chunk import Chunk, chunk_section
from rag.parse import Block
from rag.sections import Section


def _section(*texts_with_pages, path=("1. 생활요법",)):
    blocks = [Block(page, 100 + i, 110 + i, text, x0=72, x1=300) for i, (page, text) in enumerate(texts_with_pages)]
    return Section(list(path), blocks)


def test_small_section_single_chunk():
    chunks = chunk_section(_section((2, "가나다"), (3, "라마바")), max_chars=100, overlap=10)
    assert chunks == [Chunk(["1. 생활요법"], 2, 3, "가나다\n라마바")]


def test_splits_with_overlap_and_respects_limit():
    texts = [(1, "가" * 40), (1, "나" * 40), (2, "다" * 40), (2, "라" * 40)]
    chunks = chunk_section(_section(*texts), max_chars=100, overlap=45)
    assert all(len(c.body) <= 100 for c in chunks)
    assert chunks[0].body == "가" * 40 + "\n" + "나" * 40
    assert chunks[1].body.startswith("나" * 40)  # 직전 마지막 단위를 이어받음
    assert (chunks[1].page_start, chunks[1].page_end) == (1, 2)
    joined = "".join(c.body for c in chunks)
    for text in ("가" * 40, "나" * 40, "다" * 40, "라" * 40):
        assert text in joined


def test_overlong_block_without_sentence_breaks():
    chunks = chunk_section(_section((1, "가" * 3000)), max_chars=1200, overlap=150)
    assert all(len(c.body) <= 1200 for c in chunks)
    assert sum(c.body.count("가") for c in chunks) >= 3000


def test_long_block_split_on_sentences():
    sentence = "나트륨 섭취를 하루 2g 이하로 줄인다. "
    chunks = chunk_section(_section((1, sentence * 60)), max_chars=300, overlap=0)
    assert len(chunks) > 1
    assert all(len(c.body) <= 300 for c in chunks)
    assert all(c.body.split("\n")[0].startswith("나트륨") for c in chunks)


def test_empty_section_no_chunks():
    assert chunk_section(Section([], []), max_chars=100, overlap=10) == []


@pytest.mark.parametrize("overlap", [-1, 100, 150])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        chunk_section(_section((1, "가")), max_chars=100, overlap=overlap)


def test_overlap_carries_tail_of_long_paragraphs():
    paragraphs = [(1, " ".join(f"{i}번 문단 {j}번 문장을 읽는다." for j in range(12))) for i in range(10)]
    chunks = chunk_section(_section(*paragraphs), max_chars=1200, overlap=150)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        shared = max(k for k in range(151) if prev.body.endswith(nxt.body[:k]))
        assert 0 < shared <= 150
