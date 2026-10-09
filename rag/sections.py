import re
from dataclasses import dataclass

from rag.parse import Block, TocEntry

_REFERENCES = re.compile(r"^(참고문헌|references|bibliography)$", re.IGNORECASE)


@dataclass(frozen=True)
class Section:
    heading_path: list[str]
    blocks: list[Block]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _is_references(title: str) -> bool:
    return bool(_REFERENCES.match(_compact(re.sub(r"^[\d.\s]+", "", title))))


def _find_start(blocks: list[Block], entry: TocEntry, start: int) -> tuple[int, bool]:
    """(섹션 시작 인덱스, 제목 블록을 찾았는지)."""
    title = _compact(entry.title)
    for i in range(start, len(blocks)):
        block = blocks[i]
        if block.page > entry.page:
            break
        if block.page == entry.page and title in _compact(block.text)[: len(title) + 10]:
            return i, True
    for i in range(start, len(blocks)):
        if blocks[i].page >= entry.page:
            return i, False
    return len(blocks), False


def build_sections(blocks: list[Block], toc: list[TocEntry]) -> list[Section]:
    if not toc:
        return [Section([], blocks)] if blocks else []

    starts = []
    cursor = 0
    for entry in toc:
        index, found = _find_start(blocks, entry, cursor)
        starts.append((index, found))
        cursor = index + 1 if found else index

    sections = []
    if starts[0][0] > 0:
        sections.append(Section([], blocks[: starts[0][0]]))

    path: list[str] = []
    for k, entry in enumerate(toc):
        path = path[: entry.level - 1] + [entry.title]
        index, found = starts[k]
        end = starts[k + 1][0] if k + 1 < len(toc) else len(blocks)
        body = blocks[index + 1 if found else index : end]
        if body and not any(_is_references(title) for title in path):
            sections.append(Section(list(path), body))
    return sections
