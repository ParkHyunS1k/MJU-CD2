import re
from dataclasses import dataclass
from typing import Literal

from rag.parse import Block, TocEntry

_REFERENCES = re.compile(r"^(참고문헌|references|bibliography)$", re.IGNORECASE)

# heading: 제목만 있는 블록(버림) / merged: 제목+본문이 한 블록(유지) / page: 제목을 못 찾아 쪽 시작으로 대체
StartKind = Literal["heading", "merged", "page"]


@dataclass(frozen=True)
class Section:
    heading_path: list[str]
    blocks: list[Block]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _is_references(title: str) -> bool:
    return bool(_REFERENCES.match(_compact(re.sub(r"^[\d.\s]+", "", title))))


def _find_start(blocks: list[Block], entry: TocEntry, start: int) -> tuple[int, StartKind]:
    title = _compact(entry.title)
    for i in range(start, len(blocks)):
        block = blocks[i]
        if block.page > entry.page:
            break
        text = _compact(block.text)
        if block.page == entry.page and title in text[: len(title) + 10]:
            return i, "heading" if len(text) <= len(title) + 10 else "merged"
    for i in range(start, len(blocks)):
        if blocks[i].page >= entry.page:
            return i, "page"
    return len(blocks), "page"


def build_sections(blocks: list[Block], toc: list[TocEntry]) -> list[Section]:
    if not toc:
        return [Section([], blocks)] if blocks else []

    boundaries: list[tuple[int, bool, list[str]]] = []  # (시작 인덱스, 제목 블록 버림, 경로)
    path: list[str] = []
    cursor = 0
    for entry in toc:
        path = path[: entry.level - 1] + [entry.title]
        index, kind = _find_start(blocks, entry, cursor)
        if kind == "page" and boundaries:
            previous = boundaries[-1][0]
            if previous < len(blocks) and blocks[previous].page == entry.page:
                continue  # 같은 쪽의 앞 제목 본문을 빼앗지 않도록 경계를 만들지 않음
        boundaries.append((index, kind == "heading", list(path)))
        cursor = index if kind == "page" else index + 1

    sections = []
    if boundaries[0][0] > 0:
        sections.append(Section([], blocks[: boundaries[0][0]]))
    for k, (index, skip_heading, heading_path) in enumerate(boundaries):
        end = boundaries[k + 1][0] if k + 1 < len(boundaries) else len(blocks)
        body = blocks[index + 1 if skip_heading else index : end]
        if body and not any(_is_references(title) for title in heading_path):
            sections.append(Section(heading_path, body))
    return sections
