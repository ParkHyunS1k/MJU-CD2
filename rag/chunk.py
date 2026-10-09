import re
from dataclasses import dataclass

from rag.sections import Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

Unit = tuple[str, int]  # (텍스트, 쪽 번호)


@dataclass(frozen=True)
class Chunk:
    heading_path: list[str]
    page_start: int
    page_end: int
    body: str


def _units(section: Section, max_chars: int) -> list[Unit]:
    units = []
    for block in section.blocks:
        text = re.sub(r"\s+", " ", block.text).strip()
        pieces = [text] if len(text) <= max_chars else _SENTENCE_END.split(text)
        for piece in pieces:
            for i in range(0, len(piece), max_chars):
                units.append((piece[i : i + max_chars], block.page))
    return units


def _size(units: list[Unit]) -> int:
    return sum(len(text) for text, _ in units) + max(len(units) - 1, 0)


def chunk_section(section: Section, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    chunks: list[Chunk] = []
    current: list[Unit] = []

    def emit() -> None:
        pages = [page for _, page in current]
        body = "\n".join(text for text, _ in current)
        chunks.append(Chunk(list(section.heading_path), min(pages), max(pages), body))

    for unit in _units(section, max_chars):
        if current and _size(current + [unit]) > max_chars:
            emit()
            carried: list[Unit] = []
            for previous in reversed(current):
                if _size([previous] + carried) > overlap:
                    break
                carried.insert(0, previous)
            current = carried if _size(carried + [unit]) <= max_chars else []
        current.append(unit)

    if current:
        emit()
    return chunks
