import re
from dataclasses import dataclass

from rag.paragraphs import merge_paragraphs
from rag.sections import Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TAIL_START = re.compile(r"(?<=[.!?])\s+|\n")

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


def _tail(body: str, overlap: int) -> str:
    """직전 청크 끝 overlap 글자. 그 안에 문장·줄 경계가 있으면 경계 다음부터."""
    tail = body[-overlap:] if overlap else ""
    match = _TAIL_START.search(tail)
    return tail[match.end():] if match and match.end() < len(tail) else tail


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
            tail = _tail(chunks[-1].body, overlap)
            carried = [(tail, current[-1][1])] if tail else []
            current = carried if _size(carried + [unit]) <= max_chars else []
        current.append(unit)

    if current:
        emit()
    return chunks


def chunk_paragraphs(section: Section, max_chars: int = 1200) -> list[Chunk]:
    """문단을 max_chars까지 쌓는다. 문단 중간은 자르지 않고, 넘치는 문단만 문장 단위로 나눈다."""
    return chunk_section(Section(section.heading_path, merge_paragraphs(section.blocks)), max_chars, overlap=0)


def chunk_single(section: Section, max_chars: int = 1200, min_chars: int = 80) -> list[Chunk]:
    """문단 하나가 청크 하나. min_chars 미만 조각(소제목 등)은 다음 문단에 붙인다."""
    pieces = [c for p in merge_paragraphs(section.blocks)
              for c in chunk_section(Section(section.heading_path, [p]), max_chars, overlap=0)]
    chunks, pending = [], []
    for piece in pieces:
        pending.append(piece)
        if sum(len(p.body) for p in pending) >= min_chars:
            chunks.append(Chunk(list(section.heading_path), pending[0].page_start, pending[-1].page_end,
                                "\n".join(p.body for p in pending)))
            pending = []
    if pending:
        chunks.append(Chunk(list(section.heading_path), pending[0].page_start, pending[-1].page_end,
                            "\n".join(p.body for p in pending)))
    return chunks
