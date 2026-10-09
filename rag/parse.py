import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class Block:
    page: int  # 1부터 시작
    y0: float
    y1: float
    text: str


@dataclass(frozen=True)
class TocEntry:
    level: int
    title: str
    page: int  # 1부터 시작


@dataclass(frozen=True)
class ParsedDoc:
    blocks: list[Block]
    toc: list[TocEntry]
    page_heights: list[float]


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def parse_pdf(path: Path) -> ParsedDoc:
    blocks = []
    heights = []
    with pymupdf.open(path) as doc:
        for page in doc:
            heights.append(page.rect.height)
            for _x0, y0, _x1, y1, text, _no, kind in page.get_text("blocks"):
                text = _nfc(text)
                if kind == 0 and text:
                    blocks.append(Block(page.number + 1, y0, y1, text))
        toc = [
            TocEntry(level, _nfc(title), page_no)
            for level, title, page_no, *_ in doc.get_toc(simple=True)
            if page_no >= 1
        ]
    return ParsedDoc(blocks, toc, heights)
