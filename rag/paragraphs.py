"""PDF 줄 단위 블록을 문단으로 합친다. 문단 청커(rag.chunk.chunk_paragraphs, chunk_single)가 쓴다."""

import re
import statistics
from dataclasses import replace

from rag.parse import Block

_SENTENCE_MARK = (".", "!", "?", "。")
_CITATION_TAIL = re.compile(r"(\[[\d,\s\-–]+\]|\d+(?:[,\-–]\d+)*)$")
_SOFT_HYPHEN = re.compile("\u00ad\\s*")
_COLUMN_TOLERANCE = 15  # 같은 단으로 보는 x0 차이(pt)
_CAPTION = re.compile(r"^(표|그림|Table|Figure|Fig\.)\s*\d+([.\-–]\d+)*\.?\s")
# 문장 끝: 글자 뒤 마침표. 사이에 인용 번호(다363-365.)나 <표14>가 끼어도 되고, 뒤에 인용 번호가 붙어도 된다
_SENTENCE_END = re.compile(r"([가-힣]|[a-z)\]%>])\s*(\d+([,\-–]\d+)*)?[.!?](?=\s|$|\d)")
_MAX_DIGIT_RATIO = 0.2


def ends_sentence(text: str) -> bool:
    t = re.sub(r"\s+", " ", text).strip()
    if t.endswith(_SENTENCE_MARK):
        return True
    return _CITATION_TAIL.sub("", t).rstrip().endswith((*_SENTENCE_MARK, "다", "요"))


def _line_gap(blocks: list[Block]) -> float:
    height = statistics.median(b.y1 - b.y0 for b in blocks)
    tight = [b.y0 - a.y1 for a, b in zip(blocks, blocks[1:])
             if a.page == b.page and abs(a.x0 - b.x0) < _COLUMN_TOLERANCE and 0 <= b.y0 - a.y1 < height]
    return statistics.median(tight) if tight else height * 0.5


def _continues(prev: Block, block: Block, line_gap: float) -> bool:
    if ends_sentence(prev.text):
        return False
    if block.page == prev.page + 1:
        return True  # 쪽 끝에서 문장이 끝나지 않았다
    if block.page != prev.page:
        return False
    if block.y0 < prev.y0:
        return True  # 다음 단 맨 위로 이어진다
    return abs(block.x0 - prev.x0) < _COLUMN_TOLERANCE * 2 and block.y0 - prev.y1 <= line_gap + 4


def _join(a: str, b: str) -> str:
    return a[:-1] + b if a.endswith("\u00ad") else f"{a} {b}"


def merge_paragraphs(blocks: list[Block]) -> list[Block]:
    """같은 단에서 평소 줄 간격으로 이어지고 앞 줄이 문장으로 끝나지 않으면 한 문단으로 합친다."""
    if not blocks:
        return []
    line_gap = _line_gap(blocks)
    out: list[Block] = []
    prev: Block | None = None
    for block in blocks:
        text = re.sub(r"\s+", " ", _SOFT_HYPHEN.sub("\u00ad", block.text)).strip()
        if prev is not None and _continues(prev, block, line_gap):
            last = out[-1]
            out[-1] = replace(last, text=_join(last.text, text), y1=block.y1 if block.page == last.page else last.y1,
                              x0=min(last.x0, block.x0), x1=max(last.x1, block.x1))
        else:
            out.append(replace(block, text=text))
        prev = block
    return [replace(b, text=b.text.replace("\u00ad", "")) for b in out]


def is_table_or_caption(text: str) -> bool:
    """표·그림 캡션, 숫자 위주 표, 문장이 하나도 없는 조각(표 칸, 목록 제목, 소속·주소). 문단 청커가 버린다.
    권고문처럼 문장 뒤에 근거 등급(Ⅰ A)이 붙은 문단은 문장이 있으므로 남는다."""
    t = re.sub(r"\s+", " ", text).strip()
    has_sentence = bool(_SENTENCE_END.search(t))
    if _CAPTION.match(t) and (not has_sentence or len(t) < 200):  # 200자 이상이면 캡션 뒤에 본문이 붙은 문단
        return True
    if sum(c.isdigit() for c in t) >= _MAX_DIGIT_RATIO * len(t):
        return True
    return not has_sentence
