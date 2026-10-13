import re

from rag.parse import Block

_PAGE_NUMBER = re.compile(r"^[-–—\s]*\d{1,4}[-–—\s]*$")
_EDGE_RATIO = 0.08
_MIN_REPEAT_PAGES = 3
_TOC_LINE = re.compile(r"(?:[.·]{3,}|[…‥]{2,})\s*\d+\s*$")


def _signature(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text)).strip()


def _is_toc_listing(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and sum(bool(_TOC_LINE.search(line)) for line in lines) * 2 >= len(lines)


def clean_blocks(blocks: list[Block], page_heights: list[float], page_widths: list[float]) -> list[Block]:
    def at_edge(block: Block) -> bool:
        height = page_heights[block.page - 1]
        return block.y1 <= height * _EDGE_RATIO or block.y0 >= height * (1 - _EDGE_RATIO)

    def at_side(block: Block) -> bool:
        width = page_widths[block.page - 1]
        return block.x1 <= width * _EDGE_RATIO or block.x0 >= width * (1 - _EDGE_RATIO)

    pages_by_signature: dict[str, set[int]] = {}
    for block in blocks:
        if at_edge(block):
            pages_by_signature.setdefault(_signature(block.text), set()).add(block.page)

    # 장마다 바뀌는 꼬리말도 잡도록 고정 기준. 위아래 여백에만 적용하므로 본문은 영향 없음
    repeated = {sig for sig, pages in pages_by_signature.items() if len(pages) >= _MIN_REPEAT_PAGES}

    return [
        block
        for block in blocks
        if not _is_toc_listing(block.text)
        and not ((at_edge(block) or at_side(block)) and _PAGE_NUMBER.match(block.text))
        and not (at_edge(block) and _signature(block.text) in repeated)
    ]


_REFERENCES_HEADING = re.compile(r"^(참고문헌|references|bibliography)$", re.IGNORECASE)
_REFERENCE_ENTRY = re.compile(r"^\d{1,3}\.\s+[A-Z][A-Za-z'\-]+\s+[A-Z]{1,3},.*(et al|\d{4};\s*\d+)")
_HANGUL = re.compile("[가-힣]")
_LATIN = re.compile("[A-Za-z]")


def _mostly_korean(text: str) -> bool:
    hangul, latin = len(_HANGUL.findall(text)), len(_LATIN.findall(text))
    return hangul >= 2 and hangul >= 0.3 * (hangul + latin)


_BIBLIOGRAPHIC = re.compile(r"\d{4};\s*\d+|et al\.|\b[A-Z][a-z]+ [A-Z]{1,2},")
_BARE_NUMBERS = re.compile(r"^(\d{1,3}\.\s*)+$")


def drop_references(blocks: list[Block]) -> list[Block]:
    """참고문헌을 뺀다.
    - 참고문헌 제목 뒤 블록은 한글 본문이 다시 나올 때까지 뺀다.
    - 한글 문서에서는 위치와 관계없이 영문 서지 블록과 번호만 있는 블록도 뺀다
      (펼침면 PDF는 읽는 순서가 참고문헌과 본문을 오가서 제목 기준만으로는 부족하다)."""
    korean_doc = _mostly_korean(" ".join(b.text for b in blocks))
    out, in_references = [], False
    for block in blocks:
        text = re.sub(r"\s+", " ", block.text).strip()
        if _REFERENCES_HEADING.match(text.replace(" ", "")):
            in_references = True
            continue
        korean = _mostly_korean(text)
        if in_references and not korean:
            continue
        in_references = False
        if _REFERENCE_ENTRY.match(text):
            continue
        if korean_doc and not korean and (_BIBLIOGRAPHIC.search(text) or _BARE_NUMBERS.match(text)):
            continue
        out.append(block)
    return out
