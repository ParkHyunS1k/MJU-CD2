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
