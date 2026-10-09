import re

from rag.parse import Block

_PAGE_NUMBER = re.compile(r"^[-–—\s]*\d{1,4}[-–—\s]*$")
_EDGE_RATIO = 0.08


def _signature(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text)).strip()


def clean_blocks(blocks: list[Block], page_heights: list[float]) -> list[Block]:
    def at_edge(block: Block) -> bool:
        height = page_heights[block.page - 1]
        return block.y1 <= height * _EDGE_RATIO or block.y0 >= height * (1 - _EDGE_RATIO)

    pages_by_signature: dict[str, set[int]] = {}
    for block in blocks:
        if at_edge(block):
            pages_by_signature.setdefault(_signature(block.text), set()).add(block.page)

    min_pages = max(3, int(len(page_heights) * 0.3))
    repeated = {sig for sig, pages in pages_by_signature.items() if len(pages) >= min_pages}

    return [
        block
        for block in blocks
        if not _PAGE_NUMBER.match(block.text)
        and not (at_edge(block) and _signature(block.text) in repeated)
    ]
