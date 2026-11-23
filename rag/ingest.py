"""sources.json의 PDF를 청크 JSONL로 변환한다. 저장소 루트에서 `python -m rag.ingest`."""

import argparse
import json
import logging
import re
import sys
from dataclasses import replace
from functools import cache
from pathlib import Path

from rag.chunk import chunk_paragraphs, chunk_section, chunk_single
from rag.clean import clean_blocks, drop_references
from rag.kdca import load_sections as load_kdca_sections
from rag.parse import Block, parse_pdf, parse_pdf_plumber
from rag.records import build_records, dedupe
from rag.sections import build_sections, sections_from_spec
from rag.sources import SourcesError, load_sources

log = logging.getLogger("rag.ingest")

PARSERS = {"pymupdf": parse_pdf, "pdfplumber": parse_pdf_plumber}
CHUNKERS = {
    "chars": chunk_section,
    "paragraph": lambda section, max_chars, overlap: chunk_paragraphs(section, max_chars),
    "paragraph-single": lambda section, max_chars, overlap: chunk_single(section, max_chars),
}


@cache
def _kiwi():
    from kiwipiepy import Kiwi  # 띄어쓰기 실험에서만 필요한 의존성

    return Kiwi()


def _space(blocks: list[Block]) -> list[Block]:
    return [replace(b, text=_kiwi().space(b.text)) for b in blocks]


_HANGUL_PAIR_GAP = re.compile(r"(?<=[가-힣]) (?=[가-힣])")
_MIN_SPACE_RATIO = 0.1  # 정상 한국어는 한글 글자당 공백 0.2~0.3. 이보다 훨씬 적으면 PDF에 공백이 기록되지 않은 문서


def _lacks_spaces(blocks: list[Block]) -> bool:
    text = " ".join(b.text for b in blocks)
    hangul = len(re.findall("[가-힣]", text))
    return hangul > 0 and len(_HANGUL_PAIR_GAP.findall(text)) / hangul < _MIN_SPACE_RATIO


def ingest(
    sources_path: Path,
    raw_dir: Path,
    max_chars: int = 1200,
    overlap: int = 150,
    parser: str = "pymupdf",
    spacing: bool | str = False,  # True: 모든 문서, "auto": 공백이 거의 없는 문서만
    chunker: str = "chars",
) -> tuple[list[dict], dict]:
    stats = {"documents": 0, "chunks": 0, "no_toc": [], "skipped_no_text": [], "failed": []}
    records: list[dict] = []
    parse = PARSERS[parser]
    make_chunks = CHUNKERS[chunker]
    for source, path in load_sources(sources_path, raw_dir):
        if source.kind == "kdca":
            sections = load_kdca_sections(path, source.exclude_sections, source.exclude_paragraphs)
            chunks = [chunk for section in sections for chunk in make_chunks(section, max_chars, overlap)]
            # 웹 문서라 쪽 번호가 없다
            records.extend({**r, "page_start": None, "page_end": None} for r in build_records(source, chunks))
            stats["documents"] += 1
            continue
        try:
            parsed = parse(path)
        except Exception:
            log.exception("[%s] failed to parse %s", source.id, path)
            stats["failed"].append(source.id)
            continue
        blocks = drop_references(clean_blocks(parsed.blocks, parsed.page_heights, parsed.page_widths))
        if not blocks:
            log.warning("[%s] no text layer (scanned PDF?); skipped", source.id)
            stats["skipped_no_text"].append(source.id)
            continue
        if spacing is True or (spacing == "auto" and _lacks_spaces(blocks)):
            blocks = _space(blocks)
        if source.sections:
            sections = sections_from_spec(blocks, source.sections)
            found = {section.heading_path[0] for section in sections}
            for spec in source.sections:
                if spec.title not in found:
                    log.warning("[%s] section %r pages %s has no text", source.id, spec.title, spec.pages)
        else:
            if not parsed.toc:
                stats["no_toc"].append(source.id)
            sections = build_sections(blocks, parsed.toc)

        chunks = [chunk for section in sections for chunk in make_chunks(section, max_chars, overlap)]
        records.extend(build_records(source, chunks))
        stats["documents"] += 1

    records = dedupe(records)
    stats["chunks"] = len(records)
    return records, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 코퍼스 수집: PDF → 청크 JSONL")
    parser.add_argument("--sources", type=Path, default=Path("rag/sources.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("rag/raw"))
    parser.add_argument("--out", type=Path, default=Path("rag/data/chunks.jsonl"))
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--parser", choices=sorted(PARSERS), default="pymupdf")
    parser.add_argument("--spacing", choices=["off", "on", "auto"], default="off",
                        help="kiwi 띄어쓰기 복원: on은 모든 문서, auto는 공백이 거의 없는 문서만")
    parser.add_argument("--chunker", choices=sorted(CHUNKERS), default="chars",
                        help="chars: 글자 수로 쌓기, paragraph: 문단 경계를 지키며 쌓기, paragraph-single: 문단 하나씩")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        records, stats = ingest(args.sources, args.raw_dir, args.max_chars, args.overlap, args.parser,
                                {"off": False, "on": True, "auto": "auto"}[args.spacing], args.chunker)
    except SourcesError as e:
        log.error("%s", e)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("wrote %s", args.out)
    log.info("stats: %s", json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
