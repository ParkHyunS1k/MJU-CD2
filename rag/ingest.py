"""sources.json의 PDF를 청크 JSONL로 변환한다. 저장소 루트에서 `python -m rag.ingest`."""

import argparse
import json
import logging
import sys
from pathlib import Path

from rag.chunk import chunk_section
from rag.clean import clean_blocks
from rag.parse import parse_pdf
from rag.records import build_records, dedupe
from rag.sections import build_sections, sections_from_spec
from rag.sources import SourcesError, load_sources

log = logging.getLogger("rag.ingest")


def ingest(
    sources_path: Path,
    raw_dir: Path,
    max_chars: int = 1200,
    overlap: int = 150,
) -> tuple[list[dict], dict]:
    stats = {"documents": 0, "chunks": 0, "no_toc": [], "skipped_no_text": [], "failed": []}
    records: list[dict] = []
    for source, path in load_sources(sources_path, raw_dir):
        try:
            parsed = parse_pdf(path)
        except Exception:
            log.exception("[%s] failed to parse %s", source.id, path)
            stats["failed"].append(source.id)
            continue
        blocks = clean_blocks(parsed.blocks, parsed.page_heights, parsed.page_widths)
        if not blocks:
            log.warning("[%s] no text layer (scanned PDF?); skipped", source.id)
            stats["skipped_no_text"].append(source.id)
            continue
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

        chunks = [chunk for section in sections for chunk in chunk_section(section, max_chars, overlap)]
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        records, stats = ingest(args.sources, args.raw_dir, args.max_chars, args.overlap)
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
