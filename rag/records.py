import hashlib
import unicodedata

from rag.chunk import Chunk
from rag.sources import SourceDoc


def _category(source: SourceDoc, heading_path: list[str]) -> str:
    joined = " > ".join(heading_path)
    for rule in source.section_rules:
        if rule.match in joined:
            return rule.category
    return source.category


def build_records(source: SourceDoc, chunks: list[Chunk]) -> list[dict]:
    records = []
    for chunk in chunks:
        prefix = " > ".join([source.title, *chunk.heading_path])
        content = unicodedata.normalize("NFC", f"{prefix}\n{chunk.body}")
        records.append({
            "doc_id": source.id,
            "title": source.title,
            "publisher": source.publisher,
            "year": source.year,
            "url": source.url,
            "heading_path": " > ".join(chunk.heading_path) or None,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "conditions": source.conditions,
            "category": _category(source, chunk.heading_path),
            "content": content,
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        })
    return records


def dedupe(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for record in records:
        if record["content_hash"] not in seen:
            seen.add(record["content_hash"])
            unique.append(record)
    return unique
