"""선정 구성으로 청크를 만들어 Supabase(rag_documents, rag_chunks)와 맞춘다. 저장소 루트에서 `python -m rag.load`.
새 content_hash만 임베딩해 넣고, 사라진 청크·문서는 지운다. 설계: 2026-10-15-rag-runtime-design.md 4.3절."""

import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from app.services.retrieval_service import EMBEDDING_DIMS, EMBEDDING_MODEL, _vector_literal

log = logging.getLogger("rag.load")

SOURCES, RAW, SELECTED = Path("rag/sources.json"), Path("rag/raw"), Path("rag/eval/selected.json")
_PAGE = 1000  # PostgREST 기본 최대 행 수
_INSERT_BATCH = 20  # 임베딩 배치와 같게 해서, 한도에 걸려 멈춰도 넣은 만큼은 남는다
_DELETE_BATCH = 100


def plan_sync(records: list[dict], db_hashes: set[str]) -> tuple[list[dict], set[str]]:
    hashes = {r["content_hash"] for r in records}
    return [r for r in records if r["content_hash"] not in db_hashes], db_hashes - hashes


def _all(sb, table: str, column: str) -> list[dict]:
    rows, start = [], 0
    while True:
        page = sb.table(table).select(column).range(start, start + _PAGE - 1).execute().data
        rows.extend(page)
        if len(page) < _PAGE:
            return rows
        start += _PAGE


def _delete(sb, table: str, column: str, values: list[str]) -> None:
    for i in range(0, len(values), _DELETE_BATCH):
        sb.table(table).delete().in_(column, values[i : i + _DELETE_BATCH]).execute()


def sync(sb, records: list[dict], sources: list[dict], embed_fn: Callable) -> dict:
    sb.table("rag_documents").upsert([
        {"id": s["id"], "title": s["title"], "publisher": s.get("publisher"), "published_year": s.get("year"),
         "url": s.get("url"), "license_note": s.get("license_note")} for s in sources]).execute()
    stale_docs = sorted({d["id"] for d in _all(sb, "rag_documents", "id")} - {s["id"] for s in sources})
    _delete(sb, "rag_documents", "id", stale_docs)  # 청크는 on delete cascade

    to_add, to_delete = plan_sync(records, {r["content_hash"] for r in _all(sb, "rag_chunks", "content_hash")})
    _delete(sb, "rag_chunks", "content_hash", sorted(to_delete))
    model = f"{EMBEDDING_MODEL}@{EMBEDDING_DIMS}"
    for i in range(0, len(to_add), _INSERT_BATCH):
        batch = to_add[i : i + _INSERT_BATCH]
        vectors = embed_fn([r["content"] for r in batch], "document")
        sb.table("rag_chunks").insert([
            {"document_id": r["doc_id"], "content": r["content"], "heading_path": r["heading_path"],
             "page_start": r["page_start"], "page_end": r["page_end"], "conditions": r["conditions"],
             "category": r["category"], "content_hash": r["content_hash"], "embedding_model": model,
             "embedding": _vector_literal(v)} for r, v in zip(batch, vectors, strict=True)]).execute()
        log.info("inserted %d/%d", min(i + _INSERT_BATCH, len(to_add)), len(to_add))
    return {"documents": len(sources), "added": len(to_add), "deleted": len(to_delete),
            "kept": len(records) - len(to_add)}


def _embed_with_retry(texts: list[str], kind: str, attempts: int = 4) -> list[list[float]]:
    from app.services.embedding_service import embed

    for attempt in range(attempts):
        try:
            return embed(texts, kind)
        except Exception as e:
            if attempt == attempts - 1 or not ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                raise
            wait = 15 * 2**attempt
            log.warning("Gemini 한도, %d초 뒤 재시도: %s", wait, e)
            time.sleep(wait)
    raise AssertionError("unreachable")


def main() -> int:
    from app.core.supabase import get_supabase_admin
    from rag.ingest import ingest

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = json.loads(SELECTED.read_text(encoding="utf-8"))
    records, stats = ingest(SOURCES, RAW, cfg["max_chars"], cfg["overlap"], cfg["parser"], cfg["spacing"],
                            chunker=cfg["chunker"])
    log.info("ingest: %s", json.dumps(stats, ensure_ascii=False))
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    print(json.dumps(sync(get_supabase_admin(), records, sources, _embed_with_retry), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
