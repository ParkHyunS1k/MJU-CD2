"""리포트 근거 검색. 쿼리를 한 번에 임베딩하고 쿼리마다 match_rag_chunks RPC를 병렬로 부른다.
설정값은 평가에서 고른 구성(rag/eval/selected.json)과 같아야 한다. 설계: 2026-10-15-rag-runtime-design.md 5.3절."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

from app.services.embedding_service import embed
from app.services.rag_taxonomy import Query

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMS = 768
FINAL_K = 3
MIN_SIMILARITY = 0.7028
MAX_EVIDENCE = 15
TIMEOUT_S = 10.0  # 실측: 쿼리 13개 임베딩 2.1초 + RPC 0.3초. 리포트 생성(약 36초)에 비하면 작은 여유
_WORKERS = 8


class RetrievalError(Exception):
    pass


@dataclass(frozen=True)
class Evidence:
    sid: str
    chunk_id: int
    document_id: str
    title: str
    publisher: str | None
    year: int | None
    url: str | None
    heading_path: str | None
    page_start: int | None
    page_end: int | None
    content: str
    similarity: float
    condition: str
    category: str  # 이 청크를 찾은 쿼리의 분야(근거 머리글용)
    chunk_category: str | None = None  # 청크 자신의 분야. 영양제 인용은 이것으로 거른다. 옛 RPC면 None
    trait: str | None = None  # 특성 쿼리로 찾았으면 그 특성 이름


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"


def _match(vector: list[float]) -> list[dict]:
    from app.core.supabase import get_supabase_admin

    res = get_supabase_admin().rpc("match_rag_chunks", {
        "query_embedding": _vector_literal(vector), "match_count": FINAL_K, "min_similarity": MIN_SIMILARITY,
    }).execute()
    return res.data or []


def select_evidence(queries: list[Query], results: list[list[dict]], limit: int = MAX_EVIDENCE) -> list[Evidence]:
    """쿼리마다 1위를 먼저, 그다음 2위… 순서로 모은다. 한 질환이 근거를 독차지하지 않게 하기 위해서다."""
    picked: list[tuple[Query, dict]] = []
    seen: set[int] = set()
    depth = max((len(rows) for rows in results), default=0)
    for rank in range(depth):
        for query, rows in zip(queries, results, strict=True):
            if rank >= len(rows) or rows[rank]["chunk_id"] in seen:
                continue
            seen.add(rows[rank]["chunk_id"])
            picked.append((query, rows[rank]))
            if len(picked) == limit:
                break
        if len(picked) == limit:
            break
    return [Evidence(sid=f"S{i}", chunk_id=row["chunk_id"], document_id=row["document_id"], title=row["title"],
                     publisher=row.get("publisher"), year=row.get("published_year"), url=row.get("url"),
                     heading_path=row.get("heading_path"), page_start=row.get("page_start"),
                     page_end=row.get("page_end"), content=row["content"], similarity=row["similarity"],
                     condition=query.condition, category=query.category, chunk_category=row.get("category"))
            for i, (query, row) in enumerate(picked, 1)]


def retrieve(queries: list[Query], embed_fn: Callable = embed, match_fn: Callable | None = None,
             timeout: float = TIMEOUT_S) -> list[Evidence]:
    """임베딩·RPC 중 하나라도 실패하거나 timeout을 넘기면 RetrievalError."""
    if not queries:
        return []
    match_fn = match_fn or _match
    pool = ThreadPoolExecutor(max_workers=_WORKERS)

    def run() -> list[list[dict]]:
        vectors = embed_fn([q.text for q in queries], "query")
        return [f.result() for f in [pool.submit(match_fn, v) for v in vectors]]

    outer = ThreadPoolExecutor(max_workers=1)
    try:
        results = outer.submit(run).result(timeout=timeout)
    except FutureTimeout as e:
        raise RetrievalError(f"검색이 {timeout}초를 넘었습니다") from e
    except Exception as e:
        raise RetrievalError(str(e)) from e
    finally:
        # ponytail: 시간을 넘긴 스레드는 끝날 때까지 남는다. 요청은 바로 돌려주고, 스레드 누수가 문제 되면 httpx 타임아웃을 건다
        outer.shutdown(wait=False, cancel_futures=True)
        pool.shutdown(wait=False, cancel_futures=True)
    # 특성 쿼리를 더해도 질환 쿼리마다 1위 근거는 들어가도록 상한을 쿼리 수 이상으로 둔다
    return select_evidence(queries, results, limit=max(MAX_EVIDENCE, len(queries)))
