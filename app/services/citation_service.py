"""생성된 리포트의 인용을 이번 근거 목록에 맞춘다. 설계: 2026-10-15-rag-runtime-design.md 6.3절."""

import copy

from app.services.retrieval_service import Evidence
from app.services.supplement_evidence import MFDS_DOCUMENT_ID


def _clean(ids: list[str], valid: set[str]) -> list[str]:
    out: list[str] = []
    for raw in ids:
        sid = raw.strip().upper()
        if sid in valid and sid not in out:
            out.append(sid)
    return out


def _page(e: Evidence) -> str | None:
    if e.page_start is None:
        return None
    return str(e.page_start) if e.page_end in (None, e.page_start) else f"{e.page_start}–{e.page_end}"


def _cited_items(content: dict) -> list[dict]:
    return [*content["risk_report"], *content["lifestyle_plan"], *content["supplement_candidates"],
            *content["body_composition_report"]["watch_points"]]


def validate_citations(content: dict, evidence: list[Evidence]) -> tuple[dict, list[dict]]:
    """없는 번호를 지우고, 영양제 근거를 인용하지 않은 영양제를 빼고, 인용된 근거만 sources로 돌려준다."""
    content = copy.deepcopy(content)
    valid = {e.sid for e in evidence}
    for item in _cited_items(content):
        item["source_ids"] = _clean(item.get("source_ids", []), valid)
    # 영양제는 영양제(supplement) 문단을 인용할 때만 남긴다. 식사요법 문단의 식품 언급은 보충제 근거가 아니다.
    # 쿼리 분야가 아니라 청크 자신의 분야로 본다(영양제 쿼리도 식사 문단을 찾아올 수 있다)
    supplement_sids = {e.sid for e in evidence if e.chunk_category == "supplement"}
    # 원료 근거는 식약처 기능성 원료뿐이다. 포털 영양제 문서는 일반 안내라 함께 인용할 때 주의사항 근거로만 남긴다
    mfds_sids = {e.sid for e in evidence if e.document_id == MFDS_DOCUMENT_ID}
    for s in content["supplement_candidates"]:
        s["source_ids"] = [sid for sid in s["source_ids"] if sid in supplement_sids]
    content["supplement_candidates"] = [s for s in content["supplement_candidates"]
                                        if set(s["source_ids"]) & mfds_sids]
    cited = {sid for item in _cited_items(content) for sid in item["source_ids"]}
    sources = [{"sid": e.sid, "title": e.title, "publisher": e.publisher, "year": e.year, "page": _page(e),
                "heading_path": e.heading_path, "url": e.url}
               for e in evidence if e.sid in cited]
    return content, sources
