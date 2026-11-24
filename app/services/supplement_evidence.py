"""위험 질환·체성분 유형에 맞는 식약처 기능성 원료를 골라 영양제 근거(Evidence)로 만든다. 벡터 검색은 쓰지 않는다.
원료 목록은 `python -m rag.mfds`가 만든 app/data/mfds_ingredients.json.
설계: docs/superpowers/specs/2026-11-24-mfds-supplement-ingredients-design.md"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from app.services.retrieval_service import Evidence

log = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parents[1] / "data" / "mfds_ingredients.json"
MFDS_DOCUMENT_ID = "mfds"
PER_TAG, LIMIT = 5, 10
TAG_NAMES = {"blood_glucose": "혈당", "blood_pressure": "혈압", "cholesterol": "콜레스테롤",
             "triglyceride": "중성지방", "body_fat": "체지방", "muscle": "근력"}
_DISEASE_TAGS = {"hypertension": "blood_pressure", "cholesterol": "cholesterol", "diabetes": "blood_glucose",
                 "triglyceride": "triglyceride"}
_TYPE_TAGS = {"지방 감량형": ["body_fat"], "지방→근육형": ["body_fat", "muscle"], "근육 보강형": ["muscle"]}


@lru_cache(maxsize=1)
def load_ingredients() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))


def needed_tags(disease_risks: list[dict], metrics: dict) -> list[str]:
    tags = [_DISEASE_TAGS[r["disease_key"]] for r in disease_risks
            if r.get("risk_level") in ("medium", "high") and r.get("disease_key") in _DISEASE_TAGS]
    body = metrics.get("body_type") or {}
    tags += _TYPE_TAGS.get(body.get("type"), [])
    # 근육 판정을 못 해도 체지방 과다는 따로 측정된 사실이라 체지방 원료는 찾는다
    if body.get("type") == "근육 판정 불가" and body.get("fat_excess"):
        tags.append("body_fat")
    return list(dict.fromkeys(tags))


def select_ingredients(ingredients: list[dict], tags: list[str], per_tag: int = PER_TAG,
                       limit: int = LIMIT) -> list[tuple[str, dict]]:
    """태그마다 고시형 먼저(같은 그룹은 파일 순서) per_tag개, 태그별 1위부터 번갈아 limit개. 같은 이름은 한 번만."""
    groups = []
    for tag in tags:
        hits = [i for i in ingredients if tag in i["tags"]]
        groups.append((tag, ([i for i in hits if i["notified"]] + [i for i in hits if not i["notified"]])[:per_tag]))
    picked, seen = [], set()
    for rank in range(per_tag):
        for tag, group in groups:
            if rank < len(group) and group[rank]["name"] not in seen and len(picked) < limit:
                seen.add(group[rank]["name"])
                picked.append((tag, group[rank]))
    return picked


def _evidence(sid: str, index: int, tag: str, ing: dict) -> Evidence:
    kind = "고시형" if ing["notified"] else f"개별인정형 제{ing['recognition_no']}호"
    content = (f"{ing['name']}\n주된 기능성: {ing['function']}\n1일 섭취량: {ing['intake'] or '정보 없음'}\n"
               f"섭취 시 주의사항: {ing['cautions'] or '없음'}")
    return Evidence(sid=sid, chunk_id=-(index + 1), document_id=MFDS_DOCUMENT_ID, title=ing["name"],
                    publisher="식품의약품안전처", year=int(ing["updated"][:4]) if ing["updated"] else None, url=None,
                    heading_path=kind, page_start=None, page_end=None, content=content, similarity=1.0,
                    condition="general", category="supplement", chunk_category="supplement",
                    trait=f"{TAG_NAMES[tag]} 기능성 원료")


def supplement_evidence(disease_risks: list[dict], metrics: dict, start: int,
                        ingredients: list[dict] | None = None) -> list[Evidence]:
    """start는 첫 S번호(검색 근거 수 + 1). 원료 파일을 못 읽으면 영양제 근거 없이 리포트를 만든다."""
    if ingredients is None:
        try:
            ingredients = load_ingredients()
        except (OSError, ValueError):
            log.exception("failed to load mfds ingredients from %s", DATA)
            return []
    picked = select_ingredients(ingredients, needed_tags(disease_risks, metrics))
    return [_evidence(f"S{start + i}", i, tag, ing) for i, (tag, ing) in enumerate(picked)]
