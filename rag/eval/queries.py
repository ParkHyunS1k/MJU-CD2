"""평가용 검색 쿼리 세트. 저장소 루트에서 `python -m rag.eval.queries`."""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from app.services.rag_taxonomy import CONDITION_NAMES, SCENARIOS, keyword_query  # 평가와 런타임이 같은 문구를 쓴다

CATEGORY_NAMES = {"criteria": "진단 기준", "diet": "식사", "lifestyle": "생활습관", "exercise": "운동"}

_INBODY_QUERIES = [
    "체지방률 32% 40대 여성 비만 기준",
    "골격근량 감소 노인 근력운동 방법",
    "내장지방 높고 허리둘레 큰 남성 대사증후군 위험",
]
_UNANSWERABLE_QUERIES = [
    "통풍 환자 식이요법 퓨린 제한",
    "갑상선 기능저하증 운동 주의사항",
    "임신 중 철분제 복용 시기",
]


def _split(index: int) -> str:
    return "tune" if index % 2 == 0 else "test"


def build_queries(draft_question: Callable[[str, str], str]) -> list[dict]:
    queries = []
    for i, (condition, category) in enumerate(SCENARIOS):
        base = {"condition": condition, "category": category, "split": _split(i)}
        queries.append({"query_id": f"{condition}-{category}-keyword", "text": keyword_query(condition, category),
                        "form": "keyword", **base})
        queries.append({"query_id": f"{condition}-{category}-question",
                        "text": draft_question(condition, category).strip(), "form": "question", **base})
    for i, text in enumerate(_INBODY_QUERIES, 1):
        queries.append({"query_id": f"inbody-{i}", "text": text, "form": "inbody",
                        "condition": "mixed", "category": "mixed", "split": _split(i)})
    for i, text in enumerate(_UNANSWERABLE_QUERIES, 1):
        queries.append({"query_id": f"none-{i}", "text": text, "form": "unanswerable",
                        "condition": "none", "category": "none", "split": _split(i)})
    return queries


class _Question(BaseModel):
    question: str


def _gemini_drafter() -> Callable[[str, str], str]:
    from rag.eval.llm import generate_json, get_client

    client = get_client()

    def draft(condition: str, category: str) -> str:
        topic = "일반 성인" if condition == "general" else CONDITION_NAMES[condition]
        prompt = (
            "건강검진과 인바디 결과를 받은 일반인이 건강 앱에 물어볼 법한 질문을 한국어 한 문장으로 쓰세요. "
            "40자 이내, 의학 용어보다 일상어 위주, 질문만 출력하세요.\n"
            f"주제: {topic} / {CATEGORY_NAMES[category]}"
        )
        return generate_json(client, prompt, _Question).question

    return draft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="평가용 검색 쿼리 생성")
    parser.add_argument("--out", type=Path, default=Path("rag/eval/data/queries.jsonl"))
    args = parser.parse_args(argv)
    queries = build_queries(_gemini_drafter())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"wrote {len(queries)} queries to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
