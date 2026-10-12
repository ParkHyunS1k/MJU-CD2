"""평가용 검색 쿼리 세트. 저장소 루트에서 `python -m rag.eval.queries`."""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

CONDITION_NAMES = {
    "hypertension": "고혈압",
    "diabetes": "당뇨병",
    "dyslipidemia": "이상지질혈증",
    "obesity": "비만",
    "metabolic_syndrome": "대사증후군",
    "fatty_liver": "지방간",
    "sarcopenia": "근감소증",
}
CATEGORY_NAMES = {"criteria": "진단 기준", "diet": "식사", "lifestyle": "생활습관", "exercise": "운동"}

SCENARIOS: list[tuple[str, str]] = [
    ("hypertension", "criteria"), ("hypertension", "diet"), ("hypertension", "lifestyle"), ("hypertension", "exercise"),
    ("diabetes", "diet"), ("diabetes", "exercise"), ("diabetes", "lifestyle"),
    ("dyslipidemia", "criteria"), ("dyslipidemia", "diet"), ("dyslipidemia", "lifestyle"), ("dyslipidemia", "exercise"),
    ("obesity", "criteria"), ("obesity", "lifestyle"), ("obesity", "exercise"),
    ("metabolic_syndrome", "criteria"), ("metabolic_syndrome", "lifestyle"),
    ("fatty_liver", "criteria"), ("fatty_liver", "diet"), ("fatty_liver", "exercise"), ("fatty_liver", "lifestyle"),
    ("sarcopenia", "criteria"), ("sarcopenia", "exercise"),
    ("general", "diet"), ("general", "exercise"),
]

_KEYWORD_TEMPLATES = {
    "criteria": "{name} 진단 기준 분류",
    "diet": "{name} 예방 식이요법 식사 관리",
    "lifestyle": "{name} 생활습관 교정 체중 감량 금연 절주",
    "exercise": "{name} 운동요법 유산소 근력운동 권고",
}
_GENERAL_KEYWORDS = {
    "diet": "한국인 영양소 섭취기준 권장섭취량 상한섭취량",
    "exercise": "성인 신체활동 권장량 유산소 근력운동",
}
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


def keyword_query(condition: str, category: str) -> str:
    if condition == "general":
        return _GENERAL_KEYWORDS[category]
    return _KEYWORD_TEMPLATES[category].format(name=CONDITION_NAMES[condition])


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
