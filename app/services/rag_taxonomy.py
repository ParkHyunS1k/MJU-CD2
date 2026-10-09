"""RAG 공통 용어집. 코퍼스 청크 태그와 예측 모델 출력(RiskSignal)이 공유하는 키."""

from typing import Literal

CONDITION_KEYS = frozenset({
    "hypertension",
    "diabetes",
    "dyslipidemia",
    "obesity",
    "metabolic_syndrome",
    "fatty_liver",
    "sarcopenia",
    "general",
})

Category = Literal["lifestyle", "diet", "exercise", "supplement", "criteria"]
