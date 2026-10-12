"""평가 지표. 근거 문장 매칭은 청킹·파서와 무관하게 쓰도록 공백을 지우고 비교한다."""

import re
import unicodedata


def compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def evidence_coverage(evidence: str, text: str, n: int = 10) -> float:
    e, t = compact(evidence), compact(text)
    if not e:
        return 0.0
    if len(e) < n:
        return 1.0 if e in t else 0.0
    grams = [e[i : i + n] for i in range(len(e) - n + 1)]
    return sum(gram in t for gram in grams) / len(grams)


def contains_evidence(text: str, evidence: str, threshold: float = 0.5) -> bool:
    return evidence_coverage(evidence, text) >= threshold
