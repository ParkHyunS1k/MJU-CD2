"""평가 지표. 근거 문장 매칭은 청킹·파서와 무관하게 쓰도록 공백을 지우고 비교한다."""

import math
import re
import unicodedata

import numpy as np


def compact(text: str) -> str:
    # 소프트 하이픈(U+00AD)은 줄바꿈 자리에만 보이는 글자라 지운다
    return re.sub(r"[\s\u00ad]+", "", unicodedata.normalize("NFC", text))


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


def query_metrics(ranked: list[int], found: dict[int, set[int]], n_evidence: int,
                  ks=(1, 3, 5, 10, 20)) -> dict[str, float]:
    """found: 청크 번호 → 그 청크가 담은 근거 번호. 관련 청크 = found에 있는 청크."""

    def ev(top: list[int]) -> float:
        return len(set().union(*(found[i] for i in top if i in found))) / n_evidence

    out = {}
    for k in ks:
        top = ranked[:k]
        out[f"recall@{k}"] = float(any(i in found for i in top))
        out[f"ev_recall@{k}"] = ev(top)
    first = next((r for r, i in enumerate(ranked) if i in found), None)
    out["mrr"] = 0.0 if first is None else 1 / (first + 1)
    dcg = sum(1 / math.log2(r + 2) for r, i in enumerate(ranked[:10]) if i in found)
    idcg = sum(1 / math.log2(r + 2) for r in range(min(10, len(found))))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    if not values:
        return (math.nan, math.nan, math.nan)
    v = np.asarray(values, dtype=float)
    means = np.random.default_rng(seed).choice(v, size=(n, len(v))).mean(axis=1)
    return (float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
