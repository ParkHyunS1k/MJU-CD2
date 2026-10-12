"""경로 B: OCR 줄 목록 → 규칙 기반 수치 추출.

규칙
1. 라벨(체중, 체수분 …)을 찾는다. 긴 라벨 우선, 4글자 이상은 1글자 오인식 허용.
   적정체중·체중조절·저체중처럼 대상이 아닌 라벨도 등록해 '체중'으로 잘못 잡히지 않게 막는다.
2. 값 후보: 라벨 뒤 같은 줄 → 같은 행 오른쪽 줄 → (행에서 못 찾으면) 바로 아래 줄.
   다른 라벨을 만나면 멈춘다. 정상범위 "(45.0~55.0)"와 괄호 안 숫자는 제외한다.
3. 막대그래프 눈금은 후보에서 뺀다: 숫자만 있는 줄이 같은 행에 촘촘히 3개 이상 이어지거나,
   한 줄에 같은 형식의 숫자가 오름차순으로 2개 이상 있으면 눈금으로 본다.
4. 후보 우선순위: 범위 안 + 표기 자릿수 일치 > 범위 안 > 소수점 누락 복구(128 → 12.8).
5. 같은 항목이 여러 곳(체성분분석 표, 골격근·지방분석 등)에 나오면 우선순위가 가장 높은 값끼리 다수결.
6. '신체변화(History)' 이후 이력 영역은 무시한다.

InBody270 실물 1장 + 합성 데이터로만 확인했다. 기종이 늘면 tests/vectors에 추가하며 조정한다.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from .ocr_types import OcrLine
from .schema import FIELD_SPECS, InBodyExtraction

PARSER_VERSION = "0.2.0"

LABELS: dict[str, list[str]] = {
    "height_cm": ["신장", "Height"],
    "age": ["나이", "연령", "Age"],
    "weight_kg": ["체중", "Weight"],
    "total_body_water_l": ["체수분", "TotalBodyWater"],
    "protein_kg": ["단백질", "Protein"],
    "minerals_kg": ["무기질", "Minerals"],
    "body_fat_mass_kg": ["체지방량", "체지방", "BodyFatMass"],
    "fat_free_mass_kg": ["제지방량", "FatFreeMass"],
    "skeletal_muscle_mass_kg": ["골격근량", "SkeletalMuscleMass"],
    "bmi": ["BMI", "신체질량지수"],
    "percent_body_fat": ["체지방률", "PercentBodyFat", "PBF"],
    "waist_hip_ratio": ["복부지방률", "WaistHipRatio"],
    "visceral_fat_level": ["내장지방레벨", "VisceralFatLevel"],
    "basal_metabolic_rate_kcal": ["기초대사량", "BasalMetabolicRate"],
}
SEX_LABELS = ["성별", "Gender", "Sex"]
# 추출 대상은 아니지만 라벨로 인식해 값 탐색을 멈추게 하는 단어 (회원번호 등 개인정보 칸 포함)
_STOP = "_stop"
STOP_LABELS = [
    "적정체중", "체중조절", "지방조절", "근육조절", "저체중", "과체중", "심한과체중",
    "부위별체지방분석", "회원번호", "검사일시", *SEX_LABELS,
]
HISTORY_MARKERS = ["신체변화", "History"]

_ALIASES = sorted(
    [(f, a) for f, al in LABELS.items() for a in al] + [(_STOP, a) for a in STOP_LABELS],
    key=lambda t: -len(t[1]),
)

_RANGE = re.compile(r"\d+(?:[.,]\d+)?\s*[~∼〜-]\s*\d+(?:[.,]\d+)?")
_PAREN = re.compile(r"[(\[（][^)\]）]*[)\]）]")
_NUM = re.compile(r"(?<![\d.])\d{1,4}(?:\.\d{1,2})?(?![\d.])")
_NUMERIC_ONLY = re.compile(r"^[\d.\s%]+$")
_DATE = re.compile(r"(20\d{2})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})")


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _clean(text: str) -> str:
    t = re.sub(r"(\d),(\d{1,2})(?!\d)", r"\1.\2", text)  # 소수점이 쉼표로 인식된 경우
    return _PAREN.sub(" ", _RANGE.sub(" ", t))


def _numbers(text: str) -> list[str]:
    return _NUM.findall(_clean(text))


def _decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def _find_labels(text: str) -> list[tuple[str, int]]:
    """줄에서 라벨을 찾아 (field, 라벨 끝 위치) 반환.

    순서: 4글자 이상 라벨 정확 일치 → 4글자 이상 라벨 1글자 오인식 허용 → 짧은 라벨 정확 일치.
    짧은 라벨(체지방)이 오인식된 긴 라벨(체지방룰)을 가로채지 않도록 긴 라벨을 먼저 처리한다.
    """
    s = _squash(text)
    taken = [False] * len(s)
    found: list[tuple[str, int]] = []

    def claim(i: int, n: int, f: str | None) -> None:
        taken[i : i + n] = [True] * n
        if f is not None:
            found.append((f, i + n))

    def exact(aliases):
        for f, alias in aliases:
            start = 0
            while (i := s.find(alias, start)) >= 0:
                if not any(taken[i : i + len(alias)]):
                    claim(i, len(alias), f)
                start = i + 1

    long_aliases = [(f, a) for f, a in _ALIASES if len(a) >= 4]
    exact(long_aliases)
    # 1글자 오인식 허용 (예: 내장지방레벨 → 내정지방레벨).
    # 후보 항목이 둘 이상이면(체지방룰 → 체지방량? 체지방률?) 판단 보류: 자리만 막고 추출하지 않음
    for i in range(len(s)):
        hits = []
        for f, alias in long_aliases:
            n = len(alias)
            w = s[i : i + n]
            if len(w) == n and not any(taken[i : i + n]) and not any(c.isdigit() for c in w):
                if sum(a != b for a, b in zip(alias, w)) == 1:
                    hits.append((f, n))
        if hits:
            f, n = hits[0]
            claim(i, n, f if len({h[0] for h in hits}) == 1 else None)
    exact([(f, a) for f, a in _ALIASES if len(a) < 4])
    return found


def _tick_ids(lines: list[OcrLine]) -> set[int]:
    """막대그래프 눈금으로 보이는 줄의 id 집합."""
    numeric = [l for l in lines if _NUMERIC_ONLY.match(_clean(l.text).strip() or "x") and _numbers(l.text)]
    ticks: set[int] = set()
    for l in numeric:
        nums = _numbers(l.text)
        vals = [float(n) for n in nums]
        if len(nums) >= 2 and len({_decimals(n) for n in nums}) == 1 and vals == sorted(set(vals)):
            ticks.add(id(l))
    rows: list[list[OcrLine]] = []
    for l in sorted(numeric, key=lambda l: l.cy):
        if rows and abs(rows[-1][0].cy - l.cy) <= 0.004:
            rows[-1].append(l)
        else:
            rows.append([l])
    for row in rows:
        row.sort(key=lambda l: l.x0)
        chain = [row[0]]
        for prev, cur in zip(row, row[1:]):
            if cur.x0 - prev.x1 > 0.12:
                if sum(len(_numbers(c.text)) for c in chain) >= 3:
                    ticks.update(id(c) for c in chain)
                chain = []
            chain.append(cur)
        if sum(len(_numbers(c.text)) for c in chain) >= 3:
            ticks.update(id(c) for c in chain)
    return ticks


def _row_right(label: OcrLine, lines: list[OcrLine]) -> list[OcrLine]:
    tol = max(label.height * 0.8, 0.005)
    row = [l for l in lines if l is not label and abs(l.cy - label.cy) <= tol and l.x0 >= label.x1 - 0.01]
    return sorted(row, key=lambda l: l.x0)


def _below(label: OcrLine, lines: list[OcrLine]) -> list[OcrLine]:
    h = max(label.height, 0.005)
    out = [
        l for l in lines
        if l.cy > label.cy + h * 0.8 and l.y0 <= label.y1 + 2.5 * h and l.x0 >= label.x0 - 0.02
    ]
    # 아래로 한 행씩 내려가며 모으다가 다른 라벨이 있는 행에서 멈춘다
    rows: list[list[OcrLine]] = []
    for l in sorted(out, key=lambda l: l.cy):
        if rows and l.cy - rows[-1][0].cy <= h * 0.5:
            rows[-1].append(l)
        else:
            rows.append([l])
    picked: list[OcrLine] = []
    for row in rows:
        if any(_find_labels(l.text) for l in row):
            break
        picked += row
    # 값은 라벨 바로 아래(x가 가까운 곳)에, 그래프 눈금은 오른쪽으로 떨어져 있다 (InBody270 복부지방률·내장지방레벨)
    return sorted(picked, key=lambda l: abs(l.x0 - label.x0))


def _collect(neighbors: list[OcrLine], ticks: set[int]) -> list[str]:
    tokens: list[str] = []
    for other in neighbors:
        if id(other) in ticks:
            continue
        if _find_labels(other.text):  # 다른 라벨을 만나면 멈춤
            break
        tokens += _numbers(other.text)
    return tokens


def _candidates(field_key: str, tokens: list[str]) -> list[tuple[float, int]]:
    """(값, 우선순위) 목록. 2: 범위+자릿수 일치, 1: 범위 안, 0: 소수점 누락 복구."""
    spec = FIELD_SPECS[field_key]
    out = []
    for t in tokens:
        v = float(t)
        if spec.min <= v <= spec.max:
            out.append((v, 2 if _decimals(t) == spec.decimals else 1))
        elif spec.decimals > 0 and "." not in t and len(t) > spec.decimals:
            fixed = v / 10**spec.decimals
            if spec.min <= fixed <= spec.max:
                out.append((fixed, 0))
    return out


@dataclass
class ParseResult:
    extraction: InBodyExtraction
    sources: dict[str, list[str]] = field(default_factory=dict)  # 디버깅: 근거가 된 OCR 텍스트
    parser_version: str = PARSER_VERSION


def parse(lines: list[OcrLine]) -> ParseResult:
    cut = min((l.y0 for l in lines if any(m in _squash(l.text) for m in HISTORY_MARKERS)), default=None)
    if cut is not None:
        lines = [l for l in lines if l.y0 < cut]
    lines = sorted(lines, key=lambda l: (l.y0, l.x0))
    ticks = _tick_ids(lines)

    votes: dict[str, list[tuple[float, int]]] = {}
    sources: dict[str, list[str]] = {}
    for line in lines:
        for f, end in _find_labels(line.text):
            if f == _STOP:
                continue
            cands = _candidates(f, _numbers(_squash(line.text)[end:]) + _collect(_row_right(line, lines), ticks))
            if not any(tier > 0 for _, tier in cands):
                cands += _candidates(f, _collect(_below(line, lines), ticks))
            if cands:
                best = max(cands, key=lambda c: c[1])  # 같은 우선순위면 앞쪽(가까운) 후보
                votes.setdefault(f, []).append(next(c for c in cands if c[1] == best[1]))
                sources.setdefault(f, []).append(line.text)

    values = {}
    for f, vs in votes.items():
        top = max(t for _, t in vs)
        values[f] = Counter(v for v, t in vs if t == top).most_common(1)[0][0]

    return ParseResult(
        InBodyExtraction(
            is_inbody_sheet=len(values) >= 3,
            sex=_parse_sex(lines),
            measured_at=_parse_date(lines),
            **values,
        ),
        sources,
    )


def _parse_sex(lines: list[OcrLine]) -> str | None:
    def classify(s: str) -> str | None:
        if re.search(r"남|Male", s):
            return "M"
        if re.search(r"여|Female", s):
            return "F"
        return None

    for line in lines:
        s = _squash(line.text)
        if any(lb in s for lb in SEX_LABELS):
            tail = s.split(next(lb for lb in SEX_LABELS if lb in s), 1)[1]
            near = [l for l in _row_right(line, lines) + _below(line, lines) if not _find_labels(l.text)]
            for cand in [tail] + [_squash(l.text) for l in near[:2]]:
                if sex := classify(cand):
                    return sex
    for line in lines:  # 라벨 없이 단독으로 인식된 경우
        if _squash(line.text) in ("남", "남성", "Male"):
            return "M"
        if _squash(line.text) in ("여", "여성", "Female"):
            return "F"
    return None


def _parse_date(lines: list[OcrLine]) -> str | None:
    if m := _DATE.search(" ".join(l.text for l in lines)):
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return None
