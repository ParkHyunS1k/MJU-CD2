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
6. '신체변화(History)' 이력 영역은 본문과 따로 읽는다: 시각(HH:MM)으로 열, 왼쪽 라벨로 행을 정한다.
7. 정상범위·운동소비열량은 같은 라벨 규칙으로, 인바디점수("84/100")·임피던스는 전용 형식으로 읽는다.

InBody270 실물 1장 + 합성 데이터로만 확인했다. 기종이 늘면 tests/vectors에 추가하며 조정한다.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from .ocr_types import OcrLine
from .schema import (
    EXERCISE_SPECS,
    FIELD_SPECS,
    IMPEDANCE_FREQS,
    IMPEDANCE_SEGMENTS,
    RANGE_FIELDS,
    HistoryPoint,
    InBodyExtraction,
)

PARSER_VERSION = "0.3.0"

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
    "ideal_weight_kg": ["적정체중"],
    "weight_control_kg": ["체중조절"],
    "fat_control_kg": ["지방조절"],
    "muscle_control_kg": ["근육조절"],
    "obesity_degree_pct": ["비만도"],
    "recommended_calorie_kcal": ["권장섭취열량"],
    **{f"exercise.{k}": [s.label] for k, s in EXERCISE_SPECS.items()},
}
# 운동소비열량도 같은 라벨→값 규칙으로 읽는다
SPECS = {**FIELD_SPECS, **{f"exercise.{k}": s for k, s in EXERCISE_SPECS.items()}}
SEX_LABELS = ["성별", "Gender", "Sex"]
# 추출 대상은 아니지만 라벨로 인식해 값 탐색을 멈추게 하는 단어 (회원번호 등 개인정보 칸 포함)
_STOP = "_stop"
STOP_LABELS = [
    "저체중", "과체중", "심한과체중",
    "부위별체지방분석", "회원번호", "검사일시", *SEX_LABELS,
]
HISTORY_MARKERS = ["신체변화", "History"]
# 이력 그래프 아래 체크박스 글자. 이력 그래프 높이는 이 글자 높이의 약 8배 (InBody270)
HISTORY_FOOTERS = ["최근", "전체"]
# 값이 라벨 바로 아래에 있고 오른쪽에는 그래프 눈금(0.80, 0.90, 10)이 있는 항목 (InBody270).
# 라벨 바로 아래만 본다
VALUE_BELOW = {"waist_hip_ratio", "visceral_fat_level"}
# 위 항목의 라벨 줄은 사진에서 OCR이 통째로 놓치는 일이 잦다. 그때 기준점으로 쓰는 그래프 글자
GRAPH_ANCHORS = {"waist_hip_ratio": "0.80", "visceral_fat_level": "낮음"}
RIGHT_COLUMN_MARKERS = ["임피던스", "운동소비열량", "Impedance"]

_ALIASES = sorted(
    [(f, a) for f, al in LABELS.items() for a in al] + [(_STOP, a) for a in STOP_LABELS],
    key=lambda t: -len(t[1]),
)

_RANGE = re.compile(r"\d+(?:[.,]\d+)?\s*[~∼〜-]\s*\d+(?:[.,]\d+)?")
_PAREN = re.compile(r"[(\[（][^)\]）]*[)\]）]")
# 글자에 붙은 숫자는 제외 (영문 부제 오인식 "1cna"). 단 뒤에 붙은 단위는 허용 ("83.0kg", "183cm").
# 앞의 '-'는 부호로 함께 읽는다 (체중조절 "- 4.4 kg")
_NUM = re.compile(
    r"(?:(?<![\w.~])-\s?)?(?<![\d.A-Za-z])\d{1,4}(?:\.\d{1,2})?(?![\d.])(?=(?i:kg|cm|kcal|l)(?![a-z])|[^A-Za-z]|$)"
)
_DATE = re.compile(r"(20\d{2})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})")
# 출력 설정에 따라 일.월.연도로 나오기도 한다 (InBody270 "16.08.2024. 17:02")
_DATE_DMY = re.compile(r"(?<!\d)(\d{1,2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(20\d{2})(?!\d)")
_TIME = re.compile(r"[.\s]*([01]?\d|2[0-3])\s*:\s*([0-5]\d)(?!\d)")
_SHORT_DATE = re.compile(r"(?<!\d)(\d{2})\.(\d{2})\.(\d{2})(?!\d)")  # 이력 날짜 "25.02.07."
_RANGE_PARTS = re.compile(r"(\d+(?:\.\d+)?)\s*[~∼〜]\s*(\d+(?:\.\d+)?)")


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _clean(text: str) -> str:
    t = re.sub(r"(\d),(\d{1,2})(?!\d)", r"\1.\2", text)  # 소수점이 쉼표로 인식된 경우
    return _PAREN.sub(" ", _RANGE.sub(" ", t))


def _numbers(text: str) -> list[str]:
    return [t.replace(" ", "") for t in _NUM.findall(_clean(text))]


def _numeric_like(text: str) -> bool:
    """숫자만 있는 줄. 짧은 줄은 OCR 잡음 글자 1개까지 허용 ("50o", "5$0" 같은 눈금 오인식)."""
    t = _clean(text).strip()
    stray = len(re.sub(r"[\d.\s%]", "", t))
    return bool(re.search(r"\d", t)) and (stray == 0 or (stray == 1 and len(t) <= 5))


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
    # 후보 항목이 둘 이상이면(체지방룰 → 체지방량? 체지방률?) 판단 보류: 추출하지 않지만
    # 라벨이 있는 자리이므로 _STOP으로 남겨 다른 항목의 값 탐색은 여기서 멈추게 한다
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
            claim(i, n, f if len({h[0] for h in hits}) == 1 else _STOP)
    exact([(f, a) for f, a in _ALIASES if len(a) < 4])
    return found


def _tick_ids(lines: list[OcrLine]) -> set[int]:
    """막대그래프 눈금으로 보이는 줄의 id 집합."""
    numeric = [l for l in lines if _numeric_like(l.text)]
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


def _row_right(label: OcrLine, lines: list[OcrLine], max_x: float = 1.0) -> list[OcrLine]:
    """같은 행 오른쪽 줄. max_x: 이 위치에서 시작하는 줄부터는 다른 열이라 보지 않는다."""
    tol = max(label.height * 0.8, 0.005)
    row = [
        l for l in lines
        if l is not label and abs(l.cy - label.cy) <= tol and label.x1 - 0.01 <= l.x0 < max_x
    ]
    return sorted(row, key=lambda l: l.x0)


def _right_column_x(lines: list[OcrLine]) -> float:
    """오른쪽 열 시작 위치: 오른쪽 열 제목들의 왼쪽 끝. 제목을 못 찾으면 열을 나누지 않는다(1.0)."""
    return min((l.x0 for l in lines if any(m in _squash(l.text) for m in RIGHT_COLUMN_MARKERS)), default=1.0)


def _below(label: OcrLine, lines: list[OcrLine], max_dx: float = 0.3) -> list[OcrLine]:
    """라벨 아래 줄. max_dx: 라벨 오른쪽 끝에서 얼마나 오른쪽까지 볼지 (다른 열의 줄은 제외)."""
    h = max(label.height, 0.005)
    out = [
        l for l in lines
        if l.cy > label.cy + h * 0.8 and l.y0 <= label.y1 + 2.5 * h
        and label.x0 - 0.02 <= l.x0 <= label.x1 + max_dx
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
        if _find_labels(other.text):  # 다른 라벨을 만나면 멈춤. 라벨 앞의 숫자까지는 이 항목 값 ("141 게이트볼")
            tokens += _numbers(re.split(r"[가-힣A-Za-z]", other.text, maxsplit=1)[0])
            break
        tokens += _numbers(other.text)
    return tokens


def _candidates(field_key: str, tokens: list[str]) -> list[tuple[float, int]]:
    """(값, 우선순위) 목록. 2: 범위+자릿수 일치, 1: 범위 안, 0: 소수점 누락 복구."""
    spec = SPECS[field_key]
    out = []
    for t in tokens:
        v = float(t)
        if spec.min >= 0:  # 음수가 없는 항목에 붙은 '-'는 잡음
            v = abs(v)
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


def _estimate_skew(lines: list[OcrLine]) -> float:
    """행 기울기 k (y가 x에 따라 k만큼 변함). 0.2 이상 떨어진 줄 쌍의 기울기 중 가장 많이 나온 값.

    같은 행의 줄 쌍은 모두 같은 기울기를 내고, 다른 행끼리의 쌍은 제각각 흩어지므로 최빈값이 행 기울기다.
    """
    votes: Counter[int] = Counter()
    for i, a in enumerate(lines):
        for b in lines[i + 1 :]:
            dx = (b.x0 + b.x1 - a.x0 - a.x1) / 2
            if abs(dx) < 0.2:
                continue
            s = (b.cy - a.cy) / dx
            if abs(s) <= 0.2:
                votes[round(s / 0.002)] += 1
    if not votes:
        return 0.0
    # 이웃 구간까지 합쳐 최빈 구간을 고른다 (원근 왜곡으로 위치마다 기울기가 조금씩 다른 경우 대비)
    best = max(votes, key=lambda b: sum((4 - abs(d)) * votes[b + d] for d in range(-3, 4)))
    return best * 0.002


def _deskew(lines: list[OcrLine]) -> list[OcrLine]:
    """기울어진 사진 보정: 행 기울기 k를 추정해 y' = y - k·x 로 되돌린다.

    정규화 좌표라 회전 대신 y 방향 기울임(shear)으로 근사한다. 좌우 열 위치는 거의 안 바뀌어
    '같은 행 오른쪽' 판단에 필요한 행 정렬만 바로잡으면 된다. ±약 10°까지 탐색.
    """
    if len(lines) < 10:
        return lines
    k = _estimate_skew(lines)
    if abs(k) < 0.005:  # 폭 전체에서 높이 차 0.5% 미만: 보정하지 않음 (캡처 이미지는 그대로)
        return lines
    out = []
    for l in lines:
        # 기울어진 줄의 박스는 폭에 비례해 위아래로 부풀어 있으므로 원래 높이로 줄인다
        shrink = min(abs(k) * (l.x1 - l.x0), l.height * 0.6) / 2
        dy = k * (l.x0 + l.x1) / 2
        out.append(OcrLine(l.text, l.x0, l.y0 - dy + shrink, l.x1, l.y1 - dy - shrink))
    return out


def _from_graph_anchor(
    f: str, anchor: str, lines: list[OcrLine], ticks: set[int]
) -> tuple[float, list[str]] | None:
    """라벨을 OCR이 놓쳤을 때: 그래프 눈금 왼쪽 아래의 숫자를 값으로 본다 (InBody270).

    교차검증이 없는 항목이라 위치·범위·자릿수가 모두 맞는 후보가 정확히 하나일 때만 채택한다.
    """
    for a in lines:
        if not _squash(_clean(a.text)).startswith(anchor):
            continue
        near = [
            l for l in lines
            if a.x0 - 0.25 <= l.x0 <= a.x0 - 0.03 and a.cy - 0.005 <= l.cy <= a.cy + 0.03 and _numeric_like(l.text)
        ]
        cands = {v for v, tier in _candidates(f, _collect(near, ticks)) if tier == 2}
        if len(cands) == 1:
            return cands.pop(), [a.text]
    return None


def _history_split(lines: list[OcrLine]) -> tuple[list[OcrLine], list[OcrLine]]:
    """(본문 줄, 신체변화 이력 영역 줄). 이력은 왼쪽 아래에만 있고 오른쪽 열(임피던스)은 본문이다."""
    cut = min((l.y0 for l in lines if any(m in _squash(l.text) for m in HISTORY_MARKERS)), default=None)
    if cut is None:  # 제목을 놓친 경우: 이력 그래프 바로 아래의 '최근/전체'에서 글자 높이 9배 위까지가 이력
        footer = [l for l in lines if any(m in l.text for m in HISTORY_FOOTERS)]
        if footer:
            cut = min(l.y0 for l in footer) - 9 * min(l.height for l in footer)
    if cut is None:
        return lines, []
    right = _right_column_x(lines)
    hist =[l for l in lines if l.y0 >= cut and l.x1 <= right + 0.01]
    return [l for l in lines if not (l.y0 >= cut and l.x1 <= right + 0.01)], hist


def _range_of(
    f: str, label: OcrLine, end: int, lines: list[OcrLine], max_x: float
) -> tuple[float, float] | None:
    """라벨 뒤 같은 줄·같은 행 오른쪽에서 첫 정상범위 "(41.4~50.6)".

    소수점이 빠진 경계("414~50.6", "111~135")는 항목 범위를 벗어나면 표기 자릿수로 복구하고,
    복구해도 항목 범위를 벗어나면 버린다.
    """
    spec = FIELD_SPECS[f]

    def bound(tok: str) -> float:
        v = float(tok)
        if not spec.min <= v <= spec.max and "." not in tok:
            v /= 10**spec.decimals
        return v

    texts = [_squash(label.text)[end:]]
    for other in _row_right(label, lines, max_x):
        if _find_labels(other.text):
            break
        texts.append(other.text)
    for t in texts:
        if m := _RANGE_PARTS.search(t.replace(",", ".")):
            lo, hi = bound(m.group(1)), bound(m.group(2))
            if spec.min <= lo < hi <= spec.max:
                return lo, hi
    return None


def parse(lines: list[OcrLine]) -> ParseResult:
    lines, hist = _history_split(_deskew(lines))
    lines = sorted(lines, key=lambda l: (l.y0, l.x0))
    ticks = _tick_ids(lines)
    right_col = _right_column_x(lines)

    votes: dict[str, list[tuple[float, int]]] = {}
    sources: dict[str, list[str]] = {}
    ranges: dict[str, tuple[float, float]] = {}
    for line in lines:
        # 왼쪽 표의 라벨은 오른쪽 열(인바디점수·체중조절 …)까지 넘어가 값을 찾지 않는다
        max_x = right_col if line.x1 < right_col else 1.0
        for f, end in _find_labels(line.text):
            if f == _STOP:
                continue
            same_line = _numbers(_squash(line.text)[end:])
            if f in VALUE_BELOW:  # 오른쪽은 그래프라 보지 않는다. 바로 아래에 없으면 못 읽음 처리
                cands = _candidates(f, same_line + _collect(_below(line, lines, max_dx=0.0), ticks))
            else:
                cands = _candidates(f, same_line + _collect(_row_right(line, lines, max_x), ticks))
                if not any(tier == 2 for _, tier in cands):
                    cands += _candidates(f, _collect(_below(line, lines), ticks))
            if cands:
                best = max(cands, key=lambda c: c[1])  # 같은 우선순위면 앞쪽(가까운) 후보
                votes.setdefault(f, []).append(next(c for c in cands if c[1] == best[1]))
                sources.setdefault(f, []).append(line.text)
            if f in RANGE_FIELDS and f not in ranges and (r := _range_of(f, line, end, lines, max_x)):
                ranges[f] = r

    values = {}
    for f, vs in votes.items():
        top = max(t for _, t in vs)
        # 표기 자릿수로 반올림 (옆 글자가 붙어 "15.34"로 읽힌 값)
        values[f] = round(Counter(v for v, t in vs if t == top).most_common(1)[0][0], SPECS[f].decimals)

    for f, anchor in GRAPH_ANCHORS.items():
        if f not in values and (found := _from_graph_anchor(f, anchor, lines, ticks)):
            values[f], sources[f] = found

    exercise = {f.split(".", 1)[1]: values.pop(f) for f in list(values) if f.startswith("exercise.")}
    date, time, dmy = _parse_datetime(lines)
    history = _parse_history(hist, dmy)
    # 이력 값이 이번 측정과 크게 다르면 작은 글씨 오독("75.7" → "55.3")으로 보고 버린다
    for p in history:
        for f, limit in HISTORY_MAX_DIFF.items():
            v, cur = getattr(p, f), values.get(f)
            if v is not None and cur is not None and abs(v - cur) > (limit * cur if f != "percent_body_fat" else limit):
                setattr(p, f, None)
    return ParseResult(
        InBodyExtraction(
            is_inbody_sheet=len(values) >= 3,
            sex=_parse_sex(lines),
            measured_at=date,
            measured_time=time,
            inbody_score=_parse_score(lines),
            **values,
            ranges=ranges,
            exercise_kcal=exercise,
            impedance=_parse_impedance(lines),
            history=history,
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


def _parse_datetime(lines: list[OcrLine]) -> tuple[str | None, str | None, bool]:
    """(측정일 YYYY-MM-DD, 측정 시각 HH:MM, 일.월.연도 표기 여부). 시각은 날짜 바로 뒤에 있다."""
    text = " ".join(l.text for l in lines)
    for pattern, dmy in ((_DATE, False), (_DATE_DMY, True)):
        if m := pattern.search(text):
            a, mo, b = m.groups()
            y, d = (b, a) if dmy else (a, b)
            t = _TIME.match(text, m.end())
            time = f"{int(t.group(1)):02d}:{t.group(2)}" if t else None
            return f"{y}-{int(mo):02d}-{int(d):02d}", time, dmy
    return None, None, False


def _parse_score(lines: list[OcrLine]) -> float | None:
    """인바디점수 "84/100 점". 라벨 아래 안내문에 '100점'이 있어 라벨 규칙 대신 '/100' 형식으로 찾는다."""
    for l in lines:
        if m := re.match(r"\s*(\d{2,3})\s*/\s*100", l.text):
            v = float(m.group(1))
            spec = FIELD_SPECS["inbody_score"]
            if spec.min <= v <= spec.max:
                return v
    return None


_IMP_HEAD = re.compile(r"(?<![A-Za-z])(RA|LA|TR|RL|LL)(?![A-Za-z])")
_IMP_TOKEN = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])")


def _impedance_value(tok: str) -> tuple[float, int] | None:
    """(값, 우선순위). 값은 모두 소수 한 자리. 소수점이 빠진 네 자리("2604")는 복구. 그 밖('20' kHz 표기 등)은 값 아님."""
    if re.fullmatch(r"\d{1,3}\.\d", tok):
        return float(tok), 2
    if re.fullmatch(r"\d{4}", tok):
        return float(tok) / 10, 1
    return None


def _parse_impedance(lines: list[OcrLine]) -> dict[str, dict[str, float]]:
    """'임피던스' 제목 아래 값 두 행(20kHz, 100kHz) × 부위 5개 (RA LA TR RL LL).

    한 줄에 값 5개가 다 있으면 순서대로, 줄이 쪼개지거나 같은 값이 두 번 읽혔으면 머리글(RA…) 위치를 열로 써서 맞춘다.
    같은 부위는 100kHz 값이 20kHz보다 작다: 어긋난 부위는 오독으로 보고 버린다.
    """
    anchor = next((l for l in lines if any(m in _squash(l.text) for m in ("임피던스", "Impedance"))), None)
    if anchor is None:
        return {}
    h = max(anchor.height, 0.005)
    region = [l for l in lines if anchor.cy < l.cy <= anchor.cy + 8 * h and l.x1 > anchor.x0]
    heads = {}
    for l in region:
        for m, x in _token_positions(l, _IMP_HEAD):
            heads.setdefault(IMPEDANCE_SEGMENTS.index(m.group(1)), x)
    cols: list[float] = []
    if len(heads) >= 3:  # 빠진 머리글은 등간격으로 채운다
        idx = sorted(heads)
        step = (heads[idx[-1]] - heads[idx[0]]) / (idx[-1] - idx[0])
        cols = [heads.get(i, heads[idx[0]] + (i - idx[0]) * step) for i in range(5)]

    rows: list[list[OcrLine]] = []
    for l in sorted(region, key=lambda l: l.cy):
        if rows and l.cy - rows[-1][0].cy <= h * 0.5:
            rows[-1].append(l)
        else:
            rows.append([l])
    value_rows = []
    for row in rows:
        per_line = [[(x, v) for m, x in _token_positions(l, _IMP_TOKEN) if (v := _impedance_value(m.group(0)))] for l in row]
        if full := next((t for t in per_line if len(t) == 5), None):
            value_rows.append([v for _, (v, _) in full])
            continue
        toks = sorted(t for line_toks in per_line for t in line_toks)
        if len(toks) < 3:
            continue
        vals: list[float | None] = [None] * 5
        if cols:
            best = [0] * 5
            half = abs(cols[1] - cols[0]) / 2
            for x, (v, tier) in toks:
                i = min(range(5), key=lambda i: abs(cols[i] - x))
                if abs(cols[i] - x) <= half and tier > best[i]:
                    vals[i], best[i] = v, tier
        elif len(toks) == 5:
            vals = [v for _, (v, _) in toks]
        value_rows.append(vals)

    out = {freq: dict(zip(IMPEDANCE_SEGMENTS, vals)) for freq, vals in zip(IMPEDANCE_FREQS, value_rows)}
    if len(out) == 2:
        for s in IMPEDANCE_SEGMENTS:
            a, b = out["20kHz"][s], out["100kHz"][s]
            if a is not None and b is not None and b >= a:
                out["20kHz"][s] = out["100kHz"][s] = None
    return {f: {s: v for s, v in segs.items() if v is not None and 5 <= v <= 1000} for f, segs in out.items()}


HISTORY_ROWS = {"weight_kg": "체중", "skeletal_muscle_mass_kg": "골격근량", "percent_body_fat": "체지방률"}
# 이력 값과 이번 측정값의 최대 차이: 체중·골격근량은 비율(25%), 체지방률은 %p(12). 잠정값
HISTORY_MAX_DIFF = {"weight_kg": 0.25, "skeletal_muscle_mass_kg": 0.25, "percent_body_fat": 12}


def _token_positions(line: OcrLine, pattern: re.Pattern) -> list[tuple[re.Match, float]]:
    """줄 안 각 매치와 그 x 중심 (글자 위치 비례로 추정. 여러 값이 한 줄로 합쳐 인식된 경우 대비)."""
    n = max(len(line.text), 1)
    return [(m, line.x0 + (m.start() + m.end()) / 2 / n * (line.x1 - line.x0)) for m in pattern.finditer(line.text)]


def _parse_history(hist: list[OcrLine], dmy: bool) -> list[HistoryPoint]:
    """신체변화 그래프: 맨 아래 시각(HH:MM)으로 열을, 왼쪽 라벨(체중·골격근량·체지방률)로 행을 정한다.

    그래프 숫자는 작아서 OCR이 소수점을 자주 놓친다("782"). 세 자리 숫자는 소수점 복구, 자릿수가 다른 값은 버린다.
    """
    times = [(m, x, l) for l in hist for m, x in _token_positions(l, _TIME)]
    if not times:
        return []
    # 날짜·시각 줄 (그래프 숫자는 이보다 위). 날짜 줄을 못 읽었으면 시각 줄 위로 글자 높이만큼
    date_lines = [l for l in hist if _SHORT_DATE.search(l.text)]
    date_row_top = min(l.y0 for l in date_lines) if date_lines else min(l.y0 - l.height for _, _, l in times)
    labels = {}
    for f, label in HISTORY_ROWS.items():
        found = [l for l in hist if label in _squash(l.text) and not (f == "weight_kg" and "조절" in l.text)]
        if found:
            labels[f] = min(found, key=lambda l: l.x0).cy
    if not labels:
        return []
    columns = sorted(times, key=lambda t: t[1])
    xs = [x for _, x, _ in columns]
    half = (max(xs) - min(xs)) / (len(xs) - 1) / 2 if len(xs) > 1 else 0.05

    dates: dict[int, str] = {}
    for l in hist:
        for m, x in _token_positions(l, _SHORT_DATE):
            i = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
            if abs(xs[i] - x) <= half:
                a, mo, b = (int(g) for g in m.groups())
                y, d = (b, a) if dmy else (a, b)
                if 1 <= mo <= 12 and 1 <= d <= 31:
                    dates[i] = f"20{y:02d}-{mo:02d}-{d:02d}"

    points = [HistoryPoint(date=dates.get(i), time=f"{int(m.group(1)):02d}:{m.group(2)}") for i, (m, _, _) in enumerate(columns)]
    label_left = min(l.x1 for l in hist if any(lb in _squash(l.text) for lb in HISTORY_ROWS.values()))
    # (열, 행) → (우선순위, 값). 소수점이 제대로 읽힌 값을 소수점 복구 값보다 우선한다
    found: dict[tuple[int, str], tuple[int, float]] = {}
    for l in hist:
        if l.cy >= date_row_top or _TIME.search(l.text):
            continue
        for m, x in _token_positions(l, _NUM):
            if x < label_left:
                continue
            f = min(labels, key=lambda f: abs(labels[f] - l.cy))
            i = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
            if abs(xs[i] - x) > half:
                continue
            tok = m.group(0).replace(" ", "").lstrip("-")
            if re.fullmatch(r"\d{2}\.\d", tok):
                tier, v = 2, float(tok)
            elif re.fullmatch(r"\d{3}", tok):  # 소수점 누락 복구
                tier, v = 1, float(tok) / 10
            else:
                continue
            spec = FIELD_SPECS[f]
            if spec.min <= v <= spec.max and tier > found.get((i, f), (0, 0))[0]:
                found[(i, f)] = (tier, v)
    for (i, f), (_, v) in found.items():
        setattr(points[i], f, v)
    return points
