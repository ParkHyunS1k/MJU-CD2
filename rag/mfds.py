"""식품안전나라 건강기능식품 기능성 원료(고시형+개별인정형) 중 앱 질환과 관련된 것을 받아 태그를 달고
app/data/mfds_ingredients.json에 쓴다. 런타임은 이 파일만 읽는다. 저장소 루트에서 `python -m rag.mfds`.
키는 루트 .env의 FOODSAFETYKOREA_API_KEY. 설계: docs/superpowers/specs/2026-11-24-mfds-supplement-ingredients-design.md"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

OUT = Path("app/data/mfds_ingredients.json")
API = "http://openapi.foodsafetykorea.go.kr/api/{key}/{service}/json/{start}/{end}"
_PAGE = 1000
# 기능성 문구 → 태그. 근력은 "근력"만 본다: "근육"으로 잡으면 칼슘·마그네슘("신경과 근육 기능 유지")이 걸린다
TAGS = {"blood_glucose": r"혈당", "blood_pressure": r"혈압", "cholesterol": r"콜레스테롤",
        "triglyceride": r"중성지[방질]", "body_fat": r"체지방", "muscle": r"근력"}
_NO = re.compile(r"제\s*(\d{4}-\d+)\s*호")
_NO_SUFFIX = re.compile(r"\s*\([^()]*제\s*\d{4}-\d+\s*호\)\s*$")


class IncompleteFetch(RuntimeError):
    """받은 원료가 없거나 크게 줄었다. 그대로 쓰면 영양제 근거가 사라진다."""


def tags_of(function: str) -> list[str]:
    return [tag for tag, pattern in TAGS.items() if re.search(pattern, function or "")]


def recognition_no(name: str) -> str | None:
    m = _NO.search(name or "")
    return m.group(1) if m else None


def clean_name(name: str) -> str:
    return _NO_SUFFIX.sub("", name).strip()


def _text(value) -> str | None:
    value = (value or "").strip()
    return value or None


def _range(low, high) -> str | None:
    low, high = _text(low), _text(high)
    if not (low or high):
        return None
    return low or high if low == high or not (low and high) else f"{low}–{high}"


def _from_i2710(row: dict) -> dict:
    intake = _range(row.get("DAY_INTK_LOWLIMIT"), row.get("DAY_INTK_HIGHLIMIT"))
    if intake and _text(row.get("INTK_UNIT")):
        intake += f" {row['INTK_UNIT'].strip()}"
    # INTK_MEMO는 대부분 "(24.12.23.) 일일섭취량 현행화" 같은 변경 이력이라 쓰지 않는다
    no = recognition_no(row["PRDCT_NM"])
    return {"name": clean_name(row["PRDCT_NM"]), "notified": no is None, "recognition_no": no,
            "function": row["PRIMARY_FNCLTY"], "intake": intake, "cautions": _text(row.get("IFTKN_ATNT_MATR_CN")),
            "tags": tags_of(row["PRIMARY_FNCLTY"]), "updated": _text(row.get("LAST_UPDT_DTM"))}


def _from_i0050(row: dict) -> dict:
    return {"name": clean_name(row["RAWMTRL_NM"]), "notified": False,
            "recognition_no": recognition_no(row["HF_FNCLTY_MTRAL_RCOGN_NO"]) or recognition_no(row["RAWMTRL_NM"]),
            "function": row["PRIMARY_FNCLTY"],
            "intake": _range(row.get("DAY_INTK_LOWLIMIT"), row.get("DAY_INTK_HIGHLIMIT")),
            "cautions": _text(row.get("IFTKN_ATNT_MATR_CN")), "tags": tags_of(row["PRIMARY_FNCLTY"]), "updated": None}


def build_ingredients(i2710: list[dict], i0050: list[dict]) -> list[dict]:
    """관련 기능성이 있는 원료만. I2710(최신, 고시형+개별인정형) 순서대로, I2710에 없는 인정번호만 I-0050에서 더한다."""
    out = [i for i in map(_from_i2710, i2710) if i["tags"]]
    known = {i["recognition_no"] for i in out if i["recognition_no"]}
    for i in map(_from_i0050, i0050):
        if i["tags"] and i["recognition_no"] not in known:
            known.add(i["recognition_no"])
            out.append(i)
    return out


def check_counts(ingredients: list, previous: list | None) -> None:
    if not ingredients or (previous and len(ingredients) < 0.8 * len(previous)):
        raise IncompleteFetch(f"원료가 {len(ingredients)}개뿐이라(이전 {len(previous or [])}개) 파일을 바꾸지 않습니다.")


def fetch(service: str, key: str) -> list[dict]:
    rows, start = [], 1
    while True:
        url = API.format(key=key, service=service, start=start, end=start + _PAGE - 1)
        body = json.loads(urllib.request.urlopen(url, timeout=60).read())[service]
        if body["RESULT"]["CODE"] != "INFO-000":
            raise RuntimeError(f"{service}: {body['RESULT']['MSG']}".replace(key, "***"))
        rows += body.get("row", [])
        if start + _PAGE > int(body["total_count"]):
            return rows
        start += _PAGE


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(".env")
    key = os.environ["FOODSAFETYKOREA_API_KEY"]
    try:
        ingredients = build_ingredients(fetch("I2710", key), fetch("I-0050", key))
    except Exception as e:  # urllib 오류 메시지에 요청 URL(키 포함)이 들어갈 수 있다
        raise SystemExit(str(e).replace(key, "***")) from None
    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else None
    check_counts(ingredients, previous)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ingredients, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts = {tag: sum(tag in i["tags"] for i in ingredients) for tag in TAGS}
    print(f"{len(ingredients)}개 (고시형 {sum(i['notified'] for i in ingredients)}) {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
