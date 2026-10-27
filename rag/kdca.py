"""국가건강정보포털(질병관리청) 건강정보를 받아 수집 파이프라인이 읽는 JSON으로 저장한다.
API(XML)로 받고, 본문이 비어 있으면 공개 웹 페이지에서 받는다. 저장소 루트에서 `python -m rag.kdca`.
토큰은 루트 .env의 KDCA_API_TOKEN. 원문은 rag/raw/kdca/<번호>.json(git 제외)."""

import argparse
import html
import json
import os
import re
import ssl
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from pathlib import Path

from rag.parse import Block
from rag.sections import Section

API_URL = "https://api.kdca.go.kr/api/provide/healthInfo?TOKEN={token}&cntntsSn={sn}"
PAGE_URL = ("https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/"
            "gnrlzHealthInfoView.do?cntnts_sn={sn}")
SKIP_SECTIONS = {"참고문헌"}
_LINK = re.compile(r"^https?://\S+$")
_BREAKS = re.compile(r"<br\s*/?>|</p>|</li>|</h\d>|</div>|</tr>", re.I)


def _paragraphs(fragment: str) -> list[str]:
    """HTML 조각이나 줄바꿈 텍스트를 문단 목록으로. 이미지 링크만 있는 줄은 버린다."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", _BREAKS.sub("\n", fragment)))
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    return [line for line in lines if line and not _LINK.match(line)]


def _merge(pairs: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """같은 이름의 섹션이 연달아 오면 하나로 합치고, 참고문헌과 빈 섹션은 뺀다."""
    out: list[tuple[str, list[str]]] = []
    for name, paras in pairs:
        if name in SKIP_SECTIONS or not paras:
            continue
        if out and out[-1][0] == name:
            out[-1][1].extend(paras)
        else:
            out.append((name, list(paras)))
    return out


def parse_api(xml_text: str) -> tuple[str, list[tuple[str, list[str]]]]:
    root = ET.fromstring(xml_text)
    title = (root.findtext(".//CNTNTSSJ") or "").strip()
    pairs = [((cl.findtext("CNTNTS_CL_NM") or "").strip(), _paragraphs(cl.findtext("CNTNTS_CL_CN") or ""))
             for cl in root.iter("cntntsCl")]
    return title, _merge(pairs)


def parse_web(page: str) -> list[tuple[str, list[str]]]:
    """공개 페이지의 본문 블록(div.contents-Div, 제목 h3)만 읽는다."""
    pairs = []
    for block in re.findall(r'<div[^>]*class="contents-Div"[^>]*>(.*?)(?=<div[^>]*class="contents-Div"|<footer|$)',
                            page, re.S):
        head = re.search(r"<h3[^>]*>(.*?)</h3>", block, re.S)
        if not head:
            continue
        name = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", head.group(1)))).strip()
        pairs.append((name, _paragraphs(block[head.end():])))
    return _merge(pairs)


def _get(url: str) -> str:
    ctx = ssl.create_default_context()
    ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT  # api.kdca.go.kr가 옛 TLS 재협상만 지원한다
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return opener.open(req, timeout=30).read().decode("utf-8", "ignore")


def fetch(sn: int, token: str, get: Callable[[str], str] = _get) -> dict:
    title, sections = parse_api(get(API_URL.format(token=token, sn=sn)))
    source = "api"
    if not sections:
        sections, source = parse_web(get(PAGE_URL.format(sn=sn))), "web"
    return {"sn": sn, "title": title, "source": source,
            "sections": [{"name": name, "paragraphs": paras} for name, paras in sections]}


def load_sections(path: Path, exclude_sections: Iterable[str] = (),
                  exclude_paragraphs: Iterable[str] = ()) -> list[Section]:
    """저장한 JSON을 수집 파이프라인의 섹션으로. 문단 하나가 블록 하나이고, 블록 사이를 띄워 서로 합쳐지지 않게 한다.
    exclude_sections는 섹션 이름, exclude_paragraphs는 문단에 들어 있는 문구로 뺀다(학회 지침과 충돌하는 내용)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    names = {s["name"] for s in doc["sections"]}
    paras = [p for s in doc["sections"] for p in s["paragraphs"]]
    stale = [x for x in exclude_sections if x not in names] + \
            [x for x in exclude_paragraphs if not any(x in p for p in paras)]
    if stale:
        raise ValueError(f"{path}: 제외 항목이 문서에 없습니다(원문이 바뀌었나?): {stale}")
    sections, y = [], 0.0
    for section in doc["sections"]:
        if section["name"] in exclude_sections:
            continue
        blocks = []
        for para in section["paragraphs"]:
            if any(x in para for x in exclude_paragraphs):
                continue
            blocks.append(Block(1, y, y + 12, para, x0=72, x1=520))
            y += 100
        if blocks:
            sections.append(Section([section["name"]], blocks))
    return sections


def _token() -> str:
    from dotenv import load_dotenv

    load_dotenv(".env")
    token = os.environ.get("KDCA_API_TOKEN")
    if not token:
        raise SystemExit("KDCA_API_TOKEN이 .env에 없습니다")
    return token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="국가건강정보포털 건강정보 받기")
    parser.add_argument("--sources", type=Path, default=Path("rag/sources.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("rag/raw"))
    parser.add_argument("--force", action="store_true", help="이미 받은 문서도 다시 받는다")
    args = parser.parse_args(argv)

    token = _token()
    docs = [d for d in json.loads(args.sources.read_text(encoding="utf-8")) if d.get("kind") == "kdca"]
    for d in docs:
        path = args.raw_dir / d["file"]
        if path.exists() and not args.force:
            continue
        doc = fetch(d["cntnts_sn"], token)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        chars = sum(len(p) for s in doc["sections"] for p in s["paragraphs"])
        print(f"{d['id']}: {doc['title']} ({doc['source']}, 섹션 {len(doc['sections'])}, {chars}자)")
        time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
