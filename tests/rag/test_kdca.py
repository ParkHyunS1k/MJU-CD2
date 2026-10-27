import json

from rag import kdca
from rag.ingest import ingest

API_XML = """<?xml version="1.0" encoding="UTF-8"?>
<XML><HEAD><CODE>S001</CODE><MESSAGE>OK</MESSAGE></HEAD><svc>
<CNTNTSSJ><![CDATA[고혈압]]></CNTNTSSJ><CNTNTS_SN><![CDATA[6765]]></CNTNTS_SN>
<cntntsClList>
<cntntsCl><CNTNTS_CL_NM><![CDATA[개요]]></CNTNTS_CL_NM><CNTNTS_CL_CN><![CDATA[고혈압이란 수축기혈압 140 mmHg 이상인 경우를 말합니다.]]></CNTNTS_CL_CN></cntntsCl>
<cntntsCl><CNTNTS_CL_NM><![CDATA[정의]]></CNTNTS_CL_NM><CNTNTS_CL_CN><![CDATA[혈압은 정상혈압과 고혈압으로 구분합니다.
<p>원인이 명확하지 않으면 본태성 고혈압이라 합니다.</p>]]></CNTNTS_CL_CN></cntntsCl>
<cntntsCl><CNTNTS_CL_NM><![CDATA[정의]]></CNTNTS_CL_NM><CNTNTS_CL_CN><![CDATA[https://is.kdca.go.kr/cscdnhfile/health/healthNewDown/healthInfoFileDown.do?SEQ=19b6]]></CNTNTS_CL_CN></cntntsCl>
<cntntsCl><CNTNTS_CL_NM><![CDATA[비약물 치료]]></CNTNTS_CL_NM><CNTNTS_CL_CN><![CDATA[• 소금 섭취를 하루 5 g 이하로 줄입니다.
• 규칙적으로 운동합니다.]]></CNTNTS_CL_CN></cntntsCl>
<cntntsCl><CNTNTS_CL_NM><![CDATA[참고문헌]]></CNTNTS_CL_NM><CNTNTS_CL_CN><![CDATA[1. 대한고혈압학회. 2022 고혈압 진료지침.]]></CNTNTS_CL_CN></cntntsCl>
</cntntsClList></svc></XML>"""

EMPTY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<XML><HEAD><CODE>S001</CODE><MESSAGE>OK</MESSAGE></HEAD><svc>
<CNTNTSSJ><![CDATA[대사증후군]]></CNTNTSSJ><CNTNTS_SN><![CDATA[6781]]></CNTNTS_SN>
<cntntsClList></cntntsClList></svc></XML>"""

WEB_HTML = """<html><body><nav><a href="#contentsDiv2">개요</a></nav>
<div id="contentsDiv2" class="contents-Div" style="text-align:left;"><h3>개요</h3>
<p><br></p><p>대사증후군이란 위험인자가 겹쳐 있는 상태를 가리킵니다.</p>
<p><img src="https://is.kdca.go.kr/x.do?SEQ=1" alt="그림"> </p></div>
<div id="contentsDiv3" class="contents-Div"><h3>생활습관 관리</h3><ul><li>체중을 줄입니다.</li><li>금연합니다.</li></ul></div>
<div id="contentsDiv12" class="contents-Div"><h3>참고문헌</h3><p>1. 문헌</p></div>
<footer>저작권</footer></body></html>"""


def test_parse_api_merges_repeated_names_and_drops_links_and_references():
    title, sections = kdca.parse_api(API_XML)
    assert title == "고혈압"
    assert [name for name, _ in sections] == ["개요", "정의", "비약물 치료"]
    assert sections[1][1] == ["혈압은 정상혈압과 고혈압으로 구분합니다.", "원인이 명확하지 않으면 본태성 고혈압이라 합니다."]
    assert sections[2][1] == ["• 소금 섭취를 하루 5 g 이하로 줄입니다.", "• 규칙적으로 운동합니다."]


def test_parse_api_empty_body_gives_no_sections():
    assert kdca.parse_api(EMPTY_XML) == ("대사증후군", [])


def test_parse_web_reads_content_blocks():
    sections = kdca.parse_web(WEB_HTML)
    assert sections == [("개요", ["대사증후군이란 위험인자가 겹쳐 있는 상태를 가리킵니다."]),
                        ("생활습관 관리", ["체중을 줄입니다.", "금연합니다."])]


def test_fetch_falls_back_to_web_when_api_body_is_empty():
    calls = []

    def get(url):
        calls.append(url)
        return EMPTY_XML if "api.kdca.go.kr" in url else WEB_HTML

    doc = kdca.fetch(6781, "tok", get=get)
    assert doc["source"] == "web" and doc["title"] == "대사증후군"
    assert doc["sections"][0]["name"] == "개요"
    assert "cntntsSn=6781" in calls[0] and "TOKEN=tok" in calls[0]
    assert "cntnts_sn=6781" in calls[1]


def test_ingest_reads_kdca_documents_without_pages(tmp_path):
    raw = tmp_path / "raw"
    (raw / "kdca").mkdir(parents=True)
    title, sections = kdca.parse_api(API_XML)
    (raw / "kdca" / "6765.json").write_text(json.dumps(
        {"sn": 6765, "title": title, "source": "api",
         "sections": [{"name": n, "paragraphs": p} for n, p in sections]}, ensure_ascii=False), encoding="utf-8")
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps([{
        "id": "kdca-6765", "kind": "kdca", "cntnts_sn": 6765, "file": "kdca/6765.json", "title": "고혈압",
        "publisher": "질병관리청", "conditions": ["hypertension"], "category": "lifestyle",
        "section_rules": [{"match": "치료", "category": "lifestyle"}],
    }], ensure_ascii=False), encoding="utf-8")
    records, stats = ingest(sources, raw, 1200, 0, chunker="paragraph")
    assert stats["documents"] == 1 and stats["failed"] == []
    assert {r["heading_path"] for r in records} == {"개요", "정의", "비약물 치료"}
    assert all(r["page_start"] is None and r["page_end"] is None for r in records)
    joined = "\n".join(r["content"] for r in records)
    assert "본태성 고혈압" in joined and "대한고혈압학회" not in joined and "healthInfoFileDown" not in joined
