import hashlib

from rag.chunk import Chunk
from rag.records import build_records, dedupe
from rag.sources import SourceDoc

SOURCE = SourceDoc(
    id="kshguide-2022",
    file="a.pdf",
    title="고혈압 진료지침",
    publisher="대한고혈압학회",
    year=2022,
    url="https://example.org/guide.pdf",
    conditions=["hypertension"],
    category="lifestyle",
    section_rules=[{"match": "운동", "category": "exercise"}, {"match": "식이", "category": "diet"}],
)


def test_record_fields_and_content_prefix():
    [record] = build_records(SOURCE, [Chunk(["3. 생활요법", "3.2 식이요법"], 45, 46, "나트륨을 줄인다.")])
    assert record == {
        "doc_id": "kshguide-2022",
        "title": "고혈압 진료지침",
        "publisher": "대한고혈압학회",
        "year": 2022,
        "url": "https://example.org/guide.pdf",
        "heading_path": "3. 생활요법 > 3.2 식이요법",
        "page_start": 45,
        "page_end": 46,
        "conditions": ["hypertension"],
        "category": "diet",
        "content": "고혈압 진료지침 > 3. 생활요법 > 3.2 식이요법\n나트륨을 줄인다.",
        "content_hash": hashlib.sha256(
            "고혈압 진료지침 > 3. 생활요법 > 3.2 식이요법\n나트륨을 줄인다.".encode()
        ).hexdigest(),
    }


def test_first_matching_rule_wins_and_default_category():
    records = build_records(
        SOURCE,
        [Chunk(["운동과 식이"], 1, 1, "a"), Chunk([], 2, 2, "b")],
    )
    assert [r["category"] for r in records] == ["exercise", "lifestyle"]
    assert records[1]["heading_path"] is None
    assert records[1]["content"] == "고혈압 진료지침\nb"


def test_content_is_nfc_normalized():
    nfd_body = "가"  # NFD '가'
    [record] = build_records(SOURCE, [Chunk([], 1, 1, nfd_body)])
    assert record["content"].endswith("\n가")


def test_dedupe_keeps_first():
    chunks = [Chunk([], 1, 1, "같은 본문"), Chunk([], 5, 5, "같은 본문"), Chunk([], 6, 6, "다른 본문")]
    records = dedupe(build_records(SOURCE, chunks))
    assert [r["page_start"] for r in records] == [1, 6]
