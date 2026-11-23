import json

import pytest

from rag.sources import SourcesError, load_sources


def _write(tmp_path, entries, files=("a.pdf",)):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in files:
        (raw / name).write_bytes(b"%PDF-1.4")
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return sources, raw


def _entry(**overrides):
    entry = {
        "id": "kshguide-2022",
        "file": "a.pdf",
        "title": "고혈압 진료지침",
        "publisher": "대한고혈압학회",
        "year": 2022,
        "conditions": ["hypertension"],
        "category": "lifestyle",
        "section_rules": [{"match": "운동", "category": "exercise"}],
    }
    entry.update(overrides)
    return entry


def test_load_valid_sources(tmp_path):
    sources, raw = _write(tmp_path, [_entry()])
    [(doc, path)] = load_sources(sources, raw)
    assert doc.id == "kshguide-2022"
    assert doc.section_rules[0].category == "exercise"
    assert path == raw / "a.pdf"


def test_unknown_condition_rejected(tmp_path):
    sources, raw = _write(tmp_path, [_entry(conditions=["cholesterol"])])
    with pytest.raises(SourcesError, match="cholesterol"):
        load_sources(sources, raw)


def test_unknown_category_rejected(tmp_path):
    sources, raw = _write(tmp_path, [_entry(category="sleep")])
    with pytest.raises(SourcesError):
        load_sources(sources, raw)


def test_missing_file_rejected(tmp_path):
    sources, raw = _write(tmp_path, [_entry(file="missing.pdf")])
    with pytest.raises(SourcesError, match="missing.pdf"):
        load_sources(sources, raw)


def test_duplicate_ids_rejected(tmp_path):
    sources, raw = _write(tmp_path, [_entry(), _entry()])
    with pytest.raises(SourcesError, match="duplicate"):
        load_sources(sources, raw)


def test_invalid_json_rejected(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sources = tmp_path / "sources.json"
    sources.write_text("[{", encoding="utf-8")
    with pytest.raises(SourcesError):
        load_sources(sources, raw)


def test_sections_parsed(tmp_path):
    sections = [{"title": "운동요법", "pages": [59, 64], "category": "exercise"}, {"title": "식사", "pages": [50, 58]}]
    sources, raw = _write(tmp_path, [_entry(sections=sections)])
    [(doc, _)] = load_sources(sources, raw)
    assert doc.sections[0].pages == (59, 64)
    assert doc.sections[1].category is None


@pytest.mark.parametrize("sections", [
    [{"title": "a", "pages": [5, 3]}],
    [{"title": "a", "pages": [0, 3]}],
    [{"title": "a", "pages": [1, 5]}, {"title": "b", "pages": [5, 8]}],
    [{"title": "a", "pages": [1, 2]}, {"title": "a", "pages": [3, 4]}],
])
def test_invalid_sections_rejected(tmp_path, sections):
    sources, raw = _write(tmp_path, [_entry(sections=sections)])
    with pytest.raises(SourcesError):
        load_sources(sources, raw)
