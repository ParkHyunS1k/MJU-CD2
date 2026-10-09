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
