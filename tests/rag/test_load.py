from types import SimpleNamespace

from rag import load


def _rec(h, doc="a"):
    return {"doc_id": doc, "content_hash": h, "content": f"c{h}", "heading_path": None, "page_start": 1,
            "page_end": 1, "conditions": ["hypertension"], "category": "diet"}


class FakeQuery:
    def __init__(self, db, name):
        self.db, self.name, self.op, self.payload, self.filter = db, name, "select", None, None

    def select(self, *_):
        self.op = "select"
        return self

    def range(self, start, end):
        self.bounds = (start, end)
        return self

    def upsert(self, rows):
        self.op, self.payload = "upsert", rows
        return self

    def insert(self, rows):
        self.op, self.payload = "insert", rows
        return self

    def delete(self):
        self.op = "delete"
        return self

    def in_(self, column, values):
        self.filter = (column, set(values))
        return self

    def execute(self):
        rows = self.db[self.name]
        if self.op == "select":
            start, end = getattr(self, "bounds", (0, len(rows) - 1))
            return SimpleNamespace(data=rows[start : end + 1])
        if self.op in ("upsert", "insert"):
            key = "id" if self.name == "rag_documents" else "content_hash"
            keep = [r for r in rows if r[key] not in {p[key] for p in self.payload}]
            self.db[self.name] = keep + list(self.payload)
        if self.op == "delete":
            col, vals = self.filter
            self.db[self.name] = [r for r in rows if r[col] not in vals]
        return SimpleNamespace(data=[])


class FakeSB:
    def __init__(self, chunks=(), docs=()):
        self.db = {"rag_chunks": list(chunks), "rag_documents": list(docs)}

    def table(self, name):
        return FakeQuery(self.db, name)


SOURCES = [{"id": "a", "title": "A", "publisher": "P", "year": 2026, "url": None, "license_note": None}]


def test_plan_sync_adds_new_and_deletes_missing():
    add, delete = load.plan_sync([_rec("1"), _rec("2")], {"2", "3"})
    assert [r["content_hash"] for r in add] == ["1"] and delete == {"3"}


def test_sync_embeds_only_new_chunks_and_removes_stale_documents():
    sb = FakeSB(chunks=[{"content_hash": "2", "document_id": "a"}, {"content_hash": "9", "document_id": "old"}],
                docs=[{"id": "old"}])
    embedded = []

    def fake_embed(texts, kind):
        assert kind == "document"
        embedded.extend(texts)
        return [[0.1] * 768 for _ in texts]

    stats = load.sync(sb, [_rec("1"), _rec("2")], SOURCES, fake_embed)
    assert stats == {"documents": 1, "added": 1, "deleted": 1, "kept": 1}
    assert embedded == ["c1"]
    row = next(r for r in sb.db["rag_chunks"] if r["content_hash"] == "1")
    assert row["document_id"] == "a" and row["embedding_model"] == "gemini-embedding-001@768"
    assert row["embedding"].startswith("[0.100000,")
    assert [d["id"] for d in sb.db["rag_documents"]] == ["a"]
    assert sb.db["rag_documents"][0]["published_year"] == 2026


def test_sync_without_changes_calls_no_embedding():
    sb = FakeSB(chunks=[{"content_hash": "1", "document_id": "a"}], docs=[{"id": "a"}])
    stats = load.sync(sb, [_rec("1")], SOURCES, lambda *_: (_ for _ in ()).throw(AssertionError("no call")))
    assert stats == {"documents": 1, "added": 0, "deleted": 0, "kept": 1}


import pytest


@pytest.mark.parametrize("records,stats", [
    ([], {"failed": [], "skipped_no_text": []}),
    ([_rec("1")], {"failed": ["b"], "skipped_no_text": []}),
    ([_rec("1")], {"failed": [], "skipped_no_text": ["b"]}),
])
def test_incomplete_ingest_aborts_before_touching_db(records, stats):
    with pytest.raises(load.IncompleteIngest):
        load.check_ingest(records, stats)


def test_complete_ingest_passes():
    load.check_ingest([_rec("1")], {"failed": [], "skipped_no_text": []})
