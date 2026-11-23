from rag.eval.pool import pool_query


def test_pool_interleaves_and_dedupes():
    rankings = {"a": ["x", "y", "z"], "b": ["y", "w", "x"]}
    assert pool_query(rankings, per_system=2) == [
        {"content_hash": "x", "systems": ["a"]},
        {"content_hash": "y", "systems": ["b", "a"]},
        {"content_hash": "w", "systems": ["b"]},
    ]


def test_pool_respects_per_system_and_short_lists():
    rankings = {"a": ["x"], "b": ["y", "z", "w"]}
    assert [c["content_hash"] for c in pool_query(rankings, per_system=2)] == ["x", "y", "z"]


def test_pool_empty():
    assert pool_query({"a": []}, per_system=5) == []
