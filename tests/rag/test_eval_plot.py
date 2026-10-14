import json

from rag.eval.plot import plot_stage


def test_plot_stage_writes_png(tmp_path):
    src = tmp_path / "03_embedding.json"
    src.write_text(json.dumps({"primary": "ndcg@10", "rows": [
        {"name": "bge-m3", "ndcg@10": [0.6, 0.5, 0.7]},
        {"name": "kure-v1", "ndcg@10": [0.65, 0.55, 0.75]},
        {"name": "빈 값", "ndcg@10": [float("nan")] * 3},
    ]}), encoding="utf-8")
    out = tmp_path / "03_embedding.png"
    plot_stage(src, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_cli_takes_results_dir(tmp_path, monkeypatch):
    import sys

    from rag.eval import plot

    (tmp_path / "03_embedding.json").write_text(json.dumps({"primary": "m", "rows": [{"name": "a", "m": [0.5, 0.4, 0.6]}]}))
    monkeypatch.setattr(sys, "argv", ["plot", str(tmp_path)])
    plot.cli()
    assert (tmp_path / "03_embedding.png").exists()
