import joblib

import training.train as train


def test_main_writes_selected_bundles(knhanes_like, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "load_knhanes", lambda: knhanes_like)
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    train.main(["--only", "hypertension", "diabetes", "--age-spline", "diabetes"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["diabetes.pkl", "hypertension.pkl"]
    assert joblib.load(tmp_path / "hypertension.pkl")["meta"]["age_spline"] is False
    assert joblib.load(tmp_path / "diabetes.pkl")["meta"]["age_spline"] is True
