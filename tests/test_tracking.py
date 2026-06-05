import mlflow

from app import tracking
from app.config import settings


def test_init_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"file://{tmp_path}/mlruns")
    monkeypatch.setattr(settings, "mlflow_enabled", True)
    monkeypatch.setattr(tracking, "_initialized", False)

    tracking.init()
    first_state = tracking._initialized
    tracking.init()
    assert first_state is True
    assert tracking._initialized is True


def test_run_is_noop_when_disabled(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"file://{tmp_path}/mlruns")
    monkeypatch.setattr(settings, "mlflow_enabled", False)
    monkeypatch.setattr(tracking, "_initialized", False)

    tracking.init()

    with tracking.run("disabled") as r:
        r.log_param("k", 1)
        r.log_metric("m", 0.5)
        r.log_dict({"a": 1}, "x.json")

    assert not (tmp_path / "mlruns").exists()


def test_run_creates_tracked_run_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"file://{tmp_path}/mlruns")
    monkeypatch.setattr(settings, "mlflow_enabled", True)
    monkeypatch.setattr(tracking, "_initialized", False)

    tracking.init()

    with tracking.run("smoke"):
        mlflow.log_param("foo", "bar")

    df = mlflow.search_runs(experiment_names=[settings.mlflow_experiment])
    assert len(df) >= 1
