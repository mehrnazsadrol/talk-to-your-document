import os
from contextlib import contextmanager
from pathlib import Path

import mlflow

from app.config import settings

_initialized = False


class _NullRun:
    def log_param(self, *args, **kwargs) -> None:
        return None

    def log_metric(self, *args, **kwargs) -> None:
        return None

    def log_dict(self, *args, **kwargs) -> None:
        return None

    def set_tag(self, *args, **kwargs) -> None:
        return None


class _ActiveRunAdapter:
    """Wraps an mlflow.ActiveRun so callers can use instance-style logging.

    MLflow 3.x's ActiveRun doesn't expose log_param/log_metric/log_dict/set_tag
    as instance methods — they only exist at the module level. This adapter
    delegates to those module-level functions so the caller API matches _NullRun.
    """

    def __init__(self, active_run) -> None:
        self._active_run = active_run

    def log_param(self, key, value) -> None:
        mlflow.log_param(key, value)

    def log_metric(self, key, value) -> None:
        mlflow.log_metric(key, value)

    def log_dict(self, dictionary, artifact_file) -> None:
        mlflow.log_dict(dictionary, artifact_file)

    def set_tag(self, key, value) -> None:
        mlflow.set_tag(key, value)


def _resolve_tracking_uri(uri: str) -> str:
    prefix = "file://"
    if uri.startswith(prefix):
        path = uri[len(prefix) :]
        if path.startswith("./") or not path.startswith("/"):
            return f"file://{Path(path).resolve()}"
    return uri


def init() -> None:
    global _initialized
    if _initialized or not settings.mlflow_enabled:
        return
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(_resolve_tracking_uri(settings.mlflow_tracking_uri))
    mlflow.set_experiment(settings.mlflow_experiment)
    _initialized = True


@contextmanager
def run(run_name: str | None = None, tags: dict | None = None):
    if not settings.mlflow_enabled:
        yield _NullRun()
        return
    with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
        yield _ActiveRunAdapter(active_run)
