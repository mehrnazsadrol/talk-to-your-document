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
        yield active_run
