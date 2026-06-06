import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.config import settings

_DEFAULT_PATH = Path("eval/drift/queries.jsonl")


def log_query(question: str, embedding: list[float], top_distance: float) -> None:
    if not settings.drift_logging_enabled:
        return
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "embedding": embedding,
        "top_distance": top_distance,
    }
    try:
        _DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEFAULT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        print(f"drift.log_query failed: {e}", file=sys.stderr)


def load_window(path: Path, start: datetime, end: datetime) -> np.ndarray:
    rows: list[list[float]] = []
    if not path.exists():
        return np.empty((0, 0))
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ts = datetime.fromisoformat(row["ts"])
            if start <= ts < end:
                rows.append(row["embedding"])
    if not rows:
        return np.empty((0, 0))
    return np.array(rows, dtype=float)


def centroid_distance(window_a: np.ndarray, window_b: np.ndarray) -> float:
    a = window_a.mean(axis=0)
    b = window_b.mean(axis=0)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def detect(
    path: Path = _DEFAULT_PATH,
    now: datetime | None = None,
    threshold: float = 0.15,
) -> dict:
    now = now or datetime.now(timezone.utc)
    current = load_window(path, now - timedelta(days=7), now)
    baseline = load_window(path, now - timedelta(days=14), now - timedelta(days=7))

    current_n = current.shape[0]
    baseline_n = baseline.shape[0]

    if current_n < 20 or baseline_n < 20:
        return {
            "status": "insufficient_data",
            "current_n": current_n,
            "baseline_n": baseline_n,
        }

    distance = centroid_distance(current, baseline)
    result = {
        "status": "drift" if distance > threshold else "stable",
        "distance": distance,
        "threshold": threshold,
        "current_n": current_n,
        "baseline_n": baseline_n,
        "computed_at": now.isoformat(),
    }

    results_dir = path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{now.date()}.json").write_text(json.dumps(result, indent=2))

    return result
