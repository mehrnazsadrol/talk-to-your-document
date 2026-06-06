import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from app import drift


def test_centroid_distance_identical_is_zero():
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    assert drift.centroid_distance(a, b) == pytest.approx(0.0)


def test_centroid_distance_orthogonal_is_one():
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 1.0, 0.0]])
    assert drift.centroid_distance(a, b) == pytest.approx(1.0)


def test_detect_empty_file_returns_insufficient_data(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text("")
    result = drift.detect(path=path)
    assert result["status"] == "insufficient_data"
    assert result["current_n"] == 0
    assert result["baseline_n"] == 0


def _seed(path: Path, now: datetime) -> None:
    rows = []
    # 30 baseline rows centered at [1, 0, 0], in [now-14d, now-7d)
    for i in range(30):
        ts = now - timedelta(days=14) + timedelta(hours=i)
        rows.append(
            {
                "ts": ts.isoformat(),
                "question": f"baseline {i}",
                "embedding": [1.0, 0.0, 0.0],
                "top_distance": 0.1,
            }
        )
    # 30 current rows centered at [0, 1, 0], in [now-7d, now)
    for i in range(30):
        ts = now - timedelta(days=7) + timedelta(hours=i)
        rows.append(
            {
                "ts": ts.isoformat(),
                "question": f"current {i}",
                "embedding": [0.0, 1.0, 0.0],
                "top_distance": 0.2,
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_detect_flags_drift_above_threshold(tmp_path):
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "queries.jsonl"
    _seed(path, now)
    result = drift.detect(path=path, now=now, threshold=0.5)
    assert result["status"] == "drift"
    assert result["distance"] > 0.5
    assert result["current_n"] == 30
    assert result["baseline_n"] == 30
    # snapshot written to results/
    snapshot = tmp_path / "results" / f"{now.date()}.json"
    assert snapshot.exists()
    assert json.loads(snapshot.read_text())["status"] == "drift"


def test_detect_stable_below_threshold(tmp_path):
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "queries.jsonl"
    _seed(path, now)
    result = drift.detect(path=path, now=now, threshold=2.0)
    assert result["status"] == "stable"


def test_log_query_swallows_io_errors(tmp_path, monkeypatch, capsys):
    # Point _DEFAULT_PATH at a file inside a read-only directory.
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    ro_dir.chmod(0o500)
    monkeypatch.setattr(drift, "_DEFAULT_PATH", ro_dir / "queries.jsonl")
    # Must not raise.
    drift.log_query("hello", [0.1, 0.2, 0.3], 0.5)
    err = capsys.readouterr().err
    assert "drift.log_query failed" in err
    # Restore perms for cleanup.
    ro_dir.chmod(0o700)
