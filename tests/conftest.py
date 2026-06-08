"""Shared pytest fixtures.

Existing test files predate this conftest and carry their own per-file
fixtures (e.g. `tmp_store` in test_vector_store.py, test_query.py,
test_ingest_audio.py). Those are intentionally left in place — the fixtures
below cover net-new tests added by ticket 3.7 and any future test files.

The one exception is `disable_mlflow`, which is `autouse=True` so every
test runs with MLflow off by default. Tests that need MLflow enabled
(test_tracking.py, the two MLflow tests in test_query.py) re-monkeypatch
`settings.mlflow_enabled = True` explicitly — monkeypatch is per-test, so
the order is: autouse disables, then the test's own fixture re-enables.
"""

from __future__ import annotations

import hashlib

import pytest

from app import embedder, llm, transcriber, vector_store
from app.config import settings


@pytest.fixture(autouse=True)
def disable_mlflow(monkeypatch):
    """Default-off MLflow for all tests. Opt back in by re-monkeypatching."""
    monkeypatch.setattr(settings, "mlflow_enabled", False)


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    """Point ChromaDB at a tmp dir and reset the cached client."""
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(vector_store, "_client", None)
    yield tmp_path
    vector_store._client = None


@pytest.fixture
def stub_embedder(monkeypatch):
    """Deterministic 384-dim embedder. Same input -> same output."""

    def _embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()  # 32 bytes
            # Expand 32 bytes to 384 floats by repeating (32 * 12 = 384).
            vec = [(b / 255.0) for b in digest] * 12
            out.append(vec)
        return out

    monkeypatch.setattr(embedder, "embed_texts", _embed)
    # Also patch the symbol re-exported into the query router.
    from app.routers import query as query_router

    monkeypatch.setattr(query_router, "embed_texts", _embed)
    return _embed


@pytest.fixture
def stub_transcriber(monkeypatch):
    """Replace transcriber._get_model with a stub yielding two fixed segments."""

    class _Segment:
        def __init__(self, start: float, end: float, text: str) -> None:
            self.start = start
            self.end = end
            self.text = text

    class _Info:
        language = "en"
        duration = 2.5

    class _StubModel:
        def transcribe(self, path):
            segments = [
                _Segment(0.0, 1.0, "hello"),
                _Segment(1.0, 2.5, "world"),
            ]
            return segments, _Info()

    monkeypatch.setattr(transcriber, "_get_model", lambda: _StubModel())
    return _StubModel


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace llm.generate_answer with an echo stub."""

    def _generate(question, retrieved):
        return (
            f"stub answer based on {len(retrieved)} chunks",
            {"prompt_tokens": 10, "completion_tokens": 5},
        )

    monkeypatch.setattr(llm, "generate_answer", _generate)
    return _generate
