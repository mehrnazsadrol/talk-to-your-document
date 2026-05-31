import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app


class _StubModel:
    def transcribe(self, path):
        segments = [
            SimpleNamespace(start=0.0, end=1.0, text="hello"),
            SimpleNamespace(start=1.0, end=2.5, text=" world "),
        ]
        info = SimpleNamespace(duration=2.5, language="en")
        return segments, info


def _install_stub(monkeypatch):
    from app import transcriber
    monkeypatch.setattr(transcriber, "_get_model", lambda: _StubModel())


def test_ingest_audio_happy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    _install_stub(monkeypatch)

    response = TestClient(app).post(
        "/ingest_audio",
        files={"file": ("test.wav", b"fake-wav-bytes", "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["num_segments"] == 2
    assert body["source_filename"] == "test.wav"
    assert body["duration_seconds"] == 2.5
    assert body["language"] == "en"
    assert "document_id" in body

    document_id = body["document_id"]
    audio_file = tmp_path / "audio" / f"{document_id}.wav"
    txt_file = tmp_path / "transcripts" / f"{document_id}.txt"
    json_file = tmp_path / "transcripts" / f"{document_id}.json"

    assert audio_file.exists()
    assert txt_file.exists()
    assert json_file.exists()

    assert txt_file.read_text(encoding="utf-8") == "hello\nworld"
    segments = json.loads(json_file.read_text(encoding="utf-8"))
    assert segments == [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.5, "text": "world"},
    ]


def test_ingest_audio_rejects_unsupported_extension(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    _install_stub(monkeypatch)

    response = TestClient(app).post(
        "/ingest_audio",
        files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 415


def test_ingest_audio_does_not_touch_vector_store(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("app.config.settings.chroma_persist_dir", str(tmp_path / "chroma"))
    import app.vector_store as vs
    monkeypatch.setattr(vs, "_client", None)
    _install_stub(monkeypatch)

    before = vs._get_collection().count()

    response = TestClient(app).post(
        "/ingest_audio",
        files={"file": ("test.wav", b"fake-wav-bytes", "audio/wav")},
    )
    assert response.status_code == 200

    after = vs._get_collection().count()
    assert after == before
