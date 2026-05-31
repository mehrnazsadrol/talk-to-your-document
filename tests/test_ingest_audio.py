import json

from fastapi.testclient import TestClient

from app.audio import AudioSegment
from app.main import app


def _install_stub(monkeypatch):
    from app import audio

    def fake_transcribe_segments(path):
        segments = [
            AudioSegment(text="hello", start_seconds=0.0, end_seconds=1.0, segment_index=0),
            AudioSegment(text="world", start_seconds=1.0, end_seconds=2.5, segment_index=1),
        ]
        return segments, 2.5, "en"

    monkeypatch.setattr(audio, "transcribe_segments", fake_transcribe_segments)


def test_ingest_audio_happy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("app.config.settings.chroma_persist_dir", str(tmp_path / "chroma"))
    import app.vector_store as vs
    monkeypatch.setattr(vs, "_client", None)
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
    assert isinstance(body["num_chunks"], int)
    assert body["num_chunks"] >= 1

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
        {"text": "hello", "start_seconds": 0.0, "end_seconds": 1.0, "segment_index": 0},
        {"text": "world", "start_seconds": 1.0, "end_seconds": 2.5, "segment_index": 1},
    ]


def test_ingest_audio_rejects_unsupported_extension(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    _install_stub(monkeypatch)

    response = TestClient(app).post(
        "/ingest_audio",
        files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 415


def test_ingest_audio_indexes_chunks_with_timestamps(tmp_path, monkeypatch):
    """End-to-end: audio segments flow through chunker + embedder + chroma
    and come back out with source_type="audio" plus timestamp ranges."""
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("app.config.settings.chroma_persist_dir", str(tmp_path / "chroma"))
    import app.vector_store as vs
    monkeypatch.setattr(vs, "_client", None)

    from app import audio
    from app.audio import AudioSegment

    def fake_transcribe_segments(path):
        segments = [
            AudioSegment(text="first segment text", start_seconds=0.0, end_seconds=1.0, segment_index=0),
            AudioSegment(text="second segment text", start_seconds=1.0, end_seconds=2.5, segment_index=1),
            AudioSegment(text="third segment text", start_seconds=2.5, end_seconds=4.0, segment_index=2),
        ]
        return segments, 4.0, "en"

    monkeypatch.setattr(audio, "transcribe_segments", fake_transcribe_segments)

    response = TestClient(app).post(
        "/ingest_audio",
        files={"file": ("test.wav", b"fake-wav-bytes", "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["num_chunks"] >= 1

    collection = vs._get_collection()
    assert collection.count() == body["num_chunks"]

    got = collection.get(include=["metadatas"])
    metas = got["metadatas"]
    assert metas, "expected at least one indexed chunk"
    for meta in metas:
        assert meta["source_type"] == "audio"
        assert meta["document_id"] == body["document_id"]
        assert meta["source_filename"] == "test.wav"
        assert isinstance(meta["start_seconds"], (int, float))
        assert isinstance(meta["end_seconds"], (int, float))
        assert meta["end_seconds"] >= meta["start_seconds"]

    # All three segments together are ~57 chars + joiners, well below the
    # default 500-token (~2000-char) chunk size, so we expect one chunk
    # spanning all three segments: start=0.0, end=4.0.
    assert len(metas) == 1
    assert metas[0]["start_seconds"] == 0.0
    assert metas[0]["end_seconds"] == 4.0

    vs._client = None


def test_ingest_audio_empty_transcript_skips_indexing(tmp_path, monkeypatch):
    """Silent clip → empty segments → no chunking/embedding/indexing,
    transcript files still written, num_chunks=0."""
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("app.config.settings.chroma_persist_dir", str(tmp_path / "chroma"))
    import app.vector_store as vs
    monkeypatch.setattr(vs, "_client", None)

    from app import audio

    def fake_empty(path):
        return [], 0.5, ""

    monkeypatch.setattr(audio, "transcribe_segments", fake_empty)

    response = TestClient(app).post(
        "/ingest_audio",
        files={"file": ("silent.wav", b"fake", "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["num_segments"] == 0
    assert body["num_chunks"] == 0
    assert vs._get_collection().count() == 0
    vs._client = None
