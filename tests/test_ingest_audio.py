import json

import pytest
from fastapi.testclient import TestClient

import app.vector_store as vs
from app.audio import AudioSegment
from app.config import settings
from app.main import app


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(vs, "_client", None)
    yield tmp_path
    vs._client = None


def _install_stub(monkeypatch):
    from app import audio

    def fake_transcribe_segments(path):
        segments = [
            AudioSegment(text="hello", start_seconds=0.0, end_seconds=1.0, segment_index=0),
            AudioSegment(text="world", start_seconds=1.0, end_seconds=2.5, segment_index=1),
        ]
        return segments, 2.5, "en"

    monkeypatch.setattr(audio, "transcribe_segments", fake_transcribe_segments)


def test_ingest_audio_happy_path(tmp_store, monkeypatch):
    tmp_path = tmp_store
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

    assert txt_file.read_text(encoding="utf-8") == "hello\n\nworld"
    segments = json.loads(json_file.read_text(encoding="utf-8"))
    assert segments == [
        {"text": "hello", "start_seconds": 0.0, "end_seconds": 1.0, "segment_index": 0},
        {"text": "world", "start_seconds": 1.0, "end_seconds": 2.5, "segment_index": 1},
    ]


def test_ingest_audio_rejects_unsupported_extension(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    _install_stub(monkeypatch)
    # No tmp_store fixture: the 415 short-circuits before any vector-store I/O.

    response = TestClient(app).post(
        "/ingest_audio",
        files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 415


def test_ingest_audio_indexes_chunks_with_timestamps(tmp_store, monkeypatch):
    """End-to-end: audio segments flow through chunker + embedder + chroma
    and come back out with source_type="audio" plus timestamp ranges."""
    from app import audio
    from app.audio import AudioSegment

    def fake_transcribe_segments(path):
        segments = [
            AudioSegment(
                text="first segment text", start_seconds=0.0, end_seconds=1.0, segment_index=0
            ),
            AudioSegment(
                text="second segment text", start_seconds=1.0, end_seconds=2.5, segment_index=1
            ),
            AudioSegment(
                text="third segment text", start_seconds=2.5, end_seconds=4.0, segment_index=2
            ),
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


def test_ingest_audio_multi_chunk_straddling_joiners(tmp_store, monkeypatch):
    """Force multiple chunks across segment boundaries so some chunks
    start mid-joiner. Each chunk's timestamp range should still cover
    every segment it overlaps, including ones bordering the joiner."""
    from app import audio
    from app.audio import AudioSegment
    from app.chunker import chunk_text as real_chunk_text
    from app.routers import ingest_audio as ingest_audio_router

    # Three segments, each ~40 chars, joined by "\n\n" => full_text ~124 chars.
    seg_text = "x" * 40
    segments = [
        AudioSegment(text=seg_text, start_seconds=0.0, end_seconds=1.0, segment_index=0),
        AudioSegment(text=seg_text, start_seconds=1.0, end_seconds=2.0, segment_index=1),
        AudioSegment(text=seg_text, start_seconds=2.0, end_seconds=3.0, segment_index=2),
    ]

    def fake_transcribe_segments(path):
        return segments, 3.0, "en"

    monkeypatch.setattr(audio, "transcribe_segments", fake_transcribe_segments)

    # Force the chunker to a tiny size so we get multiple chunks with
    # boundaries that fall in/near the "\n\n" joiners.
    def small_chunk_text(text):
        return real_chunk_text(text, chunk_size=12, chunk_overlap=2)

    monkeypatch.setattr(ingest_audio_router, "chunk_text", small_chunk_text)

    response = TestClient(app).post(
        "/ingest_audio",
        files={"file": ("multi.wav", b"fake-wav-bytes", "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["num_chunks"] >= 2

    collection = vs._get_collection()
    metas = collection.get(include=["metadatas"])["metadatas"]

    # Sanity: every chunk should have a non-degenerate timestamp range
    # (start < end) because each chunk covers at least one full segment
    # or part of one. With the fix, chunks straddling a joiner pick up
    # both the previous and next segment's timestamps.
    for meta in metas:
        assert meta["end_seconds"] > meta["start_seconds"], f"chunk timestamps collapsed: {meta}"

    # At least one chunk must span more than a single segment (i.e. cross
    # a joiner) — otherwise the test isn't exercising the boundary case.
    spans_multiple = [m for m in metas if (m["end_seconds"] - m["start_seconds"]) > 1.0]
    assert spans_multiple, "expected at least one chunk to span a joiner"


def test_ingest_audio_empty_transcript_skips_indexing(tmp_store, monkeypatch):
    """Silent clip → empty segments → no chunking/embedding/indexing,
    transcript files still written, num_chunks=0."""
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
