import pytest

import app.vector_store as vs
from app.chunker import Chunk


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr(vs, "_client", None)
    yield tmp_path
    vs._client = None


def _chunks() -> list[Chunk]:
    return [
        Chunk(text="alpha apple", start_char=0, end_char=11, chunk_index=0),
        Chunk(text="beta banana", start_char=11, end_char=22, chunk_index=1),
        Chunk(text="gamma grape", start_char=22, end_char=33, chunk_index=2),
    ]


def _embeddings() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_round_trip(tmp_store):
    chunks = _chunks()
    embeddings = _embeddings()
    vs.add_chunks("doc1", "sample.pdf", chunks, embeddings, source_type="pdf")

    results = vs.query([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "alpha apple"
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-5)


def test_metadata_round_trip(tmp_store):
    chunks = _chunks()
    embeddings = _embeddings()
    vs.add_chunks("doc1", "sample.pdf", chunks, embeddings, source_type="pdf")

    results = vs.query([0.0, 1.0, 0.0], top_k=1)
    meta = results[0]["metadata"]
    assert meta["document_id"] == "doc1"
    assert meta["source_filename"] == "sample.pdf"
    assert meta["chunk_index"] == 1
    assert meta["start_char"] == 11
    assert meta["end_char"] == 22
    assert meta["source_type"] == "pdf"


def test_idempotent_upsert(tmp_store):
    chunks = _chunks()
    embeddings = _embeddings()
    vs.add_chunks("doc1", "sample.pdf", chunks, embeddings, source_type="pdf")
    vs.add_chunks("doc1", "sample.pdf", chunks, embeddings, source_type="pdf")

    collection = vs._get_collection()
    assert collection.count() == len(chunks)


def test_audio_source_type_and_timestamps_round_trip(tmp_store):
    chunks = _chunks()
    embeddings = _embeddings()
    extra = [
        {"start_seconds": 0.0, "end_seconds": 1.0},
        {"start_seconds": 1.0, "end_seconds": 2.5},
        {"start_seconds": 2.5, "end_seconds": 4.0},
    ]
    vs.add_chunks(
        "doc-audio",
        "lecture.mp3",
        chunks,
        embeddings,
        source_type="audio",
        extra_metadata=extra,
    )

    results = vs.query([0.0, 1.0, 0.0], top_k=1)
    meta = results[0]["metadata"]
    assert meta["source_type"] == "audio"
    assert meta["start_seconds"] == 1.0
    assert meta["end_seconds"] == 2.5
    assert meta["document_id"] == "doc-audio"


def test_extra_metadata_length_mismatch_raises(tmp_store):
    chunks = _chunks()
    embeddings = _embeddings()
    with pytest.raises(ValueError):
        vs.add_chunks(
            "doc1",
            "lecture.mp3",
            chunks,
            embeddings,
            source_type="audio",
            extra_metadata=[{"start_seconds": 0.0, "end_seconds": 1.0}],
        )


def test_extra_metadata_cannot_overwrite_base_fields(tmp_store):
    chunks = _chunks()
    embeddings = _embeddings()
    extra = [
        {"document_id": "evil", "start_seconds": 0.0, "end_seconds": 1.0},
        {"document_id": "evil", "start_seconds": 1.0, "end_seconds": 2.0},
        {"document_id": "evil", "start_seconds": 2.0, "end_seconds": 3.0},
    ]
    vs.add_chunks(
        "doc-real",
        "lecture.mp3",
        chunks,
        embeddings,
        source_type="audio",
        extra_metadata=extra,
    )
    results = vs.query([1.0, 0.0, 0.0], top_k=1)
    assert results[0]["metadata"]["document_id"] == "doc-real"


def test_multi_segment_span_offset_mapping(tmp_store):
    """A chunk spanning segments 2 and 3 should take seg2.start_seconds
    and seg3.end_seconds — the audio router precomputes this, but we
    exercise the storage layer directly with hand-built metadata.
    """
    # Simulate three segments with timestamps; build a single chunk that
    # spans the last two of them.
    seg_offsets = [
        (0, 5, 0.0, 1.0),    # seg 1
        (7, 12, 1.0, 2.5),   # seg 2
        (14, 19, 2.5, 4.0),  # seg 3
    ]
    # Chunk covers chars 7..19 — overlaps segs 2 and 3.
    chunk_start, chunk_end = 7, 19
    overlapping = [
        (s, e, ts, te) for (s, e, ts, te) in seg_offsets
        if s < chunk_end and e > chunk_start
    ]
    assert len(overlapping) == 2
    chunk_start_s = overlapping[0][2]
    chunk_end_s = overlapping[-1][3]
    assert chunk_start_s == 1.0
    assert chunk_end_s == 4.0

    chunk = Chunk(
        text="seg2text seg3text",
        start_char=chunk_start,
        end_char=chunk_end,
        chunk_index=0,
    )
    vs.add_chunks(
        "doc-span",
        "lecture.mp3",
        [chunk],
        [[1.0, 0.0, 0.0]],
        source_type="audio",
        extra_metadata=[{"start_seconds": chunk_start_s, "end_seconds": chunk_end_s}],
    )

    results = vs.query([1.0, 0.0, 0.0], top_k=1)
    meta = results[0]["metadata"]
    assert meta["start_seconds"] == 1.0
    assert meta["end_seconds"] == 4.0
    assert meta["source_type"] == "audio"


def test_persistence_across_client_reset(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr(vs, "_client", None)

    chunks = _chunks()
    embeddings = _embeddings()
    vs.add_chunks("doc1", "sample.pdf", chunks, embeddings, source_type="pdf")

    # Simulate server restart: drop cached client; same on-disk path.
    vs._client = None

    results = vs.query([0.0, 0.0, 1.0], top_k=1)
    assert results[0]["text"] == "gamma grape"
    assert results[0]["metadata"]["chunk_index"] == 2

    vs._client = None
