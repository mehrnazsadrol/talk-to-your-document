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
    vs.add_chunks("doc1", "sample.pdf", chunks, embeddings)

    results = vs.query([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "alpha apple"
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-5)


def test_metadata_round_trip(tmp_store):
    chunks = _chunks()
    embeddings = _embeddings()
    vs.add_chunks("doc1", "sample.pdf", chunks, embeddings)

    results = vs.query([0.0, 1.0, 0.0], top_k=1)
    meta = results[0]["metadata"]
    assert meta["document_id"] == "doc1"
    assert meta["source_filename"] == "sample.pdf"
    assert meta["chunk_index"] == 1
    assert meta["start_char"] == 11
    assert meta["end_char"] == 22


def test_idempotent_upsert(tmp_store):
    chunks = _chunks()
    embeddings = _embeddings()
    vs.add_chunks("doc1", "sample.pdf", chunks, embeddings)
    vs.add_chunks("doc1", "sample.pdf", chunks, embeddings)

    collection = vs._get_collection()
    assert collection.count() == len(chunks)


def test_persistence_across_client_reset(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr(vs, "_client", None)

    chunks = _chunks()
    embeddings = _embeddings()
    vs.add_chunks("doc1", "sample.pdf", chunks, embeddings)

    # Simulate server restart: drop cached client; same on-disk path.
    vs._client = None

    results = vs.query([0.0, 0.0, 1.0], top_k=1)
    assert results[0]["text"] == "gamma grape"
    assert results[0]["metadata"]["chunk_index"] == 2

    vs._client = None
