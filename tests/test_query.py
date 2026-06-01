import pytest
from fastapi.testclient import TestClient

import app.llm as llm
import app.vector_store as vs
from app.chunker import Chunk
from app.config import settings
from app.embedder import embed_texts
from app.main import app


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr(vs, "_client", None)
    yield tmp_path
    vs._client = None


def _seed(num: int = 3) -> None:
    texts = [
        "alpha apple is a fruit",
        "beta banana is yellow",
        "gamma grape grows on vines",
    ][:num]
    chunks = [
        Chunk(text=t, start_char=0, end_char=len(t), chunk_index=i)
        for i, t in enumerate(texts)
    ]
    embeddings = embed_texts(texts)
    vs.add_chunks("doc1", "sample.pdf", chunks, embeddings, source_type="pdf")


def test_empty_corpus_returns_dont_know_without_calling_llm(tmp_store, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("LLM must not be called when corpus is empty")

    monkeypatch.setattr(llm, "generate_answer", boom)

    response = TestClient(app).post("/query", json={"question": "anything?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "I don't know."
    assert body["sources"] == []


def test_happy_path_returns_answer_and_sources(tmp_store, monkeypatch):
    _seed()
    monkeypatch.setattr(llm, "generate_answer", lambda q, r: "mocked answer [chunk_index=0]")

    response = TestClient(app).post("/query", json={"question": "what is alpha?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "mocked answer [chunk_index=0]"
    assert len(body["sources"]) >= 1
    for src in body["sources"]:
        assert set(src.keys()) == {
            "document_id",
            "chunk_index",
            "source_filename",
            "snippet",
            "distance",
        }
        assert src["document_id"] == "doc1"
        assert src["source_filename"] == "sample.pdf"
        assert isinstance(src["chunk_index"], int)
        assert isinstance(src["snippet"], str)
        assert len(src["snippet"]) <= 200
        assert isinstance(src["distance"], float)


def test_top_k_override_limits_sources(tmp_store, monkeypatch):
    _seed()
    monkeypatch.setattr(llm, "generate_answer", lambda q, r: "mocked")

    response = TestClient(app).post(
        "/query", json={"question": "what is alpha?", "top_k": 1}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) == 1
