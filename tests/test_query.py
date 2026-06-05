import mlflow
import pytest
from fastapi.testclient import TestClient

import app.llm as llm
import app.tracking as tracking
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


@pytest.fixture
def mlflow_disabled(monkeypatch):
    monkeypatch.setattr(settings, "mlflow_enabled", False)
    monkeypatch.setattr(tracking, "_initialized", False)


@pytest.fixture
def mlflow_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        settings, "mlflow_tracking_uri", f"file://{tmp_path}/mlruns"
    )
    monkeypatch.setattr(settings, "mlflow_enabled", True)
    monkeypatch.setattr(tracking, "_initialized", False)
    tracking.init()
    yield tmp_path


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
    monkeypatch.setattr(
        llm,
        "generate_answer",
        lambda q, r: ("mocked answer [chunk_index=0]", {"prompt_tokens": 1, "completion_tokens": 2}),
    )

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
    monkeypatch.setattr(
        llm, "generate_answer", lambda q, r: ("mocked", {"prompt_tokens": 0, "completion_tokens": 0})
    )

    response = TestClient(app).post(
        "/query", json={"question": "what is alpha?", "top_k": 1}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) == 1


def test_mlflow_disabled_creates_no_mlruns(tmp_store, mlflow_disabled, monkeypatch):
    _seed()
    monkeypatch.setattr(
        llm, "generate_answer", lambda q, r: ("answer", {"prompt_tokens": 1, "completion_tokens": 2})
    )

    response = TestClient(app).post("/query", json={"question": "what is alpha?"})

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert not (tmp_store / "mlruns").exists()


def test_mlflow_enabled_logs_one_run_with_metrics_params_artifacts(
    tmp_store, mlflow_tmp, monkeypatch
):
    _seed()
    monkeypatch.setattr(
        llm,
        "generate_answer",
        lambda q, r: ("mocked answer", {"prompt_tokens": 7, "completion_tokens": 9}),
    )

    response = TestClient(app).post("/query", json={"question": "what is alpha?"})
    assert response.status_code == 200
    body = response.json()

    df = mlflow.search_runs(experiment_names=[settings.mlflow_experiment])
    assert len(df) == 1
    row = df.iloc[0]

    assert row["metrics.embed_ms"] > 0
    assert row["metrics.retrieve_ms"] > 0
    assert row["metrics.llm_ms"] > 0
    assert row["metrics.prompt_tokens"] == 7
    assert row["metrics.completion_tokens"] == 9

    assert row["params.top_k"] == str(settings.top_k)
    assert row["params.embedding_model"] == settings.embedding_model
    assert row["params.llm_model"] == settings.llm_model
    assert row["params.corpus_size"] == "3"

    assert row["tags.source_types"] == "pdf"

    run_id = row["run_id"]
    retrieved_artifact = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/retrieved.json"
    )
    artifact_chunk_ids = {
        (r["metadata"]["document_id"], r["metadata"]["chunk_index"])
        for r in retrieved_artifact
    }
    response_chunk_ids = {
        (s["document_id"], s["chunk_index"]) for s in body["sources"]
    }
    assert artifact_chunk_ids == response_chunk_ids

    request_artifact = mlflow.artifacts.load_dict(f"runs:/{run_id}/request.json")
    assert request_artifact == {"question": "what is alpha?", "top_k": settings.top_k}

    response_artifact = mlflow.artifacts.load_dict(f"runs:/{run_id}/response.json")
    assert response_artifact["answer"] == body["answer"]


def test_mlflow_enabled_logs_run_for_empty_corpus(tmp_store, mlflow_tmp, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("LLM must not be called when corpus is empty")

    monkeypatch.setattr(llm, "generate_answer", boom)

    response = TestClient(app).post("/query", json={"question": "anything?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "I don't know."
    assert body["sources"] == []

    df = mlflow.search_runs(experiment_names=[settings.mlflow_experiment])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["params.corpus_size"] == "0"
    assert row["metrics.llm_ms"] == 0
