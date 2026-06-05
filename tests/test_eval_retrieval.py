import json
from pathlib import Path

import pytest

import app.vector_store as vs
from app.chunker import Chunk
from app.config import settings
from eval import eval_retrieval


def _retrieved(items: list[tuple[str, int]]) -> list[dict]:
    return [
        {
            "text": f"text-{i}",
            "metadata": {"source_filename": fn, "chunk_index": idx},
            "distance": 0.1 * i,
        }
        for i, (fn, idx) in enumerate(items)
    ]


def test_hit_rate_at_k_true_when_match_at_position_zero():
    retrieved = _retrieved([("paper.pdf", 12), ("other.pdf", 0), ("paper.pdf", 99)])
    assert eval_retrieval.hit_rate_at_k(retrieved, "paper.pdf", [12, 13], k=3) is True
    assert eval_retrieval.hit_rate_at_k(retrieved, "paper.pdf", [12, 13], k=1) is True


def test_hit_rate_at_k_false_when_match_is_outside_top_k():
    retrieved = _retrieved(
        [
            ("other.pdf", 1),
            ("other.pdf", 2),
            ("other.pdf", 3),
            ("paper.pdf", 12),
            ("other.pdf", 5),
        ]
    )
    assert eval_retrieval.hit_rate_at_k(retrieved, "paper.pdf", [12], k=3) is False
    assert eval_retrieval.hit_rate_at_k(retrieved, "paper.pdf", [12], k=4) is True


def test_hit_rate_at_k_requires_filename_match():
    retrieved = _retrieved([("other.pdf", 12)])
    assert eval_retrieval.hit_rate_at_k(retrieved, "paper.pdf", [12], k=3) is False


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(vs, "_client", None)
    yield tmp_path
    vs._client = None


def test_run_eval_end_to_end(tmp_store, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "mlflow_enabled", False)

    fake_chunks = [
        Chunk(text="alpha apple", start_char=0, end_char=11, chunk_index=0),
        Chunk(text="beta banana", start_char=11, end_char=22, chunk_index=1),
        Chunk(text="gamma grape", start_char=22, end_char=33, chunk_index=2),
    ]
    fake_embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    vs.add_chunks(
        "doc-eval", "paper.pdf", fake_chunks, fake_embeddings, source_type="pdf"
    )

    monkeypatch.setattr(eval_retrieval, "seed_corpus", lambda persist_dir: None)

    def fake_embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            if "alpha" in t:
                out.append([1.0, 0.0, 0.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0, 0.0])
        return out

    monkeypatch.setattr(eval_retrieval, "embed_texts", fake_embed)

    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "alpha question",
                "expected_source_filename": "paper.pdf",
                "expected_chunk_indices": [0],
                "notes": "",
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "q2",
                "question": "beta question",
                "expected_source_filename": "paper.pdf",
                "expected_chunk_indices": [99],
                "notes": "expected chunk doesn't exist → guaranteed miss",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    results_dir = tmp_path / "results"
    aggregate = eval_retrieval.run_eval(golden, results_dir, top_k_max=5)

    assert 0.0 <= aggregate["hit_rate_at_3"] <= 1.0
    assert 0.0 <= aggregate["hit_rate_at_5"] <= 1.0
    # q1 hits, q2 misses → 0.5.
    assert aggregate["hit_rate_at_3"] == 0.5
    assert aggregate["hit_rate_at_5"] == 0.5

    # JSON written and matches returned aggregate.
    written = list(results_dir.glob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["aggregate"]["hit_rate_at_3"] == aggregate["hit_rate_at_3"]
    assert payload["aggregate"]["hit_rate_at_5"] == aggregate["hit_rate_at_5"]
    assert payload["params"]["golden_set_size"] == 2
    assert len(payload["per_question"]) == 2
    assert payload["per_question"][0]["id"] == "q1"
    assert payload["per_question"][0]["hit@3"] is True
    assert payload["per_question"][1]["hit@3"] is False
    assert len(payload["per_question"][0]["top5"]) <= 5


def test_load_golden_set_skips_blank_lines(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "a",
                "question": "?",
                "expected_source_filename": "x",
                "expected_chunk_indices": [0],
                "notes": "",
            }
        )
        + "\n\n"
        + json.dumps(
            {
                "id": "b",
                "question": "?",
                "expected_source_filename": "x",
                "expected_chunk_indices": [1],
                "notes": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = eval_retrieval.load_golden_set(path)
    assert [r["id"] for r in records] == ["a", "b"]
