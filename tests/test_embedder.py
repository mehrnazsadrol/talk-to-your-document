import pytest

from app import embedder
from app.embedder import embed_texts


def test_embed_single_text_returns_one_vector():
    vectors = embed_texts(["hello"])
    assert len(vectors) == 1


def test_embed_vectors_have_dim_384():
    vectors = embed_texts(["hello", "world"])
    for v in vectors:
        assert len(v) == 384


def test_embed_is_deterministic():
    a = embed_texts(["the quick brown fox"])
    b = embed_texts(["the quick brown fox"])
    assert a == b


def test_embed_empty_list_skips_model_call(monkeypatch):
    def boom():
        raise AssertionError("_get_model should not be called for empty input")

    monkeypatch.setattr(embedder, "_get_model", boom)
    assert embed_texts([]) == []
