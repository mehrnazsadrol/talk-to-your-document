"""Pure unit tests for the ``append_turn`` helper in ``frontend/app.py``."""

from __future__ import annotations

from frontend.app import append_turn


def test_append_user_then_assistant():
    messages: list = []
    append_turn(messages, "user", "What is RAG?")
    sources = [{"document_id": "abc", "chunk_index": 0, "snippet": "..."}]
    append_turn(messages, "assistant", "RAG is...", sources=sources)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is RAG?"
    assert messages[0]["sources"] is None
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "RAG is..."
    assert messages[1]["sources"] is sources


def test_append_default_sources_is_none():
    messages: list = []
    append_turn(messages, "user", "hi")
    assert messages[0]["sources"] is None
