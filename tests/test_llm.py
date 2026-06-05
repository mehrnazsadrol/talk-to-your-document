from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.llm as llm
from app.config import settings


def test_generate_answer_uses_system_prompt_and_returns_content(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="the answer"))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=34),
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr(llm, "_client", fake_client)

    retrieved = [
        {
            "text": "alpha apple is a fruit",
            "metadata": {
                "document_id": "doc1",
                "source_filename": "sample.pdf",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 22,
            },
            "distance": 0.1,
        }
    ]

    answer, usage = llm.generate_answer("what is alpha?", retrieved)

    assert answer == "the answer"
    assert usage == {"prompt_tokens": 12, "completion_tokens": 34}
    messages = captured["kwargs"]["messages"]
    assert messages[0]["role"] == "system"
    assert "strictly from the provided context" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "alpha apple is a fruit" in messages[1]["content"]
    assert "what is alpha?" in messages[1]["content"]


def test_generate_answer_returns_empty_string_when_content_is_none(monkeypatch):
    def fake_create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=None,
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr(llm, "_client", fake_client)

    answer, usage = llm.generate_answer("anything?", [])
    assert answer == ""
    assert usage is None


def test_generate_answer_raises_503_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", None)
    monkeypatch.setattr(llm, "_client", None)

    with pytest.raises(HTTPException) as excinfo:
        llm.generate_answer("anything?", [])
    assert excinfo.value.status_code == 503
