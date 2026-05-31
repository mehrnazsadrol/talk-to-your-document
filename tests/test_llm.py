from types import SimpleNamespace

import app.llm as llm


def test_generate_answer_uses_system_prompt_and_returns_content(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="the answer"))]
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

    result = llm.generate_answer("what is alpha?", retrieved)

    assert result == "the answer"
    messages = captured["kwargs"]["messages"]
    assert messages[0]["role"] == "system"
    assert "strictly from the provided context" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "alpha apple is a fruit" in messages[1]["content"]
    assert "what is alpha?" in messages[1]["content"]
