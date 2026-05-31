from __future__ import annotations

from openai import OpenAI

from app.config import settings

_SYSTEM_PROMPT = (
    "You answer questions strictly from the provided context. "
    "If the answer isn't in the context, say you don't know. "
    "Cite chunk indices inline like [chunk_index=3]."
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
    return _client


def generate_answer(question: str, retrieved: list[dict]) -> str:
    parts = []
    for r in retrieved:
        meta = r["metadata"]
        parts.append(
            f"Chunk [chunk_index={meta['chunk_index']}] "
            f"(source={meta['source_filename']}):\n{r['text']}"
        )
    context = "\n\n".join(parts)
    user_message = f"{context}\n\nQuestion: {question}"

    response = _get_client().chat.completions.create(
        model=settings.llm_model,
        max_tokens=1024,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
