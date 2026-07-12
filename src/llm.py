"""Thin, retrying wrapper around the OpenAI-compatible chat API."""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from config import TEACHER_MODEL, get_client

_client = None


def _client_lazy():
    global _client
    if _client is None:
        _client = get_client()
    return _client


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
def chat(system: str, user: str, temperature: float = 0.9, model: str | None = None) -> str:
    resp = _client_lazy().chat.completions.create(
        model=model or TEACHER_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
