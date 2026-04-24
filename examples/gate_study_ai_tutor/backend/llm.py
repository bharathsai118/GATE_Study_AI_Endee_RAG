"""LLM wrapper with no-key fallback behavior."""

from __future__ import annotations

from typing import Optional

from backend.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_OPENAI_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)


def _client(api_key: str, base_url: Optional[str] = None):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def generate_response(
    system_prompt: str,
    user_prompt: str,
    fallback_text: str,
    temperature: float = 0.2,
) -> str:
    try:
        if OPENAI_API_KEY:
            response = _client(OPENAI_API_KEY, OPENAI_BASE_URL).chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or fallback_text
        if GEMINI_API_KEY:
            response = _client(GEMINI_API_KEY, GEMINI_OPENAI_BASE_URL).chat.completions.create(
                model=GEMINI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or fallback_text
    except Exception as exc:
        return f"{fallback_text}\n\nLLM call failed, so fallback mode was used. Error: {exc}"
    return fallback_text
