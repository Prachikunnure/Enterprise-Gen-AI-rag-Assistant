"""
Modular LLM client using an OpenAI-compatible chat completions API.
"""

from typing import Any

import requests

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE
from src.utils import setup_logging

logger = setup_logging()

SYSTEM_PROMPT = """You are an enterprise document assistant.

Answer the user's question using ONLY the provided context.

Do not use outside knowledge.

If the answer cannot be found in the provided context, clearly say:
"I could not find this information in the uploaded documents."

Do not invent facts.

When possible, give a concise answer and mention the relevant source document name."""


class LLMConfigurationError(Exception):
    """Missing or invalid LLM configuration."""


def _check_api_key() -> None:
    if not LLM_API_KEY:
        raise LLMConfigurationError(
            "LLM API key is not configured. "
            "Copy .env.example to .env and set LLM_API_KEY=your_key_here, then restart the app."
        )


def format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Build context block with source and page for the prompt."""
    if not retrieved_chunks:
        return "(No relevant document excerpts were retrieved.)"

    parts: list[str] = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        source = ch.get("source", "unknown")
        page = ch.get("page", "?")
        text = ch.get("text", "")
        parts.append(f"[Excerpt {i}] Source: {source} | Page: {page}\n{text}")
    return "\n\n".join(parts)


def build_messages(
    question: str,
    context: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Assemble chat messages for the API."""
    user_content = f"""CONTEXT:
{context}

QUESTION:
{question}"""

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        for turn in conversation_history[-8:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_content})
    return messages


def generate_answer(
    question: str,
    context: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Call the LLM with grounded context. Requires LLM_API_KEY in .env.
    """
    _check_api_key()
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    messages = build_messages(question, context, conversation_history)

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
    }

    logger.info("LLM request started (model=%s)", LLM_MODEL)
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        logger.error("LLM request failed: %s", exc)
        raise RuntimeError(
            "Could not reach the LLM API. Check your network and LLM_BASE_URL in .env."
        ) from exc

    if response.status_code == 401:
        raise LLMConfigurationError(
            "LLM API rejected the key (401). Check LLM_API_KEY in your .env file."
        )
    if response.status_code >= 400:
        logger.error("LLM API error %s: %s", response.status_code, response.text[:500])
        raise RuntimeError(
            f"LLM API returned error {response.status_code}. "
            "Check LLM_MODEL and your account limits."
        )

    data = response.json()
    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        logger.error("Unexpected LLM response shape: %s", data)
        raise RuntimeError("Unexpected response from LLM API.") from exc

    logger.info("LLM response received")
    return answer


def get_llm_status_message() -> str:
    """Human-readable LLM configuration status for the UI."""
    if not LLM_API_KEY:
        return "LLM not configured — add LLM_API_KEY to .env"
    return f"LLM ready — model: {LLM_MODEL}"
