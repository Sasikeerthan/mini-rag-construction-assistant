"""OpenRouter-based LLM answer generator for the Mini RAG pipeline."""

import json
import os
import requests

from rag.generator import SYSTEM_PROMPT, _format_context

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def _build_messages(query: str, context_chunks: list[dict]) -> list[dict]:
    context_block = _format_context(context_chunks)
    user_content = (
        f"--- Context ---\n{context_block}\n--- End Context ---\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def check_openrouter() -> bool:
    """Check if OpenRouter API key is set and the API is reachable."""
    if not OPENROUTER_API_KEY:
        return False
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=_headers(),
            timeout=10,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate an answer using OpenRouter, grounded in provided context."""
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": _build_messages(query, context_chunks),
        "temperature": 0.3,
        "stream": False,
    }

    try:
        response = requests.post(
            OPENROUTER_URL, headers=_headers(), json=payload, timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.ConnectionError:
        return "Unable to reach OpenRouter API. Please check your internet connection."
    except requests.RequestException as exc:
        return f"OpenRouter error: {exc}"
    except (KeyError, IndexError):
        return "Unexpected response format from OpenRouter API."


def generate_answer_stream(query: str, context_chunks: list[dict]):
    """Generate an answer using OpenRouter with streaming. Yields token strings."""
    if not OPENROUTER_API_KEY:
        yield "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."
        return

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": _build_messages(query, context_chunks),
        "temperature": 0.3,
        "stream": True,
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=_headers(),
            json=payload,
            timeout=120,
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield token
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    except requests.ConnectionError:
        yield "Unable to reach OpenRouter API."
    except requests.RequestException as exc:
        yield f"OpenRouter error: {exc}"
