"""Ollama-based LLM answer generator for the Mini RAG pipeline."""

import os
import time
import json
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")

SYSTEM_PROMPT = (
    "You are a helpful AI assistant for Indecimal, a construction marketplace. "
    "Answer the user's question using ONLY the provided context below. "
    "Do NOT use any outside knowledge. If the answer is not found in the context, "
    'say "I don\'t have enough information in the provided documents to answer this question." '
    "Be specific and cite relevant details from the context."
)


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate an answer using Ollama, grounded in the provided context chunks.

    Args:
        query: The user's question.
        context_chunks: Retrieved chunks, each a dict with keys
            "doc", "header", "content", "score".

    Returns:
        The generated answer text.
    """
    context_block = _format_context(context_chunks)

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Context ---\n{context_block}\n--- End Context ---\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.3},
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.ConnectionError:
        return (
            "Sorry, I'm unable to reach the language model service right now. "
            "Please try again later."
        )
    except requests.RequestException as exc:
        return f"An error occurred while generating the answer: {exc}"


def generate_answer_stream(query: str, context_chunks: list[dict]):
    """Generate an answer using Ollama with streaming, grounded in provided context.

    Yields:
        str: Token chunks as they arrive from the model.
    """
    context_block = _format_context(context_chunks)

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Context ---\n{context_block}\n--- End Context ---\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": True,
        "options": {"temperature": 0.3},
    }

    try:
        response = requests.post(url, json=payload, timeout=120, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
    except requests.ConnectionError:
        yield (
            "Sorry, I'm unable to reach the language model service right now. "
            "Please try again later."
        )
    except requests.RequestException as exc:
        yield f"An error occurred while generating the answer: {exc}"


def wait_for_ollama(timeout: int = 300) -> bool:
    """Poll the Ollama server until it is ready.

    Args:
        timeout: Maximum seconds to wait.

    Returns:
        True when the server is ready, False if the timeout is reached.
    """
    url = f"{OLLAMA_HOST}/api/tags"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)

    return False


def _format_context(chunks: list[dict]) -> str:
    """Format context chunks into a readable text block."""
    if not chunks:
        return "(no context provided)"

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        doc = chunk.get("doc", "unknown")
        header = chunk.get("header", "")
        content = chunk.get("content", "")
        score = chunk.get("score", 0.0)

        header_line = f" > {header}" if header else ""
        parts.append(
            f"[{i}] (source: {doc}{header_line}, relevance: {score:.2f})\n{content}"
        )

    return "\n\n".join(parts)
