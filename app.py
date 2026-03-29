import time
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from rag.chunker import chunk_documents
from rag.embedder import get_embeddings
from rag.retriever import build_rag_index, search
from rag.generator import generate_answer_stream as ollama_stream
from rag.generator import generate_answer as ollama_generate
from rag.generator import wait_for_ollama, OLLAMA_HOST, OLLAMA_MODEL
from rag.openrouter_generator import (
    generate_answer_stream as openrouter_stream,
    generate_answer as openrouter_generate,
    check_openrouter,
)

import requests as _requests

st.set_page_config(
    page_title="Indecimal AI Assistant",
    layout="wide",
    page_icon="\U0001f3e0",
)

MODEL_OPTIONS = {
    "Local (phi3:mini)": "ollama",
    "OpenRouter (Llama 3.1 8B)": "openrouter",
}

# --- Sidebar ---
with st.sidebar:
    st.title("Indecimal RAG Assistant")
    st.markdown(
        "Ask questions about Indecimal and get answers grounded in our internal "
        "documentation."
    )
    st.divider()
    top_k = st.slider("Top-K Results", min_value=1, max_value=10, value=5, key="top_k")

    st.subheader("Model")
    selected_model_label = st.selectbox(
        "Choose LLM", list(MODEL_OPTIONS.keys()), key="model_label"
    )
    compare_mode = st.checkbox("Compare Models (side-by-side)", key="compare_mode")

    st.divider()
    st.subheader("Status")
    ollama_ready = st.session_state.get("ollama_ready", False)
    openrouter_ready = st.session_state.get("openrouter_ready", False)
    st.markdown(f"**Ollama:** {'Connected' if ollama_ready else 'Not connected'}")
    st.markdown(
        f"**OpenRouter:** {'Connected' if openrouter_ready else 'No API key'}"
    )
    num_chunks = len(st.session_state.get("chunks", []))
    st.markdown(f"**Chunks indexed:** {num_chunks}")

# --- Initialization ---
if "initialized" not in st.session_state:
    with st.spinner("Waiting for Ollama LLM to be ready..."):
        try:
            wait_for_ollama()
            st.session_state["ollama_ready"] = True
        except Exception as e:
            st.error(f"Failed to connect to Ollama: {e}")
            st.stop()

    # Pre-warm Ollama model
    with st.spinner("Pre-warming Ollama model..."):
        try:
            _requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": "hello", "stream": False},
                timeout=120,
            )
        except Exception:
            pass

    # Pre-warm embedding model
    with st.spinner("Loading embedding model..."):
        get_embeddings(["warmup"])

    with st.spinner("Indexing documents..."):
        try:
            index, chunks = build_rag_index("documents")
            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
        except Exception as e:
            st.error(f"Failed to index documents: {e}")
            st.stop()

    # Check OpenRouter
    st.session_state["openrouter_ready"] = check_openrouter()

    st.session_state["chat_history"] = []
    st.session_state["initialized"] = True
    st.rerun()


def _render_context(relevant_chunks: list[dict]):
    """Render retrieved context in an expander."""
    with st.expander(f"\U0001f4c4 Retrieved Context ({len(relevant_chunks)} chunks)"):
        for i, chunk in enumerate(relevant_chunks):
            doc_name = chunk.get("doc", "Unknown")
            section = chunk.get("header", "N/A")
            score = chunk.get("score", 0.0)
            text = chunk.get("content", "")
            st.markdown(
                f"**Chunk {i + 1}** | Document: `{doc_name}` | "
                f"Section: `{section}` | Similarity: `{score:.4f}`"
            )
            st.markdown(f"> {text}")
            if i < len(relevant_chunks) - 1:
                st.divider()


def handle_query_single(query: str, model_key: str):
    """Process a query using a single selected model with streaming."""
    st.session_state["chat_history"].append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    index = st.session_state["index"]
    chunks = st.session_state["chunks"]

    with st.spinner("Searching relevant documents..."):
        relevant_chunks = search(query, index, chunks, top_k=st.session_state["top_k"])

    stream_fn = ollama_stream if model_key == "ollama" else openrouter_stream
    model_name = "phi3:mini" if model_key == "ollama" else "Llama 3.1 8B"

    with st.chat_message("assistant"):
        start = time.time()
        answer = st.write_stream(stream_fn(query, relevant_chunks))
        elapsed = time.time() - start
        st.caption(f"Model: {model_name} | Latency: {elapsed:.1f}s")
        _render_context(relevant_chunks)

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer,
        "context": relevant_chunks,
        "model": model_name,
        "latency": elapsed,
    })


def handle_query_compare(query: str):
    """Process a query through both models side-by-side."""
    st.session_state["chat_history"].append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    index = st.session_state["index"]
    chunks = st.session_state["chunks"]

    with st.spinner("Searching relevant documents..."):
        relevant_chunks = search(query, index, chunks, top_k=st.session_state["top_k"])

    with st.chat_message("assistant"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Local (phi3:mini)**")
            start1 = time.time()
            answer1 = st.write_stream(ollama_stream(query, relevant_chunks))
            elapsed1 = time.time() - start1
            st.caption(f"Latency: {elapsed1:.1f}s")

        with col2:
            st.markdown("**OpenRouter (Llama 3.1 8B)**")
            start2 = time.time()
            answer2 = st.write_stream(openrouter_stream(query, relevant_chunks))
            elapsed2 = time.time() - start2
            st.caption(f"Latency: {elapsed2:.1f}s")

        _render_context(relevant_chunks)

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": f"**Local (phi3:mini):**\n{answer1}\n\n**OpenRouter (Llama 3.1 8B):**\n{answer2}",
        "context": relevant_chunks,
        "model": "comparison",
        "latency_ollama": elapsed1,
        "latency_openrouter": elapsed2,
    })


# --- Display chat history ---
for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "context" in msg:
            ctx = msg["context"]
            _render_context(ctx)

# --- Example questions (shown when chat is empty) ---
if not st.session_state.get("chat_history"):
    st.markdown("### Get started with an example question:")
    examples = [
        "What packages does Indecimal offer?",
        "How does the payment system work?",
        "What quality checks are performed?",
        "What is included in the maintenance program?",
    ]
    cols = st.columns(len(examples))
    for col, question in zip(cols, examples):
        if col.button(question, use_container_width=True):
            if st.session_state.get("compare_mode"):
                handle_query_compare(question)
            else:
                model_key = MODEL_OPTIONS[st.session_state["model_label"]]
                handle_query_single(question, model_key)
            st.rerun()

# --- Chat input ---
user_input = st.chat_input("Ask a question about Indecimal...")
if user_input:
    if st.session_state.get("compare_mode"):
        handle_query_compare(user_input)
    else:
        model_key = MODEL_OPTIONS[st.session_state["model_label"]]
        handle_query_single(user_input, model_key)
    st.rerun()
