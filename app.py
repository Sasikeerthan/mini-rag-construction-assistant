import streamlit as st
from rag.chunker import chunk_documents
from rag.embedder import get_embeddings
from rag.retriever import build_rag_index, search
from rag.generator import generate_answer, wait_for_ollama

st.set_page_config(
    page_title="Indecimal AI Assistant",
    layout="wide",
    page_icon="\U0001f3e0",
)

# --- Sidebar ---
with st.sidebar:
    st.title("Indecimal RAG Assistant")
    st.markdown(
        "Ask questions about Indecimal and get answers grounded in our internal "
        "documentation. The assistant retrieves relevant document chunks and uses "
        "a local LLM to generate accurate responses."
    )
    st.divider()
    top_k = st.slider("Top-K Results", min_value=1, max_value=10, value=5, key="top_k")
    st.divider()
    st.subheader("Status")
    ollama_ready = st.session_state.get("ollama_ready", False)
    st.markdown(f"**Ollama connected:** {'Yes' if ollama_ready else 'No'}")
    num_chunks = len(st.session_state.get("chunks", []))
    st.markdown(f"**Document chunks indexed:** {num_chunks}")

# --- Initialization ---
if "initialized" not in st.session_state:
    with st.spinner("Waiting for Ollama LLM to be ready..."):
        try:
            wait_for_ollama()
            st.session_state["ollama_ready"] = True
        except Exception as e:
            st.error(f"Failed to connect to Ollama: {e}")
            st.stop()

    with st.spinner("Indexing documents..."):
        try:
            index, chunks = build_rag_index("documents")
            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
        except Exception as e:
            st.error(f"Failed to index documents: {e}")
            st.stop()

    st.session_state["chat_history"] = []
    st.session_state["initialized"] = True
    st.rerun()


def handle_query(query: str):
    """Process a user query through the RAG pipeline and update chat history."""
    st.session_state["chat_history"].append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    index = st.session_state["index"]
    chunks = st.session_state["chunks"]

    try:
        with st.spinner("Searching relevant documents..."):
            relevant_chunks = search(query, index, chunks, top_k=st.session_state["top_k"])
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error during retrieval: {e}")
        return

    try:
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, relevant_chunks)
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error during generation: {e}")
        return

    with st.chat_message("assistant"):
        st.markdown(answer)
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

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer,
        "context": relevant_chunks,
    })


# --- Display chat history ---
for msg in st.session_state.get("chat_history", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "context" in msg:
            ctx = msg["context"]
            with st.expander(f"\U0001f4c4 Retrieved Context ({len(ctx)} chunks)"):
                for i, chunk in enumerate(ctx):
                    doc_name = chunk.get("doc", "Unknown")
                    section = chunk.get("header", "N/A")
                    score = chunk.get("score", 0.0)
                    text = chunk.get("content", "")
                    st.markdown(
                        f"**Chunk {i + 1}** | Document: `{doc_name}` | "
                        f"Section: `{section}` | Similarity: `{score:.4f}`"
                    )
                    st.markdown(f"> {text}")
                    if i < len(ctx) - 1:
                        st.divider()

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
            handle_query(question)
            st.rerun()

# --- Chat input ---
user_input = st.chat_input("Ask a question about Indecimal...")
if user_input:
    handle_query(user_input)
    st.rerun()
