# Indecimal RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for **Indecimal**, a construction marketplace. The assistant answers user questions using internal documents (policies, FAQs, package specs) rather than relying on general LLM knowledge.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Streamlit UI (port 8501)                   │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Chunker │→ │ Embedder │→ │ Retriever │  │
│  └─────────┘  └──────────┘  └─────┬─────┘  │
│                                   │         │
│  User Query → Retrieve Top-K → Generate     │
│                                   │         │
│                          ┌────────▼───────┐ │
│                          │  Generator     │ │
│                          └────────┬───────┘ │
└───────────────────────────────────┼─────────┘
                                    │
                           ┌────────▼───────┐
                           │  Ollama (LLM)  │
                           │  phi3:mini     │
                           └────────────────┘
```

## Models Used

### Embedding Model: `all-MiniLM-L6-v2` (sentence-transformers)
- **Why**: Free, runs locally (no API keys), lightweight (~80MB), produces high-quality 384-dimensional embeddings. Excellent for semantic similarity tasks at this document scale. Normalized embeddings enable efficient cosine similarity via FAISS inner product.

### LLM: `phi3:mini` (3.8B parameters, via Ollama)
- **Why**: Open-source local LLM (bonus points), strong instruction-following capability, runs efficiently on CPU via Ollama. Good at grounded QA tasks where the model must stick to provided context. No external API dependencies.

## Document Chunking & Retrieval

### Chunking Strategy
- **Markdown-aware splitting**: Documents are split by `##` and `###` headers, preserving the section hierarchy (e.g., "Package Pricing > Steel")
- **Context preservation**: Each chunk includes its parent header path for context
- **Size control**: Target 200-400 tokens per chunk. Oversized sections are further split by paragraphs
- **Output**: Each chunk carries metadata: `doc` (filename), `header` (section path), `content` (text)

### Retrieval
- **FAISS IndexFlatIP**: Inner product index on L2-normalized embeddings (equivalent to cosine similarity)
- **Top-K search**: Configurable via sidebar slider (default: 5 chunks)
- **Semantic matching**: Queries are embedded with the same model and matched against all document chunks

## Grounding Enforcement

The LLM is constrained to answer only from retrieved context through:

1. **System prompt**: Explicitly instructs the model to use ONLY the provided context and to decline if the answer isn't found
2. **Context formatting**: Retrieved chunks are clearly delineated with source attribution (document, section, relevance score)
3. **Low temperature** (0.3): Reduces creative/hallucinated outputs
4. **Transparency**: The UI displays all retrieved chunks alongside the answer, allowing users to verify grounding

## Running Locally

### Prerequisites
- Docker and Docker Compose installed
- ~4GB disk space (for Ollama + phi3:mini model + sentence-transformers)

### Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd RAG

# Start all services
docker compose up --build
```

This will:
1. Start the Ollama LLM server
2. Pull the `phi3:mini` model (~2.3GB, first run only)
3. Build and start the Streamlit RAG app

Once ready, open **http://localhost:8501** in your browser.

### Stopping

```bash
docker compose down
```

To also remove the downloaded model volume:
```bash
docker compose down -v
```

## Project Structure

```
RAG/
├── app.py                  # Streamlit chat UI
├── rag/
│   ├── __init__.py
│   ├── chunker.py          # Markdown-aware document chunking
│   ├── embedder.py         # Sentence-transformers embedding
│   ├── retriever.py        # FAISS index + semantic search
│   └── generator.py        # Ollama LLM integration
├── documents/
│   ├── doc1.md             # Company overview & customer journey
│   ├── doc2.md             # Package comparison & specifications
│   └── doc3.md             # Policies, quality, guarantees
├── Dockerfile
├── docker-compose.yml
├── pip-requirements.txt    # Python dependencies
└── README.md
```

## Sample Results
![image1](https://github.com/user-attachments/assets/4e60bf84-0fc0-486b-9351-d9e0eb2124a0)

