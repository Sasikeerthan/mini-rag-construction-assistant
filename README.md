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

### Cloud LLM: `meta-llama/llama-3.1-8b-instruct:free` (via OpenRouter)
- **Why**: Free tier on OpenRouter, 8B parameters, strong instruction-following. Provides a cloud-based comparison point against the local phi3:mini model. No local GPU required.

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

### OpenRouter Setup (Optional — for model comparison)

1. Get a free API key at [openrouter.ai](https://openrouter.ai)
2. Set the environment variable before running:

```bash
OPENROUTER_API_KEY=your-key-here docker compose up --build
```

Or create a `.env` file (see `.env.example`).

### Running the Evaluation

```bash
# Inside the container or locally with dependencies installed
OPENROUTER_API_KEY=your-key-here python evaluate.py
```

This runs 15 test questions through both models and outputs `evaluation_results.md`.

## Project Structure

```
RAG/
├── app.py                  # Streamlit chat UI (streaming, model comparison)
├── evaluate.py             # Automated evaluation script
├── rag/
│   ├── __init__.py
│   ├── chunker.py          # Markdown-aware document chunking
│   ├── embedder.py         # Sentence-transformers embedding
│   ├── retriever.py        # FAISS index + semantic search
│   ├── generator.py        # Ollama LLM integration (local)
│   └── openrouter_generator.py  # OpenRouter LLM integration (cloud)
├── documents/
│   ├── doc1.md             # Company overview & customer journey
│   ├── doc2.md             # Package comparison & specifications
│   └── doc3.md             # Policies, quality, guarantees
├── test_questions.csv      # 15 test questions with expected answers
├── evaluation_results.md   # Model comparison results (generated)
├── .env.example            # Environment variable template
├── Dockerfile
├── docker-compose.yml
├── pip-requirements.txt    # Python dependencies
└── README.md
```

## Model Comparison Findings

Comparison of Local (phi3:mini, 3.8B) vs OpenRouter (Llama 3.1 8B) across 15 test questions:

| Metric | Local (phi3:mini) | OpenRouter (Llama 3.1 8B) |
|--------|-------------------|---------------------------|
| Avg Latency | Higher (CPU inference) | Lower (cloud GPU) |
| Groundedness | Good — stays within context | Good — stays within context |
| Key-Point Coverage | Moderate | Higher (larger model captures more detail) |

**Key Observations:**
- Both models respect the grounding constraint well due to the explicit system prompt
- The cloud model (Llama 3.1 8B) produces more detailed answers and covers more expected key points
- The local model (phi3:mini) has higher latency on CPU but requires no API key or internet
- Streaming responses significantly improve perceived latency for both models

See `evaluation_results.md` for detailed per-question results.

## Sample Results
![image1](https://github.com/user-attachments/assets/4e60bf84-0fc0-486b-9351-d9e0eb2124a0)

