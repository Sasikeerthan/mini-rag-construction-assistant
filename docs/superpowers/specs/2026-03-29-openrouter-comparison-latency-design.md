# Design: OpenRouter Integration, Model Comparison, Streaming & Evaluation

## Overview

Add OpenRouter LLM support, side-by-side model comparison UI, streaming responses for latency reduction, and an automated evaluation script. These address the bonus requirements from the assignment spec.

## Architecture Changes

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit UI (app.py)                                  │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │ Single Mode  │  │ Compare Mode │                     │
│  │ (dropdown)   │  │ (side-by-side)│                    │
│  └──────┬───────┘  └──────┬───────┘                     │
│         │                 │                              │
│         ▼                 ▼                              │
│  ┌─────────────────────────────────────┐                │
│  │  Retriever (shared - same chunks)   │                │
│  └─────────────┬───────────────────────┘                │
│                │                                         │
│       ┌────────┴────────┐                               │
│       ▼                 ▼                                │
│  ┌──────────┐   ┌──────────────┐                        │
│  │ Ollama   │   │ OpenRouter   │                        │
│  │ (stream) │   │ (stream)     │                        │
│  └──────────┘   └──────────────┘                        │
└─────────────────────────────────────────────────────────┘

evaluate.py (standalone) ─── runs both generators ─── outputs evaluation_results.md
```

## Component Details

### 1. OpenRouter Generator (`rag/openrouter_generator.py`)

New module, same interface as `generator.py`:

- **API**: `POST https://openrouter.ai/api/v1/chat/completions` (OpenAI-compatible)
- **Model**: `meta-llama/llama-3.1-8b-instruct:free`
- **Auth**: `OPENROUTER_API_KEY` env var, validated at startup
- **System prompt**: Same `SYSTEM_PROMPT` as Ollama generator (import from `generator.py`)
- **Context formatting**: Reuse `_format_context()` from `generator.py`
- **Temperature**: 0.3 (same as Ollama for fair comparison)

Functions:
- `generate_answer(query, context_chunks) -> str` — non-streaming, for evaluation script
- `generate_answer_stream(query, context_chunks) -> Iterator[str]` — yields tokens for UI
- `check_openrouter() -> bool` — validates API key and connectivity

### 2. Streaming for Ollama Generator (`rag/generator.py` changes)

Add streaming function alongside existing non-streaming one:

- `generate_answer_stream(query, context_chunks) -> Iterator[str]` — new function
  - Uses `stream: True` in Ollama API payload
  - Iterates over response lines, parses JSON, yields `response` field tokens
  - Existing `generate_answer()` stays unchanged (used by evaluation script)

### 3. Shared Prompt Utilities

Extract shared constants/functions to avoid duplication:

- Move `SYSTEM_PROMPT` and `_format_context()` to be importable from `generator.py`
- `openrouter_generator.py` imports these from `generator.py`

### 4. App UI Changes (`app.py`)

**Sidebar additions:**
- Model selector dropdown: "Local (phi3:mini)" / "OpenRouter (Llama 3.1 8B)"
- "Compare Models" toggle checkbox
- OpenRouter status indicator (connected/API key missing)

**Single mode (compare off):**
- Works like current app but uses selected model
- Streams tokens into `st.chat_message` using `st.write_stream()`
- Shows latency timer after response completes

**Compare mode (compare on):**
- Query goes through retrieval once (shared chunks)
- Two `st.columns` side by side
- Both models stream simultaneously using `concurrent.futures.ThreadPoolExecutor`
- Each column shows: model name, streamed answer, latency, retrieved context expander
- Chat history stores both answers

**Pre-warming at startup:**
- After `wait_for_ollama()`, send a dummy generate request: `{"model": "phi3:mini", "prompt": "hello", "stream": false}` with short timeout
- Call `get_embeddings(["warmup"])` to eagerly load the embedding model

### 5. Evaluation Script (`evaluate.py`)

Standalone CLI script, not part of the Streamlit app:

**Input:** `test_questions.csv` (15 questions with expected key points)

**Process for each question:**
1. Embed query, retrieve top-5 chunks
2. Run through Ollama generator (non-streaming) — measure latency
3. Run through OpenRouter generator (non-streaming) — measure latency
4. Score groundedness: check if answer content references/matches retrieved chunk content (simple keyword overlap ratio)
5. Score against expected key points: what fraction of expected key points appear in the answer

**Output:** `evaluation_results.md` with:
- Per-question comparison table (Ollama answer summary, OpenRouter answer summary, latency, groundedness score, key-point coverage)
- Aggregate summary: average latency, average groundedness, average key-point coverage per model
- Observations section for README

**Dependencies:** Requires both Ollama running and `OPENROUTER_API_KEY` set.

**Usage:**
```bash
# Inside the Docker container or locally with dependencies
OPENROUTER_API_KEY=sk-... python evaluate.py
```

### 6. Docker/Config Changes

**`docker-compose.yml`:**
- Add `OPENROUTER_API_KEY` env var pass-through to `rag-app` service

**`pip-requirements.txt`:**
- No new dependencies needed (already has `requests`)

**`.env.example`:** (new file)
- Document `OPENROUTER_API_KEY=your-key-here`

### 7. README Updates

Add sections:
- OpenRouter model details and why Llama 3.1 8B was chosen
- How to set up OpenRouter API key
- Model comparison findings (from evaluation_results.md)
- Updated architecture diagram showing both generators
- Updated project structure

## Implementation Order (Parallelizable Workstreams)

### WS1: Streaming + Pre-warming (no external deps)
1. Add `generate_answer_stream()` to `generator.py`
2. Add pre-warming logic to `app.py` initialization
3. Update `handle_query()` to use streaming with `st.write_stream()`

### WS2: OpenRouter Generator (independent module)
1. Create `rag/openrouter_generator.py` with both `generate_answer()` and `generate_answer_stream()`
2. Add `check_openrouter()` health check
3. Update `docker-compose.yml` with env var

### WS3: Comparison UI (depends on WS1 + WS2)
1. Add sidebar controls (model selector, compare toggle)
2. Implement single-model mode with model selection
3. Implement compare mode with side-by-side columns
4. Update chat history to store model info

### WS4: Evaluation Script (depends on WS2)
1. Create `evaluate.py`
2. Run evaluation, generate `evaluation_results.md`
3. Update README with findings

**Parallel execution plan:**
- WS1 and WS2 run in parallel (independent)
- WS3 runs after WS1 + WS2 complete
- WS4 can start as soon as WS2 is done (doesn't need streaming)

## Files Changed/Created

| File | Action | Workstream |
|------|--------|------------|
| `rag/generator.py` | Edit — add `generate_answer_stream()` | WS1 |
| `app.py` | Edit — streaming, pre-warming, comparison UI, sidebar | WS1 + WS3 |
| `rag/openrouter_generator.py` | Create | WS2 |
| `docker-compose.yml` | Edit — add env var | WS2 |
| `.env.example` | Create | WS2 |
| `evaluate.py` | Create | WS4 |
| `evaluation_results.md` | Generated output | WS4 |
| `README.md` | Edit — add comparison section | WS4 |
