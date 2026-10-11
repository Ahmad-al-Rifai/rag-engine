# RAG Engine

Local retrieval engine for an Obsidian markdown vault.

## Pipeline

Obsidian Vault → Markdown ingestion → Embeddings → ChromaDB → retrieval → local LLM (Ollama) → answer with sources → FastAPI / CLI

## Files

- `config.py` — all settings, read from env vars / `.env` (defaults in `.env.example`)
- `ingest.py` — indexes the vault: heading-aware chunks, idempotent, removes stale chunks (`--dry-run` to preview)
- `store.py` — shared, lazily loaded embedding model and ChromaDB collection
- `retrieval.py` — `retrieve(question, top_k)`: the single retrieval path
- `query.py` — inspect retrieval from the terminal
- `rag.py` — builds the grounded prompt (notes as untrusted data), calls the LLM, maps [n] citations to notes
- `ask.py` — ask a question from the terminal, get an answer with sources
- `api.py` — HTTP API: `/health`, `/query` (retrieval only), `/ask` (answer + sources)
- `eval_retrieval.py` — scores retrieval quality (hit@1, hit@3, MRR) on a fixed question set
- `llm.py` — `LLMProvider` interface + `OllamaProvider`; all failures raise `LLMError`

## Setup

Runs inside WSL Ubuntu. Keep the project on the Linux filesystem (`~/rag-engine`), not under `/mnt/c`, or imports take minutes.

```bash
cd ~/rag-engine
python3 -m venv venv
venv/bin/pip install -r requirements.txt   # CPU-only torch
```

## Ollama (local LLM)

Ollama runs on Windows. WSL reaches it at `localhost:11434` only with mirrored networking, set in `C:\Users\user\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

Then run `wsl --shutdown` once from Windows. Check from WSL: `curl -s localhost:11434/api/tags`.

## Configuration

Copy `.env.example` to `.env` and edit as needed. Real environment variables override `.env`. Relative paths (like `CHROMA_PATH=chroma_db`) are resolved against the project folder, so scripts work from any directory.

## Run

```bash
venv/bin/python ingest.py --dry-run               # preview what would change
venv/bin/python ingest.py                         # index the vault (safe to re-run)
venv/bin/uvicorn api:app --reload --port 8000     # GET /health, POST /query, POST /ask
venv/bin/python ask.py "what are my rules?"       # answer with sources (needs Ollama)
venv/bin/python query.py "what are my rules"      # one-shot retrieval (no arg = interactive)
venv/bin/python eval_retrieval.py --live          # retrieval quality report
```

## Test

```bash
venv/bin/python -m pytest -q
```
