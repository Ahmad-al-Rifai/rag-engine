# RAG Engine

Local retrieval engine for an Obsidian markdown vault.

## Pipeline

Obsidian Vault → Markdown ingestion → Embeddings → ChromaDB → FastAPI API

## Files

- `config.py` — all settings, read from env vars / `.env` (defaults in `.env.example`)
- `ingest.py` — reads markdown notes and stores embeddings
- `query.py` — tests retrieval from terminal
- `api.py` — exposes retrieval over HTTP
- `llm.py` — Ollama provider (not wired in yet)

## Setup

Runs inside WSL Ubuntu. Keep the project on the Linux filesystem (`~/rag-engine`), not under `/mnt/c`, or imports take minutes.

```bash
cd ~/rag-engine
python3 -m venv venv
venv/bin/pip install -r requirements.txt   # CPU-only torch
```

## Configuration

Copy `.env.example` to `.env` and edit as needed. Real environment variables override `.env`. Relative paths (like `CHROMA_PATH=chroma_db`) are resolved against the project folder, so scripts work from any directory.

## Run

```bash
venv/bin/python ingest.py                         # index the vault
venv/bin/uvicorn api:app --reload --port 8000     # GET /health, POST /query
venv/bin/python query.py                          # interactive retrieval
```

## Test

```bash
venv/bin/python -m pytest -q
```
