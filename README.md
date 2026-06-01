# RAG Engine

Local retrieval engine for an Obsidian markdown vault.

## Pipeline

Obsidian Vault → Markdown ingestion → Embeddings → ChromaDB → FastAPI API

## Files

- `ingest.py` — reads markdown notes and stores embeddings
- `query.py` — tests retrieval from terminal
- `api.py` — exposes retrieval over HTTP

## Run

```bash
source venv/bin/activate
python ingest.py
uvicorn api:app --reload --port 8000

