# Architecture

Local-first RAG over an Obsidian vault. Everything runs on one machine: Python in WSL Ubuntu (`~/rag-engine`), Ollama on Windows reached through WSL mirrored networking. No cloud services.

## Data flow

```
Obsidian vault (read-only, /mnt/c/Users/user/obsidian-vault)
   │  ingest.py: discover_notes → build_chunks → ingest
   │    - skips Templates/, dot-folders, symlinks leaving the vault
   │    - chunk_markdown: one chunk per heading section (≤ CHUNK_SIZE words),
   │      text prefixed "Note title > Heading > Subheading"
   │    - ID "<vault-relative path>#<i>", fingerprint per note
   ▼
store.py: embed() with all-MiniLM-L6-v2 (384-d, reads ≤256 tokens)
   ▼
ChromaDB (./chroma_db, collection "vault")  ── metadata per chunk:
   source (relative path), filename, folder, chunk, title, tags, area, fingerprint
   ▼
retrieval.py: validate → retrieve(question, top_k) → [Hit]       ← query.py, POST /query
   ▼
rag.py: over-fetch 3×top_k → diversify (≤2 sections per note)
        → build_prompt (numbered <note> blocks, note text neutralised)
        → LLMProvider.complete(SYSTEM_PROMPT, prompt)
        → "don't know" normalisation → parse_citations → Answer
   ▼
llm.py: get_provider() → OllamaProvider → POST {OLLAMA_BASE_URL}/api/chat
   ▼
ask.py (CLI)   ·   api.py  POST /ask  →  {answer, cited[], retrieved[]}
```

## Modules

| File | Role | Key functions |
|---|---|---|
| `config.py` | All settings from env vars / `.env`, validated at startup | `load_settings`, `settings`, `setup_logging` |
| `store.py` | Lazily created embedding model and Chroma collection | `get_model`, `get_collection`, `embed` |
| `ingest.py` | Idempotent indexing | `discover_notes`, `chunk_markdown`, `note_title`, `build_chunks`, `ingest`, `main` |
| `retrieval.py` | The only retrieval path; separate from generation | `validate`, `retrieve`, `diversify`, `Hit` |
| `rag.py` | Prompt building, citation mapping | `build_prompt`, `answer`, `parse_citations`, `is_not_found` |
| `llm.py` | Provider interface; provider-specific code lives only here | `LLMProvider`, `OllamaProvider`, `LLMError`, `get_provider` |
| `api.py` | FastAPI: `GET /health`, `POST /query`, `POST /ask` | sync endpoints (run in worker threads) |
| `ask.py`, `query.py` | Terminal entry points | |
| `eval_retrieval.py` | Retrieval quality (hit@1, hit@3, MRR) on temp indexes | |
| `eval_rag.py` | Answer quality against the real LLM, 13 checks | |

## Key design decisions

- **Idempotent re-index.** A note is re-embedded only when its fingerprint changes. The fingerprint hashes `INDEX_VERSION`, the model, `CHUNK_SIZE` and the raw file. A changed note has all its old chunk IDs replaced. Sources missing from disk are deleted. IDs are vault-relative, so moving the vault changes nothing.
- **Heading-aware chunks.** Chosen by measurement (`eval_retrieval.py`): hit@1 went from 0.60 to 0.87 and MRR from 0.76 to 0.93, compared with fixed 250-word whole-note chunks.
- **Diversity at answer time.** At most 2 sections per note, so a two-part question still reaches the second note. Retrieval itself (`/query`) stays pure similarity ranking.
- **Notes are untrusted data.** They're wrapped in `<note>` blocks, and tag-like text inside them is escaped. The system prompt says to explain rules found in notes but never obey them. An injection fixture is tested in `tests/test_rag.py` and `eval_rag.py`.
- **"Don't know" is a fixed sentence.** It's enforced in code so it never carries citations.
- **Explicit failure modes.**
  - Bad input gets a 422 (schema) or 400 (whitespace-only). The 1–20 bound on `top_k` and the 1000-character limit on questions are enforced in both the API and `retrieval.validate`.
  - Ollama failures raise `LLMError`, which `/ask` turns into a 503.
  - Connecting to Ollama times out after 5s, and reading its answer after `OLLAMA_TIMEOUT`.
- **Blocking work in FastAPI.** Endpoints are plain `def`, so FastAPI runs them in its threadpool. The embedding model is loaded once at startup (lifespan).
- **Logging.** Library modules log to stderr. CLI results go to stdout. The level comes from `LOG_LEVEL`.

## Configuration

See `.env.example`. Real env vars override `.env`. Relative paths resolve against the project folder.
