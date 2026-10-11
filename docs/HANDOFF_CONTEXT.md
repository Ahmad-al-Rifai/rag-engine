# AIOS — Historical Context (UNVERIFIED)

Collected from earlier ChatGPT/Claude conversations. **None of this is verified.** Confirm each item against the repo and runtime, then move confirmed facts into `docs/STATUS.md` / `docs/ARCHITECTURE.md`.

## Claimed history
- Built an Obsidian-based RAG setup with FastAPI and ChromaDB.
- A `/health` endpoint once reported 5 indexed vault documents.
- Vault path previously used: `/mnt/c/Users/user/obsidian-vault` (WSL path to a Windows folder).
- The Obsidian vault is reportedly synced to GitHub at `Ahmad-al-Rifai/Obsidian-Vault`.
- Ollama provider explored with model `mistral`.
- A provider abstraction was discussed so local models could later be swapped for hosted ones. It may be partial or nonexistent.
- Intended long-term design: a model-agnostic multi-agent layer with 5 agents (Research, Security, Builder, Review, Documentation). Treat this as a **future milestone (7)** only.

## Verified from GitHub (2026-10-10, `Ahmad-al-Rifai/rag-engine`, branch `main`)
Verified by reading the repo only. Nothing was executed, and the local checkout may differ.
- 3 commits, all on 2026-05-31 ("phase 2 complete"). Files: `ingest.py`, `query.py`, `api.py`, `requirements.txt`, `.gitignore`.
- Embeddings: `sentence-transformers` `all-MiniLM-L6-v2`. ChromaDB `PersistentClient` at `./chroma_db`, collection `vault`.
- Ingest: recursive `*.md` scan (skips `Templates/`), frontmatter parsed, 250-word chunks, IDs = md5(`filepath_i`), upserted one chunk at a time.
- API: `GET /health` (returns `collection.count()`, which is the **chunk** count, not the document count), `POST /query` (retrieval only).
- **There is no LLM, Ollama, Mistral, or provider code at all.** Generation (Milestone 4) and provider abstraction (Milestone 5) have not been started.
- No tests, README, `.env.example`, or config handling. `VAULT_PATH` is hard-coded in `ingest.py`. `python-dotenv` and `pydantic-settings` are installed but unused.
- `requirements.txt` is a full `pip freeze` that includes CUDA, torch, and kubernetes packages (multi-GB).

## Known issues found in code review
- Stale chunks: if a note shrinks, its old high-index chunks stay in the DB. Deleted notes are never removed.
- IDs include the absolute path, so changing `VAULT_PATH` re-indexes everything as duplicates.
- Embedding and upsert happen per chunk instead of in batches, which is slow for large vaults.
- Model and DB are initialised at import time in all 3 files. Config is duplicated across them.
- `top_k` is unvalidated (0, negative, or huge values are all accepted).
- The API returns absolute filesystem paths (`source`, `folder`).
- Ingest errors are only `print`ed. There is no summary or exit code.

## Open questions for Ahmad (cannot be discovered from code)
<!-- Ahmad: fill these in, or Claude will ask -->
- Code repo = `rag-engine` (confirmed on GitHub). Local path: `C:\Users\user\rag-engine` (`/mnt/c/Users/user/rag-engine` in WSL). Local checkout had 4 commits (latest `ff49270`, adds README) plus an untracked `llm.py` (`OllamaProvider`) on 2026-10-10. It is at least that far ahead of the GitHub snapshot above. Not fetched, so exact sync is unknown.
- `Obsidian-Vault` repo: private, renamed, or not created? (Not readable anonymously.)
- Machine: RAM / GPU / free disk (decides whether Mistral 7B runs locally)
- Windows + WSL: verified. The venv is a Linux venv created in WSL (`/usr/bin/python3.13`). WSL distros: Ubuntu, kali-linux, docker-desktop. Vault at `C:\Users\user\obsidian-vault` holds 8 `.md` files.
- Should the vault ever be written to by AIOS (e.g. generated notes), or stay read-only?
- What is the first real question/workflow AIOS should answer well? (Defines "done" for Milestone 4.)
