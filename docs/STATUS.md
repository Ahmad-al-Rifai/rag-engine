# Status

Last verified: 2026-10-11, in `~/rag-engine` (WSL Ubuntu). Every item below comes from a command that was actually run. The evidence is listed next to it.

## Milestones

| # | Milestone | Status | Evidence |
|---|---|---|---|
| 1 | Reproducible local setup | **Done** | Lean `requirements.txt` with CPU torch. `import api` takes 17s (it took 235s+ from `/mnt/c`). All config comes from env vars plus `.env.example`. README commands verified. |
| 2 | Reliable ingestion | **Done** | `tests/test_ingest.py` (12 tests): no duplicates, stale and deleted chunks removed, vault move changes no IDs, errors keep the last good chunks. A real second run reports `5 unchanged, 0 written`. |
| 3 | Reliable retrieval with source metadata | **Done** | `eval_retrieval.py --live`: hit@1 0.87, hit@3 1.00, MRR 0.93 on 15 questions. Sources are vault-relative. `top_k` is limited to 1–20 (out-of-range values get 422). |
| 4 | RAG answers with source references | **Done, with one known miss** | `eval_rag.py`: 12/13. Live `/ask` answered the CVE-rules question correctly and cited `master-rules.md`. |
| 5 | Clean provider abstraction | **Done** | `LLMProvider` + `get_provider()`. All failures become `LLMError`. 12 mocked tests. Live: unknown model → 404 message, unreachable → "cannot reach" in 5s. |
| 6 | Tests, error handling, logging, docs | **Done** | 85 pytest tests pass. Logging goes to stderr (`LOG_LEVEL`). README, ARCHITECTURE, STATUS. |
| 7 | Agent workflows | Not started | Out of scope until 1–6 have been stable for a while. |

## Answer-quality eval (`eval_rag.py`, mistral 7B, temperature 0)

| Run | Score | Change |
|---|---|---|
| Baseline | 9/13 | |
| After fixes | 12/13 | Prompt treats AI-rule notes as content; temperature 0; ≤2 sections per note; "don't know" never cited; parser accepts "note n" |

Passing angles: single-note facts, section-level fact, paraphrase, explaining rules without obeying them, synthesis across the capture log and weekly review, two unanswerable questions, prompt injection, empty and overlong input, LLM unreachable.

## Known issues

1. **One citation is attributed to the wrong note.** On "career goals + CTF folder", mistral cites the profile note for the `07_Labs` fact, which actually comes from `README.md`. A per-sentence citation instruction was tried, made no difference, and was reverted.
2. **Answers are slow.** 46–204s per answer on this machine. mistral runs on CPU, and Windows had about 0.5 GB of RAM free during the tests, so the model gets paged out. `OLLAMA_TIMEOUT=300` and `OLLAMA_KEEP_ALIVE=30m` absorb this. A smaller model would be faster, but it would need a download and its own eval run.
3. **ChromaDB writes bookkeeping on every open.** The file checksum changes even on read-only use. The data doesn't change.
4. **Hugging Face Hub check.** Without `HF_HUB_OFFLINE=1`, loading the model sends one unauthenticated request to the HF Hub. Setting `HF_HUB_OFFLINE=1` works fully offline once the model is cached.
5. **Test warning.** `StarletteDeprecationWarning` (httpx) comes from FastAPI's TestClient. It's harmless, and fixing it would need a new package.

## Operational notes

- The real repo is `~/rag-engine` in WSL. `C:\Users\user\rag-engine` is a stale copy.
- Backups of the DB from before the re-indexes: `~/chroma_db_backup_2026-10-11`, `~/chroma_db_backup_pre_inc3`.
- WSL reaches Windows Ollama only with `networkingMode=mirrored` in `C:\Users\user\.wslconfig`.

## Next step

Use AIOS on real questions for a while and add any failures as `eval_rag.py` cases. If latency gets in the way, compare a smaller Ollama model with `eval_rag.py` before switching.
