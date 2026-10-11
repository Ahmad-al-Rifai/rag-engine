---
description: Read-only audit of the AIOS repo, environment, and runtime. No changes.
---

Run the Phase One audit of AIOS. **Read-only: do not modify files, git state, packages, models, or ChromaDB data.**

Read `CLAUDE.md` and `docs/HANDOFF_CONTEXT.md` first. Treat the handoff context as claims to verify.

## 1. Environment
Determine from evidence: OS (Windows / WSL / Linux), shell, Python version, dependency manager, and any existing venv. Do not print env var values.

## 2. Git
Report the repo root, remotes (strip any credentials from URLs), current branch, `git status`, and the last ~10 commits. Do **not** fetch or pull. Say whether local-vs-GitHub sync can be determined without fetching.

## 3. Codebase map
Find what actually exists for each component:
entry points · FastAPI routes · Markdown discovery · parsing/chunking · metadata · embeddings (which model?) · ChromaDB init/persistence path/collection names · retrieval · prompt building · LLM provider(s) · config handling · dependencies · tests · docs.

Draw the real data flow as a short text diagram. Mark missing components as missing.

## 4. Safe runtime checks
Run only checks that need no installs, downloads, or writes:
- Does the venv import the app?
- Run existing tests if present.
- Is the configured vault path readable? (Report counts only; don't read note contents.)
- Can ChromaDB be opened **read-only** and its collection count reported?
- Is Ollama reachable (`/api/tags`), and is `mistral` present?
- If the API can start without side effects, hit `/health`.

For each check you skipped, say why.

## 5. Report
Use these sections:
- **Verified facts**
- **Historical claims**: confirmed / contradicted / unverified
- **Inferences** (labelled as such)
- **Unknowns**, noting which ones only Ahmad can answer
- **Working / broken / missing**, mapped to Milestones 1–7 in CLAUDE.md
- **Biggest blocker**
- **Recommended next step** (one), then a short prioritized plan (≤5 items)

Then propose concrete values for the `## Commands` section of CLAUDE.md and wait for approval before editing anything.
