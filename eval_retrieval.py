"""Measure retrieval quality: does the right note come back for a question?

Builds a throwaway index (temp folder) per chunking strategy from the real
vault and real embedding model, then scores each strategy on QUESTIONS.
The live ChromaDB is never touched; the vault is only read.

Usage: venv/bin/python eval_retrieval.py [--live]
  --live  also score the live index (chroma_db) as currently built
Metrics: hit@1 / hit@3 (an expected note is in the top 1 / 3 distinct notes)
and MRR (mean reciprocal rank of the first expected note).
"""
import argparse
import tempfile

import chromadb
import frontmatter

from config import settings
from ingest import chunk_markdown, chunk_text, discover_notes, note_title
from store import embed, get_collection

RULES = "00_System/Rules/master-rules.md"
WEEKLY = "00_System/Workflows/weekly-review.md"
CAPTURE = "01_Inbox/capture-log.md"
PROFILE = "05_Agents/Briefs/user-profile.md"
README = "README.md"

# (question, acceptable notes). Mix of direct, paraphrased (little word
# overlap with the note) and section-level questions.
QUESTIONS = [
    ("what are my rules", {RULES}),
    ("how should AI agents behave when context is missing?", {RULES}),
    ("is it ok for the assistant to make up vulnerability IDs?", {RULES}),
    ("how should research results be formatted?", {RULES}),
    ("what are the principles for writing notes?", {RULES}),
    ("what do I do during my weekly review?", {WEEKLY}),
    ("checklist for cleaning up old projects", {WEEKLY}),
    ("how do I check notes for hallucination risk?", {WEEKLY}),
    ("where do I put quick ideas before organizing them?", {CAPTURE}),
    ("how often should the inbox be processed?", {CAPTURE, WEEKLY}),
    ("what internship am I looking for?", {PROFILE}),
    ("which tools do I use?", {PROFILE}),
    ("what is my background?", {PROFILE}),
    ("which folder holds experiments and CTFs?", {README}),
    ("how is the vault organized?", {README}),
]

STRATEGIES = {
    "baseline: 250-word chunks": lambda content, title: chunk_text(content, 250),
    "250-word chunks + title": lambda content, title: [f"{title}\n{c}" for c in chunk_text(content, 250)],
    "heading sections (<=120 words)": lambda content, title: chunk_markdown(content, title, 120),
}


def build_index(strategy, client, name):
    vault = settings.vault_path
    col = client.create_collection(name)
    for path in discover_notes(vault):
        post = frontmatter.load(path)
        content = post.content.strip()
        if not content:
            continue
        source = path.relative_to(vault).as_posix()
        chunks = strategy(content, note_title(post, path))
        col.add(
            ids=[f"{source}#{i}" for i in range(len(chunks))],
            embeddings=embed(chunks),
            documents=chunks,
            metadatas=[{"source": source}] * len(chunks),
        )
    return col


def ranked_sources(col, question):
    n = min(col.count(), 20)
    res = col.query(query_embeddings=embed([question]), n_results=n, include=["metadatas"])
    seen = []
    for meta in res["metadatas"][0]:
        if meta["source"] not in seen:
            seen.append(meta["source"])
    return seen


def score(col):
    hit1 = hit3 = rr = 0.0
    misses = []
    for question, expected in QUESTIONS:
        ranked = ranked_sources(col, question)
        rank = next((i + 1 for i, s in enumerate(ranked) if s in expected), None)
        hit1 += rank == 1
        hit3 += rank is not None and rank <= 3
        rr += 1 / rank if rank else 0
        if rank != 1:
            misses.append(f"    rank {rank}: {question!r} -> got {ranked[0]}")
    n = len(QUESTIONS)
    return hit1 / n, hit3 / n, rr / n, misses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="also score the live index")
    args = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        client = chromadb.PersistentClient(path=tmp)
        for i, (label, strategy) in enumerate(STRATEGIES.items()):
            col = build_index(strategy, client, f"eval{i}")
            results.append((label, col.count(), *score(col)))
    if args.live:
        col = get_collection()
        results.append(("LIVE index (chroma_db)", col.count(), *score(col)))

    print(f"{len(QUESTIONS)} questions\n")
    print(f"{'strategy':34} {'chunks':>6} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}")
    for label, count, h1, h3, mrr, _ in results:
        print(f"{label:34} {count:>6} {h1:>6.2f} {h3:>6.2f} {mrr:>6.2f}")
    for label, *_, misses in results:
        if misses:
            print(f"\n{label} - not ranked first:")
            print("\n".join(misses))


if __name__ == "__main__":
    main()
