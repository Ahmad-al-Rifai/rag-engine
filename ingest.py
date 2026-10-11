"""Index the Obsidian vault into ChromaDB.

Re-running is safe (idempotent):
- chunk IDs come from the vault-relative path + chunk index, so they don't
  change when the vault moves;
- unchanged notes are skipped (fingerprint check);
- changed notes have all their old chunks replaced, so a shrunk note leaves
  no stale chunks behind;
- chunks whose note no longer exists (deleted, renamed, or indexed under an
  old ID scheme) are removed.

Usage: venv/bin/python ingest.py [--dry-run]
"""
import argparse
import hashlib
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import frontmatter

from config import settings, setup_logging

log = logging.getLogger("ingest")  # stable name even when run as a script

# Bump when the chunk text or metadata format changes, to force a re-embed.
# 3: heading-aware chunks prefixed with title + heading trail.
INDEX_VERSION = "3"

EXCLUDED_DIRS = {"Templates"}

Embedder = Callable[[list[str]], list[list[float]]]


@dataclass
class Summary:
    notes_seen: int = 0
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed_sources: list[str] = field(default_factory=list)
    chunks_written: int = 0
    chunks_removed: int = 0
    errors: list[str] = field(default_factory=list)

    def report(self, dry_run: bool = False) -> str:
        prefix = "[dry run, nothing written] " if dry_run else ""
        lines = [
            f"{prefix}{self.notes_seen} notes: "
            f"{len(self.added)} added, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged, {len(self.removed_sources)} sources removed",
            f"{prefix}chunks: {self.chunks_written} written, {self.chunks_removed} removed",
        ]
        lines += [f"  added: {s}" for s in self.added]
        lines += [f"  updated: {s}" for s in self.updated]
        lines += [f"  removed: {s}" for s in self.removed_sources]
        lines += [f"  ERROR: {e}" for e in self.errors]
        return "\n".join(lines)


def chunk_text(text: str, size: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def note_title(post, path: Path) -> str:
    """Frontmatter title, else the first H1, else the file name."""
    if post.metadata.get("title"):
        return str(post.metadata["title"])
    for line in post.content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def chunk_markdown(content: str, title: str, max_words: int) -> list[str]:
    """Split a note on its Markdown headings.

    Each chunk is one section, prefixed with the note title and the heading
    trail (e.g. "Master Rules > AI Agent Rules > Reliability") so a section
    still says what it's about on its own. Sections longer than max_words are
    split by words; heading-only sections are folded into the trail.
    """
    trail: list[tuple[int, str]] = []  # (level, heading text)
    sections: list[tuple[str, list[str]]] = []
    body: list[str] = []

    def flush():
        text = "\n".join(body).strip()
        if text:
            sections.append((" > ".join(h for _, h in trail), text))
        body.clear()

    for line in content.splitlines():
        stripped = line.lstrip()
        level = len(stripped) - len(stripped.lstrip("#"))
        if 1 <= level <= 6 and stripped[level:level + 1] == " ":
            flush()
            trail[:] = [(lv, h) for lv, h in trail if lv < level]
            trail.append((level, stripped[level:].strip()))
        else:
            body.append(line)
    flush()

    def norm(s: str) -> str:
        return " ".join(s.lower().replace("-", " ").replace("_", " ").split())

    chunks = []
    for heading, text in sections:
        parts = [title]
        for h in heading.split(" > ") if heading else []:
            if norm(h) != norm(parts[-1]):  # skip an H1 that just repeats the title
                parts.append(h)
        header = " > ".join(parts)
        for piece in chunk_text(text, max_words):
            chunks.append(f"{header}\n{piece}")
    return chunks


def discover_notes(vault: Path) -> list[Path]:
    """All .md files inside the vault, skipping Templates and dot-folders
    (.obsidian, .trash, ...) and anything that resolves outside the vault."""
    vault = vault.resolve()
    notes = []
    for path in sorted(vault.rglob("*.md")):
        rel_parts = path.relative_to(vault).parts
        if any(p in EXCLUDED_DIRS or p.startswith(".") for p in rel_parts[:-1]):
            continue
        if not path.resolve().is_relative_to(vault):  # symlink escaping the vault
            continue
        notes.append(path)
    return notes


def fingerprint(raw: str, chunk_size: int, embed_model: str) -> str:
    key = f"{INDEX_VERSION}|{embed_model}|{chunk_size}|{raw}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def build_chunks(path: Path, vault: Path, chunk_size: int, embed_model: str):
    """Return (source, fingerprint, ids, texts, metadatas) for one note."""
    rel = path.resolve().relative_to(vault.resolve()).as_posix()
    raw = path.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    content = post.content.strip()
    fp = fingerprint(raw, chunk_size, embed_model)

    title = note_title(post, path)
    chunks = chunk_markdown(content, title, chunk_size) if content else []
    folder = Path(rel).parent.as_posix()
    ids, metas = [], []
    for i in range(len(chunks)):
        ids.append(f"{rel}#{i}")
        metas.append({
            "source": rel,
            "filename": path.name,
            "folder": "" if folder == "." else folder,
            "chunk": i,
            "title": title,
            "tags": str(post.metadata.get("tags", [])),
            "area": str(post.metadata.get("area", "")),
            "fingerprint": fp,
        })
    return rel, fp, ids, chunks, metas


def ingest(vault: Path, collection, embed: Embedder, chunk_size: int,
           embed_model: str, dry_run: bool = False) -> Summary:
    summary = Summary()
    vault = vault.resolve()
    start = time.time()

    # What's indexed now: source -> (set of chunk ids, set of fingerprints)
    existing = collection.get(include=["metadatas"])
    indexed: dict[str, tuple[set, set]] = {}
    for chunk_id, meta in zip(existing["ids"], existing["metadatas"]):
        meta = meta or {}
        ids, fps = indexed.setdefault(meta.get("source", ""), (set(), set()))
        ids.add(chunk_id)
        fps.add(meta.get("fingerprint"))

    seen_sources = set()
    for path in discover_notes(vault):
        summary.notes_seen += 1
        try:
            source, fp, ids, texts, metas = build_chunks(path, vault, chunk_size, embed_model)
        except Exception as e:  # unreadable / bad frontmatter: keep old chunks
            rel = path.relative_to(vault).as_posix()
            seen_sources.add(rel)
            summary.errors.append(f"{rel}: {e}")
            log.warning("skipped %s (kept its previous chunks): %s", rel, e)
            continue
        seen_sources.add(source)

        old_ids, old_fps = indexed.get(source, (set(), set()))
        nothing_to_do = not ids and not old_ids  # empty note, never indexed
        if nothing_to_do or (old_fps == {fp} and old_ids == set(ids)):
            summary.unchanged.append(source)
            continue

        (summary.updated if old_ids else summary.added).append(source)
        summary.chunks_written += len(ids)
        summary.chunks_removed += len(old_ids - set(ids))
        if dry_run:
            continue
        if old_ids - set(ids):
            collection.delete(ids=sorted(old_ids - set(ids)))
        if ids:
            collection.upsert(ids=ids, embeddings=embed(texts), documents=texts, metadatas=metas)

    # Sources in the DB that no longer exist on disk (deleted notes, old
    # absolute-path IDs from the previous scheme).
    for source, (old_ids, _) in sorted(indexed.items()):
        if source in seen_sources:
            continue
        summary.removed_sources.append(source)
        summary.chunks_removed += len(old_ids)
        if not dry_run:
            collection.delete(ids=sorted(old_ids))

    log.info("ingest%s: %d notes, %d added, %d updated, %d removed, %d errors in %.1fs",
             " (dry run)" if dry_run else "", summary.notes_seen, len(summary.added),
             len(summary.updated), len(summary.removed_sources), len(summary.errors),
             time.time() - start)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Index the Obsidian vault into ChromaDB.")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would change without writing to ChromaDB")
    args = parser.parse_args(argv)
    setup_logging()

    vault = settings.vault_path
    if not vault.is_dir():
        print(f"Vault not found: {vault} (set VAULT_PATH)", file=sys.stderr)
        return 2

    from store import embed, get_collection

    summary = ingest(vault, get_collection(), embed, settings.chunk_size,
                     settings.embed_model, dry_run=args.dry_run)
    print(summary.report(dry_run=args.dry_run))
    return 1 if summary.errors else 0


if __name__ == "__main__":
    sys.exit(main())
