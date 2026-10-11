import hashlib
import shutil

import chromadb
import pytest

from ingest import discover_notes, ingest

CHUNK = 10  # small chunk size so tests can create multi-chunk notes cheaply
MODEL = "fake-model"


class FakeEmbedder:
    """Deterministic 8-dim vectors; records how many texts it embedded."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += len(texts)
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]


def words(n, word="w"):
    return " ".join(f"{word}{i}" for i in range(n))


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    (v / "projects").mkdir(parents=True)
    (v / "a.md").write_text("---\ntitle: Note A\ntags: [x]\n---\n" + words(25))  # 3 chunks
    (v / "projects" / "b.md").write_text(words(5))  # 1 chunk
    return v


@pytest.fixture
def collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    return client.get_or_create_collection("test")


def run(vault, collection, embedder=None, **kw):
    return ingest(vault, collection, embedder or FakeEmbedder(), CHUNK, MODEL, **kw)


def ids_for(collection, source):
    return sorted(collection.get(where={"source": source})["ids"])


def test_first_run_indexes_with_relative_ids_and_metadata(vault, collection):
    s = run(vault, collection)

    assert sorted(s.added) == ["a.md", "projects/b.md"]
    assert collection.count() == 4
    assert ids_for(collection, "a.md") == ["a.md#0", "a.md#1", "a.md#2"]
    meta = collection.get(ids=["projects/b.md#0"])["metadatas"][0]
    assert meta["source"] == "projects/b.md"
    assert meta["folder"] == "projects"
    assert meta["filename"] == "b.md"
    assert collection.get(ids=["a.md#0"])["metadatas"][0]["title"] == "Note A"


def test_second_run_is_a_noop(vault, collection):
    run(vault, collection)
    emb = FakeEmbedder()
    s = run(vault, collection, emb)

    assert sorted(s.unchanged) == ["a.md", "projects/b.md"]
    assert s.added == s.updated == s.removed_sources == []
    assert emb.calls == 0  # nothing re-embedded
    assert collection.count() == 4


def test_shrunk_note_leaves_no_stale_chunks(vault, collection):
    run(vault, collection)
    (vault / "a.md").write_text(words(5))  # 3 chunks -> 1

    s = run(vault, collection)

    assert s.updated == ["a.md"]
    assert ids_for(collection, "a.md") == ["a.md#0"]
    assert collection.count() == 2


def test_edited_note_is_re_embedded(vault, collection):
    run(vault, collection)
    (vault / "projects" / "b.md").write_text(words(5, word="changed"))
    emb = FakeEmbedder()

    s = run(vault, collection, emb)

    assert s.updated == ["projects/b.md"]
    assert emb.calls == 1
    assert "changed0" in collection.get(ids=["projects/b.md#0"])["documents"][0]


def test_deleted_note_is_removed(vault, collection):
    run(vault, collection)
    (vault / "a.md").unlink()

    s = run(vault, collection)

    assert s.removed_sources == ["a.md"]
    assert ids_for(collection, "a.md") == []
    assert collection.count() == 1


def test_moving_the_vault_changes_no_ids(vault, collection, tmp_path):
    run(vault, collection)
    before = sorted(collection.get()["ids"])
    moved = tmp_path / "elsewhere" / "vault"
    shutil.copytree(vault, moved)

    s = run(moved, collection)

    assert s.added == s.updated == s.removed_sources == []
    assert sorted(collection.get()["ids"]) == before


def test_legacy_absolute_path_chunks_are_removed(vault, collection):
    collection.add(ids=["5d41402abc4b2a76b9719d911017c592"], embeddings=[[0.0] * 8],
                   documents=["old"], metadatas=[{"source": "/mnt/c/old/vault/a.md"}])

    s = run(vault, collection)

    assert s.removed_sources == ["/mnt/c/old/vault/a.md"]
    assert collection.get(ids=["5d41402abc4b2a76b9719d911017c592"])["ids"] == []
    assert collection.count() == 4


def test_dry_run_writes_nothing(vault, collection):
    s = run(vault, collection, dry_run=True)

    assert sorted(s.added) == ["a.md", "projects/b.md"]
    assert s.chunks_written == 4
    assert collection.count() == 0


def test_unreadable_note_reports_error_and_keeps_old_chunks(vault, collection):
    run(vault, collection)
    (vault / "a.md").write_bytes(b"\xff\xfe not utf-8 \xff")

    s = run(vault, collection)

    assert len(s.errors) == 1 and s.errors[0].startswith("a.md:")
    assert s.removed_sources == []
    assert len(ids_for(collection, "a.md")) == 3  # last good version kept


def test_empty_note_is_not_reported_as_added_every_run(vault, collection):
    (vault / "empty.md").write_text("---\ntitle: nothing\n---\n")
    run(vault, collection)

    s = run(vault, collection)

    assert "empty.md" in s.unchanged
    assert s.added == []


def test_discover_skips_templates_dotfolders_and_escaping_symlinks(vault, tmp_path):
    for d in ["Templates", ".obsidian", ".trash"]:
        (vault / d).mkdir()
        (vault / d / "x.md").write_text("skip me")
    outside = tmp_path / "outside.md"
    outside.write_text("secret")
    (vault / "link.md").symlink_to(outside)

    found = [p.relative_to(vault).as_posix() for p in discover_notes(vault)]

    assert found == ["a.md", "projects/b.md"]


def test_errors_and_summary_are_logged(vault, collection, caplog):
    run(vault, collection)
    (vault / "a.md").write_bytes(b"\xff bad")
    with caplog.at_level("INFO", logger="ingest"):
        run(vault, collection)
    messages = [r.getMessage() for r in caplog.records]
    assert any(r.levelname == "WARNING" and "a.md" in r.getMessage() for r in caplog.records)
    assert any("1 errors" in m for m in messages)
