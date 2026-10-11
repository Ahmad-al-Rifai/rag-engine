import chromadb
import pytest
from fastapi.testclient import TestClient

import api
from ingest import chunk_markdown
from retrieval import MAX_TOP_K, Hit, retrieve, validate


# --- chunk_markdown -------------------------------------------------------

NOTE = """# Master Rules

## Purpose
Knowledge base for AI work.

## AI Agent Rules

### Reliability
- Do not invent CVE numbers.

### Output
- Use Markdown.
"""


def test_chunks_follow_headings_with_title_trail():
    chunks = chunk_markdown(NOTE, "master-rules", 120)
    assert chunks == [
        "master-rules > Purpose\nKnowledge base for AI work.",
        "master-rules > AI Agent Rules > Reliability\n- Do not invent CVE numbers.",
        "master-rules > AI Agent Rules > Output\n- Use Markdown.",
    ]


def test_h1_equal_to_title_is_not_repeated():
    chunks = chunk_markdown("# Weekly Review\n## Inbox\n- clear it", "Weekly Review", 120)
    assert chunks == ["Weekly Review > Inbox\n- clear it"]


def test_long_section_is_split_by_words_and_keeps_header():
    body = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_markdown(f"## Big\n{body}", "T", 10)
    assert len(chunks) == 3
    assert all(c.startswith("T > Big\n") for c in chunks)


def test_note_without_headings_gets_title_only():
    assert chunk_markdown("just text", "T", 10) == ["T\njust text"]


def test_heading_only_note_produces_no_chunks():
    assert chunk_markdown("# A\n## B\n", "T", 10) == []


# --- retrieve -------------------------------------------------------------

def fake_embed(texts):
    # 2-d vectors: "rules" texts point one way, everything else the other.
    return [[1.0, 0.0] if "rules" in t else [0.0, 1.0] for t in texts]


@pytest.fixture
def collection(tmp_path):
    col = chromadb.PersistentClient(path=str(tmp_path)).get_or_create_collection("test")
    col.add(
        ids=["rules.md#0", "weekly.md#0"],
        embeddings=fake_embed(["rules", "weekly"]),
        documents=["the rules text", "weekly text"],
        metadatas=[
            {"source": "00_System/rules.md", "title": "Rules", "folder": "00_System", "chunk": 0},
            {"source": "weekly.md", "title": "Weekly", "folder": "", "chunk": 0},
        ],
    )
    return col


def test_retrieve_ranks_and_maps_metadata(collection):
    hits = retrieve("my rules", 2, collection=collection, embed=fake_embed)
    assert [h.source for h in hits] == ["00_System/rules.md", "weekly.md"]
    assert hits[0] == Hit(source="00_System/rules.md", title="Rules", folder="00_System",
                          chunk=0, distance=0.0, text="the rules text")


def test_top_k_larger_than_index_returns_what_exists(collection):
    assert len(retrieve("rules", MAX_TOP_K, collection=collection, embed=fake_embed)) == 2


def test_empty_index_returns_nothing(tmp_path):
    col = chromadb.PersistentClient(path=str(tmp_path)).get_or_create_collection("empty")
    assert retrieve("anything", 3, collection=col, embed=fake_embed) == []


@pytest.mark.parametrize("question, top_k, message", [
    ("", 3, "empty"),
    ("   ", 3, "empty"),
    ("x" * 1001, 3, "at most"),
    ("ok", 0, "top_k"),
    ("ok", MAX_TOP_K + 1, "top_k"),
])
def test_validate_rejects_bad_input(question, top_k, message):
    with pytest.raises(ValueError, match=message):
        validate(question, top_k)


# --- API ------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, collection):
    monkeypatch.setattr(api, "retrieve",
                        lambda q, k: retrieve(q, k, collection=collection, embed=fake_embed))
    monkeypatch.setattr(api, "get_collection", lambda: collection)
    return TestClient(api.app)  # not used as a context manager: skips model preload


def test_query_returns_relative_sources(client):
    r = client.post("/query", json={"question": "rules", "top_k": 1})
    assert r.status_code == 200
    result = r.json()["results"][0]
    assert result["source"] == "00_System/rules.md"
    assert not result["source"].startswith("/")
    assert result["title"] == "Rules"


@pytest.mark.parametrize("body", [
    {"question": "rules", "top_k": 0},
    {"question": "rules", "top_k": MAX_TOP_K + 1},
    {"question": ""},
    {"question": "x" * 1001},
    {},
])
def test_query_rejects_invalid_requests(client, body):
    assert client.post("/query", json=body).status_code == 422


def test_whitespace_question_is_400(client):
    r = client.post("/query", json={"question": "   "})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]


def test_health_reports_chunks_and_notes(client):
    assert client.get("/health").json() == {"status": "ok", "chunks_indexed": 2, "notes_indexed": 2}
