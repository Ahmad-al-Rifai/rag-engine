from pathlib import Path

import pytest

from config import PROJECT_DIR, load_settings

ALL_VARS = [
    "VAULT_PATH", "CHROMA_PATH", "COLLECTION_NAME", "EMBED_MODEL", "CHUNK_SIZE",
    "DEFAULT_TOP_K", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_TIMEOUT",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Make every test start from an environment with none of our variables set."""
    for name in ALL_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_match_original_hardcoded_values():
    s = load_settings(env_file=None)
    assert s.vault_path == Path("/mnt/c/Users/user/obsidian-vault")
    assert s.chroma_path == PROJECT_DIR / "chroma_db"
    assert s.collection_name == "vault"
    assert s.embed_model == "all-MiniLM-L6-v2"
    assert s.chunk_size == 120  # changed from 250 in increment 3 (eval_retrieval.py)
    assert s.default_top_k == 3
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_model == "mistral"
    assert s.ollama_timeout == 120


def test_env_vars_override_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "vault"))
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.local:1234/")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")

    s = load_settings(env_file=None)

    assert s.vault_path == tmp_path / "vault"
    assert s.chunk_size == 100
    assert s.ollama_base_url == "http://example.local:1234"  # trailing slash stripped
    assert s.ollama_model == "llama3"


def test_relative_chroma_path_resolves_under_project(monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", "data/db")
    assert load_settings(env_file=None).chroma_path == PROJECT_DIR / "data" / "db"


@pytest.mark.parametrize("value", ["abc", "1.5"])
def test_invalid_int_names_the_variable(monkeypatch, value):
    monkeypatch.setenv("CHUNK_SIZE", value)
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        load_settings(env_file=None)


@pytest.mark.parametrize("value", ["0", "-3"])
def test_non_positive_int_is_rejected(monkeypatch, value):
    monkeypatch.setenv("DEFAULT_TOP_K", value)
    with pytest.raises(ValueError, match="DEFAULT_TOP_K"):
        load_settings(env_file=None)


def test_env_file_is_read_and_real_env_wins(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_MODEL=from-file\nCHUNK_SIZE=50\n")
    monkeypatch.setenv("CHUNK_SIZE", "75")

    s = load_settings(env_file=env_file)

    assert s.ollama_model == "from-file"
    assert s.chunk_size == 75
