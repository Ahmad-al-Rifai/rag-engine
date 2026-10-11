"""Central configuration, read from environment variables (and an optional .env).

Every setting has a default matching the original hard-coded values, so the app
behaves the same with no .env at all. See .env.example for the full list.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    vault_path: Path
    chroma_path: Path
    collection_name: str
    embed_model: str
    chunk_size: int
    default_top_k: int
    ollama_base_url: str
    ollama_model: str
    ollama_timeout: int


def _get_int(env: dict, name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _get_path(env: dict, name: str, default: str) -> Path:
    path = Path(env.get(name) or default).expanduser()
    # Relative paths are relative to the project, not the current directory,
    # so running a script from elsewhere can't create a second empty DB.
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def load_settings(env_file: str | Path | None = PROJECT_DIR / ".env") -> Settings:
    """Build Settings from the process environment, falling back to env_file.

    Real environment variables win over the .env file. Pass env_file=None to
    ignore any .env (used by tests).
    """
    env = {}
    if env_file is not None and Path(env_file).is_file():
        env.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    env.update(os.environ)

    return Settings(
        vault_path=_get_path(env, "VAULT_PATH", "/mnt/c/Users/user/obsidian-vault"),
        chroma_path=_get_path(env, "CHROMA_PATH", "chroma_db"),
        collection_name=env.get("COLLECTION_NAME") or "vault",
        embed_model=env.get("EMBED_MODEL") or "all-MiniLM-L6-v2",
        chunk_size=_get_int(env, "CHUNK_SIZE", 120),
        default_top_k=_get_int(env, "DEFAULT_TOP_K", 3),
        ollama_base_url=(env.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/"),
        ollama_model=env.get("OLLAMA_MODEL") or "mistral",
        ollama_timeout=_get_int(env, "OLLAMA_TIMEOUT", 120),
    )


settings = load_settings()
