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
    rag_top_k: int
    ollama_base_url: str
    ollama_model: str
    ollama_timeout: int
    ollama_keep_alive: str
    ollama_temperature: float
    log_level: str


LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def setup_logging(level: str | None = None) -> None:
    """Log to stderr (CLI output on stdout stays clean). Call once per entry point."""
    import logging

    logging.basicConfig(
        level=level or settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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


def _get_float(env: dict, name: str, default: float, low: float, high: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
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
        rag_top_k=_get_int(env, "RAG_TOP_K", 5),
        ollama_base_url=(env.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/"),
        ollama_model=env.get("OLLAMA_MODEL") or "mistral",
        ollama_timeout=_get_int(env, "OLLAMA_TIMEOUT", 300),
        ollama_keep_alive=env.get("OLLAMA_KEEP_ALIVE") or "30m",
        ollama_temperature=_get_float(env, "OLLAMA_TEMPERATURE", 0.0, 0.0, 2.0),
        log_level=_get_log_level(env),
    )


def _get_log_level(env: dict) -> str:
    level = (env.get("LOG_LEVEL") or "INFO").upper()
    if level not in LOG_LEVELS:
        raise ValueError(f"LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}, got {level!r}")
    return level


settings = load_settings()
