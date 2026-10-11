"""LLM providers behind one small interface.

Everything outside this file talks to `LLMProvider` (via get_provider()), so a
different backend can be added here without touching the RAG code. Every
provider failure surfaces as LLMError with a readable message.
"""
import logging
import time
from typing import Protocol

import requests

from config import settings

CONNECT_TIMEOUT = 5  # seconds to establish the connection

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The LLM backend failed: unreachable, timed out, HTTP error, or bad reply."""


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's reply to one system + user message."""

    def health(self) -> dict:
        """Return {"reachable": bool, "model_available": bool, "detail": str}. Never raises."""


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str = settings.ollama_model,
        base_url: str = settings.ollama_base_url,
        timeout: int = settings.ollama_timeout,
        keep_alive: str = settings.ollama_keep_alive,
        temperature: float = settings.ollama_temperature,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        start = time.time()
        try:
            reply = self._chat(system, user)
        except LLMError as e:
            log.warning("ollama %s failed after %.1fs: %s", self.model, time.time() - start, e)
            raise
        log.info("ollama %s answered in %.1fs (%d chars)", self.model, time.time() - start, len(reply))
        return reply

    def _chat(self, system: str, user: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {"temperature": self.temperature},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                # (connect, read): fail fast if Ollama isn't there, but give
                # a slow CPU model the full read timeout to answer.
                timeout=(CONNECT_TIMEOUT, self.timeout),
            )
        except requests.ConnectTimeout:  # subclass of Timeout: must come first
            raise LLMError(f"cannot reach Ollama at {self.base_url} (is it running?)") from None
        except requests.Timeout:
            raise LLMError(f"Ollama did not answer within {self.timeout}s "
                           f"(model {self.model!r}; raise OLLAMA_TIMEOUT if the model is slow)") from None
        except requests.ConnectionError:
            raise LLMError(f"cannot reach Ollama at {self.base_url} (is it running?)") from None

        if response.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError):
            raise LLMError(f"unexpected reply from Ollama: {response.text[:200]}") from None
        if not isinstance(content, str):
            raise LLMError("unexpected reply from Ollama: message content is not text")
        return content

    def health(self) -> dict:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=CONNECT_TIMEOUT)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except Exception as e:
            return {"reachable": False, "model_available": False,
                    "detail": f"cannot reach Ollama at {self.base_url}: {type(e).__name__}"}
        available = any(n == self.model or n.split(":")[0] == self.model for n in names)
        detail = "ok" if available else f"model {self.model!r} not pulled (have: {', '.join(names) or 'none'})"
        return {"reachable": True, "model_available": available, "detail": detail}


def get_provider() -> LLMProvider:
    """The configured provider. The one place to switch backends."""
    return OllamaProvider()
