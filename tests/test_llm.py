import pytest
import requests

import llm
from llm import LLMError, OllamaProvider


class FakeResponse:
    def __init__(self, status=200, json_data=None, text=""):
        self.status_code = status
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("not json")
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


@pytest.fixture
def provider():
    return OllamaProvider(model="mistral", base_url="http://ollama.test:11434", timeout=7, keep_alive="10m",
                          temperature=0.0)


def patch_post(monkeypatch, result):
    calls = []

    def fake_post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(llm.requests, "post", fake_post)
    return calls


def test_complete_sends_chat_request_and_returns_text(monkeypatch, provider):
    calls = patch_post(monkeypatch, FakeResponse(json_data={"message": {"content": "hi"}}))

    assert provider.complete("sys", "user q") == "hi"
    call = calls[0]
    assert call["url"] == "http://ollama.test:11434/api/chat"
    assert call["timeout"] == (llm.CONNECT_TIMEOUT, 7)  # (connect, read)
    assert call["json"]["model"] == "mistral"
    assert call["json"]["stream"] is False
    assert call["json"]["keep_alive"] == "10m"
    assert call["json"]["options"] == {"temperature": 0.0}
    assert call["json"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user q"},
    ]


@pytest.mark.parametrize("error, message", [
    (requests.ReadTimeout(), "within 7s"),
    (requests.ConnectionError(), "cannot reach Ollama at http://ollama.test:11434"),
    # ConnectTimeout is also a Timeout; it must still read as "cannot reach"
    (requests.ConnectTimeout(), "cannot reach Ollama at http://ollama.test:11434"),
])
def test_network_failures_become_llm_error(monkeypatch, provider, error, message):
    patch_post(monkeypatch, error)
    with pytest.raises(LLMError, match=message):
        provider.complete("s", "u")


def test_http_error_includes_status_and_body(monkeypatch, provider):
    patch_post(monkeypatch, FakeResponse(status=404, text='{"error":"model not found"}'))
    with pytest.raises(LLMError, match="HTTP 404.*model not found"):
        provider.complete("s", "u")


@pytest.mark.parametrize("response", [
    FakeResponse(json_data=None, text="<html>"),              # not JSON
    FakeResponse(json_data={"unexpected": True}),             # missing keys
    FakeResponse(json_data={"message": {"content": None}}),   # wrong type
])
def test_malformed_reply_becomes_llm_error(monkeypatch, provider, response):
    patch_post(monkeypatch, response)
    with pytest.raises(LLMError, match="unexpected reply"):
        provider.complete("s", "u")


def test_health_reports_model_available(monkeypatch, provider):
    monkeypatch.setattr(llm.requests, "get", lambda url, timeout: FakeResponse(
        json_data={"models": [{"name": "mistral:latest"}, {"name": "llama3:8b"}]}))
    assert provider.health() == {"reachable": True, "model_available": True, "detail": "ok"}


def test_health_reports_missing_model(monkeypatch, provider):
    monkeypatch.setattr(llm.requests, "get", lambda url, timeout: FakeResponse(
        json_data={"models": [{"name": "llama3:8b"}]}))
    h = provider.health()
    assert h["reachable"] is True and h["model_available"] is False
    assert "not pulled" in h["detail"]


def test_health_never_raises_when_unreachable(monkeypatch, provider):
    def boom(url, timeout):
        raise requests.ConnectionError()
    monkeypatch.setattr(llm.requests, "get", boom)
    assert provider.health()["reachable"] is False


def test_get_provider_returns_configured_ollama():
    p = llm.get_provider()
    assert isinstance(p, OllamaProvider)
    assert p.model == llm.settings.ollama_model
