import httpx
import pytest

from app.llm import OllamaProvider


def test_embed_calls_ollama(monkeypatch):
    captured = {}
    url = "http://x:11434/api/embeddings"

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    p = OllamaProvider("http://x:11434", "embed-m", "gen-m")
    assert p.embed("hello") == [0.1, 0.2, 0.3]
    assert captured["url"].endswith("/api/embeddings")
    assert captured["json"]["model"] == "embed-m"


def test_generate_returns_response(monkeypatch):
    url = "http://x:11434/api/generate"

    def fake_post(url, json, timeout):
        return httpx.Response(200, json={"response": "the answer"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    p = OllamaProvider("http://x:11434", "embed-m", "gen-m")
    assert p.generate("q") == "the answer"
    assert p.model_name == "gen-m"


def test_generate_raises_on_server_error(monkeypatch):
    url = "http://x:11434/api/generate"

    def fake_post(url, json, timeout):
        return httpx.Response(503, json={}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    p = OllamaProvider("http://x:11434", "embed-m", "gen-m")
    with pytest.raises(httpx.HTTPStatusError):
        p.generate("q")
