from typing import Protocol

import httpx


class LLMProvider(Protocol):
    model_name: str

    def embed(self, text: str) -> list[float]: ...

    def generate(self, prompt: str) -> str: ...


class OllamaProvider:
    def __init__(self, url: str, embed_model: str, gen_model: str):
        self.url = url.rstrip("/")
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.model_name = gen_model

    def embed(self, text: str) -> list[float]:
        r = httpx.post(
            f"{self.url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["embedding"]

    def generate(self, prompt: str) -> str:
        r = httpx.post(
            f"{self.url}/api/generate",
            json={"model": self.gen_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"]
