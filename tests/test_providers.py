import pytest

from app.config import Settings
from app.llm import BedrockProvider, OllamaProvider
from app.providers import build_llm, build_store
from app.vectorstore import ChromaStore, OpenSearchStore


def test_local_backend_builds_ollama_and_chroma():
    s = Settings(backend="local")
    assert isinstance(build_llm(s), OllamaProvider)
    assert isinstance(build_store(s), ChromaStore)


def test_aws_backend_builds_bedrock_and_opensearch():
    s = Settings(backend="aws")
    assert isinstance(build_llm(s), BedrockProvider)
    assert isinstance(build_store(s), OpenSearchStore)


def test_unknown_backend_raises():
    s = Settings(backend="azure")
    with pytest.raises(ValueError):
        build_llm(s)
    with pytest.raises(ValueError):
        build_store(s)
