from app.llm import BedrockProvider, OllamaProvider
from app.vectorstore import ChromaStore, OpenSearchStore


def build_llm(settings):
    if settings.backend == "local":
        return OllamaProvider(settings.ollama_url, settings.embed_model, settings.gen_model)
    if settings.backend == "aws":
        return BedrockProvider(settings.embed_model, settings.gen_model, settings.aws_region)
    raise ValueError(f"unknown RCE_BACKEND: {settings.backend!r}")


def build_store(settings):
    if settings.backend == "local":
        return ChromaStore(path=settings.chroma_path)
    if settings.backend == "aws":
        return OpenSearchStore(settings.opensearch_url, settings.embed_dim)
    raise ValueError(f"unknown RCE_BACKEND: {settings.backend!r}")
