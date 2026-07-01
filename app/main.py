from app.api import create_app
from app.audit import make_session_factory
from app.config import settings
from app.lexical import BM25Index
from app.llm import OllamaProvider
from app.rerank import CrossEncoderReranker
from app.vectorstore import ChromaStore

store = ChromaStore(path=settings.chroma_path)
app = create_app(
    store,
    BM25Index(store),
    OllamaProvider(settings.ollama_url, settings.embed_model, settings.gen_model),
    CrossEncoderReranker(settings.rerank_model),
    make_session_factory(settings.database_url),
)
