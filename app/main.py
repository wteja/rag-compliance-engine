from app.api import create_app
from app.audit import make_session_factory
from app.config import settings
from app.lexical import BM25Index
from app.providers import build_llm, build_store
from app.rerank import CrossEncoderReranker

store = build_store(settings)
app = create_app(
    store,
    BM25Index(store),
    build_llm(settings),
    CrossEncoderReranker(settings.rerank_model),
    make_session_factory(settings.database_url),
)
