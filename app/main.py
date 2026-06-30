from app.api import create_app
from app.audit import make_session_factory
from app.config import settings
from app.llm import OllamaProvider
from app.vectorstore import ChromaStore

app = create_app(
    ChromaStore(path=settings.chroma_path),
    OllamaProvider(settings.ollama_url, settings.embed_model, settings.gen_model),
    make_session_factory(settings.database_url),
)
