from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RCE_")

    jwt_secret: str = "dev-secret-change-me-in-production-32+"
    jwt_algorithm: str = "HS256"
    database_url: str = "sqlite:///./audit.db"
    chroma_path: str = "./chroma"
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    gen_model: str = "llama3"
    top_k: int = 4
    dense_n: int = 20
    lexical_n: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 20
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


settings = Settings()
