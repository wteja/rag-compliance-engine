"""Seed sample users (tokens) and tagged documents. Run after the stack is up:
    python seed.py
Requires the embedding model pulled in Ollama:
    docker compose exec ollama ollama pull nomic-embed-text
    docker compose exec ollama ollama pull llama3
"""
from app.audit import make_session_factory
from app.auth import make_token
from app.config import settings
from app.ingest import ingest
from app.llm import OllamaProvider
from app.vectorstore import ChromaStore

store = ChromaStore(path=settings.chroma_path)
llm = OllamaProvider(settings.ollama_url, settings.embed_model, settings.gen_model)
session = make_session_factory(settings.database_url)()

DOCS = [
    (b"Q3 finance: revenue $4.2M, margin 38%. Owner cfo-team.", "finance-q3.txt", ["finance"]),
    (b"Marketing plan: launch campaign in August across social channels.", "marketing-plan.txt", ["marketing"]),
]
for content, name, groups in DOCS:
    doc_id = ingest(content, name, groups, "admin", "acme", store, llm, session)
    print(f"ingested {name} -> {doc_id} (groups={groups})")

print("\n--- tokens (use as 'Authorization: Bearer <token>') ---")
print("admin   :", make_token("admin", ["finance", "marketing"], "admin", "acme"))
print("alice   :", make_token("alice", ["marketing"], "user", "acme"))
print("bob     :", make_token("bob", ["finance"], "user", "acme"))
print("auditor :", make_token("auditor", [], "auditor", "acme"))
