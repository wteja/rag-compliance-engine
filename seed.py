"""Seed two tenants with overlapping groups + per-tenant tokens. Run after the stack is up:
    python seed.py
Requires the models pulled in Ollama:
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

# Same group name ("finance") in both tenants — the tenant boundary keeps them apart.
DOCS = [
    (b"Acme Q3 finance: revenue $4.2M, margin 38%.", "acme-finance.txt", ["finance"], "acme"),
    (b"Globex Q3 finance: revenue $9.9M, margin 51%.", "globex-finance.txt", ["finance"], "globex"),
]
for content, name, groups, tenant in DOCS:
    doc_id = ingest(content, name, groups, "admin", tenant, store, llm, session)
    print(f"[{tenant}] ingested {name} -> {doc_id} (groups={groups})")

print("\n--- tokens (Authorization: Bearer <token>) ---")
print("acme   user   :", make_token("alice", ["finance"], "user", "acme"))
print("acme   auditor:", make_token("auditor-a", [], "auditor", "acme"))
print("globex user   :", make_token("bob", ["finance"], "user", "globex"))
print("globex auditor:", make_token("auditor-g", [], "auditor", "globex"))
print("\nTry: query as the acme user for 'revenue' — you get Acme's $4.2M, never Globex's $9.9M.")
