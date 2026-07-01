import uuid

import chromadb

from app.lexical import BM25Index
from app.vectorstore import ChromaStore


def _store():
    store = ChromaStore(client=chromadb.EphemeralClient(), prefix=f"test-{uuid.uuid4().hex}")
    store.add("fin", [0.1, 0.2, 0.3], "quarterly revenue and profit margin figures",
              {"doc_id": "d1", "source": "fin.txt", "page": 1, "groups": "finance", "chunk_id": "fin"}, "acme")
    store.add("mkt", [0.4, 0.5, 0.6], "social media launch campaign plan",
              {"doc_id": "d2", "source": "mkt.txt", "page": 1, "groups": "marketing", "chunk_id": "mkt"}, "acme")
    return store


def test_bm25_ranks_lexical_match_first():
    idx = BM25Index(_store())
    hits = idx.query("revenue margin", n=2, tenant="acme")
    assert hits[0].chunk_id == "fin"
    assert hits[0].score > 0
    assert hits[0].group == "finance"


def test_bm25_returns_empty_on_empty_corpus():
    store = ChromaStore(client=chromadb.EphemeralClient(), prefix=f"test-{uuid.uuid4().hex}")
    assert BM25Index(store).query("anything", n=4, tenant="acme") == []


def test_bm25_isolates_by_tenant():
    store = ChromaStore(client=chromadb.EphemeralClient(), prefix=f"test-{uuid.uuid4().hex}")
    store.add("a", [0.1, 0.2, 0.3], "revenue margin figures",
              {"doc_id": "d", "source": "a.txt", "page": 1, "groups": "finance", "chunk_id": "a"}, "acme")
    store.add("b", [0.1, 0.2, 0.3], "revenue margin figures",
              {"doc_id": "d", "source": "b.txt", "page": 1, "groups": "finance", "chunk_id": "b"}, "globex")
    idx = BM25Index(store)
    hits = idx.query("revenue margin", n=5, tenant="acme")
    assert {h.chunk_id for h in hits} == {"a"}  # globex's index is separate
