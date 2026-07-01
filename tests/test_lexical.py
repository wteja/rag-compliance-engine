import uuid

import chromadb

from app.lexical import BM25Index
from app.vectorstore import ChromaStore


def _store():
    store = ChromaStore(client=chromadb.EphemeralClient(), collection=f"test-{uuid.uuid4().hex}")
    store.add("fin", [0.1, 0.2, 0.3], "quarterly revenue and profit margin figures",
              {"doc_id": "d1", "source": "fin.txt", "page": 1, "groups": "finance", "chunk_id": "fin"})
    store.add("mkt", [0.4, 0.5, 0.6], "social media launch campaign plan",
              {"doc_id": "d2", "source": "mkt.txt", "page": 1, "groups": "marketing", "chunk_id": "mkt"})
    return store


def test_bm25_ranks_lexical_match_first():
    idx = BM25Index(_store())
    hits = idx.query("revenue margin", n=2)
    assert hits[0].chunk_id == "fin"
    assert hits[0].score > 0
    assert hits[0].group == "finance"  # unfiltered; caller applies access control


def test_bm25_returns_empty_on_empty_corpus():
    store = ChromaStore(client=chromadb.EphemeralClient(), collection=f"test-{uuid.uuid4().hex}")
    assert BM25Index(store).query("anything", n=4) == []
