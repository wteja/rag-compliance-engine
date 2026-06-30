import uuid

import chromadb

from app.vectorstore import ChromaStore


def _store():
    return ChromaStore(client=chromadb.EphemeralClient(), collection=f"test-{uuid.uuid4().hex}")


def _add(store, cid, group, vec):
    store.add(cid, vec, f"text-{cid}",
              {"doc_id": "d", "source": "s.txt", "page": 1, "groups": group, "chunk_id": cid})


def test_filtered_query_excludes_other_group():
    store = _store()
    _add(store, "fin1", "finance", [0.1, 0.2, 0.3])
    _add(store, "mkt1", "marketing", [0.1, 0.2, 0.3])

    allowed = store.query([0.1, 0.2, 0.3], k=4, groups=["marketing"])
    ids = {r.chunk_id for r in allowed}
    assert ids == {"mkt1"}
    assert allowed[0].group == "marketing"


def test_unfiltered_query_returns_all():
    store = _store()
    _add(store, "fin1", "finance", [0.1, 0.2, 0.3])
    _add(store, "mkt1", "marketing", [0.1, 0.2, 0.3])

    everything = store.query([0.1, 0.2, 0.3], k=4, groups=None)
    assert {r.chunk_id for r in everything} == {"fin1", "mkt1"}
