import uuid

import chromadb

from app.vectorstore import ChromaStore, Retrieved


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


def test_corpus_returns_all_chunks_with_text_and_group():
    store = _store()
    store.add("c1", [0.1, 0.2, 0.3], "finance text",
              {"doc_id": "d1", "source": "fin.txt", "page": 1, "groups": "finance", "chunk_id": "c1"})
    store.add("c2", [0.4, 0.5, 0.6], "marketing text",
              {"doc_id": "d2", "source": "mkt.txt", "page": 1, "groups": "marketing", "chunk_id": "c2"})

    corpus = store.corpus()

    assert {c.chunk_id for c in corpus} == {"c1", "c2"}
    by_id = {c.chunk_id: c for c in corpus}
    assert by_id["c1"].text == "finance text"
    assert by_id["c1"].group == "finance"
    assert by_id["c2"].source == "mkt.txt"


def test_retrieved_has_provenance_defaults():
    r = Retrieved(chunk_id="c", doc_id="d", source="s", page=1, group="g", score=1.0, text="t")
    assert r.dense_rank is None and r.lexical_rank is None
    assert r.rrf_score is None and r.rerank_score is None
