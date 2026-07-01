from app.vectorstore import OpenSearchStore, Retrieved


class FakeIndices:
    def __init__(self): self.created = []
    def exists(self, index): return False
    def create(self, index, body): self.created.append((index, body))


class FakeOSClient:
    def __init__(self, hits=None):
        self.indices = FakeIndices()
        self.indexed = []
        self.searches = []
        self._hits = hits or []

    def index(self, index, id, body):
        self.indexed.append((index, id, body))

    def search(self, index, body):
        self.searches.append((index, body))
        return {"hits": {"hits": self._hits}}


def _hit(cid, group, score=1.0):
    return {"_score": score, "_source": {
        "chunk_id": cid, "doc_id": "d", "source": "s.txt",
        "page": 1, "groups": group, "text": f"text-{cid}"}}


def test_add_creates_tenant_index_and_indexes_doc():
    c = FakeOSClient()
    store = OpenSearchStore(url="x", dim=3, client=c)
    store.add("c1", [0.1, 0.2, 0.3], "hello",
              {"doc_id": "d", "source": "s.txt", "page": 1, "groups": "finance", "chunk_id": "c1"},
              "acme")
    idx, cid, body = c.indexed[0]
    assert idx == "chunks__acme"
    assert cid == "c1"
    assert body["embedding"] == [0.1, 0.2, 0.3]
    assert body["groups"] == "finance"
    ci_index, ci_body = c.indices.created[0]
    assert ci_index == "chunks__acme"
    assert ci_body["mappings"]["properties"]["embedding"] == {"type": "knn_vector", "dimension": 3}


def test_query_builds_knn_with_group_filter_for_tenant_index():
    c = FakeOSClient(hits=[_hit("c1", "marketing")])
    store = OpenSearchStore(url="x", dim=3, client=c)
    out = store.query([0.1, 0.2, 0.3], k=4, groups=["marketing"], tenant="acme")
    idx, body = c.searches[0]
    assert idx == "chunks__acme"
    assert body["size"] == 4
    knn = body["query"]["knn"]["embedding"]
    assert knn["vector"] == [0.1, 0.2, 0.3]
    assert knn["k"] == 4
    assert knn["filter"] == {"terms": {"groups": ["marketing"]}}
    assert out[0].chunk_id == "c1" and out[0].group == "marketing"


def test_query_without_groups_omits_filter():
    c = FakeOSClient(hits=[])
    store = OpenSearchStore(url="x", dim=3, client=c)
    store.query([0.1, 0.2, 0.3], k=4, groups=None, tenant="acme")
    _, body = c.searches[0]
    assert "filter" not in body["query"]["knn"]["embedding"]


def test_corpus_scans_tenant_index():
    c = FakeOSClient(hits=[_hit("c1", "finance"), _hit("c2", "finance")])
    store = OpenSearchStore(url="x", dim=3, client=c)
    corpus = store.corpus("acme")
    idx, body = c.searches[0]
    assert idx == "chunks__acme"
    assert body["query"] == {"match_all": {}}
    assert {r.chunk_id for r in corpus} == {"c1", "c2"}
    assert all(isinstance(r, Retrieved) for r in corpus)
