import uuid

import chromadb
from fastapi.testclient import TestClient

from app.api import create_app
from app.audit import make_session_factory
from app.auth import make_token
from app.vectorstore import ChromaStore


class FakeLLM:
    model_name = "fake"

    def embed(self, text):
        return [0.1, 0.2, 0.3]

    def generate(self, prompt):
        return "generated answer"


def _client(llm=None):
    store = ChromaStore(client=chromadb.EphemeralClient(), collection=f"test-{uuid.uuid4().hex}")
    Session = make_session_factory("sqlite:///:memory:")
    app = create_app(store, llm or FakeLLM(), Session)
    return TestClient(app)


def _bearer(role, groups):
    return {"Authorization": f"Bearer {make_token('u-' + role, groups, role)}"}


def _ingest(client, content, filename, groups):
    return client.post(
        "/documents",
        headers=_bearer("admin", ["finance", "marketing"]),
        files={"file": (filename, content, "text/plain")},
        data={"groups": groups},
    )


def test_access_filter_end_to_end():
    client = _client()
    assert _ingest(client, b"Finance figures Q3.", "fin.txt", ["finance"]).status_code == 200
    assert _ingest(client, b"Marketing plan.", "mkt.txt", ["marketing"]).status_code == 200

    r = client.post("/query", headers=_bearer("user", ["marketing"]), json={"query": "q"})
    assert r.status_code == 200
    sources = {c["source"] for c in r.json()["citations"]}
    assert sources == {"mkt.txt"}  # finance never reached the user

    audit = client.get("/audit", headers=_bearer("auditor", [])).json()
    assert audit[0]["filtered_out_count"] == 1  # proof the finance chunk was withheld


def test_missing_token_401():
    client = _client()
    assert client.post("/query", json={"query": "q"}).status_code == 401


def test_wrong_role_403():
    client = _client()
    r = client.post("/query", headers=_bearer("auditor", []), json={"query": "q"})
    assert r.status_code == 403


def test_unsupported_format_422():
    client = _client()
    assert _ingest(client, b"x", "data.csv", ["finance"]).status_code == 422


def test_llm_down_503():
    class DownLLM(FakeLLM):
        def generate(self, prompt):
            raise RuntimeError("down")

    client = _client(llm=DownLLM())
    _ingest(client, b"Marketing plan.", "mkt.txt", ["marketing"])
    r = client.post("/query", headers=_bearer("user", ["marketing"]), json={"query": "q"})
    assert r.status_code == 503
