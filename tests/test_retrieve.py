import uuid

import chromadb
import pytest

from app.audit import make_session_factory, read_audit
from app.auth import Principal
from app.ingest import ingest
from app.retrieve import retrieve, answer_query, ABSTAIN, LLMUnavailable
from app.vectorstore import ChromaStore


class FakeLLM:
    model_name = "fake"

    def __init__(self, fail_generate=False):
        self.fail_generate = fail_generate

    def embed(self, text):
        return [0.1, 0.2, 0.3]

    def generate(self, prompt):
        if self.fail_generate:
            raise RuntimeError("ollama down")
        return "generated answer"


def _seeded_store():
    store = ChromaStore(client=chromadb.EphemeralClient(), collection=f"test-{uuid.uuid4().hex}")
    Session = make_session_factory("sqlite:///:memory:")
    s = Session()
    ingest(b"Finance figures for Q3.", "fin.txt", ["finance"], "admin", store, FakeLLM(), s)
    ingest(b"Marketing campaign plan.", "mkt.txt", ["marketing"], "admin", store, FakeLLM(), s)
    return store, s


def test_retrieve_filters_and_counts():
    store, _ = _seeded_store()
    res = retrieve("anything", ["marketing"], store, FakeLLM(), k=4)
    assert {c.group for c in res.chunks} == {"marketing"}
    assert res.filtered_out_count == 1


def test_answer_query_abstains_when_no_access():
    store, s = _seeded_store()
    p = Principal("bob", ["hr"], "user")
    out = answer_query("anything", p, store, FakeLLM(), s, k=4)
    assert out["answer"] == ABSTAIN
    assert out["citations"] == []
    assert read_audit(s)[0].filtered_out_count == 2
    assert read_audit(s)[0].response == ABSTAIN


def test_answer_query_audits_on_llm_failure_then_raises():
    store, s = _seeded_store()
    p = Principal("alice", ["marketing"], "user")
    with pytest.raises(LLMUnavailable):
        answer_query("anything", p, store, FakeLLM(fail_generate=True), s, k=4)
    row = read_audit(s)[0]
    assert row.response is None
    assert row.filtered_out_count == 1
