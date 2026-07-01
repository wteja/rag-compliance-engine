import uuid
from dataclasses import replace

import chromadb
import pytest

from app.audit import make_session_factory, read_audit
from app.auth import Principal
from app.config import Settings
from app.ingest import ingest
from app.lexical import BM25Index
from app.retrieve import retrieve, answer_query, rrf, ABSTAIN, LLMUnavailable
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


class FakeReranker:
    """Deterministic: preserve fused order, copy rrf_score into rerank_score."""
    def rerank(self, query, candidates):
        return [replace(c, rerank_score=c.rrf_score or 0.0) for c in candidates]


CFG = Settings()


def _seeded():
    store = ChromaStore(client=chromadb.EphemeralClient(), collection=f"test-{uuid.uuid4().hex}")
    Session = make_session_factory("sqlite:///:memory:")
    s = Session()
    ingest(b"Quarterly revenue and profit margin for Q3.", "fin.txt", ["finance"], "admin", store, FakeLLM(), s)
    ingest(b"Marketing campaign launch plan.", "mkt.txt", ["marketing"], "admin", store, FakeLLM(), s)
    return store, BM25Index(store), s


def test_lexical_arm_honors_access_control():
    # headline: a marketing user searching a term that lexically matches the
    # finance chunk must never see it via the BM25 arm.
    store, lexical, s = _seeded()
    res = retrieve("revenue margin", ["marketing"], store, lexical, FakeLLM(), FakeReranker(), CFG)
    assert {c.group for c in res.chunks} == {"marketing"}
    assert all(c.source != "fin.txt" for c in res.chunks)
    assert res.filtered_out_count == 1  # the finance chunk existed but was withheld


def test_retrieve_populates_provenance():
    store, lexical, s = _seeded()
    res = retrieve("campaign", ["marketing"], store, lexical, FakeLLM(), FakeReranker(), CFG)
    c = res.chunks[0]
    assert c.rrf_score is not None
    assert c.rerank_score is not None
    assert c.dense_rank is not None or c.lexical_rank is not None


def test_reranker_determines_final_order():
    store, lexical, s = _seeded()

    class LenReranker:  # rank by text length desc, proving rerank controls order
        def rerank(self, query, candidates):
            return sorted((replace(c, rerank_score=float(len(c.text))) for c in candidates),
                          key=lambda c: c.rerank_score, reverse=True)

    # admin sees both groups; both chunks are candidates
    res = retrieve("plan", ["finance", "marketing"], store, lexical, FakeLLM(), LenReranker(), CFG)
    lengths = [len(c.text) for c in res.chunks]
    assert lengths == sorted(lengths, reverse=True)


def test_answer_query_abstains_when_no_access():
    store, lexical, s = _seeded()
    p = Principal("bob", ["hr"], "user")
    out = answer_query("anything", p, store, lexical, FakeLLM(), FakeReranker(), s, CFG)
    assert out["answer"] == ABSTAIN
    assert out["citations"] == []
    assert read_audit(s)[0].filtered_out_count == 2
    assert read_audit(s)[0].response == ABSTAIN


def test_answer_query_audits_on_llm_failure_then_raises():
    store, lexical, s = _seeded()
    p = Principal("alice", ["marketing"], "user")
    with pytest.raises(LLMUnavailable):
        answer_query("anything", p, store, lexical, FakeLLM(fail_generate=True), FakeReranker(), s, CFG)
    row = read_audit(s)[0]
    assert row.response is None
    assert row.filtered_out_count == 1


def test_rerank_failure_falls_back_to_fused_order():
    store, lexical, s = _seeded()

    class BrokenReranker:
        def rerank(self, query, candidates):
            raise RuntimeError("model unavailable")

    # must not raise; returns fused-ordered results
    res = retrieve("campaign", ["marketing"], store, lexical, FakeLLM(), BrokenReranker(), CFG)
    assert res.chunks  # fell back, still answered
    assert res.chunks[0].rerank_score is None  # never reranked


def test_rrf_rewards_agreement_across_arms():
    scores = rrf({"dense": ["a", "b", "c"], "lexical": ["b", "a", "d"]}, k=60)
    # 'a' and 'b' appear in both arms near the top → outrank single-arm 'c'/'d'
    assert scores["b"] > scores["c"]
    assert scores["a"] > scores["d"]
    # exact: b = 1/(60+2) + 1/(60+1)
    assert scores["b"] == 1 / 62 + 1 / 61
