import uuid

import chromadb
import pytest

from app.audit import make_session_factory
from app.ingest import chunk_text, redact, ingest
from app.vectorstore import ChromaStore


class FakeLLM:
    model_name = "fake"

    def embed(self, text):
        return [0.1, 0.2, 0.3]

    def generate(self, prompt):
        return "answer"


def test_chunk_text_overlaps():
    chunks = chunk_text("x" * 1200, size=500, overlap=50)
    assert len(chunks) == 3
    assert chunks[0][-50:] == chunks[1][:50]


def test_redact_masks_email_and_phone():
    out = redact("Contact me at jane@acme.com or 415-555-0199.")
    assert "jane@acme.com" not in out
    assert "415-555-0199" not in out


def test_ingest_stores_one_copy_per_group_redacted():
    store = ChromaStore(client=chromadb.EphemeralClient(), prefix=f"test-{uuid.uuid4().hex}")
    Session = make_session_factory("sqlite:///:memory:")
    s = Session()
    doc_id = ingest(b"Email jane@acme.com about Q3.", "memo.txt",
                    ["finance", "legal"], "admin", "acme", store, FakeLLM(), s)

    fin = store.query([0.1, 0.2, 0.3], k=10, groups=["finance"], tenant="acme")
    legal = store.query([0.1, 0.2, 0.3], k=10, groups=["legal"], tenant="acme")
    assert len(fin) == 1 and len(legal) == 1
    assert fin[0].doc_id == doc_id
    assert "jane@acme.com" not in fin[0].text


def test_parse_rejects_unsupported_format():
    with pytest.raises(ValueError):
        ingest(b"x", "data.csv", ["finance"], "admin", "acme", None, FakeLLM(), None)
