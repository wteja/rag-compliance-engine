from app.audit import make_session_factory, write_audit, read_audit


def test_write_and_read_audit_scoped_to_tenant():
    Session = make_session_factory("sqlite:///:memory:")
    s = Session()
    write_audit(s, tenant_id="acme", user_id="alice", role="user", query="q",
                retrieved_chunks=[{"chunk_id": "c1", "score": 0.4}],
                filtered_out_count=1, prompt_sent="p", model="m", model_version="m",
                response="r", output_redactions={})
    write_audit(s, tenant_id="globex", user_id="bob", role="user", query="q2",
                retrieved_chunks=[], filtered_out_count=0, prompt_sent="p", model="m",
                model_version="m", response="r2", output_redactions=None)

    acme_rows = read_audit(s, "acme")
    assert len(acme_rows) == 1
    assert acme_rows[0].user_id == "alice"
    assert all(r.tenant_id == "acme" for r in acme_rows)  # globex row excluded
