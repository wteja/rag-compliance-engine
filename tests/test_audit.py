from app.audit import make_session_factory, write_audit, read_audit


def test_write_and_read_audit():
    Session = make_session_factory("sqlite:///:memory:")
    s = Session()
    write_audit(
        s,
        user_id="alice", role="user", query="q",
        retrieved_chunks=[{"chunk_id": "c1", "score": 0.4}],
        filtered_out_count=1, prompt_sent="p", model="m", model_version="m", response="r",
    )
    rows = read_audit(s)
    assert len(rows) == 1
    assert rows[0].user_id == "alice"
    assert rows[0].filtered_out_count == 1
    assert rows[0].retrieved_chunks == [{"chunk_id": "c1", "score": 0.4}]
