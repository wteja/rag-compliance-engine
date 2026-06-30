import pytest
from app.auth import make_token, decode_token, Principal


def test_roundtrip_token():
    tok = make_token("alice", ["marketing"], "user")
    p = decode_token(tok)
    assert p == Principal(sub="alice", groups=["marketing"], role="user")


def test_bad_token_raises():
    with pytest.raises(Exception):
        decode_token("not-a-real-token")
