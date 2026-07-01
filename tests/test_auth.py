import pytest
import jwt
from app.auth import make_token, decode_token, Principal
from app.config import settings


def test_roundtrip_token_carries_tenant():
    tok = make_token("alice", ["marketing"], "user", "acme")
    p = decode_token(tok)
    assert p == Principal(sub="alice", groups=["marketing"], role="user", tenant="acme")


def test_bad_token_raises():
    with pytest.raises(Exception):
        decode_token("not-a-real-token")


def test_missing_tenant_claim_raises():
    tok = jwt.encode({"sub": "x", "groups": [], "role": "user"},
                     settings.jwt_secret, algorithm=settings.jwt_algorithm)
    with pytest.raises(KeyError):
        decode_token(tok)
