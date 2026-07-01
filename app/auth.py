from dataclasses import dataclass

import jwt

from app.config import settings


@dataclass
class Principal:
    sub: str
    groups: list[str]
    role: str
    tenant: str


def decode_token(token: str) -> Principal:
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    tenant = payload["tenant"]
    if not tenant or not tenant.strip():
        raise ValueError("empty tenant claim")
    return Principal(
        sub=payload["sub"],
        groups=payload["groups"],
        role=payload["role"],
        tenant=tenant,
    )


def make_token(sub: str, groups: list[str], role: str, tenant: str) -> str:
    return jwt.encode(
        {"sub": sub, "groups": groups, "role": role, "tenant": tenant},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
