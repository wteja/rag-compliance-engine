from dataclasses import dataclass

import jwt

from app.config import settings


@dataclass
class Principal:
    sub: str
    groups: list[str]
    role: str


def decode_token(token: str) -> Principal:
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    return Principal(sub=payload["sub"], groups=payload["groups"], role=payload["role"])


def make_token(sub: str, groups: list[str], role: str) -> str:
    return jwt.encode(
        {"sub": sub, "groups": groups, "role": role},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
