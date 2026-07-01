from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from app.vectorstore import Retrieved


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name)
        self.model = model

    def rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c.text) for c in candidates])
        scored = [replace(c, rerank_score=float(s)) for c, s in zip(candidates, scores)]
        return sorted(scored, key=lambda c: c.rerank_score, reverse=True)
