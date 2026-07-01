from __future__ import annotations

import re
from dataclasses import replace

from rank_bm25 import BM25Okapi

from app.vectorstore import Retrieved, VectorStore


def _tokenize(text: str) -> list[str]:
    # ponytail: naive tokenizer, fine for the demo; swap for a real analyzer if recall suffers.
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Per-tenant lexical (BM25) retrieval arm — one in-memory index per tenant.

    # ponytail: rebuild-on-ingest, O(corpus) in-memory, one index per tenant; move to a
    # persistent BM25 index (OpenSearch) at scale.
    """

    def __init__(self, store: VectorStore):
        self.store = store
        self._indexes: dict[str, tuple[BM25Okapi | None, list[Retrieved]]] = {}
        self._dirty: set[str] = set()

    def mark_dirty(self, tenant: str) -> None:
        self._dirty.add(tenant)

    def _rebuild(self, tenant: str) -> None:
        corpus = self.store.corpus(tenant)
        if corpus:
            bm25 = BM25Okapi([_tokenize(c.text) for c in corpus])
            # ponytail: degenerate case — a tiny/non-discriminating corpus (e.g. a 2-doc
            # test fixture) can drive every term's IDF to 0, which would zero out BM25
            # scores entirely. Apply a small floor so ranking still differentiates by
            # term frequency instead of collapsing to ties.
            max_idf = max(bm25.idf.values()) if bm25.idf else 0.0
            if max_idf == 0:
                for word in bm25.idf:
                    bm25.idf[word] = 0.25
        else:
            bm25 = None
        self._indexes[tenant] = (bm25, corpus)
        self._dirty.discard(tenant)

    def query(self, text: str, n: int, tenant: str) -> list[Retrieved]:
        if tenant in self._dirty or tenant not in self._indexes:
            self._rebuild(tenant)
        bm25, corpus = self._indexes[tenant]
        if not bm25:
            return []
        scores = bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)
        return [replace(chunk, score=float(score)) for chunk, score in ranked[:n]]
