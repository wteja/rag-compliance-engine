from __future__ import annotations

import re
from dataclasses import replace

from rank_bm25 import BM25Okapi

from app.vectorstore import Retrieved, VectorStore


def _tokenize(text: str) -> list[str]:
    # ponytail: naive tokenizer, fine for the demo; swap for a real analyzer if recall suffers.
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Lexical (BM25) retrieval arm over the vector store's corpus.

    # ponytail: rebuild-on-ingest, O(corpus) in-memory; move to a persistent
    # BM25 index (OpenSearch) at scale.
    """

    def __init__(self, store: VectorStore):
        self.store = store
        self._bm25: BM25Okapi | None = None
        self._corpus: list[Retrieved] = []
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    def _rebuild(self) -> None:
        self._corpus = self.store.corpus()
        if self._corpus:
            self._bm25 = BM25Okapi([_tokenize(c.text) for c in self._corpus])
            # Ensure minimum IDF for small corpora where all IDF values might be 0
            min_idf = max(self._bm25.idf.values()) if self._bm25.idf else 0.0
            if min_idf == 0:
                min_idf = 0.25  # fallback minimum for degenerate cases
                for word in self._bm25.idf:
                    self._bm25.idf[word] = min_idf
        else:
            self._bm25 = None
        self._dirty = False

    def query(self, text: str, n: int) -> list[Retrieved]:
        if self._dirty:
            self._rebuild()
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(self._corpus, scores), key=lambda pair: pair[1], reverse=True)
        return [replace(chunk, score=float(score)) for chunk, score in ranked[:n]]
