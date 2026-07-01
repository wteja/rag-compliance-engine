from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import chromadb


@dataclass
class Retrieved:
    chunk_id: str
    doc_id: str
    source: str
    page: int
    group: str
    score: float
    text: str
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class VectorStore(Protocol):
    def add(self, chunk_id: str, embedding: list[float], text: str, metadata: dict, tenant: str) -> None: ...

    def query(self, embedding: list[float], k: int, groups: list[str] | None, tenant: str) -> list[Retrieved]: ...

    def corpus(self, tenant: str) -> list[Retrieved]: ...


class ChromaStore:
    def __init__(self, path: str | None = None, client=None, prefix: str = "chunks"):
        self.client = client or chromadb.PersistentClient(path=path)
        self.prefix = prefix
        # ponytail: one Chroma collection handle per tenant, unbounded; add eviction / a
        # persistent per-tenant store (OpenSearch) at scale.
        self._cols: dict[str, object] = {}

    def _col(self, tenant: str):
        if tenant not in self._cols:
            self._cols[tenant] = self.client.get_or_create_collection(f"{self.prefix}__{tenant}")
        return self._cols[tenant]

    def add(self, chunk_id, embedding, text, metadata, tenant):
        self._col(tenant).add(ids=[chunk_id], embeddings=[embedding], documents=[text], metadatas=[metadata])

    def query(self, embedding, k, groups, tenant):
        where = {"groups": {"$in": groups}} if groups is not None else None
        res = self._col(tenant).query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        out = []
        ids = res["ids"][0]
        for i, cid in enumerate(ids):
            m = res["metadatas"][0][i]
            out.append(Retrieved(
                chunk_id=cid,
                doc_id=m["doc_id"],
                source=m["source"],
                page=m["page"],
                group=m["groups"],
                score=res["distances"][0][i],
                text=res["documents"][0][i],
            ))
        return out

    def corpus(self, tenant):
        res = self._col(tenant).get(include=["documents", "metadatas"])
        out = []
        for cid, doc, m in zip(res["ids"], res["documents"], res["metadatas"]):
            out.append(Retrieved(
                chunk_id=cid,
                doc_id=m["doc_id"],
                source=m["source"],
                page=m["page"],
                group=m["groups"],
                score=0.0,
                text=doc,
            ))
        return out
