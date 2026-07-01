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


class OpenSearchStore:
    def __init__(self, url: str, dim: int, prefix: str = "chunks", client=None):
        if client is None:
            from opensearchpy import OpenSearch
            client = OpenSearch(hosts=[url])
        self.client = client
        self.dim = dim
        self.prefix = prefix
        self._ensured: set[str] = set()

    def _index(self, tenant: str) -> str:
        return f"{self.prefix}__{tenant}"

    def _ensure_index(self, tenant: str) -> None:
        name = self._index(tenant)
        if name in self._ensured:
            return
        if not self.client.indices.exists(index=name):
            self.client.indices.create(index=name, body={
                "settings": {"index.knn": True},
                "mappings": {"properties": {
                    "embedding": {"type": "knn_vector", "dimension": self.dim},
                    "groups": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_id": {"type": "keyword"},
                    "text": {"type": "text"},
                }},
            })
        self._ensured.add(name)

    @staticmethod
    def _to_retrieved(hit, score=None) -> Retrieved:
        s = hit["_source"]
        return Retrieved(
            chunk_id=s["chunk_id"], doc_id=s["doc_id"], source=s["source"],
            page=s["page"], group=s["groups"],
            score=hit["_score"] if score is None else score,
            text=s["text"],
        )

    def add(self, chunk_id, embedding, text, metadata, tenant):
        self._ensure_index(tenant)
        self.client.index(index=self._index(tenant), id=chunk_id,
                          body={**metadata, "text": text, "embedding": embedding})

    def query(self, embedding, k, groups, tenant):
        knn = {"vector": embedding, "k": k}
        if groups is not None:
            knn["filter"] = {"terms": {"groups": groups}}
        res = self.client.search(index=self._index(tenant),
                                 body={"size": k, "query": {"knn": {"embedding": knn}}})
        return [self._to_retrieved(h) for h in res["hits"]["hits"]]

    def corpus(self, tenant):
        # ponytail: single match_all page (size 10000); use scroll/PIT for large corpora.
        res = self.client.search(index=self._index(tenant),
                                 body={"size": 10000, "query": {"match_all": {}}})
        return [self._to_retrieved(h, score=0.0) for h in res["hits"]["hits"]]
