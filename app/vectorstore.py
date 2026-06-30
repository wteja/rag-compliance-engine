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


class VectorStore(Protocol):
    def add(self, chunk_id: str, embedding: list[float], text: str, metadata: dict) -> None: ...

    def query(self, embedding: list[float], k: int, groups: list[str] | None) -> list[Retrieved]: ...


class ChromaStore:
    def __init__(self, path: str | None = None, client=None, collection: str = "chunks"):
        self.client = client or chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(collection)

    def add(self, chunk_id, embedding, text, metadata):
        self.col.add(ids=[chunk_id], embeddings=[embedding], documents=[text], metadatas=[metadata])

    def query(self, embedding, k, groups=None):
        where = {"groups": {"$in": groups}} if groups is not None else None
        res = self.col.query(
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
