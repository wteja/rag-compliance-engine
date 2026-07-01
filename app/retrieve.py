from __future__ import annotations

from dataclasses import dataclass

from app.audit import write_audit
from app.vectorstore import Retrieved

ABSTAIN = "No accessible information answers this."


class LLMUnavailable(Exception):
    pass


@dataclass
class RetrievalResult:
    chunks: list[Retrieved]
    filtered_out_count: int
    prompt: str


def rrf(arm_rankings: dict[str, list[str]], k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ids in arm_rankings.values():
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def _build_prompt(query: str, chunks: list[Retrieved]) -> str:
    context = "\n\n".join(f"[{c.source} p{c.page}] {c.text}" for c in chunks)
    return (
        "Answer the question using only the context below. "
        "If the context is insufficient, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )


def retrieve(query, user_groups, store, llm, k) -> RetrievalResult:
    embedding = llm.embed(query)
    allowed = store.query(embedding, k, groups=user_groups)
    unfiltered = store.query(embedding, k, groups=None)
    filtered_out = sum(1 for r in unfiltered if r.group not in user_groups)
    prompt = _build_prompt(query, allowed) if allowed else ""
    return RetrievalResult(chunks=allowed, filtered_out_count=filtered_out, prompt=prompt)


def answer_query(query, principal, store, llm, session, k) -> dict:
    res = retrieve(query, principal.groups, store, llm, k)
    citations = [
        {"source": c.source, "page": c.page, "chunk_id": c.chunk_id, "score": c.score}
        for c in res.chunks
    ]
    record = dict(
        user_id=principal.sub,
        role=principal.role,
        query=query,
        retrieved_chunks=[
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "source": c.source,
             "page": c.page, "score": c.score, "allowed": True}
            for c in res.chunks
        ],
        filtered_out_count=res.filtered_out_count,
        prompt_sent=res.prompt,
        model=llm.model_name,
        model_version=llm.model_name,  # ponytail: Ollama exposes no distinct version; tag == version for now
    )

    if not res.chunks:
        write_audit(session, **record, response=ABSTAIN)
        return {"answer": ABSTAIN, "citations": citations}

    try:
        answer = llm.generate(res.prompt)
    except Exception as e:
        write_audit(session, **record, response=None)
        raise LLMUnavailable() from e

    write_audit(session, **record, response=answer)
    return {"answer": answer, "citations": citations}
