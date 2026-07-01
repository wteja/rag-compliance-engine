from __future__ import annotations

from dataclasses import dataclass, replace

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


def _rerank_or_fallback(query, candidates, reranker):
    try:
        return reranker.rerank(query, candidates)
    except Exception:
        return sorted(candidates, key=lambda c: c.rrf_score or 0.0, reverse=True)


def retrieve(query, user_groups, store, lexical, llm, reranker, cfg) -> RetrievalResult:
    embedding = llm.embed(query)
    dense_unf = store.query(embedding, cfg.dense_n, groups=None)   # unfiltered — used ONLY for filtered_out counting
    lex_unf = lexical.query(query, cfg.lexical_n)

    # filtered_out: distinct chunks either arm surfaced (unfiltered) that the user can't see
    groups_by_id: dict[str, str] = {}
    for r in dense_unf + lex_unf:
        groups_by_id.setdefault(r.chunk_id, r.group)
    filtered_out = sum(1 for g in groups_by_id.values() if g not in user_groups)

    dense_allowed = store.query(embedding, cfg.dense_n, groups=user_groups)   # access enforced in the vector store
    lex_allowed = [r for r in lex_unf if r.group in user_groups]
    dense_rank = {r.chunk_id: i for i, r in enumerate(dense_allowed)}
    lex_rank = {r.chunk_id: i for i, r in enumerate(lex_allowed)}

    fused = rrf(
        {"dense": [r.chunk_id for r in dense_allowed],
         "lexical": [r.chunk_id for r in lex_allowed]},
        cfg.rrf_k,
    )

    merged: dict[str, Retrieved] = {}
    for r in dense_allowed + lex_allowed:
        merged.setdefault(r.chunk_id, r)  # identical text/meta across arms; keep dense copy
    candidates = [
        replace(r, dense_rank=dense_rank.get(cid), lexical_rank=lex_rank.get(cid),
                rrf_score=fused[cid])
        for cid, r in merged.items()
    ]
    candidates.sort(key=lambda c: c.rrf_score, reverse=True)
    candidates = candidates[:cfg.rerank_top_n]

    ranked = _rerank_or_fallback(query, candidates, reranker)
    top = ranked[:cfg.top_k]
    top = [replace(c, score=c.rerank_score if c.rerank_score is not None else c.rrf_score) for c in top]
    prompt = _build_prompt(query, top) if top else ""
    return RetrievalResult(chunks=top, filtered_out_count=filtered_out, prompt=prompt)


def answer_query(query, principal, store, lexical, llm, reranker, session, cfg) -> dict:
    res = retrieve(query, principal.groups, store, lexical, llm, reranker, cfg)
    citations = [
        {"source": c.source, "page": c.page, "chunk_id": c.chunk_id, "score": c.score}
        for c in res.chunks
    ]
    record = dict(
        user_id=principal.sub,
        role=principal.role,
        query=query,
        retrieved_chunks=[
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "source": c.source, "page": c.page,
             "dense_rank": c.dense_rank, "lexical_rank": c.lexical_rank,
             "rrf_score": c.rrf_score, "rerank_score": c.rerank_score,
             "score": c.score, "allowed": True}
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
