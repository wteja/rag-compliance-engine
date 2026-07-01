from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from app.auth import Principal, decode_token
from app.audit import read_audit
from app.config import settings
from app.ingest import ingest
from app.retrieve import LLMUnavailable, OutputRedactionError, answer_query


class QueryBody(BaseModel):
    query: str


def _principal(authorization: str | None) -> Principal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        return decode_token(authorization.split(" ", 1)[1])
    except Exception:
        raise HTTPException(status_code=401, detail="invalid token")


def create_app(store, lexical, llm, reranker, session_factory) -> FastAPI:
    app = FastAPI(title="RAG Compliance Engine")

    def get_principal(authorization: str | None = Header(default=None)) -> Principal:
        return _principal(authorization)

    def require(role: str):
        def dep(p: Principal = Depends(get_principal)) -> Principal:
            if p.role != role:
                raise HTTPException(status_code=403, detail="forbidden")
            return p
        return dep

    def get_session():
        s = session_factory()
        try:
            yield s
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    @app.post("/documents")
    def upload(
        file: UploadFile = File(...),
        groups: list[str] = Form(...),
        principal: Principal = Depends(require("admin")),
        session=Depends(get_session),
    ):
        try:
            doc_id = ingest(file.file.read(), file.filename, groups, principal.sub, store, llm, session, lexical=lexical)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"doc_id": doc_id}

    @app.post("/query")
    def query(
        body: QueryBody,
        principal: Principal = Depends(require("user")),
        session=Depends(get_session),
    ):
        try:
            return answer_query(body.query, principal, store, lexical, llm, reranker, session, settings)
        except LLMUnavailable:
            raise HTTPException(status_code=503, detail="generation backend unavailable")
        except OutputRedactionError:
            raise HTTPException(status_code=500, detail="output redaction failed")

    @app.get("/audit")
    def audit(
        principal: Principal = Depends(require("auditor")),
        session=Depends(get_session),
    ):
        return [
            {
                "id": r.id, "ts": r.ts.isoformat() if r.ts else None,
                "user_id": r.user_id, "role": r.role, "query": r.query,
                "retrieved_chunks": r.retrieved_chunks, "filtered_out_count": r.filtered_out_count,
                "prompt_sent": r.prompt_sent, "model": r.model,
                "model_version": r.model_version, "response": r.response,
                "output_redactions": r.output_redactions,
            }
            for r in read_audit(session)
        ]

    return app
