import io
from uuid import uuid4

from pypdf import PdfReader

from app.audit import Document
from app.pii import redact

_SUPPORTED = ("pdf", "txt", "md")


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def parse(file_bytes: bytes, filename: str) -> list[tuple[int, str]]:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if ext in ("txt", "md"):
        return [(1, file_bytes.decode("utf-8"))]
    raise ValueError(f"unsupported format: .{ext}")


def ingest(file_bytes, filename, groups, uploaded_by, tenant, store, llm, session, lexical=None) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in _SUPPORTED:
        raise ValueError(f"unsupported format: .{ext}")

    doc_id = uuid4().hex
    pages = parse(file_bytes, filename)

    # Stage everything first; a redact/embed failure must leave the index untouched.
    staged = []
    for page, text in pages:
        for idx, raw in enumerate(chunk_text(text)):
            clean = redact(raw)
            emb = llm.embed(clean)
            for group in groups:
                cid = f"{doc_id}:{page}:{idx}:{group}"
                meta = {"doc_id": doc_id, "source": filename, "page": page,
                        "groups": group, "chunk_id": cid}
                staged.append((cid, emb, clean, meta))

    for cid, emb, clean, meta in staged:
        store.add(cid, emb, clean, meta, tenant)

    if lexical is not None:
        lexical.mark_dirty(tenant)

    session.add(Document(id=doc_id, source=filename, uploaded_by=uploaded_by,
                          groups=",".join(groups), tenant_id=tenant))
    session.commit()
    return doc_id
