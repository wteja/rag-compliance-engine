from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


def _now():
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    tenant_id = Column(String)
    source = Column(String)
    uploaded_by = Column(String)
    groups = Column(String)
    created_at = Column(DateTime, default=_now)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=_now)
    tenant_id = Column(String)
    user_id = Column(String)
    role = Column(String)
    query = Column(Text)
    retrieved_chunks = Column(JSON)
    filtered_out_count = Column(Integer)
    prompt_sent = Column(Text)
    model = Column(String)
    model_version = Column(String)
    response = Column(Text)
    output_redactions = Column(JSON)


def make_session_factory(database_url: str) -> sessionmaker:
    kwargs: dict = {}
    if ":memory:" in database_url:
        kwargs = {"connect_args": {"check_same_thread": False}, "poolclass": StaticPool}
    engine = create_engine(database_url, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def write_audit(session, **fields) -> None:
    session.add(AuditLog(**fields))
    session.commit()


def read_audit(session, tenant, limit: int = 100):
    return (
        session.query(AuditLog)
        .filter(AuditLog.tenant_id == tenant)
        .order_by(AuditLog.ts.desc(), AuditLog.id.desc())
        .limit(limit)
        .all()
    )
