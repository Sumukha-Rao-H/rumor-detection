"""
SQLAlchemy Engine + Session Setup
==================================
Provides a configured SQLAlchemy engine and session factory.

The database URL is constructed from individual DB_* environment variables
loaded from .env.  If a full DATABASE_URL var is already set, that takes
precedence over the individual vars.

Usage:
    from database import SessionLocal, engine

    with SessionLocal() as session:
        posts = session.query(RedditPost).all()
"""

import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Load .env from project root (same file collector.py uses)
load_dotenv()


def _build_database_url() -> str:
    """
    Build a PostgreSQL connection URL.

    Priority:
      1. DATABASE_URL env var (if set and non-empty)
      2. Individual DB_* vars with sensible defaults
    """
    explicit_url = os.getenv("DATABASE_URL", "").strip()
    if explicit_url:
        return explicit_url

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "stock_rumors")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")

    return f"postgresql+psycopg2://{user}:{quote_plus(password)}@{host}:{port}/{name}"


# ---------- Engine ----------
# pool_pre_ping=True drops stale connections before re-using them
DATABASE_URL = _build_database_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)

# ---------- Session factory ----------
# expire_on_commit=False lets us read attributes after commit without
# triggering an implicit refresh (useful for returning data from inserts).
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session():
    """
    Context manager that yields a SQLAlchemy session and handles
    commit / rollback lifecycle automatically.

    Example:
        with get_session() as session:
            session.add(obj)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
