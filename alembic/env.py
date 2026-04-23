"""
Alembic env.py -- Migration Environment
=========================================
Reads the database URL from .env (same vars as database.py and
reddit_data_fetcher.py) and wires Alembic to our SQLAlchemy models.
"""

import os
import sys
from logging.config import fileConfig
from urllib.parse import quote_plus

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Make the project root importable so we can access models.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Import our models -- this registers all tables with Base.metadata
from models import Base  # noqa: E402

# ---------------------------------------------------------------------------
# Alembic Config object (provides access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# ---------------------------------------------------------------------------
# Build the database URL from .env variables
# ---------------------------------------------------------------------------
def _get_url() -> str:
    """
    Same priority logic as database.py:
      1. DATABASE_URL env var (if set and non-empty)
      2. Individual DB_* vars with sensible defaults
    """
    explicit = os.getenv("DATABASE_URL", "").strip()
    if explicit:
        return explicit

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "stock_rumors")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    return f"postgresql+psycopg2://{user}:{quote_plus(password)}@{host}:{port}/{name}"


# Build the URL once at module level (do NOT pass through config.set_main_option
# because configparser's % interpolation breaks on URL-encoded passwords).
_url = _get_url()

# Target metadata for autogenerate support
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline migrations (generate SQL without connecting to the DB)
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    Emits SQL to stdout instead of executing against the database.
    """
    context.configure(
        url=_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations (connect to the DB and apply changes)
# ---------------------------------------------------------------------------
def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    Connects to the database and applies migration operations directly.
    """
    from sqlalchemy import create_engine

    connectable = create_engine(_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
