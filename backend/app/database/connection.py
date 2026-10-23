"""Database Connection and Session Management for MeetWise AI."""

import logging
from typing import Generator
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session

from ..config import settings
from .models import Base

logger = logging.getLogger(__name__)

# Configure connect_args based on DB dialect (SQLite vs PostgreSQL)
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _add_missing_columns():
    """Add columns that exist on the models but not in an older database.

    create_all() only creates missing tables; it never alters existing ones. When a
    model gains a column (e.g. Meeting.status), every query on that table fails with
    "no such column" until the column is added.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                col_type = column.type.compile(dialect=engine.dialect)
                ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}'
                default = column.default.arg if column.default is not None and column.default.is_scalar else None
                if default is not None:
                    literal = f"'{default}'" if isinstance(default, str) else str(default)
                    ddl += f" DEFAULT {literal}"
                    if not column.nullable:
                        ddl += " NOT NULL"
                logger.warning(f"Schema upgrade: adding missing column {table.name}.{column.name}")
                conn.execute(text(ddl))


def init_db():
    """Create all tables defined in models if they do not exist, and add new columns."""
    logger.info(f"Initializing database tables using {settings.DATABASE_URL.split('@')[-1]}...")
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    logger.info("Database schema initialized successfully.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Ensure API clients that do not enter FastAPI's lifespan context (for example,
# short-lived workers and test clients) still receive an initialized schema.
init_db()

