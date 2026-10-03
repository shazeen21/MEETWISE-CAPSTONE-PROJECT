"""Database Connection and Session Management for MeetWise AI."""

import logging
from typing import Generator
from sqlalchemy import create_engine
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


def init_db():
    """Create all tables defined in models if they do not exist."""
    logger.info(f"Initializing database tables using {settings.DATABASE_URL.split('@')[-1]}...")
    Base.metadata.create_all(bind=engine)
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

