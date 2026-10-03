"""MeetWise AI — Backend Application Gateway.

Intelligent Meeting Intelligence and Organizational Memory Platform for Physical Meetings.
FastAPI application orchestrating audio ingestion, WhisperX, pyannote diarization,
Gemini intelligence extraction, PostgreSQL storage, BGE embeddings, ChromaDB, and RAG.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .config import settings
from .database.connection import init_db, engine
from .api.meetings import router as meetings_router
from .api.search import router as search_router
from .api.employees import router as employees_router
from .api.admin import router as admin_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("meetwise")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize database tables on startup."""
    logger.info("Starting MeetWise AI Backend...")
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization warning: {e}")
    yield
    logger.info("Shutting down MeetWise AI Backend...")


app = FastAPI(
    title="MeetWise AI",
    description=(
        "An Intelligent Meeting Intelligence and Organizational Memory Platform for Physical Meetings.\n\n"
        "Features:\n"
        "- Multi-format audio upload (.wav, .mp3, .m4a)\n"
        "- Deep denoising & audio preprocessing\n"
        "- WhisperX speech-to-text with word alignment\n"
        "- pyannote.audio multi-speaker diarization\n"
        "- Speaker + timestamp alignment\n"
        "- Google Gemini structured meeting intelligence (MoM, decisions, action items)\n"
        "- PostgreSQL relational persistence\n"
        "- Semantic transcript chunking & BAAI BGE dense embeddings\n"
        "- ChromaDB vector storage\n"
        "- Grounded RAG query answering across past meetings"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware (configured for local dev and future frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(meetings_router)
app.include_router(search_router)
app.include_router(employees_router)
app.include_router(admin_router)


@app.get("/", tags=["System"])
def root():
    """Welcome endpoint with service information."""
    return {
        "service": "MeetWise AI Backend",
        "status": "operational",
        "version": "1.0.0",
        "documentation": "/docs",
        "openapi_spec": "/openapi.json",
    }


@app.get("/health", tags=["System"])
def health_check():
    """System health check verifying database connectivity and configuration status."""
    db_status = "healthy"
    db_error = None
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        db_status = "unhealthy"
        db_error = str(e)

    chroma_status = "configured"
    try:
        from .vector_store.chroma_service import ChromaService
        cs = ChromaService(persist_dir=settings.CHROMA_PERSIST_DIR)
        cs._get_client()
        chroma_status = "healthy"
    except Exception as e:
        chroma_status = f"unhealthy ({e})"

    return {
        "status": "ok" if db_status == "healthy" else "degraded",
        "database": {
            "status": db_status,
            "dialect": settings.DATABASE_URL.split(":")[0],
            "error": db_error,
        },
        "vector_store": {
            "engine": "ChromaDB",
            "status": chroma_status,
            "persist_dir": settings.CHROMA_PERSIST_DIR,
        },
        "models": {
            "whisper_model": settings.WHISPER_MODEL,
            "whisper_device": settings.WHISPER_DEVICE,
            "bge_embedding_model": settings.BGE_MODEL,
            "denoiser_type": settings.DENOISER_TYPE,
            "gemini_configured": bool(settings.GEMINI_API_KEY),
            "hf_token_configured": bool(settings.HF_TOKEN),
        },
    }

