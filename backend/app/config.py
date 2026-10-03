"""Application Configuration for MeetWise AI."""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

# Load .env from backend/.env or root .env if present
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "backend" / ".env")
load_dotenv(BASE_DIR / ".env")

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")

        # Database
        DATABASE_URL: str = Field(
            default="sqlite:///./data/meetwise.db",
            description="PostgreSQL or SQLite connection string"
        )

        # External APIs & Authentication
        GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key")
        GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Google Gemini model identifier")
        HF_TOKEN: str = Field(default="", description="Hugging Face token for pyannote.audio")

        # WhisperX Configuration
        WHISPER_MODEL: str = Field(default="base", description="Whisper model size")
        WHISPER_DEVICE: Literal["cpu", "cuda"] = Field(default="cpu", description="Compute device")
        WHISPER_COMPUTE_TYPE: str = Field(default="int8", description="Compute precision")

        # Denoising
        DENOISER_TYPE: Literal["spectral", "deep", "passthrough", "auto"] = Field(
            default="spectral", description="Denoising strategy"
        )

        # Embeddings & Vector DB
        BGE_MODEL: str = Field(
            default="BAAI/bge-small-en-v1.5",
            description="HuggingFace BGE model identifier"
        )
        CHROMA_PERSIST_DIR: str = Field(
            default=str(BASE_DIR / "data" / "chromadb"),
            description="ChromaDB persistence path"
        )

        # Storage Directories
        DATA_DIR: str = Field(default=str(BASE_DIR / "data"))
        AUDIO_DIR: str = Field(default=str(BASE_DIR / "data" / "audio"))
        PROCESSED_DIR: str = Field(default=str(BASE_DIR / "data" / "processed"))
        TRANSCRIPTS_DIR: str = Field(default=str(BASE_DIR / "data" / "transcripts"))

        # Server
        HOST: str = Field(default="0.0.0.0")
        PORT: int = Field(default=8000)

except ImportError:
    # Fallback if pydantic-settings is not yet installed
    class Settings:  # type: ignore
        def __init__(self):
            self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/meetwise.db")
            self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
            self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.HF_TOKEN = os.getenv("HF_TOKEN", "")
            self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
            self.WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
            self.WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
            self.DENOISER_TYPE = os.getenv("DENOISER_TYPE", "spectral")
            self.BGE_MODEL = os.getenv("BGE_MODEL", "BAAI/bge-small-en-v1.5")
            self.CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chromadb"))
            self.DATA_DIR = str(BASE_DIR / "data")
            self.AUDIO_DIR = str(BASE_DIR / "data" / "audio")
            self.PROCESSED_DIR = str(BASE_DIR / "data" / "processed")
            self.TRANSCRIPTS_DIR = str(BASE_DIR / "data" / "transcripts")
            self.HOST = os.getenv("HOST", "0.0.0.0")
            self.PORT = int(os.getenv("PORT", "8000"))

# Singleton instance
settings = Settings()

# Ensure directories exist
for directory in [
    settings.DATA_DIR,
    settings.AUDIO_DIR,
    settings.PROCESSED_DIR,
    settings.TRANSCRIPTS_DIR,
    settings.CHROMA_PERSIST_DIR,
]:
    Path(directory).mkdir(parents=True, exist_ok=True)
