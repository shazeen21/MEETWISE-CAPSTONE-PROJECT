"""Embeddings and chunking package for MeetWise AI."""

from .chunker import TranscriptChunker, TranscriptChunk
from .bge_service import BGEEmbeddingService

__all__ = ["TranscriptChunker", "TranscriptChunk", "BGEEmbeddingService"]

