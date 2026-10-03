"""Search and RAG API Router for MeetWise AI.

Allows natural language query answering across historical meeting transcripts
using BGE dense embeddings and Gemini grounded generation.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..config import settings
from ..embeddings.bge_service import BGEEmbeddingService
from ..vector_store.chroma_service import ChromaService
from ..rag.rag_pipeline import RAGPipeline, RAGResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG Search"])

_rag_pipeline_instance = None


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        bge = BGEEmbeddingService(model_name=settings.BGE_MODEL)
        chroma = ChromaService(persist_dir=settings.CHROMA_PERSIST_DIR)
        _rag_pipeline_instance = RAGPipeline(
            bge_service=bge,
            chroma_service=chroma,
            gemini_api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_MODEL,
        )
    return _rag_pipeline_instance


class SearchRequest(BaseModel):
    """Natural language question payload."""
    query: str = Field(..., json_schema_extra={"example": "What action items were assigned regarding the frontend?"})
    top_k: int = Field(5, ge=1, le=20, description="Number of relevant transcript passages to retrieve")
    meeting_id: Optional[str] = Field(None, description="Optional meeting ID to narrow query scope")


@router.post("/search", response_model=RAGResponse)
def rag_search_meetings(
    request: SearchRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Execute a grounded RAG query over indexed meeting transcripts.

    - Embeds question using BAAI BGE embedding model
    - Retrieves top matching chunks from ChromaDB
    - Prompts Gemini to synthesize a strictly grounded answer with citations
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    try:
        response = rag_pipeline.query(
            question=request.query,
            top_k=request.top_k,
            meeting_id=request.meeting_id,
        )
        return response
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {e}")

