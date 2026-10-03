"""Retrieval-Augmented Generation (RAG) Pipeline for MeetWise AI.

Connects BGE query embedding, ChromaDB semantic similarity search,
context assembly, and Gemini grounded question answering with verifiable citations.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from ..embeddings.bge_service import BGEEmbeddingService
from ..vector_store.chroma_service import ChromaService
from ..meeting_intelligence.gemini_service import LLMIntelligenceError, CANDIDATE_MODELS

logger = logging.getLogger(__name__)


class SourceCitation(BaseModel):
    """Citation linking an answer back to its original meeting transcript."""
    meeting_id: str
    meeting_title: str
    speaker: str
    start_time: float
    end_time: float
    text_snippet: str


class RAGResponse(BaseModel):
    """Grounded RAG answer with source citations."""
    query: str
    answer: str
    sources: List[SourceCitation] = Field(default_factory=list)


class RAGPipeline:
    """End-to-End RAG system for organizational memory across meetings."""

    def __init__(
        self,
        bge_service: BGEEmbeddingService,
        chroma_service: ChromaService,
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        self.bge_service = bge_service
        self.chroma_service = chroma_service
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self._gemini_client = None

    def _get_gemini_client(self):
        """Lazy load Gemini generative model with fallback candidates."""
        if self._gemini_client is None:
            if not self.gemini_api_key:
                raise LLMIntelligenceError(
                    "GEMINI_API_KEY is not configured. Required for answering RAG queries."
                )
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.gemini_api_key)
                models_to_try = [self.model_name] + [m for m in CANDIDATE_MODELS if m != self.model_name]
                last_err = None

                for m in models_to_try:
                    try:
                        self._gemini_client = genai.GenerativeModel(m)
                        self.model_name = m
                        logger.info(f"RAG Gemini client initialized with {m}.")
                        break
                    except Exception as e:
                        last_err = e

                if self._gemini_client is None:
                    raise LLMIntelligenceError(f"Failed to initialize any Gemini model candidate for RAG: {last_err}")
            except Exception as e:
                raise LLMIntelligenceError(f"Failed to initialize Gemini for RAG: {e}")
        return self._gemini_client

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into mm:ss."""
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def query(
        self,
        question: str,
        top_k: int = 5,
        meeting_id: Optional[str] = None,
    ) -> RAGResponse:
        """Answer question using grounded RAG over meeting transcripts.

        Args:
            question: Natural language question from user.
            top_k: Number of transcript passages to retrieve.
            meeting_id: Optional meeting ID to restrict search.

        Returns:
            RAGResponse with grounded answer and list of source citations.
        """
        clean_question = question.strip()
        if not clean_question:
            return RAGResponse(
                query=question,
                answer="Please provide a valid non-empty question.",
                sources=[],
            )

        # Step 1: Generate BGE query embedding with retrieval instruction
        logger.info(f"Embedding search query: '{clean_question}'...")
        query_vector = self.bge_service.embed_query(clean_question)

        # Step 2: Retrieve top-k similar chunks from ChromaDB
        logger.info(f"Retrieving top {top_k} matching chunks from ChromaDB...")
        retrieved_chunks = self.chroma_service.query_similar(
            query_embedding=query_vector,
            top_k=top_k,
            meeting_id=meeting_id,
        )

        if not retrieved_chunks:
            return RAGResponse(
                query=clean_question,
                answer="I could not find any meeting records relevant to your question in the knowledge base.",
                sources=[],
            )

        # Step 3: Construct context and citations
        context_passages: List[str] = []
        citations: List[SourceCitation] = []

        for idx, item in enumerate(retrieved_chunks, 1):
            meta = item.get("metadata", {})
            m_id = meta.get("meeting_id", "UNKNOWN")
            m_title = meta.get("meeting_title", "Untitled Meeting")
            spk = meta.get("speakers", "UNKNOWN")
            start = float(meta.get("start_time", 0.0))
            end = float(meta.get("end_time", start))
            passage_text = item.get("text", "").strip()

            time_range = f"{self._format_time(start)} - {self._format_time(end)}"
            context_passages.append(
                f"--- SOURCE [{idx}] ---\n"
                f"Meeting: {m_title} (ID: {m_id})\n"
                f"Time: {time_range}\n"
                f"Speakers: {spk}\n"
                f"Transcript:\n{passage_text}\n"
            )

            citations.append(
                SourceCitation(
                    meeting_id=m_id,
                    meeting_title=m_title,
                    speaker=spk,
                    start_time=start,
                    end_time=end,
                    text_snippet=passage_text[:200] + ("..." if len(passage_text) > 200 else ""),
                )
            )

        assembled_context = "\n\n".join(context_passages)

        # Step 4: Strict grounded prompt for Gemini
        system_prompt = f"""You are the MeetWise AI Assistant, an organizational memory system for physical meetings.
Your duty is to provide an accurate, helpful, and strictly grounded answer to the user's question using ONLY the provided meeting excerpts.

CRITICAL RULES:
1. Base your answer SOLELY on the excerpts below.
2. If the excerpts do not contain enough information to answer the question, explicitly state:
   "Based on the available meeting records, this information was not found."
   DO NOT fabricate or assume any facts outside the context.
3. Attribute your statements to specific meetings, speakers, and timestamps whenever possible (e.g. "During the Project Alpha Review at 04:15, SPEAKER_01 stated...").

RETRIEVED MEETING CONTEXT:
{assembled_context}

USER QUESTION:
{clean_question}

GROUNDED ANSWER:"""

        client = self._get_gemini_client()
        logger.info("Generating grounded answer from Gemini...")

        try:
            response = client.generate_content(
                system_prompt,
                generation_config={"temperature": 0.2},
            )
            answer_text = response.text.strip()
            return RAGResponse(
                query=clean_question,
                answer=answer_text,
                sources=citations,
            )
        except Exception as e:
            logger.warning(f"RAG answer generation failed with {self.model_name}: {e}. Trying fallback candidates...")
            models_to_try = [m for m in CANDIDATE_MODELS if m != self.model_name]
            last_exception = e
            for candidate_name in models_to_try:
                try:
                    import google.generativeai as genai
                    alt_model = genai.GenerativeModel(candidate_name)
                    response = alt_model.generate_content(
                        system_prompt,
                        generation_config={"temperature": 0.2},
                    )
                    self._gemini_client = alt_model
                    self.model_name = candidate_name
                    return RAGResponse(
                        query=clean_question,
                        answer=response.text.strip(),
                        sources=citations,
                    )
                except Exception as cand_err:
                    last_exception = cand_err
            raise LLMIntelligenceError(f"Failed to generate RAG answer with Gemini: {last_exception}")
