"""Unit tests for ChromaDB Vector Store & RAG Pipeline (Sections 11 & 12)."""

from unittest.mock import MagicMock
from backend.app.embeddings.chunker import TranscriptChunk
from backend.app.vector_store.chroma_service import ChromaService
from backend.app.rag.rag_pipeline import RAGPipeline, RAGResponse


def test_chroma_service_and_rag(tmp_path):
    chroma_dir = str(tmp_path / "chromadb_test")
    chroma = ChromaService(persist_dir=chroma_dir)

    # 1. Create test chunks
    chunks = [
        TranscriptChunk(
            chunk_id="chunk_001",
            meeting_id="meeting_alpha",
            meeting_title="Project Alpha Review",
            text="Meeting: Project Alpha Review\n[02:10] SPEAKER_00: We decided to deploy to AWS on Friday.",
            start_time=130.0,
            end_time=145.0,
            speakers=["SPEAKER_00"],
        ),
        TranscriptChunk(
            chunk_id="chunk_002",
            meeting_id="meeting_beta",
            meeting_title="Design Sync",
            text="Meeting: Design Sync\n[05:00] SPEAKER_01: The mobile interface design will use dark mode.",
            start_time=300.0,
            end_time=320.0,
            speakers=["SPEAKER_01"],
        ),
    ]

    # Create synthetic embeddings (e.g. 4-dim vectors for test)
    embeddings = [
        [0.8, 0.2, 0.1, 0.0],
        [0.1, 0.0, 0.9, 0.3],
    ]

    # Add chunks to ChromaDB
    chroma.add_chunks(chunks, embeddings)

    # Query ChromaDB with embedding similar to chunk_001
    query_vector = [0.75, 0.25, 0.05, 0.0]
    results = chroma.query_similar(query_vector, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_001"
    assert "Project Alpha" in results[0]["text"]
    assert results[0]["metadata"]["speakers"] == "SPEAKER_00"

    # Test RAG pipeline with mock BGE and mock Gemini
    mock_bge = MagicMock()
    mock_bge.embed_query.return_value = query_vector

    rag = RAGPipeline(
        bge_service=mock_bge,
        chroma_service=chroma,
        gemini_api_key="test_key",
    )

    # Mock Gemini generative response
    mock_genai_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "They decided to deploy Project Alpha to AWS on Friday."
    mock_genai_model.generate_content.return_value = mock_response
    rag._gemini_client = mock_genai_model

    rag_output = rag.query("What was decided about Project Alpha?", top_k=1)

    assert isinstance(rag_output, RAGResponse)
    assert "deploy Project Alpha to AWS" in rag_output.answer
    assert len(rag_output.sources) == 1
    assert rag_output.sources[0].meeting_id == "meeting_alpha"
    assert rag_output.sources[0].speaker == "SPEAKER_00"
    assert rag_output.sources[0].start_time == 130.0



def _rag_with_chunks(chunks, answer_text):
    chroma = MagicMock()
    chroma.query_similar.return_value = chunks
    rag = RAGPipeline(bge_service=MagicMock(), chroma_service=chroma, gemini_api_key="k")
    model = MagicMock()
    model.generate_content.return_value = MagicMock(text=answer_text)
    rag._gemini_client = model
    return rag, model


def _chunk(n, similarity):
    return {
        "chunk_id": f"c{n}", "text": f"[00:0{n}] SPEAKER_00: passage {n}", "similarity": similarity,
        "metadata": {"meeting_id": "m1", "meeting_title": "Sync", "speakers": "SPEAKER_00", "start_time": n, "end_time": n + 1},
    }


def test_rag_does_not_call_llm_when_nothing_is_relevant():
    rag, model = _rag_with_chunks([_chunk(1, 0.40), _chunk(2, 0.30)], "made-up answer")
    out = rag.query("What is the capital of France?")
    assert out.answer.startswith("Based on the available meeting records")
    assert out.sources == []
    model.generate_content.assert_not_called()


def test_rag_returns_only_cited_sources():
    rag, _ = _rag_with_chunks([_chunk(1, 0.70), _chunk(2, 0.66), _chunk(3, 0.20)], "SPEAKER_00 said it [2].")
    out = rag.query("What was said?")
    assert [s.start_time for s in out.sources] == [2.0]
