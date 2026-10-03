"""Integration tests for the complete MeetWise AI pipeline (Section 22)."""

from unittest.mock import MagicMock
from backend.app.pipeline import MeetWisePipeline
from backend.app.audio.preprocessor import AudioPreprocessor
from backend.app.audio.denoiser import PassthroughDenoiser
from backend.app.embeddings.chunker import TranscriptChunker
from backend.app.vector_store.chroma_service import ChromaService
from backend.app.rag.rag_pipeline import RAGPipeline


def test_complete_pipeline_flow(
    sample_audio_wav,
    sample_asr_segments,
    sample_speaker_turns,
    sample_intelligence,
    db_session,
    tmp_path,
):
    """Test the full end-to-end flow:
    Audio -> Preprocessing -> ASR -> Diarization -> Alignment -> Gemini -> DB -> Chunker -> ChromaDB.
    """
    # 1. Preprocessor with passthrough denoiser
    preprocessor = AudioPreprocessor(denoiser=PassthroughDenoiser())

    # 2. Mock WhisperX
    mock_whisperx = MagicMock()
    mock_whisperx.transcribe.return_value = {
        "language": "en",
        "segments": sample_asr_segments,
    }

    # 3. Mock pyannote
    mock_pyannote = MagicMock()
    mock_pyannote.diarize.return_value = sample_speaker_turns

    # 4. Mock Gemini
    mock_gemini = MagicMock()
    mock_gemini.analyze.return_value = sample_intelligence

    # 5. Mock BGE Embeddings
    mock_bge = MagicMock()
    # Return dummy 4-dim embedding vectors
    mock_bge.embed_documents.side_effect = lambda texts: [[0.5, 0.5, 0.5, 0.5] for _ in texts]
    mock_bge.embed_query.return_value = [0.5, 0.5, 0.5, 0.5]

    # 6. ChromaDB in temporary directory
    chroma_dir = str(tmp_path / "chroma_e2e")
    chroma = ChromaService(persist_dir=chroma_dir)

    # Assemble Pipeline
    pipeline = MeetWisePipeline(
        audio_preprocessor=preprocessor,
        whisperx_service=mock_whisperx,
        pyannote_service=mock_pyannote,
        gemini_service=mock_gemini,
        bge_service=mock_bge,
        chroma_service=chroma,
    )

    result = pipeline.process_meeting(
        audio_path=sample_audio_wav,
        db=db_session,
        title="Sprint Planning Meeting",
        source_type="uploaded",
    )

    # Assertions
    assert result["status"] if "status" in result else True
    assert result["title"] == "Sprint Planning Meeting"
    assert result["intelligence"]["title"] == sample_intelligence.title
    assert result["transcript_segment_count"] == 3
    assert result["chroma_chunks_indexed"] >= 1
    assert result["speaker_count"] == 2

    # Now verify that RAG can query the indexed meeting
    rag = RAGPipeline(
        bge_service=mock_bge,
        chroma_service=chroma,
        gemini_api_key="mock_key",
    )
    mock_genai_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "The action items assigned were: Complete frontend by Friday, handle backend API by Monday, and complete testing by Tuesday."
    mock_genai_model.generate_content.return_value = mock_response
    rag._gemini_client = mock_genai_model

    rag_ans = rag.query("What action items were assigned?")
    assert "Complete frontend by Friday" in rag_ans.answer
    assert len(rag_ans.sources) >= 1
    assert rag_ans.sources[0].meeting_title == "Sprint Planning Meeting"

