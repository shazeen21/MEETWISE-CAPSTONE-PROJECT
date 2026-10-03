"""FastAPI Endpoint Integration Tests (Section 13)."""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.api.meetings import get_pipeline
from backend.app.api.search import get_rag_pipeline
from backend.app.database.connection import get_db
from backend.app.rag.rag_pipeline import RAGResponse, SourceCitation


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "vector_store" in data


def test_root_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "MeetWise AI Backend"


def test_meetings_api_flow(db_session, sample_audio_wav, sample_intelligence):
    # Override get_db to use test session
    app.dependency_overrides[get_db] = lambda: db_session

    # Mock pipeline
    mock_pipe = MagicMock()
    mock_pipe.process_meeting.return_value = {
        "meeting_id": "test-meeting-123",
        "title": "API Test Meeting",
        "duration_seconds": 12.5,
        "source_type": "uploaded",
        "audio_file_name": "test.wav",
        "transcript_json_path": "data/transcripts/test-meeting-123.json",
        "intelligence": sample_intelligence.model_dump(),
        "speaker_count": 2,
        "transcript_segment_count": 3,
        "chroma_chunks_indexed": 1,
    }
    app.dependency_overrides[get_pipeline] = lambda: mock_pipe

    # Mock RAG pipeline
    mock_rag = MagicMock()
    mock_rag.query.return_value = RAGResponse(
        query="What action items?",
        answer="Action items: Finish frontend by Friday.",
        sources=[
            SourceCitation(
                meeting_id="test-meeting-123",
                meeting_title="API Test Meeting",
                speaker="SPEAKER_00",
                start_time=0.0,
                end_time=4.5,
                text_snippet="Finish frontend by Friday",
            )
        ],
    )
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag

    client = TestClient(app)

    # 1. Test upload
    with open(sample_audio_wav, "rb") as audio_f:
        upload_resp = client.post(
            "/meetings/upload",
            files={"file": ("meeting.wav", audio_f, "audio/wav")},
            data={"title": "API Test Meeting", "source_type": "uploaded"},
        )
    assert upload_resp.status_code == 201
    assert upload_resp.json()["status"] == "completed"

    # 2. Test search
    search_resp = client.post("/search", json={"query": "What action items?"})
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert "Finish frontend by Friday" in search_data["answer"]
    assert len(search_data["sources"]) == 1

    # Cleanup overrides
    app.dependency_overrides.clear()

