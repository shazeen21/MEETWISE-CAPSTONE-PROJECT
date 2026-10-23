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



def test_background_processing_and_busy_guards(db_session, sample_audio_wav):
    from backend.app.pipeline import PIPELINE_PROGRESS
    import backend.app.api.meetings as meetings_api

    app.dependency_overrides[get_db] = lambda: db_session
    mock_pipe = MagicMock()
    app.dependency_overrides[get_pipeline] = lambda: mock_pipe
    started = []
    # Record the background job instead of running the (mocked) pipeline thread.
    original = meetings_api._run_in_background
    meetings_api._run_in_background = lambda pipe, mid, **kw: started.append(mid)
    try:
        client = TestClient(app)
        with open(sample_audio_wav, "rb") as f:
            mid = client.post(
                "/meetings/upload",
                files={"file": ("m.wav", f, "audio/wav")},
                data={"process_immediately": "false"},
            ).json()["meeting_id"]

        resp = client.post(f"/meetings/{mid}/process", params={"background": True})
        assert resp.status_code == 200 and resp.json()["status"] == "processing"
        assert started == [mid]

        # While queued/running: no second run and no deletion of the audio in use.
        PIPELINE_PROGRESS[mid] = {**PIPELINE_PROGRESS[mid], "state": "running", "stage": 2}
        assert client.post(f"/meetings/{mid}/process", params={"background": True}).status_code == 409
        assert client.delete(f"/meetings/{mid}").status_code == 409

        PIPELINE_PROGRESS[mid]["state"] = "completed"
        assert client.delete(f"/meetings/{mid}").status_code == 200
        assert client.get(f"/meetings/{mid}").status_code == 404
    finally:
        meetings_api._run_in_background = original
        app.dependency_overrides.clear()
