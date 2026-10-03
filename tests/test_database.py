"""Unit tests for PostgreSQL / SQLAlchemy Models and Repository (Sections 8 & 9)."""

import uuid
from backend.app.database.repository import MeetingRepository
from backend.app.database.models import Meeting, Speaker, Transcript, ActionItemModel, DecisionModel


def test_meeting_pipeline_persistence(db_session, sample_intelligence):
    repo = MeetingRepository(db_session)
    meeting_id = str(uuid.uuid4())

    aligned_segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.5, "text": "We need to finish the frontend by Friday."},
        {"speaker": "SPEAKER_01", "start": 4.5, "end": 8.2, "text": "I will handle the backend API."},
    ]

    meeting = repo.save_meeting_pipeline_results(
        meeting_id=meeting_id,
        title="Architecture Sync",
        audio_file_name="sync_audio.wav",
        duration_seconds=8.2,
        source_type="uploaded",
        aligned_segments=aligned_segments,
        intelligence=sample_intelligence,
    )

    # 1. Verify meeting fields
    assert meeting.id == meeting_id
    assert meeting.title == sample_intelligence.title
    assert meeting.duration_seconds == 8.2
    assert meeting.source_type == "uploaded"

    # 2. Verify speakers
    speakers = db_session.query(Speaker).filter(Speaker.meeting_id == meeting_id).all()
    assert len(speakers) == 2
    spk_labels = {s.speaker_label for s in speakers}
    assert spk_labels == {"SPEAKER_00", "SPEAKER_01"}

    # 3. Verify transcripts
    transcripts = repo.get_transcripts(meeting_id)
    assert len(transcripts) == 2
    assert transcripts[0].start_time == 0.0
    assert "frontend" in transcripts[0].text
    assert transcripts[1].start_time == 4.5
    assert "backend" in transcripts[1].text

    # 4. Verify action items
    action_items = db_session.query(ActionItemModel).filter(ActionItemModel.meeting_id == meeting_id).all()
    assert len(action_items) == 3
    tasks = [a.task for a in action_items]
    assert any("frontend" in t.lower() for t in tasks)
    assert any("backend" in t.lower() for t in tasks)
    assert any("testing" in t.lower() for t in tasks)

    # 5. Verify decisions
    decisions = db_session.query(DecisionModel).filter(DecisionModel.meeting_id == meeting_id).all()
    assert len(decisions) == 1
    assert "PostgreSQL" in decisions[0].decision


def test_cascade_delete(db_session, sample_intelligence):
    repo = MeetingRepository(db_session)
    meeting_id = str(uuid.uuid4())

    repo.save_meeting_pipeline_results(
        meeting_id=meeting_id,
        title="Temp Meeting",
        audio_file_name="temp.wav",
        duration_seconds=10.0,
        source_type="uploaded",
        aligned_segments=[{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "Sample text"}],
        intelligence=sample_intelligence,
    )

    # Ensure records exist
    assert db_session.query(Transcript).filter(Transcript.meeting_id == meeting_id).count() == 1
    assert db_session.query(ActionItemModel).filter(ActionItemModel.meeting_id == meeting_id).count() == 3

    # Delete meeting
    deleted = repo.delete_meeting(meeting_id)
    assert deleted is True

    # Check cascade
    assert db_session.query(Meeting).filter(Meeting.id == meeting_id).count() == 0
    assert db_session.query(Transcript).filter(Transcript.meeting_id == meeting_id).count() == 0
    assert db_session.query(ActionItemModel).filter(ActionItemModel.meeting_id == meeting_id).count() == 0
    assert db_session.query(Speaker).filter(Speaker.meeting_id == meeting_id).count() == 0

