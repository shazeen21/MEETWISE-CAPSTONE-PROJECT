"""Pytest configuration and shared test fixtures for MeetWise AI."""

import os
import sys
import tempfile
import wave
import struct
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database.models import Base
from backend.app.meeting_intelligence.schemas import (
    MeetingIntelligence,
    ActionItem,
    DecisionItem,
)


@pytest.fixture
def db_session():
    """In-memory SQLite database session for unit tests."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_audio_wav(tmp_path):
    """Creates a temporary 1-second 16kHz mono WAV file."""
    wav_path = tmp_path / "test_sample.wav"
    sample_rate = 16000
    n_samples = sample_rate * 1  # 1 second
    pcm = bytearray()
    for _ in range(n_samples):
        pcm.extend(struct.pack("<h", 0))

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)

    return str(wav_path)


@pytest.fixture
def sample_asr_segments():
    """Synthetic WhisperX transcript segments."""
    return [
        {
            "start": 0.0,
            "end": 4.5,
            "text": "We need to finish the frontend by Friday.",
        },
        {
            "start": 4.6,
            "end": 8.0,
            "text": "I will handle the backend API and database.",
        },
        {
            "start": 8.2,
            "end": 12.0,
            "text": "Testing needs to be completed by Tuesday.",
        },
    ]


@pytest.fixture
def sample_speaker_turns():
    """Synthetic pyannote speaker turns."""
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.5},
        {"speaker": "SPEAKER_01", "start": 4.5, "end": 8.1},
        {"speaker": "SPEAKER_00", "start": 8.1, "end": 12.0},
    ]


@pytest.fixture
def sample_intelligence():
    """Valid MeetingIntelligence structured object matching Section 15."""
    return MeetingIntelligence(
        title="Sprint Planning and API Architecture Review",
        summary="The team aligned on the engineering deliverables for the upcoming release.",
        topics=["Frontend Completion", "Backend API & Database", "Testing Schedule"],
        decisions=[
            DecisionItem(
                decision="Adopt PostgreSQL with ChromaDB for the backend architecture",
                timestamp="06:30",
            ),
        ],
        action_items=[
            ActionItem(
                task="Finish frontend development",
                owner="SPEAKER_00",
                deadline="Friday",
                status="pending",
            ),
            ActionItem(
                task="Handle backend API and database implementation",
                owner="SPEAKER_01",
                deadline="Monday",
                status="pending",
            ),
            ActionItem(
                task="Complete system and integration testing",
                owner="SPEAKER_00",
                deadline="Tuesday",
                status="pending",
            ),
        ],
        pending_issues=["Determine production server hosting budget"],
    )

