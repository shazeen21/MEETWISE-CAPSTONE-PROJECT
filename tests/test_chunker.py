"""Unit tests for Transcript Semantic Chunker (Section 10)."""

from backend.app.embeddings.chunker import TranscriptChunker


def test_chunker_basic():
    chunker = TranscriptChunker(target_chunk_words=10, overlap_turns=1)
    aligned_segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0, "text": "Good morning team, let us review the sprint."},
        {"speaker": "SPEAKER_01", "start": 3.5, "end": 7.0, "text": "I completed the database integration yesterday."},
        {"speaker": "SPEAKER_02", "start": 7.5, "end": 11.0, "text": "I will handle the API endpoints next."},
    ]

    chunks = chunker.chunk_transcript(
        meeting_id="m-123",
        meeting_title="Daily Standup",
        aligned_segments=aligned_segments,
    )

    assert len(chunks) >= 1
    chunk = chunks[0]
    assert chunk.meeting_id == "m-123"
    assert chunk.meeting_title == "Daily Standup"
    assert chunk.start_time == 0.0
    assert "SPEAKER_00" in chunk.speakers
    assert "Meeting: Daily Standup" in chunk.text
    assert "sprint" in chunk.text


def test_chunker_empty():
    chunker = TranscriptChunker()
    chunks = chunker.chunk_transcript("m-empty", "Empty Meeting", [])
    assert chunks == []

