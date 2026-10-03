"""Unit tests for Transcript & Speaker Alignment (Section 6)."""

import json
from pathlib import Path
from backend.app.diarization.aligner import TranscriptAligner


def test_transcript_alignment(sample_asr_segments, sample_speaker_turns, tmp_path):
    aligner = TranscriptAligner(max_merge_gap_seconds=1.0)
    aligned = aligner.align(sample_asr_segments, sample_speaker_turns)

    assert len(aligned) == 3

    # Verify speaker assignment and text match
    assert aligned[0]["speaker"] == "SPEAKER_00"
    assert "frontend" in aligned[0]["text"]
    assert aligned[0]["start"] == 0.0
    assert aligned[0]["end"] == 4.5

    assert aligned[1]["speaker"] == "SPEAKER_01"
    assert "backend" in aligned[1]["text"]

    assert aligned[2]["speaker"] == "SPEAKER_00"
    assert "Testing" in aligned[2]["text"]

    # Verify JSON export
    json_path = tmp_path / "test_merged.json"
    saved_path = aligner.save_transcript_json(aligned, str(json_path))
    assert Path(saved_path).exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 3
    assert loaded[0]["speaker"] == "SPEAKER_00"


def test_consecutive_speaker_merging():
    """Verify consecutive turns by the same speaker within gap threshold are merged."""
    aligner = TranscriptAligner(max_merge_gap_seconds=1.5)
    raw_segments = [
        {"start": 1.0, "end": 2.0, "text": "Hello world."},
        {"start": 2.2, "end": 3.5, "text": "This is a continuation."},
    ]
    speaker_turns = [
        {"speaker": "SPEAKER_00", "start": 0.5, "end": 4.0},
    ]

    aligned = aligner.align(raw_segments, speaker_turns)
    assert len(aligned) == 1
    assert aligned[0]["speaker"] == "SPEAKER_00"
    assert aligned[0]["text"] == "Hello world. This is a continuation."
    assert aligned[0]["start"] == 1.0
    assert aligned[0]["end"] == 3.5


def test_empty_speaker_turns_fallback(sample_asr_segments):
    """Verify alignment falls back gracefully to SPEAKER_00 when no turns are detected."""
    aligner = TranscriptAligner()
    aligned = aligner.align(sample_asr_segments, [])
    assert len(aligned) == len(sample_asr_segments)
    for item in aligned:
        assert item["speaker"] == "SPEAKER_00"

