import pytest
import numpy as np
from unittest.mock import MagicMock
from backend.app.speakers.voice_biometrics import VoiceBiometricsService, SpeakerRecognitionEngine, VoiceEmbeddingError

@pytest.fixture
def biometrics_service():
    return VoiceBiometricsService()

@pytest.fixture
def recognition_engine(biometrics_service):
    return SpeakerRecognitionEngine(biometrics_service=biometrics_service)

def test_identify_speaker_match(recognition_engine, biometrics_service):
    # Mock embeddings to have high similarity
    emp_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    turn_vec = np.array([0.99, 0.01, 0.0], dtype=np.float32)

    # Mock the extraction to return our controlled vector
    biometrics_service.extract_embedding = MagicMock(return_value=turn_vec)

    enrolled = [{
        "id": "emp1",
        "employee_id": "E001",
        "name": "Alice",
        "voice_embedding": emp_vec.tolist()
    }]

    result = recognition_engine.identify_speaker("dummy_audio", enrolled)
    assert result["name"] == "Alice"
    assert result["is_unknown"] is False
    assert result["confidence"] > 0.25

def test_identify_speaker_unknown(recognition_engine, biometrics_service):
    # Mock embeddings to have low similarity
    emp_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    turn_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    biometrics_service.extract_embedding = MagicMock(return_value=turn_vec)

    enrolled = [{
        "id": "emp1",
        "employee_id": "E001",
        "name": "Alice",
        "voice_embedding": emp_vec.tolist()
    }]

    result = recognition_engine.identify_speaker("dummy_audio", enrolled)
    assert result["name"] == "Unknown Speaker"
    assert result["is_unknown"] is True
    assert result["confidence"] < 0.25

def test_identify_speaker_no_enrollment(recognition_engine):
    # Pass a numpy array instead of a string to avoid file loading errors
    dummy_audio = np.zeros(16000)
    result = recognition_engine.identify_speaker(dummy_audio, [])
    assert result["name"] == "Unknown Speaker"
    assert result["is_unknown"] is True

def test_minimum_duration_validation(biometrics_service):
    # 0.5s is too short for a reliable voiceprint: must raise, never return a placeholder
    short_audio = np.zeros(8000)
    with pytest.raises(VoiceEmbeddingError):
        biometrics_service.extract_embedding(short_audio, sample_rate=16000)

def test_silence_validation(biometrics_service):
    silent_audio = np.zeros(16000)
    with pytest.raises(VoiceEmbeddingError):
        biometrics_service.extract_embedding(silent_audio, sample_rate=16000)

def test_unusable_audio_is_unknown_not_matched(recognition_engine, biometrics_service):
    # Regression: failed extraction used to return [1,0,0,...] which matched any
    # employee whose voiceprint had also fallen back, at similarity 1.0.
    biometrics_service.extract_embedding = MagicMock(side_effect=VoiceEmbeddingError("bad audio"))
    enrolled = [{"id": "emp1", "employee_id": "E001", "name": "Alice", "voice_embedding": [1.0] + [0.0] * 191}]
    result = recognition_engine.identify_speaker("dummy_audio", enrolled)
    assert result["is_unknown"] is True
    assert result["confidence"] == 0.0

def test_ambiguous_match_is_unknown(recognition_engine, biometrics_service):
    turn_vec = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    biometrics_service.extract_embedding = MagicMock(return_value=turn_vec)
    enrolled = [
        {"id": "a", "employee_id": "E1", "name": "Alice", "voice_embedding": [1.0, 0.98, 0.0]},
        {"id": "b", "employee_id": "E2", "name": "Bob", "voice_embedding": [0.98, 1.0, 0.0]},
    ]
    result = recognition_engine.identify_speaker("dummy_audio", enrolled)
    assert result["is_unknown"] is True
    assert "ambiguous" in result["reason"]


def test_pure_tone_rejected_for_enrollment(biometrics_service):
    # Regression: a 440 Hz test tone was enrolled as an employee's "voice".
    t = np.arange(48000) / 16000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    with pytest.raises(VoiceEmbeddingError, match="tone"):
        biometrics_service.check_speech_like(tone, sample_rate=16000)

def test_speech_like_audio_accepted(biometrics_service):
    # Harmonic-rich signal with a gliding pitch, like a voice, spreads its energy.
    rng = np.random.default_rng(0)
    t = np.arange(48000) / 16000
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / 16000
    voice = sum(np.sin(k * phase) / k for k in range(1, 15)) + 0.05 * rng.standard_normal(len(t))
    biometrics_service.check_speech_like(voice.astype(np.float32), sample_rate=16000)
