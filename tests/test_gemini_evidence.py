import pytest
import json
from unittest.mock import MagicMock, patch
from backend.app.meeting_intelligence.gemini_service import GeminiIntelligenceService, LLMIntelligenceError
from backend.app.meeting_intelligence.schemas import MeetingIntelligence

@pytest.fixture
def gemini_service():
    return GeminiIntelligenceService(api_key="fake-key")

def test_evidence_verification_success(gemini_service):
    # Setup: a transcript and a matching quote
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "We must finish the report by Friday."},
        {"speaker": "Bob", "start": 2.0, "end": 4.0, "text": "I agree, I will handle the charts."}
    ]
    formatted = gemini_service.format_transcript_for_llm(segments)

    # Mock a response with verbatim evidence
    mock_data = {
        "title": "Report Planning",
        "summary": "Planning the report.",
        "agenda": [],
        "discussion_points": [],
        "topics": [],
        "decisions": [
            {"decision": "Finish report by Friday", "timestamp": "00:00", "evidence": "We must finish the report by Friday.", "confidence": 1.0}
        ],
        "action_items": [
            {"task": "Handle charts", "owner": "Bob", "deadline": "Friday", "priority": "High", "status": "pending", "evidence": "I will handle the charts.", "confidence": 1.0}
        ],
        "keywords": [],
        "emotions": [
            {"speaker": "Alice", "dominant_emotion": "professional", "speaking_style": "assertive", "confidence": 1.0, "evidence": "We must finish the report by Friday."}
        ],
        "sentiment_analysis": None,
        "accents": [],
        "employee_reports": [],
        "manager_summary": None,
        "pending_issues": [],
        "next_steps": []
    }

    intelligence = MeetingIntelligence(**mock_data)

    # Run verification
    verified = gemini_service._verify_evidence(intelligence, segments)

    assert verified.decisions[0].confidence == 1.0
    assert verified.action_items[0].confidence == 1.0
    assert verified.emotions[0].confidence == 1.0

def test_evidence_verification_fabricated(gemini_service):
    # Setup: a transcript
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "The project is on track."},
    ]
    formatted = gemini_service.format_transcript_for_llm(segments)

    # Mock a response with FABRICATED evidence (not in transcript)
    mock_data = {
        "title": "Project Update",
        "summary": "Update.",
        "agenda": [],
        "discussion_points": [],
        "topics": [],
        "decisions": [
            {"decision": "Project is on track", "timestamp": "00:00", "evidence": "The project is totally awesome and on track!", "confidence": 1.0}
        ],
        "action_items": [],
        "keywords": [],
        "emotions": [],
        "sentiment_analysis": None,
        "accents": [],
        "employee_reports": [],
        "manager_summary": None,
        "pending_issues": [],
        "next_steps": []
    }

    intelligence = MeetingIntelligence(**mock_data)

    # Run verification
    verified = gemini_service._verify_evidence(intelligence, segments)

    # Fabricated evidence: the decision must be removed, not shown to users
    assert verified.decisions == []

def test_evidence_verification_missing(gemini_service):
    # Setup
    segments = [{"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "Hello."}]
    formatted = gemini_service.format_transcript_for_llm(segments)

    # Mock a response with MISSING evidence
    mock_data = {
        "title": "Greeting",
        "summary": "Greeting.",
        "agenda": [],
        "discussion_points": [],
        "topics": [],
        "decisions": [
            {"decision": "Something", "timestamp": "00:00", "evidence": None, "confidence": 1.0}
        ],
        "action_items": [],
        "keywords": [],
        "emotions": [],
        "sentiment_analysis": None,
        "accents": [],
        "employee_reports": [],
        "manager_summary": None,
        "pending_issues": [],
        "next_steps": []
    }

    intelligence = MeetingIntelligence(**mock_data)
    verified = gemini_service._verify_evidence(intelligence, segments)

    # No evidence: the decision cannot be grounded and is removed
    assert verified.decisions == []


def test_evidence_matching_ignores_case_and_punctuation(gemini_service):
    segments = [{"speaker": "Bob", "start": 0.0, "end": 2.0, "text": "Okay, I'll send the invoice tomorrow."}]
    intelligence = MeetingIntelligence(
        title="t", summary="s",
        action_items=[{"task": "Send invoice", "owner": "Bob", "evidence": "i'll send the invoice tomorrow"}],
    )
    verified = gemini_service._verify_evidence(intelligence, segments)
    assert len(verified.action_items) == 1


def test_sentiment_is_derived_from_segment_emotions(gemini_service):
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 6.0, "text": "This bug has blocked us for three weeks again."},
        {"speaker": "Bob", "start": 6.0, "end": 8.0, "text": "Great news, the client signed."},
        {"speaker": "Alice", "start": 8.0, "end": 10.0, "text": "Next item is the budget."},
    ]
    intelligence = MeetingIntelligence(
        title="t", summary="s",
        segment_emotions=[
            {"segment": 0, "emotion": "frustrated"},  # synonym -> frustration
            {"segment": 1, "emotion": "joy"},
            {"segment": 2, "emotion": "neutral"},
        ],
        sentiment_analysis={"overall_sentiment": "positive", "positive_percentage": 99.0},
    )
    result = gemini_service._derive_sentiment(intelligence, segments)
    sa = result.sentiment_analysis
    assert sa.emotions_breakdown == {"neutral": 20.0, "frustration": 60.0, "joy": 20.0}
    assert sa.negative_percentage == 60.0
    assert sa.positive_percentage == 20.0  # LLM's invented 99% is discarded
    assert sa.overall_sentiment == "negative"
    dominant = {e.speaker: e.dominant_emotion for e in result.emotions}
    assert dominant == {"Alice": "frustration", "Bob": "joy"}
