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


def test_short_or_partial_word_quotes_are_not_evidence(gemini_service):
    segments = [{"speaker": "Alice", "start": 0.0, "end": 9.0, "text": "Okay so I think we should restart the deployment later today."}]
    intelligence = MeetingIntelligence(
        title="t", summary="s",
        decisions=[
            {"decision": "Generic", "evidence": "okay so"},          # too short to prove anything
            {"decision": "Partial word", "evidence": "start the deployment later"},  # only inside "restart"
        ],
    )
    verified = gemini_service._verify_evidence(intelligence, segments)
    assert verified.decisions == []


def test_decision_timestamp_comes_from_matched_segment(gemini_service):
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 5.0, "text": "Let's review the numbers."},
        {"speaker": "Bob", "start": 125.0, "end": 130.0, "text": "We agreed to move the launch to March."},
    ]
    intelligence = MeetingIntelligence(
        title="t", summary="s",
        decisions=[{"decision": "Launch moved", "timestamp": "00:00", "evidence": "agreed to move the launch"}],
    )
    verified = gemini_service._verify_evidence(intelligence, segments)
    assert verified.decisions[0].timestamp == "02:05"


def test_ground_details_removes_unsupported_details(gemini_service):
    segments = [
        {"speaker": "Alice", "start": 0.0, "end": 5.0, "text": "Bob, please update the pricing page by Friday."},
        {"speaker": "Bob", "start": 5.0, "end": 9.0, "text": "Sure, and I will email the vendor too."},
    ]
    intelligence = MeetingIntelligence(
        title="t", summary="s",
        decisions=[{"decision": "Pricing page update", "evidence": "please update the pricing page"}],
        action_items=[
            {"task": "Update pricing page", "owner": "Bob", "deadline": "Friday", "evidence": "please update the pricing page"},
            {"task": "Email vendor", "owner": "Charlie", "deadline": "Monday", "evidence": "I will email the vendor"},
        ],
        keywords=[{"keyword": "pricing page"}, {"keyword": "Kubernetes"}],
        topics=[{"topic": "Pricing", "start_time": -3.0, "end_time": 500.0, "participants": ["Alice", "Zed"]}],
        accents=[{"speaker": "Alice", "detected_accent": "British"}],
        employee_reports=[
            {"employee_name": "Bob", "assigned_tasks": ["Invented task"]},
            {"employee_name": "Charlie", "assigned_tasks": ["Email vendor"]},
        ],
        manager_summary={"executive_summary": "x", "key_decisions": ["Invented decision"], "assigned_tasks": ["Invented"]},
        sentiment_analysis={"key_emotional_moments": ["[#1] Bob agrees", "[#99] nonexistent", "no reference"]},
    )
    result = gemini_service._ground_details(gemini_service._verify_evidence(intelligence, segments), segments)

    owners = {a.task: (a.owner, a.deadline) for a in result.action_items}
    assert owners["Update pricing page"] == ("Bob", "Friday")
    assert owners["Email vendor"] == ("Unassigned", "Not specified")  # Charlie never spoke or was named; Monday never said
    assert [k.keyword for k in result.keywords] == ["pricing page"]
    topic = result.topics[0]
    assert (topic.start_time, topic.end_time, topic.participants) == (0.0, 9.0, ["Alice"])
    assert result.accents == []
    assert [r.employee_name for r in result.employee_reports] == ["Bob"]
    assert result.employee_reports[0].assigned_tasks == ["Update pricing page"]
    assert result.manager_summary.key_decisions == ["Pricing page update"]
    assert result.sentiment_analysis.key_emotional_moments == ["[#1] Bob agrees"]
