"""Pydantic Schemas for Enterprise Meeting Intelligence and Structured Extraction.

Covers:
- Granular Action Items with Priority & Ownership
- Formal Decisions
- Topics with Time Boundaries & Participants
- Ranked & Categorized Keywords
- Emotion & Speaking Style Analysis
- Accent Classification
- Personalized Employee Reports
- 1-Page Manager Executive Summary
- Complete Minutes of Meeting (MOM) Data Contract
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class ActionItem(BaseModel):
    """Action item assigned during the meeting with priority."""
    task: str = Field(..., description="Actionable task or deliverable")
    owner: Optional[str] = Field(default="Unassigned", description="Assigned responsible person or team")
    deadline: Optional[str] = Field(default="Not specified", description="Deadline or target completion date")
    priority: str = Field(default="Medium", description="Priority: High, Medium, Low")
    status: str = Field(default="pending", description="Status: pending, in_progress, completed")


class DecisionItem(BaseModel):
    """Decision reached during the meeting."""
    decision: str = Field(..., description="Description of the agreed decision")
    timestamp: Optional[str] = Field(default=None, description="Approximate time or context when decided")


class TopicItem(BaseModel):
    """Major discussion topic with timeline and participants."""
    topic: str = Field(..., description="Topic title or theme")
    start_time: float = Field(default=0.0, description="Start time in seconds")
    end_time: float = Field(default=0.0, description="End time in seconds")
    duration_seconds: float = Field(default=0.0, description="Duration in seconds")
    participants: List[str] = Field(default_factory=list, description="Participants active in this topic")


class KeywordItem(BaseModel):
    """Extracted keyword with business category and relevance rank."""
    keyword: str = Field(..., description="Entity or keyword")
    category: str = Field(
        default="general",
        description="Category: product, client, risk, budget, hiring, revenue, deadline, deliverable, custom",
    )
    relevance_score: float = Field(default=1.0, description="Relevance score from 0.0 to 1.0")


class EmotionItem(BaseModel):
    """Per-speaker emotion and speaking style assessment."""
    speaker: str = Field(..., description="Speaker identifier or employee name")
    dominant_emotion: str = Field(
        default="neutral",
        description="Emotion: neutral, happy, excited, sad, angry, frustrated, concerned, nervous, confident, empathetic, professional",
    )
    speaking_style: str = Field(default="professional", description="Speaking style (e.g. collaborative, assertive, hesitant, formal)")
    confidence: float = Field(default=0.9, description="Confidence score 0.0 to 1.0")


class SentimentAnalysis(BaseModel):
    """Meeting-wide sentiment and emotional dynamics breakdown."""
    overall_sentiment: str = Field(default="neutral", description="positive, neutral, concerned, negative")
    positive_percentage: float = Field(default=0.0, description="Percentage of positive sentiment (0-100)")
    neutral_percentage: float = Field(default=100.0, description="Percentage of neutral sentiment (0-100)")
    negative_percentage: float = Field(default=0.0, description="Percentage of negative/concerned sentiment (0-100)")
    emotions_breakdown: Dict[str, float] = Field(default_factory=dict, description="Distribution of emotions across meeting")
    key_emotional_moments: List[str] = Field(default_factory=list, description="Notable emotional shifts or heated discussions")


class AccentItem(BaseModel):
    """Detected speaker accent."""
    speaker: str = Field(..., description="Speaker identifier or employee name")
    detected_accent: str = Field(
        default="International English",
        description="Accent: Indian, British, American, Australian, International English, or Regional",
    )
    confidence: float = Field(default=0.85, description="Accent detection confidence 0.0 to 1.0")


class EmployeeSpecificReport(BaseModel):
    """Personalized summary and obligations report tailored for an individual participant."""
    employee_name: str = Field(..., description="Employee or speaker name")
    topics_discussed: List[str] = Field(default_factory=list, description="Topics this employee engaged in")
    decisions_affecting: List[str] = Field(default_factory=list, description="Decisions impacting this employee or team")
    assigned_tasks: List[str] = Field(default_factory=list, description="Tasks assigned to or owned by this employee")
    mentioned_deadlines: List[str] = Field(default_factory=list, description="Deadlines this employee committed to")
    follow_ups: List[str] = Field(default_factory=list, description="Personal follow-up questions or items")


class ManagerExecutiveSummary(BaseModel):
    """1-Page high-impact executive brief for directors and managers."""
    executive_summary: str = Field(..., description="Concise high-level overview of the meeting")
    key_decisions: List[str] = Field(default_factory=list, description="Crucial business or technical agreements")
    risks: List[str] = Field(default_factory=list, description="Identified project, timeline, or resource risks")
    delays: List[str] = Field(default_factory=list, description="Mentioned slips or delays")
    team_blockers: List[str] = Field(default_factory=list, description="Blockers impeding delivery")
    assigned_tasks: List[str] = Field(default_factory=list, description="High-priority deliverables assigned")
    follow_up_actions: List[str] = Field(default_factory=list, description="Next steps requiring executive tracking")


class MeetingIntelligence(BaseModel):
    """Validated master structured intelligence extracted from meeting transcript."""
    title: str = Field(..., description="Concise, descriptive title for the meeting")
    summary: str = Field(..., description="Comprehensive executive summary of the meeting")
    agenda: List[str] = Field(default_factory=list, description="Inferred agenda topics")
    discussion_points: List[str] = Field(default_factory=list, description="Key discussion points and debate flow")
    topics: List[Union[TopicItem, str]] = Field(default_factory=list, description="Structured discussion topics")
    decisions: List[DecisionItem] = Field(default_factory=list, description="Formal decisions reached")
    action_items: List[ActionItem] = Field(default_factory=list, description="Action items with owner, deadline, and priority")
    keywords: List[KeywordItem] = Field(default_factory=list, description="Ranked and categorized keywords")
    emotions: List[EmotionItem] = Field(default_factory=list, description="Per-speaker emotion and style assessments")
    sentiment_analysis: Optional[SentimentAnalysis] = Field(default=None, description="Overall meeting sentiment dynamics")
    accents: List[AccentItem] = Field(default_factory=list, description="Detected speaker accents")
    employee_reports: List[EmployeeSpecificReport] = Field(default_factory=list, description="Personalized attendee reports")
    manager_summary: Optional[ManagerExecutiveSummary] = Field(default=None, description="1-Page concise manager summary")
    pending_issues: List[str] = Field(default_factory=list, description="Unresolved questions or pending issues")
    next_steps: List[str] = Field(default_factory=list, description="Next immediate organizational steps")

    @field_validator("topics", mode="before")
    @classmethod
    def normalize_topics(cls, v):
        if not v:
            return []
        normalized = []
        for item in v:
            if isinstance(item, str):
                normalized.append(TopicItem(topic=item, start_time=0.0, end_time=0.0, duration_seconds=0.0, participants=[]))
            elif isinstance(item, dict):
                normalized.append(TopicItem(**item))
            else:
                normalized.append(item)
        return normalized
