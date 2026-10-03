"""SQLAlchemy ORM Models for MeetWise AI Enterprise Platform.

Defines schemas for:
- Meetings, Media, Manager Summaries
- Employees, Voice Biometrics, Voice Samples
- Speakers with Recognition Confidence & Employee Mapping
- Transcripts with Emotion & Speaking Style
- Action Items with Priority & Ownership
- Decisions, Topics, Keywords, Configured Keywords
- Employee-Specific Personalized Reports
- Notifications & Audit Logs
"""

import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    Integer,
    DateTime,
    ForeignKey,
    Index,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a unique string UUID."""
    return str(uuid.uuid4())


class Employee(Base):
    """Registered employee profile with voice biometrics and metadata."""

    __tablename__ = "employees"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    employee_id = Column(String(64), unique=True, nullable=False)  # e.g. EMP-101
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    department = Column(String(100), default="General", nullable=False)
    team = Column(String(100), default="Core", nullable=False)
    designation = Column(String(100), default="Team Member", nullable=False)
    voice_embedding = Column(Text, nullable=True)  # JSON-encoded 192-d centroid vector
    accent_metadata = Column(Text, nullable=True)  # JSON-encoded accent details
    voice_samples_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    voice_samples = relationship("VoiceSample", back_populates="employee", cascade="all, delete-orphan")
    employee_reports = relationship("EmployeeReport", back_populates="employee", cascade="all, delete-orphan")
    speakers = relationship("Speaker", back_populates="employee")

    __table_args__ = (
        Index("idx_employees_emp_id", "employee_id"),
        Index("idx_employees_email", "email"),
        Index("idx_employees_dept", "department"),
    )


class VoiceSample(Base):
    """Enrolled speech audio sample for employee voice biometrics."""

    __tablename__ = "voice_samples"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    employee_id = Column(String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    audio_file_name = Column(String(255), nullable=False)
    duration_seconds = Column(Float, default=0.0, nullable=False)
    prompt_text = Column(Text, nullable=True)
    embedding_vector = Column(Text, nullable=True)  # JSON-encoded vector
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    employee = relationship("Employee", back_populates="voice_samples")

    __table_args__ = (
        Index("idx_voice_samples_emp_id", "employee_id"),
    )


class Meeting(Base):
    """Meeting entity representing an ingested physical, virtual, or recorded session."""

    __tablename__ = "meetings"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    title = Column(String(255), nullable=False)
    meeting_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    duration_seconds = Column(Float, default=0.0, nullable=False)
    audio_file_name = Column(String(255), nullable=False)
    media_type = Column(String(20), default="audio", nullable=False)  # audio or video
    source_type = Column(String(50), default="uploaded", nullable=False)  # uploaded or live
    summary = Column(Text, nullable=True)
    manager_summary = Column(Text, nullable=True)  # JSON-encoded 1-page executive brief
    overall_sentiment = Column(Text, nullable=True)  # JSON-encoded sentiment analytics
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships with CASCADE delete
    speakers = relationship("Speaker", back_populates="meeting", cascade="all, delete-orphan")
    transcripts = relationship("Transcript", back_populates="meeting", cascade="all, delete-orphan")
    action_items = relationship("ActionItemModel", back_populates="meeting", cascade="all, delete-orphan")
    decisions = relationship("DecisionModel", back_populates="meeting", cascade="all, delete-orphan")
    topics = relationship("Topic", back_populates="meeting", cascade="all, delete-orphan")
    keywords = relationship("Keyword", back_populates="meeting", cascade="all, delete-orphan")
    employee_reports = relationship("EmployeeReport", back_populates="meeting", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="meeting", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("source_type IN ('uploaded', 'live')", name="chk_source_type"),
        Index("idx_meetings_date", "meeting_date"),
        Index("idx_meetings_source", "source_type"),
    )


class Speaker(Base):
    """Speaker identified through pyannote diarization & biometric recognition."""

    __tablename__ = "speakers"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    meeting_id = Column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    employee_id = Column(String(36), ForeignKey("employees.id", ondelete="SET NULL"), nullable=True)
    speaker_label = Column(String(64), nullable=False)  # e.g., SPEAKER_00
    speaker_name = Column(String(255), nullable=True)   # e.g. "John Doe" or "Unknown Speaker"
    confidence_score = Column(Float, default=1.0, nullable=False)  # e.g. 0.97
    is_unknown = Column(Integer, default=0, nullable=False)
    detected_accent = Column(String(100), nullable=True)  # e.g. Indian, American, British
    dominant_emotion = Column(String(100), nullable=True)  # e.g. Confident, Empathetic
    total_speaking_time = Column(Float, default=0.0, nullable=False)

    # Relationships
    meeting = relationship("Meeting", back_populates="speakers")
    employee = relationship("Employee", back_populates="speakers")
    transcripts = relationship("Transcript", back_populates="speaker")

    __table_args__ = (
        UniqueConstraint("meeting_id", "speaker_label", name="uq_meeting_speaker_label"),
        Index("idx_speakers_meeting_id", "meeting_id"),
        Index("idx_speakers_emp_id", "employee_id"),
        Index("idx_speakers_label", "speaker_label"),
    )


class Transcript(Base):
    """Timestamped speech transcript segments attributed to speakers with emotions."""

    __tablename__ = "transcripts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    meeting_id = Column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    speaker_id = Column(String(36), ForeignKey("speakers.id", ondelete="SET NULL"), nullable=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    emotion = Column(String(100), default="neutral", nullable=False)
    speaking_style = Column(String(100), default="professional", nullable=False)
    segment_index = Column(Integer, default=0, nullable=False)

    # Relationships
    meeting = relationship("Meeting", back_populates="transcripts")
    speaker = relationship("Speaker", back_populates="transcripts")

    __table_args__ = (
        CheckConstraint("end_time >= start_time", name="chk_transcript_time"),
        Index("idx_transcripts_meeting_id", "meeting_id"),
        Index("idx_transcripts_speaker_id", "speaker_id"),
        Index("idx_transcripts_start_time", "meeting_id", "start_time"),
    )


class ActionItemModel(Base):
    """Action item extracted with ownership, deadline, priority, and status."""

    __tablename__ = "action_items"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    meeting_id = Column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    task = Column(Text, nullable=False)
    owner = Column(String(255), nullable=True, default="Unassigned")
    deadline = Column(String(100), nullable=True, default="Not specified")
    priority = Column(String(20), default="Medium", nullable=False)  # High, Medium, Low
    status = Column(String(50), default="pending", nullable=False)   # pending, in_progress, completed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    meeting = relationship("Meeting", back_populates="action_items")

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'in_progress', 'completed')", name="chk_action_status"),
        Index("idx_action_items_meeting_id", "meeting_id"),
        Index("idx_action_items_owner", "owner"),
        Index("idx_action_items_status", "status"),
        Index("idx_action_items_priority", "priority"),
    )


class DecisionModel(Base):
    """Formal decision reached in the meeting."""

    __tablename__ = "decisions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    meeting_id = Column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    decision = Column(Text, nullable=False)
    timestamp = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    meeting = relationship("Meeting", back_populates="decisions")

    __table_args__ = (
        Index("idx_decisions_meeting_id", "meeting_id"),
    )


class Topic(Base):
    """Major discussion topic with time boundaries and participants."""

    __tablename__ = "topics"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    meeting_id = Column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    topic_name = Column(String(255), nullable=False)
    start_time = Column(Float, default=0.0, nullable=False)
    end_time = Column(Float, default=0.0, nullable=False)
    duration_seconds = Column(Float, default=0.0, nullable=False)
    participants = Column(Text, default="[]", nullable=False)  # JSON list of speaker names
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    meeting = relationship("Meeting", back_populates="topics")

    __table_args__ = (
        Index("idx_topics_meeting_id", "meeting_id"),
    )


class Keyword(Base):
    """Categorized keyword detected during meeting discussions."""

    __tablename__ = "keywords"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    meeting_id = Column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    keyword = Column(String(255), nullable=False)
    category = Column(String(50), default="general", nullable=False)  # product, client, risk, budget, etc.
    relevance_score = Column(Float, default=1.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    meeting = relationship("Meeting", back_populates="keywords")

    __table_args__ = (
        Index("idx_keywords_meeting_id", "meeting_id"),
        Index("idx_keywords_category", "category"),
    )


class EmployeeReport(Base):
    """Personalized report generated for an employee participant."""

    __tablename__ = "employee_reports"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    meeting_id = Column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    employee_id = Column(String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=True)
    employee_name = Column(String(255), nullable=False)
    topics_discussed = Column(Text, default="[]", nullable=False)  # JSON list
    decisions_affecting = Column(Text, default="[]", nullable=False)  # JSON list
    assigned_tasks = Column(Text, default="[]", nullable=False)  # JSON list
    mentioned_deadlines = Column(Text, default="[]", nullable=False)  # JSON list
    follow_ups = Column(Text, default="[]", nullable=False)  # JSON list
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    meeting = relationship("Meeting", back_populates="employee_reports")
    employee = relationship("Employee", back_populates="employee_reports")

    __table_args__ = (
        Index("idx_emp_reports_meeting", "meeting_id"),
        Index("idx_emp_reports_emp", "employee_id"),
    )


class ConfiguredKeyword(Base):
    """Admin-configured keyword to track across meetings."""

    __tablename__ = "configured_keywords"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    keyword = Column(String(255), unique=True, nullable=False)
    category = Column(String(50), default="custom", nullable=False)
    is_active = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_cfg_keywords_kw", "keyword"),
    )


class Notification(Base):
    """Notification record generated after meeting processing."""

    __tablename__ = "notifications"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    meeting_id = Column(String(36), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=True)
    recipient = Column(String(255), nullable=False)  # Email or user handle
    channel = Column(String(50), default="internal", nullable=False)  # internal, email, slack, teams
    status = Column(String(50), default="sent", nullable=False)
    subject = Column(String(255), nullable=True)
    payload = Column(Text, nullable=True)  # JSON summary payload
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    meeting = relationship("Meeting", back_populates="notifications")

    __table_args__ = (
        Index("idx_notifications_recipient", "recipient"),
        Index("idx_notifications_status", "status"),
    )


class AuditLog(Base):
    """Audit log entry for security and compliance tracking."""

    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    action = Column(String(100), nullable=False)  # e.g. UPLOAD_MEETING, VOICE_ENROLL, SEARCH_RAG
    user_id = Column(String(100), default="system", nullable=False)
    details = Column(Text, nullable=True)
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_audit_logs_action", "action"),
        Index("idx_audit_logs_date", "created_at"),
    )
