"""Database Repository for MeetWise AI Enterprise Platform.

Handles transactions for:
- Meetings, Media Metadata, Manager Summaries, Overall Sentiment
- Employees, Voice Biometrics, Voice Samples
- Speakers with Recognition Confidence and Employee FK Mapping
- Transcripts with Emotion and Speaking Style
- Action Items with Priority & Ownership
- Decisions, Topics, Keywords, Configured Keywords
- Personalized Employee Reports
- Notifications & Security Audit Logs
"""

import json
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from .models import (
    Meeting,
    Speaker,
    Transcript,
    ActionItemModel,
    DecisionModel,
    Employee,
    VoiceSample,
    Topic,
    Keyword,
    EmployeeReport,
    ConfiguredKeyword,
    Notification,
    AuditLog,
)
from ..meeting_intelligence.schemas import MeetingIntelligence

logger = logging.getLogger(__name__)


class MeetingRepository:
    """Repository handling persistence and retrieval of meeting entities and enterprise intelligence."""

    def __init__(self, db: Session):
        self.db = db

    # =========================================================================
    # Employee & Voice Biometrics Operations
    # =========================================================================

    def create_employee(
        self,
        employee_id: str,
        name: str,
        email: str,
        department: str = "General",
        team: str = "Core",
        designation: str = "Team Member",
        accent_metadata: Optional[Dict[str, Any]] = None,
    ) -> Employee:
        """Create a new employee profile for onboarding."""
        emp = Employee(
            employee_id=employee_id,
            name=name,
            email=email,
            department=department,
            team=team,
            designation=designation,
            accent_metadata=json.dumps(accent_metadata) if accent_metadata else None,
            voice_samples_count=0,
        )
        self.db.add(emp)
        self.db.commit()
        self.db.refresh(emp)
        self.log_audit_event("EMPLOYEE_CREATED", user_id="system", details=f"Created {name} ({employee_id})")
        return emp

    def get_employee(self, id_or_emp_id: str) -> Optional[Employee]:
        """Fetch employee by primary UUID or employee_id string."""
        return (
            self.db.query(Employee)
            .filter((Employee.id == id_or_emp_id) | (Employee.employee_id == id_or_emp_id))
            .first()
        )

    def get_employee_by_email(self, email: str) -> Optional[Employee]:
        """Fetch employee by email."""
        return self.db.query(Employee).filter(Employee.email == email).first()

    def list_employees(self, department: Optional[str] = None, limit: int = 100) -> List[Employee]:
        """List employees with optional department filtering."""
        q = self.db.query(Employee)
        if department:
            q = q.filter(Employee.department == department)
        return q.order_by(Employee.name.asc()).limit(limit).all()

    def get_all_enrolled_employees(self) -> List[Employee]:
        """Fetch all employees who have a voice embedding registered."""
        return (
            self.db.query(Employee)
            .filter(Employee.voice_embedding.isnot(None))
            .all()
        )

    def add_voice_sample(
        self,
        employee_id: str,
        audio_file_name: str,
        duration_seconds: float,
        prompt_text: Optional[str] = None,
        embedding_vector: Optional[List[float]] = None,
    ) -> VoiceSample:
        """Register a voice recording sample for an employee and increment sample count."""
        emp = self.get_employee(employee_id)
        if not emp:
            raise ValueError(f"Employee {employee_id} not found.")

        sample = VoiceSample(
            employee_id=emp.id,
            audio_file_name=audio_file_name,
            duration_seconds=duration_seconds,
            prompt_text=prompt_text,
            embedding_vector=json.dumps(embedding_vector) if embedding_vector else None,
        )
        self.db.add(sample)
        emp.voice_samples_count += 1
        self.db.commit()
        self.db.refresh(sample)
        self.log_audit_event("VOICE_SAMPLE_ADDED", user_id=emp.employee_id, details=f"Sample {audio_file_name} added")
        return sample

    def update_employee_voice_profile(
        self,
        employee_id: str,
        centroid_embedding: List[float],
        accent_metadata: Optional[Dict[str, Any]] = None,
    ) -> Employee:
        """Update the aggregated voice centroid embedding and accent details for an employee."""
        emp = self.get_employee(employee_id)
        if not emp:
            raise ValueError(f"Employee {employee_id} not found.")

        emp.voice_embedding = json.dumps(centroid_embedding)
        if accent_metadata:
            emp.accent_metadata = json.dumps(accent_metadata)
        self.db.commit()
        self.db.refresh(emp)
        self.log_audit_event("VOICE_PROFILE_UPDATED", user_id=emp.employee_id, details="Voice centroid updated")
        return emp

    # =========================================================================
    # Meeting Operations & Enterprise Pipeline Persistence
    # =========================================================================

    def create_meeting(
        self,
        meeting_id: str,
        title: str,
        audio_file_name: str,
        duration_seconds: float = 0.0,
        media_type: str = "audio",
        source_type: str = "uploaded",
        summary: Optional[str] = None,
    ) -> Meeting:
        """Create a new meeting record."""
        meeting = Meeting(
            id=meeting_id,
            title=title,
            audio_file_name=audio_file_name,
            duration_seconds=duration_seconds,
            media_type=media_type,
            source_type=source_type,
            summary=summary,
        )
        self.db.add(meeting)
        self.db.commit()
        self.db.refresh(meeting)
        return meeting

    def save_meeting_pipeline_results(
        self,
        meeting_id: str,
        title: str,
        audio_file_name: str,
        duration_seconds: float,
        source_type: str,
        aligned_segments: List[Dict[str, Any]],
        intelligence: MeetingIntelligence,
        speaker_identifications: Optional[Dict[str, Dict[str, Any]]] = None,
        media_type: str = "audio",
    ) -> Meeting:
        """Persist full enterprise pipeline results atomically in one database transaction."""
        try:
            speaker_identifications = speaker_identifications or {}

            # 1. Create or update meeting record
            meeting = self.db.query(Meeting).filter(Meeting.id == meeting_id).first()
            manager_summary_json = None
            if hasattr(intelligence, "manager_summary") and intelligence.manager_summary:
                manager_summary_json = intelligence.manager_summary.model_dump_json()

            overall_sentiment_json = None
            if hasattr(intelligence, "sentiment_analysis") and intelligence.sentiment_analysis:
                overall_sentiment_json = intelligence.sentiment_analysis.model_dump_json()

            if not meeting:
                meeting = Meeting(
                    id=meeting_id,
                    title=intelligence.title or title,
                    audio_file_name=audio_file_name,
                    duration_seconds=duration_seconds,
                    media_type=media_type,
                    source_type=source_type,
                    summary=intelligence.summary,
                    manager_summary=manager_summary_json,
                    overall_sentiment=overall_sentiment_json,
                )
                self.db.add(meeting)
            else:
                meeting.title = intelligence.title or title
                meeting.summary = intelligence.summary
                meeting.duration_seconds = duration_seconds
                meeting.media_type = media_type
                meeting.manager_summary = manager_summary_json
                meeting.overall_sentiment = overall_sentiment_json

            # 2. Extract and create speakers with biometric recognition metadata
            unique_speakers = sorted(list(set(seg["speaker"] for seg in aligned_segments if "speaker" in seg)))
            speaker_map = {}  # speaker_label -> Speaker model

            for spk_label in unique_speakers:
                id_info = speaker_identifications.get(spk_label, {})
                recognized_name = id_info.get("name") or spk_label
                confidence = float(id_info.get("confidence", 1.0))
                emp_id_fk = id_info.get("employee_id")
                is_unknown = 1 if id_info.get("is_unknown", False) else 0
                accent = id_info.get("accent")
                emotion = id_info.get("dominant_emotion")

                # Compute total speaking time for this speaker
                total_spk_time = sum(
                    float(s.get("end", 0.0)) - float(s.get("start", 0.0))
                    for s in aligned_segments
                    if s.get("speaker") == spk_label
                )

                existing_spk = (
                    self.db.query(Speaker)
                    .filter(Speaker.meeting_id == meeting_id, Speaker.speaker_label == spk_label)
                    .first()
                )
                if not existing_spk:
                    existing_spk = Speaker(
                        meeting_id=meeting_id,
                        employee_id=emp_id_fk,
                        speaker_label=spk_label,
                        speaker_name=recognized_name,
                        confidence_score=confidence,
                        is_unknown=is_unknown,
                        detected_accent=accent,
                        dominant_emotion=emotion,
                        total_speaking_time=round(total_spk_time, 2),
                    )
                    self.db.add(existing_spk)
                    self.db.flush()
                else:
                    existing_spk.speaker_name = recognized_name
                    existing_spk.confidence_score = confidence
                    existing_spk.employee_id = emp_id_fk
                    existing_spk.is_unknown = is_unknown
                    existing_spk.total_speaking_time = round(total_spk_time, 2)
                    if accent:
                        existing_spk.detected_accent = accent
                    if emotion:
                        existing_spk.dominant_emotion = emotion

                speaker_map[spk_label] = existing_spk

            # 3. Insert transcript segments with emotions and speaking style
            for idx, seg in enumerate(aligned_segments):
                spk_obj = speaker_map.get(seg.get("speaker"))
                transcript_seg = Transcript(
                    meeting_id=meeting_id,
                    speaker_id=spk_obj.id if spk_obj else None,
                    start_time=float(seg.get("start", 0.0)),
                    end_time=float(seg.get("end", seg.get("start", 0.0))),
                    text=seg.get("text", "").strip(),
                    emotion=seg.get("emotion", "neutral"),
                    speaking_style=seg.get("speaking_style", "professional"),
                    segment_index=idx,
                )
                self.db.add(transcript_seg)

            # 4. Insert action items with priority
            for item in intelligence.action_items:
                priority_val = getattr(item, "priority", "Medium") or "Medium"
                action_model = ActionItemModel(
                    meeting_id=meeting_id,
                    task=item.task,
                    owner=item.owner or "Unassigned",
                    deadline=item.deadline or "Not specified",
                    priority=priority_val,
                    status=item.status or "pending",
                )
                self.db.add(action_model)

            # 5. Insert decisions
            for dec in intelligence.decisions:
                decision_model = DecisionModel(
                    meeting_id=meeting_id,
                    decision=dec.decision,
                    timestamp=dec.timestamp,
                )
                self.db.add(decision_model)

            # 6. Insert Topics
            if hasattr(intelligence, "topics") and intelligence.topics:
                for top in intelligence.topics:
                    if isinstance(top, str):
                        topic_obj = Topic(
                            meeting_id=meeting_id,
                            topic_name=top,
                            start_time=0.0,
                            end_time=duration_seconds,
                            duration_seconds=duration_seconds,
                            participants=json.dumps([]),
                        )
                    else:
                        topic_obj = Topic(
                            meeting_id=meeting_id,
                            topic_name=top.topic,
                            start_time=top.start_time,
                            end_time=top.end_time,
                            duration_seconds=top.duration_seconds,
                            participants=json.dumps(top.participants),
                        )
                    self.db.add(topic_obj)

            # 7. Insert Keywords
            if hasattr(intelligence, "keywords") and intelligence.keywords:
                for kw in intelligence.keywords:
                    kw_obj = Keyword(
                        meeting_id=meeting_id,
                        keyword=kw.keyword,
                        category=kw.category,
                        relevance_score=kw.relevance_score,
                    )
                    self.db.add(kw_obj)

            # 8. Insert Personalized Employee Reports
            if hasattr(intelligence, "employee_reports") and intelligence.employee_reports:
                for rep in intelligence.employee_reports:
                    matched_emp = self.get_employee(rep.employee_name)
                    emp_report = EmployeeReport(
                        meeting_id=meeting_id,
                        employee_id=matched_emp.id if matched_emp else None,
                        employee_name=rep.employee_name,
                        topics_discussed=json.dumps(rep.topics_discussed),
                        decisions_affecting=json.dumps(rep.decisions_affecting),
                        assigned_tasks=json.dumps(rep.assigned_tasks),
                        mentioned_deadlines=json.dumps(rep.mentioned_deadlines),
                        follow_ups=json.dumps(rep.follow_ups),
                    )
                    self.db.add(emp_report)

            # Commit all additions atomically
            self.db.commit()
            self.db.refresh(meeting)
            self.log_audit_event("MEETING_PROCESSED", user_id="system", details=f"Meeting {meeting_id} saved")
            logger.info(f"Successfully saved enterprise meeting results to DB: {meeting_id}")
            return meeting

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to persist meeting pipeline results: {e}")
            raise

    def get_meeting(self, meeting_id: str) -> Optional[Meeting]:
        """Fetch meeting with relationships by ID."""
        return self.db.query(Meeting).filter(Meeting.id == meeting_id).first()

    def list_meetings(self, limit: int = 50, offset: int = 0) -> List[Meeting]:
        """List all meetings chronologically ordered by creation date."""
        return (
            self.db.query(Meeting)
            .order_by(desc(Meeting.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_transcripts(self, meeting_id: str) -> List[Transcript]:
        """Fetch all transcript segments for a meeting sorted by start time."""
        return (
            self.db.query(Transcript)
            .filter(Transcript.meeting_id == meeting_id)
            .order_by(Transcript.start_time.asc())
            .all()
        )

    def get_topics(self, meeting_id: str) -> List[Topic]:
        """Fetch discussion topics for a meeting."""
        return self.db.query(Topic).filter(Topic.meeting_id == meeting_id).all()

    def get_keywords(self, meeting_id: str) -> List[Keyword]:
        """Fetch keywords for a meeting."""
        return self.db.query(Keyword).filter(Keyword.meeting_id == meeting_id).all()

    def get_employee_reports(self, meeting_id: str) -> List[EmployeeReport]:
        """Fetch personalized reports for all participants in a meeting."""
        return self.db.query(EmployeeReport).filter(EmployeeReport.meeting_id == meeting_id).all()

    def get_employee_report_for_user(self, meeting_id: str, employee_id: str) -> Optional[EmployeeReport]:
        """Fetch personalized report for a specific employee."""
        return (
            self.db.query(EmployeeReport)
            .filter(
                EmployeeReport.meeting_id == meeting_id,
                (EmployeeReport.employee_id == employee_id) | (EmployeeReport.employee_name == employee_id),
            )
            .first()
        )

    def delete_meeting(self, meeting_id: str) -> bool:
        """Delete meeting and cascade delete all related records."""
        meeting = self.get_meeting(meeting_id)
        if meeting:
            self.db.delete(meeting)
            self.db.commit()
            self.log_audit_event("MEETING_DELETED", user_id="system", details=f"Meeting {meeting_id} deleted")
            return True
        return False

    # =========================================================================
    # Configured Keywords Management
    # =========================================================================

    def list_configured_keywords(self) -> List[ConfiguredKeyword]:
        """List all configured keywords for enterprise monitoring."""
        return self.db.query(ConfiguredKeyword).order_by(ConfiguredKeyword.keyword.asc()).all()

    def add_configured_keyword(self, keyword: str, category: str = "custom") -> ConfiguredKeyword:
        """Add a keyword to active enterprise monitoring."""
        kw_clean = keyword.strip()
        existing = (
            self.db.query(ConfiguredKeyword)
            .filter(func.lower(ConfiguredKeyword.keyword) == kw_clean.lower())
            .first()
        )
        if existing:
            existing.is_active = 1
            existing.category = category
            self.db.commit()
            return existing

        cfg = ConfiguredKeyword(keyword=kw_clean, category=category, is_active=1)
        self.db.add(cfg)
        self.db.commit()
        self.db.refresh(cfg)
        self.log_audit_event("KEYWORD_ADDED", user_id="admin", details=f"Keyword '{kw_clean}' added")
        return cfg

    def delete_configured_keyword(self, keyword_id: str) -> bool:
        """Delete a configured keyword."""
        item = self.db.query(ConfiguredKeyword).filter(ConfiguredKeyword.id == keyword_id).first()
        if item:
            self.db.delete(item)
            self.db.commit()
            return True
        return False

    # =========================================================================
    # Notifications & Audit Logs
    # =========================================================================

    def create_notification(
        self,
        meeting_id: Optional[str],
        recipient: str,
        channel: str = "internal",
        subject: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Notification:
        """Record a notification sent to a participant or manager."""
        notif = Notification(
            meeting_id=meeting_id,
            recipient=recipient,
            channel=channel,
            status="sent",
            subject=subject,
            payload=json.dumps(payload) if payload else None,
        )
        self.db.add(notif)
        self.db.commit()
        self.db.refresh(notif)
        return notif

    def list_notifications(self, recipient: Optional[str] = None, limit: int = 50) -> List[Notification]:
        """Fetch notification feed."""
        q = self.db.query(Notification)
        if recipient:
            q = q.filter(Notification.recipient == recipient)
        return q.order_by(desc(Notification.created_at)).limit(limit).all()

    def log_audit_event(
        self,
        action: str,
        user_id: str = "system",
        details: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> AuditLog:
        """Record an event to the security audit trail."""
        try:
            log = AuditLog(
                action=action,
                user_id=user_id,
                details=details,
                ip_address=ip_address,
            )
            self.db.add(log)
            self.db.commit()
            return log
        except Exception as e:
            logger.warning(f"Failed to record audit log: {e}")
            self.db.rollback()

    def list_audit_logs(self, limit: int = 100) -> List[AuditLog]:
        """Fetch audit trail."""
        return self.db.query(AuditLog).order_by(desc(AuditLog.created_at)).limit(limit).all()
