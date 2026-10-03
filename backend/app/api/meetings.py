"""Meetings API Router for MeetWise AI Enterprise Platform.

Handles:
- Audio and Video uploads (.wav, .mp3, .m4a, .mp4, .webm)
- 10-phase enterprise pipeline execution
- Meeting detail & list queries with speaker recognition confidence
- Downloadable Minutes of Meeting (MOM) PDF export
- 1-Page Manager Executive Summary
- Personalized Employee-Specific Reports
- Emotion, Sentiment, Accent, and Keyword Analytics
"""

import os
import json
import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..config import settings
from ..database.connection import get_db
from ..database.repository import MeetingRepository
from ..audio.preprocessor import SUPPORTED_EXTENSIONS
from ..pipeline import MeetWisePipeline, PipelineExecutionError
from ..reports.pdf_generator import MOMPDFGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meetings", tags=["Meetings"])

_pipeline_instance = None
pdf_gen = MOMPDFGenerator()


def get_pipeline() -> MeetWisePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MeetWisePipeline()
    return _pipeline_instance


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_meeting_media(
    file: UploadFile = File(..., description="Meeting audio or video recording (.wav, .mp3, .m4a, .mp4, .webm)"),
    title: Optional[str] = Form(None, description="Optional custom meeting title"),
    source_type: str = Form("uploaded", description="Audio source: 'uploaded' or 'live'"),
    process_immediately: bool = Form(True, description="Whether to trigger the AI pipeline immediately"),
    db: Session = Depends(get_db),
    pipeline: MeetWisePipeline = Depends(get_pipeline),
):
    """Upload a meeting audio or video file and execute the MeetWise enterprise pipeline."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported media extension '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    if source_type not in ("uploaded", "live"):
        raise HTTPException(status_code=400, detail="source_type must be either 'uploaded' or 'live'.")

    meeting_id = str(uuid.uuid4())
    safe_filename = f"{meeting_id}_{Path(file.filename).name}"
    stored_media_path = Path(settings.AUDIO_DIR) / safe_filename

    # Save uploaded media safely
    try:
        with open(stored_media_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to write uploaded media file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store media file on server: {e}")

    if not process_immediately:
        repo = MeetingRepository(db)
        repo.create_meeting(
            meeting_id=meeting_id,
            title=title or f"Meeting ({file.filename})",
            audio_file_name=safe_filename,
            source_type=source_type,
        )
        return {
            "meeting_id": meeting_id,
            "status": "uploaded",
            "message": "Media file uploaded successfully. Ready for processing via POST /meetings/{meeting_id}/process.",
            "audio_file_name": safe_filename,
        }

    # Execute complete enterprise AI Pipeline
    try:
        result = pipeline.process_meeting(
            audio_path=str(stored_media_path),
            db=db,
            meeting_id=meeting_id,
            title=title,
            source_type=source_type,
        )
        return {
            "status": "completed",
            "meeting": result,
        }
    except PipelineExecutionError as p_err:
        logger.error(f"Pipeline error for meeting {meeting_id}: {p_err}")
        raise HTTPException(status_code=500, detail=str(p_err))
    except Exception as e:
        logger.error(f"Unexpected error processing meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Meeting processing failed: {e}")


@router.post("/{meeting_id}/process")
def process_existing_meeting(
    meeting_id: str,
    db: Session = Depends(get_db),
    pipeline: MeetWisePipeline = Depends(get_pipeline),
):
    """Trigger the complete AI processing pipeline for a previously uploaded file."""
    repo = MeetingRepository(db)
    meeting = repo.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting with ID '{meeting_id}' not found.")

    media_path = Path(settings.AUDIO_DIR) / meeting.audio_file_name
    if not media_path.exists():
        raise HTTPException(status_code=404, detail=f"Media file '{meeting.audio_file_name}' not found on server.")

    try:
        result = pipeline.process_meeting(
            audio_path=str(media_path),
            db=db,
            meeting_id=meeting.id,
            title=meeting.title,
            source_type=meeting.source_type,
        )
        return {"status": "completed", "meeting": result}
    except Exception as e:
        logger.error(f"Pipeline error for meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@router.get("")
def list_meetings(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List all processed meetings with summary and metadata."""
    repo = MeetingRepository(db)
    meetings = repo.list_meetings(limit=limit, offset=offset)
    return [
        {
            "id": m.id,
            "title": m.title,
            "meeting_date": m.meeting_date.isoformat() if m.meeting_date else None,
            "duration_seconds": m.duration_seconds,
            "audio_file_name": m.audio_file_name,
            "media_type": m.media_type,
            "source_type": m.source_type,
            "summary": m.summary,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "speaker_count": len(m.speakers),
        }
        for m in meetings
    ]


@router.get("/{meeting_id}")
def get_meeting_details(meeting_id: str, db: Session = Depends(get_db)):
    """Retrieve complete meeting intelligence, topics, decisions, action items, and biometric speaker identification."""
    repo = MeetingRepository(db)
    meeting = repo.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found.")

    manager_summary = None
    if meeting.manager_summary:
        try:
            manager_summary = json.loads(meeting.manager_summary)
        except Exception:
            manager_summary = meeting.manager_summary

    overall_sentiment = None
    if meeting.overall_sentiment:
        try:
            overall_sentiment = json.loads(meeting.overall_sentiment)
        except Exception:
            overall_sentiment = meeting.overall_sentiment

    return {
        "id": meeting.id,
        "title": meeting.title,
        "meeting_date": meeting.meeting_date.isoformat() if meeting.meeting_date else None,
        "duration_seconds": meeting.duration_seconds,
        "audio_file_name": meeting.audio_file_name,
        "media_type": meeting.media_type,
        "source_type": meeting.source_type,
        "summary": meeting.summary,
        "manager_summary": manager_summary,
        "overall_sentiment": overall_sentiment,
        "created_at": meeting.created_at.isoformat() if meeting.created_at else None,
        "speakers": [
            {
                "id": s.id,
                "label": s.speaker_label,
                "name": s.speaker_name,
                "employee_id": s.employee_id,
                "confidence": s.confidence_score,
                "is_unknown": bool(s.is_unknown),
                "detected_accent": s.detected_accent,
                "dominant_emotion": s.dominant_emotion,
                "total_speaking_time": s.total_speaking_time,
            }
            for s in meeting.speakers
        ],
        "decisions": [
            {
                "id": d.id,
                "decision": d.decision,
                "timestamp": d.timestamp,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in meeting.decisions
        ],
        "action_items": [
            {
                "id": a.id,
                "task": a.task,
                "owner": a.owner,
                "deadline": a.deadline,
                "priority": a.priority,
                "status": a.status,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in meeting.action_items
        ],
        "topics": [
            {
                "id": t.id,
                "topic": t.topic_name,
                "start_time": t.start_time,
                "end_time": t.end_time,
                "duration_seconds": t.duration_seconds,
                "participants": json.loads(t.participants) if t.participants else [],
            }
            for t in meeting.topics
        ],
        "keywords": [
            {
                "id": k.id,
                "keyword": k.keyword,
                "category": k.category,
                "relevance_score": k.relevance_score,
            }
            for k in meeting.keywords
        ],
    }


@router.get("/{meeting_id}/transcript")
def get_meeting_transcript(meeting_id: str, db: Session = Depends(get_db)):
    """Retrieve speaker-attributed transcript segments with emotion and speaking style annotations."""
    repo = MeetingRepository(db)
    meeting = repo.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found.")

    transcripts = repo.get_transcripts(meeting_id)
    return {
        "meeting_id": meeting_id,
        "title": meeting.title,
        "segments": [
            {
                "id": t.id,
                "speaker": (t.speaker.speaker_name or t.speaker.speaker_label) if t.speaker else "UNKNOWN",
                "start_time": t.start_time,
                "end_time": t.end_time,
                "text": t.text,
                "emotion": t.emotion,
                "speaking_style": t.speaking_style,
                "segment_index": t.segment_index,
            }
            for t in transcripts
        ],
    }


@router.get("/{meeting_id}/mom/pdf")
def download_mom_pdf(meeting_id: str, db: Session = Depends(get_db)):
    """Download the corporate Minutes of Meeting (MOM) as a formatted PDF."""
    repo = MeetingRepository(db)
    meeting = repo.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found.")

    pdf_file = Path(f"data/reports/MOM_{meeting_id}.pdf")
    if not pdf_file.exists():
        # Generate on the fly
        mom_payload = {
            "id": meeting.id,
            "title": meeting.title,
            "meeting_date": str(meeting.meeting_date),
            "duration_seconds": meeting.duration_seconds,
            "audio_file_name": meeting.audio_file_name,
            "media_type": meeting.media_type,
            "summary": meeting.summary,
            "speakers": [
                {
                    "speaker_label": s.speaker_label,
                    "speaker_name": s.speaker_name,
                    "confidence_score": s.confidence_score,
                    "total_speaking_time": s.total_speaking_time,
                    "detected_accent": s.detected_accent or "International English",
                }
                for s in meeting.speakers
            ],
            "decisions": [{"decision": d.decision, "timestamp": d.timestamp} for d in meeting.decisions],
            "action_items": [{"task": a.task, "owner": a.owner, "deadline": a.deadline, "priority": a.priority, "status": a.status} for a in meeting.action_items],
            "pending_issues": [],
        }
        pdf_path = pdf_gen.generate(mom_payload)
        pdf_file = Path(pdf_path)

    return FileResponse(
        path=str(pdf_file),
        filename=f"MOM_{meeting.title.replace(' ', '_')}_{meeting_id[:8]}.pdf",
        media_type="application/pdf",
    )


@router.get("/{meeting_id}/manager-summary")
def get_manager_summary(meeting_id: str, db: Session = Depends(get_db)):
    """Retrieve concise 1-page manager executive summary for directors and leadership."""
    repo = MeetingRepository(db)
    meeting = repo.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found.")

    if not meeting.manager_summary:
        return {
            "meeting_id": meeting_id,
            "executive_summary": meeting.summary or "Summary not yet generated.",
            "key_decisions": [d.decision for d in meeting.decisions],
            "risks": [],
            "delays": [],
            "team_blockers": [],
            "assigned_tasks": [f"{a.task} ({a.owner} - {a.deadline})" for a in meeting.action_items],
            "follow_up_actions": [],
        }

    return json.loads(meeting.manager_summary)


@router.get("/{meeting_id}/employee-reports")
def get_all_employee_reports(meeting_id: str, db: Session = Depends(get_db)):
    """Retrieve personalized attendee reports for all participants in a meeting."""
    repo = MeetingRepository(db)
    reports = repo.get_employee_reports(meeting_id)
    return [
        {
            "id": r.id,
            "employee_id": r.employee_id,
            "employee_name": r.employee_name,
            "topics_discussed": json.loads(r.topics_discussed) if r.topics_discussed else [],
            "decisions_affecting": json.loads(r.decisions_affecting) if r.decisions_affecting else [],
            "assigned_tasks": json.loads(r.assigned_tasks) if r.assigned_tasks else [],
            "mentioned_deadlines": json.loads(r.mentioned_deadlines) if r.mentioned_deadlines else [],
            "follow_ups": json.loads(r.follow_ups) if r.follow_ups else [],
            "created_at": r.created_at.isoformat(),
        }
        for r in reports
    ]


@router.get("/{meeting_id}/employee-reports/{employee_id_or_name}")
def get_single_employee_report(
    meeting_id: str,
    employee_id_or_name: str,
    db: Session = Depends(get_db),
):
    """Retrieve personalized report for a specific employee."""
    repo = MeetingRepository(db)
    rep = repo.get_employee_report_for_user(meeting_id, employee_id_or_name)
    if not rep:
        raise HTTPException(status_code=404, detail=f"Report for employee '{employee_id_or_name}' not found.")

    return {
        "id": rep.id,
        "employee_id": rep.employee_id,
        "employee_name": rep.employee_name,
        "topics_discussed": json.loads(rep.topics_discussed) if rep.topics_discussed else [],
        "decisions_affecting": json.loads(rep.decisions_affecting) if rep.decisions_affecting else [],
        "assigned_tasks": json.loads(rep.assigned_tasks) if rep.assigned_tasks else [],
        "mentioned_deadlines": json.loads(rep.mentioned_deadlines) if rep.mentioned_deadlines else [],
        "follow_ups": json.loads(rep.follow_ups) if rep.follow_ups else [],
        "created_at": rep.created_at.isoformat(),
    }


@router.get("/{meeting_id}/analytics")
def get_meeting_analytics(meeting_id: str, db: Session = Depends(get_db)):
    """Retrieve meeting emotion breakdown, sentiment distribution, accents, and topic durations."""
    repo = MeetingRepository(db)
    meeting = repo.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found.")

    sentiment = json.loads(meeting.overall_sentiment) if meeting.overall_sentiment else None
    speakers_data = [
        {
            "name": s.speaker_name or s.speaker_label,
            "speaking_time": s.total_speaking_time,
            "confidence": s.confidence_score,
            "accent": s.detected_accent or "International English",
            "dominant_emotion": s.dominant_emotion or "neutral",
        }
        for s in meeting.speakers
    ]

    return {
        "meeting_id": meeting_id,
        "title": meeting.title,
        "duration_seconds": meeting.duration_seconds,
        "overall_sentiment": sentiment,
        "speakers": speakers_data,
        "topics_count": len(meeting.topics),
        "keywords_count": len(meeting.keywords),
        "decisions_count": len(meeting.decisions),
        "action_items_count": len(meeting.action_items),
    }
