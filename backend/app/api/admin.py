"""Admin, Keywords Configuration, and Audit Logs API Router for MeetWise AI."""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..database.repository import MeetingRepository
from ..security.security import require_admin, CurrentUser, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Enterprise Administration & Compliance"])


class KeywordCreateRequest(BaseModel):
    keyword: str = Field(..., description="Keyword or entity to track")
    category: str = Field(default="custom", description="Category: product, client, risk, budget, revenue, deliverable, custom")


class KeywordResponse(BaseModel):
    id: str
    keyword: str
    category: str
    is_active: int
    created_at: str


class AuditLogResponse(BaseModel):
    id: str
    action: str
    user_id: str
    details: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: str


class NotificationResponse(BaseModel):
    id: str
    meeting_id: Optional[str] = None
    recipient: str
    channel: str
    status: str
    subject: Optional[str] = None
    payload: Optional[str] = None
    created_at: str


# =============================================================================
# Configured Keywords Management
# =============================================================================

@router.get("/admin/keywords", response_model=List[KeywordResponse])
def list_monitored_keywords(
    db: Session = Depends(get_db),
    admin_user: CurrentUser = Depends(require_admin),
):
    """List all enterprise-monitored keywords across meetings."""
    repo = MeetingRepository(db)
    items = repo.list_configured_keywords()
    return [
        KeywordResponse(
            id=item.id,
            keyword=item.keyword,
            category=item.category,
            is_active=item.is_active,
            created_at=item.created_at.isoformat(),
        )
        for item in items
    ]


@router.post("/admin/keywords", response_model=KeywordResponse, status_code=status.HTTP_200_OK)
def add_monitored_keyword(
    payload: KeywordCreateRequest,
    db: Session = Depends(get_db),
    admin_user: CurrentUser = Depends(require_admin),
):
    """Add a keyword to active enterprise monitoring across all future meetings."""
    repo = MeetingRepository(db)
    kw = repo.add_configured_keyword(keyword=payload.keyword, category=payload.category)
    return KeywordResponse(
        id=kw.id,
        keyword=kw.keyword,
        category=kw.category,
        is_active=kw.is_active,
        created_at=kw.created_at.isoformat(),
    )


@router.delete("/admin/keywords/{keyword_id}", status_code=status.HTTP_200_OK)
def delete_monitored_keyword(
    keyword_id: str,
    db: Session = Depends(get_db),
    admin_user: CurrentUser = Depends(require_admin),
):
    """Remove a keyword from active enterprise monitoring."""
    repo = MeetingRepository(db)
    if not repo.delete_configured_keyword(keyword_id):
        raise HTTPException(status_code=404, detail="Keyword not found.")
    return {"message": "Keyword deleted successfully."}


# =============================================================================
# Audit Trails & Compliance
# =============================================================================

@router.get("/admin/audit-logs", response_model=List[AuditLogResponse])
def get_audit_trail(
    limit: int = 100,
    db: Session = Depends(get_db),
    admin_user: CurrentUser = Depends(require_admin),
):
    """Retrieve security and activity audit logs for enterprise compliance."""
    repo = MeetingRepository(db)
    logs = repo.list_audit_logs(limit=limit)
    return [
        AuditLogResponse(
            id=log.id,
            action=log.action,
            user_id=log.user_id,
            details=log.details,
            ip_address=log.ip_address,
            created_at=log.created_at.isoformat(),
        )
        for log in logs
    ]


# =============================================================================
# Notifications Feed
# =============================================================================

@router.get("/notifications", response_model=List[NotificationResponse])
@router.get("/admin/notifications", response_model=List[NotificationResponse], include_in_schema=False)
def get_notifications(
    recipient: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Retrieve internal notification feed for current user or meeting."""
    repo = MeetingRepository(db)
    notifs = repo.list_notifications(recipient=recipient, limit=limit)
    return [
        NotificationResponse(
            id=n.id,
            meeting_id=n.meeting_id,
            recipient=n.recipient,
            channel=n.channel,
            status=n.status,
            subject=n.subject,
            payload=n.payload,
            created_at=n.created_at.isoformat(),
        )
        for n in notifs
    ]
