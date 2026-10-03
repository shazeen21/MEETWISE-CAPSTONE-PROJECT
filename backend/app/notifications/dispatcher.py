"""Multi-Channel Notification Dispatcher for MeetWise AI Enterprise.

Dispatches personalized post-meeting intelligence to participants via:
- Internal Notification Feed (In-App Notification Center)
- Email (SMTP / Outbox logging)
- Slack Webhook integration
- Microsoft Teams Webhook integration
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import httpx

from ..database.repository import MeetingRepository

logger = logging.getLogger(__name__)


class NotificationDispatcher:
    """Orchestrates notification fan-out after meeting processing."""

    def __init__(
        self,
        repository: MeetingRepository,
        slack_webhook_url: Optional[str] = None,
        teams_webhook_url: Optional[str] = None,
    ):
        self.repo = repository
        self.slack_webhook_url = slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.teams_webhook_url = teams_webhook_url or os.getenv("TEAMS_WEBHOOK_URL")

    def dispatch_meeting_summary(
        self,
        meeting_id: str,
        meeting_title: str,
        intelligence_data: Dict[str, Any],
        participant_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Dispatch notifications across all configured enterprise channels."""
        dispatched_logs = []

        summary_text = intelligence_data.get("summary", "")
        action_items = intelligence_data.get("action_items", [])
        decisions = intelligence_data.get("decisions", [])
        emp_reports = intelligence_data.get("employee_reports", [])

        # 1. Internal Notification for every participant
        for person in participant_names:
            # Find personalized report if available
            personal_report = next((r for r in emp_reports if r.get("employee_name") == person), None)
            personal_tasks = [a for a in action_items if a.get("owner") == person]

            payload = {
                "meeting_id": meeting_id,
                "title": meeting_title,
                "summary": summary_text[:300] + ("..." if len(summary_text) > 300 else ""),
                "your_tasks": [t.get("task") for t in personal_tasks],
                "personal_report": personal_report,
            }

            subject = f"Meeting Summary & Action Items: {meeting_title}"
            notif = self.repo.create_notification(
                meeting_id=meeting_id,
                recipient=person,
                channel="internal",
                subject=subject,
                payload=payload,
            )
            dispatched_logs.append({"recipient": person, "channel": "internal", "id": notif.id})

            # Also create an email dispatch record
            email_notif = self.repo.create_notification(
                meeting_id=meeting_id,
                recipient=f"{person.lower().replace(' ', '.')}@company.internal",
                channel="email",
                subject=subject,
                payload=payload,
            )
            dispatched_logs.append({"recipient": person, "channel": "email", "id": email_notif.id})

        # 2. Slack Webhook notification (if configured)
        if self.slack_webhook_url:
            self._send_slack_webhook(meeting_id, meeting_title, summary_text, len(action_items), len(decisions))

        # 3. Microsoft Teams Webhook notification (if configured)
        if self.teams_webhook_url:
            self._send_teams_webhook(meeting_id, meeting_title, summary_text, len(action_items), len(decisions))

        logger.info(f"Dispatched {len(dispatched_logs)} post-meeting notifications for '{meeting_title}'")
        return dispatched_logs

    def _send_slack_webhook(self, meeting_id: str, title: str, summary: str, actions_count: int, decisions_count: int):
        """Post meeting brief to Slack channel."""
        try:
            message = {
                "text": f"*MeetWise AI Summary:* {title}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f":memo: *{title}* (ID: `{meeting_id}`)\n{summary[:400]}..."},
                    },
                    {
                        "type": "context",
                        "elements": [
                            {"type": "mrkdwn", "text": f":white_check_mark: *{actions_count}* Action Items | :dart: *{decisions_count}* Decisions"},
                        ],
                    },
                ],
            }
            with httpx.Client(timeout=5.0) as client:
                res = client.post(self.slack_webhook_url, json=message)
                logger.info(f"Slack webhook response status: {res.status_code}")
        except Exception as e:
            logger.warning(f"Slack webhook dispatch skipped: {e}")

    def _send_teams_webhook(self, meeting_id: str, title: str, summary: str, actions_count: int, decisions_count: int):
        """Post meeting card to MS Teams webhook."""
        try:
            card = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "summary": f"Meeting Summary: {title}",
                "themeColor": "0076D7",
                "title": f"MeetWise AI: {title}",
                "sections": [
                    {
                        "text": summary[:400] + "...",
                        "facts": [
                            {"name": "Action Items", "value": str(actions_count)},
                            {"name": "Decisions", "value": str(decisions_count)},
                            {"name": "Meeting ID", "value": meeting_id},
                        ],
                    }
                ],
            }
            with httpx.Client(timeout=5.0) as client:
                res = client.post(self.teams_webhook_url, json=card)
                logger.info(f"Teams webhook response status: {res.status_code}")
        except Exception as e:
            logger.warning(f"Teams webhook dispatch skipped: {e}")
