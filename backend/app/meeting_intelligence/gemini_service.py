"""Gemini Meeting Intelligence Service for MeetWise AI Enterprise Platform.

Transforms multi-speaker meeting transcripts into enterprise structured intelligence:
- Title, Executive Summary, Inferred Agenda & Discussion Points
- Granular Topics with Time Boundaries & Participants
- Formal Decisions
- Prioritized Action Items (Owner, Deadline, Priority, Status)
- Ranked & Categorized Keywords (Product, Client, Risk, Budget, Revenue, etc.)
- Per-Speaker Emotion & Speaking Style Analysis + Overall Sentiment Dynamics
- Accent Classification (Indian, British, American, Australian, International English)
- Personalized Employee-Specific Reports
- 1-Page Manager Executive Summary
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

from .schemas import (
    MeetingIntelligence,
    ActionItem,
    DecisionItem,
    TopicItem,
    KeywordItem,
    EmotionItem,
    SentimentAnalysis,
    AccentItem,
    EmployeeSpecificReport,
    ManagerExecutiveSummary,
)

logger = logging.getLogger(__name__)

CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


class LLMIntelligenceError(Exception):
    """Exception raised when LLM meeting intelligence extraction fails."""
    pass


class GeminiIntelligenceService:
    """Extracts comprehensive enterprise meeting intelligence using Google Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key
        self.model_name = model_name
        self._client = None
        self._genai = None

    def _init_client(self):
        """Initialize Google Gemini client with intelligent fallback."""
        if not self.api_key:
            raise LLMIntelligenceError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your environment or .env file."
            )

        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._genai = genai

                model_to_try = [self.model_name] + [m for m in CANDIDATE_MODELS if m != self.model_name]
                last_err = None

                for m in model_to_try:
                    try:
                        self._client = genai.GenerativeModel(m)
                        self.model_name = m
                        logger.info(f"Initialized Gemini model: {self.model_name}")
                        break
                    except Exception as e:
                        last_err = e
                        logger.warning(f"Could not initialize candidate model {m}: {e}")

                if self._client is None:
                    raise LLMIntelligenceError(f"Failed to initialize any Gemini model candidate: {last_err}")

            except ImportError:
                raise LLMIntelligenceError(
                    "google-generativeai is not installed. Please install it via 'pip install google-generativeai'."
                )
            except Exception as e:
                raise LLMIntelligenceError(f"Failed to initialize Gemini client: {e}")

    @staticmethod
    def format_transcript_for_llm(aligned_segments: List[Dict[str, Any]]) -> str:
        """Format speaker-labelled segments into clean text for LLM ingestion."""
        lines = []
        for seg in aligned_segments:
            speaker = seg.get("speaker", "UNKNOWN")
            start = seg.get("start", 0.0)
            text = seg.get("text", "").strip()
            m_s = f"{int(start // 60):02d}:{int(start % 60):02d}"
            lines.append(f"[{m_s}] {speaker}: {text}")

        return "\n".join(lines)

    @staticmethod
    def clean_json_response(raw_text: str) -> str:
        """Remove markdown code fences and extraneous text surrounding JSON."""
        text = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            return brace_match.group(1).strip()

        return text

    def analyze(
        self,
        aligned_segments: List[Dict[str, Any]],
        known_employees: Optional[List[str]] = None,
        custom_keywords: Optional[List[str]] = None,
    ) -> MeetingIntelligence:
        """Analyze speaker-labelled transcript and extract complete enterprise meeting intelligence.

        Args:
            aligned_segments: List of dicts with 'speaker', 'start', 'end', and 'text'.
            known_employees: Optional list of registered employee names to align ownership.
            custom_keywords: Optional list of admin keywords to track specifically.

        Returns:
            Validated MeetingIntelligence instance.
        """
        if not aligned_segments:
            raise LLMIntelligenceError("Cannot analyze empty transcript.")

        self._init_client()
        formatted_transcript = self.format_transcript_for_llm(aligned_segments)

        emp_context = ""
        if known_employees:
            emp_context = f"\nKNOWN REGISTERED EMPLOYEES: {', '.join(known_employees)}\n"

        kw_context = ""
        if custom_keywords:
            kw_context = f"\nADMIN CONFIGURED KEYWORDS TO TRACK: {', '.join(custom_keywords)}\n"

        prompt = f"""You are an elite Enterprise Meeting Intelligence & Organizational Memory AI for MeetWise.
Analyze the following multi-speaker meeting transcript recorded in a conference room or video call.
{emp_context}{kw_context}
You must extract an exhaustive intelligence payload strictly conforming to this JSON structure:
{{
    "title": "A concise, meaningful corporate title for the meeting",
    "summary": "A comprehensive 2-3 paragraph executive summary of the meeting context, core debates, and outcomes",
    "agenda": ["Inferred agenda item 1", "Inferred agenda item 2"],
    "discussion_points": ["Detailed point covering what was debated or presented"],
    "topics": [
        {{
            "topic": "Topic Name (e.g. Q3 Roadmap, Server Costs)",
            "start_time": 0.0,
            "end_time": 45.0,
            "duration_seconds": 45.0,
            "participants": ["Participant Name or Speaker Label"]
        }}
    ],
    "decisions": [
        {{
            "decision": "Specific agreement or decision reached",
            "timestamp": "Timestamp mm:ss or null"
        }}
    ],
    "action_items": [
        {{
            "task": "Explicit action item or deliverable",
            "owner": "Person responsible (use known employee name if matched, else speaker label)",
            "deadline": "Deadline or target date (e.g. Friday, Next sprint, or 'Not specified')",
            "priority": "High | Medium | Low",
            "status": "pending"
        }}
    ],
    "keywords": [
        {{
            "keyword": "Entity or Term",
            "category": "product | client | risk | budget | hiring | revenue | deadline | deliverable | custom",
            "relevance_score": 0.95
        }}
    ],
    "emotions": [
        {{
            "speaker": "Speaker Name or Label",
            "dominant_emotion": "neutral | happy | excited | sad | angry | frustrated | concerned | nervous | confident | empathetic | professional",
            "speaking_style": "assertive | collaborative | hesitant | formal | authoritative",
            "confidence": 0.9
        }}
    ],
    "sentiment_analysis": {{
        "overall_sentiment": "positive | neutral | concerned | negative",
        "positive_percentage": 25.0,
        "neutral_percentage": 65.0,
        "negative_percentage": 10.0,
        "emotions_breakdown": {{"professional": 50.0, "confident": 30.0, "concerned": 20.0}},
        "key_emotional_moments": ["Brief note on any tension, agreement, or enthusiasm"]
    }},
    "accents": [
        {{
            "speaker": "Speaker Name or Label",
            "detected_accent": "Indian | British | American | Australian | International English",
            "confidence": 0.88
        }}
    ],
    "employee_reports": [
        {{
            "employee_name": "Speaker or Employee Name",
            "topics_discussed": ["Topics this person specifically spoke on"],
            "decisions_affecting": ["Decisions impacting their area"],
            "assigned_tasks": ["Tasks assigned to this person"],
            "mentioned_deadlines": ["Deadlines they agreed to"],
            "follow_ups": ["Personal follow-up questions"]
        }}
    ],
    "manager_summary": {{
        "executive_summary": "1-Page concise summary for Directors and VP",
        "key_decisions": ["Crucial decision 1"],
        "risks": ["Identified operational or project risk"],
        "delays": ["Any slipped deadline or delay discussed"],
        "team_blockers": ["Bottlenecks impeding the team"],
        "assigned_tasks": ["Critical deliverables"],
        "follow_up_actions": ["Next steps requiring manager sign-off"]
    }},
    "pending_issues": ["Unresolved open questions or blockers requiring follow-up"],
    "next_steps": ["Immediate next organizational milestones"]
}}

GUIDELINES:
1. Map tasks and owners accurately to the actual speaker who committed to or was assigned the task.
2. Ensure every participant in the transcript has an entry in employee_reports, emotions, and accents.
3. Categorize keywords into appropriate business buckets: product, client, risk, budget, hiring, revenue, deadline, deliverable, custom.
4. Output ONLY valid JSON with no extraneous markdown or conversational filler.

TRANSCRIPT:
{formatted_transcript}
"""

        try:
            logger.info(f"Sending transcript to Gemini ({self.model_name}) for enterprise intelligence extraction...")
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                },
            )

            raw_text = response.text
            cleaned_json_text = self.clean_json_response(raw_text)

            try:
                data = json.loads(cleaned_json_text)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode failed on Gemini output: {raw_text}")
                raise LLMIntelligenceError(f"Malformed JSON returned by Gemini: {json_err}")

            intelligence = MeetingIntelligence(**data)
            logger.info(
                f"Successfully extracted enterprise intelligence: '{intelligence.title}' with "
                f"{len(intelligence.action_items)} actions, {len(intelligence.topics)} topics, "
                f"and {len(intelligence.employee_reports)} employee reports."
            )
            return intelligence

        except ValidationError as val_err:
            logger.error(f"Pydantic validation failed on Gemini response: {val_err}")
            raise LLMIntelligenceError(f"Gemini response did not match enterprise schema: {val_err}")
        except Exception as e:
            logger.warning(f"Gemini model {self.model_name} failed: {e}. Trying fallback candidates...")
            models_to_try = [m for m in CANDIDATE_MODELS if m != self.model_name]
            last_exception = e

            for candidate_name in models_to_try:
                try:
                    import google.generativeai as genai
                    alt_model = genai.GenerativeModel(candidate_name)
                    response = alt_model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.2,
                            "response_mime_type": "application/json",
                        },
                    )
                    raw_text = response.text
                    cleaned_json_text = self.clean_json_response(raw_text)
                    data = json.loads(cleaned_json_text)
                    intelligence = MeetingIntelligence(**data)
                    self._client = alt_model
                    self.model_name = candidate_name
                    return intelligence
                except Exception as cand_err:
                    last_exception = cand_err

            raise LLMIntelligenceError(f"Gemini API request failed across all model candidates: {last_exception}")
