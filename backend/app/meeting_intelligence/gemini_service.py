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
    EMOTION_TAXONOMY,
    EMOTION_VALENCE,
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
        for idx, seg in enumerate(aligned_segments):
            speaker = seg.get("speaker", "UNKNOWN")
            start = seg.get("start", 0.0)
            text = seg.get("text", "").strip()
            m_s = f"{int(start // 60):02d}:{int(start % 60):02d}"
            lines.append(f"[#{idx} {m_s}] {speaker}: {text}")

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

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        """Lowercase, drop punctuation and collapse whitespace so quotes match despite formatting."""
        text = re.sub(r"[^\w\s']", " ", (text or "").lower())
        return re.sub(r"\s+", " ", text).strip()

    def _verify_evidence(self, intelligence: MeetingIntelligence, aligned_segments: List[Dict[str, Any]]) -> MeetingIntelligence:
        """Drop decisions / action items whose quoted evidence is not in the transcript.

        The quote must appear verbatim (ignoring case and punctuation) in the spoken
        text. Items without a verifiable quote are removed rather than shown to users
        with a hidden low confidence score.
        """
        spoken = self._normalize_for_match(" ".join(seg.get("text", "") for seg in aligned_segments))

        def verified(evidence: Optional[str]) -> bool:
            quote = self._normalize_for_match(evidence or "")
            return len(quote.split()) >= 2 and quote in spoken

        kept_actions = []
        for item in intelligence.action_items:
            if verified(item.evidence):
                kept_actions.append(item)
            else:
                logger.warning(f"Dropping action item without verbatim evidence: {item.task!r} (evidence={item.evidence!r})")
        intelligence.action_items = kept_actions

        kept_decisions = []
        for item in intelligence.decisions:
            if verified(item.evidence):
                kept_decisions.append(item)
            else:
                logger.warning(f"Dropping decision without verbatim evidence: {item.decision!r} (evidence={item.evidence!r})")
        intelligence.decisions = kept_decisions

        for item in intelligence.emotions:
            if not verified(item.evidence):
                item.confidence = 0.0
                item.evidence = None

        return intelligence

    @staticmethod
    def _derive_sentiment(intelligence: MeetingIntelligence, aligned_segments: List[Dict[str, Any]]) -> MeetingIntelligence:
        """Compute emotion distribution and sentiment from per-segment labels.

        Percentages are weighted by how long each segment was spoken, so the charts
        reflect what was actually said instead of numbers invented by the LLM.
        """
        labels = {se.segment: se.emotion for se in intelligence.segment_emotions if 0 <= se.segment < len(aligned_segments)}
        totals: Dict[str, float] = {}
        per_speaker: Dict[str, Dict[str, float]] = {}
        for idx, seg in enumerate(aligned_segments):
            emotion = labels.get(idx, "neutral")
            weight = max(float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)), 0.1)
            totals[emotion] = totals.get(emotion, 0.0) + weight
            spk = per_speaker.setdefault(seg.get("speaker", "UNKNOWN"), {})
            spk[emotion] = spk.get(emotion, 0.0) + weight

        grand = sum(totals.values()) or 1.0
        breakdown = {e: round(100.0 * totals[e] / grand, 1) for e in EMOTION_TAXONOMY if totals.get(e)}
        valence = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for e, pct in breakdown.items():
            valence[EMOTION_VALENCE[e]] += pct

        if valence["negative"] >= 40:
            overall = "negative"
        elif valence["negative"] >= 20:
            overall = "concerned"
        elif valence["positive"] > valence["negative"] and valence["positive"] >= 20:
            overall = "positive"
        else:
            overall = "neutral"

        moments = intelligence.sentiment_analysis.key_emotional_moments if intelligence.sentiment_analysis else []
        intelligence.sentiment_analysis = SentimentAnalysis(
            overall_sentiment=overall,
            positive_percentage=round(valence["positive"], 1),
            neutral_percentage=round(valence["neutral"], 1),
            negative_percentage=round(valence["negative"], 1),
            emotions_breakdown=breakdown,
            key_emotional_moments=moments,
        )

        # A speaker's dominant emotion is what they expressed most, by speaking time.
        by_speaker = {e.speaker: e for e in intelligence.emotions}
        for spk, dist in per_speaker.items():
            dominant = max(dist, key=dist.get)
            if spk in by_speaker:
                by_speaker[spk].dominant_emotion = dominant
            else:
                intelligence.emotions.append(EmotionItem(speaker=spk, dominant_emotion=dominant, confidence=0.0))
        return intelligence

    def _parse_response(self, raw_text: str, aligned_segments: List[Dict[str, Any]]) -> MeetingIntelligence:
        """Parse Gemini JSON, validate against the schema, and apply grounding checks."""
        cleaned_json_text = self.clean_json_response(raw_text)
        try:
            data = json.loads(cleaned_json_text)
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON decode failed on Gemini output: {raw_text}")
            raise LLMIntelligenceError(f"Malformed JSON returned by Gemini: {json_err}")

        intelligence = MeetingIntelligence(**data)
        intelligence = self._verify_evidence(intelligence, aligned_segments)
        return self._derive_sentiment(intelligence, aligned_segments)

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

        emotion_choices = " | ".join(EMOTION_TAXONOMY)
        emotion_list = ", ".join(EMOTION_TAXONOMY)

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
            "timestamp": "Timestamp mm:ss or null",
            "evidence": "Direct quote from the transcript supporting this decision",
            "confidence": 0.95
        }}
    ],
    "action_items": [
        {{
            "task": "Explicit action item or deliverable",
            "owner": "Person responsible (use known employee name if matched, else speaker label)",
            "deadline": "Deadline or target date (e.g. Friday, Next sprint, or 'Not specified')",
            "priority": "High | Medium | Low",
            "status": "pending",
            "evidence": "Direct quote from the transcript where this task was assigned",
            "confidence": 0.9
        }}
    ],
    "keywords": [
        {{
            "keyword": "Entity or Term",
            "category": "product | client | risk | budget | hiring | revenue | deadline | deliverable | custom",
            "relevance_score": 0.95
        }}
    ],
    "segment_emotions": [
        {{"segment": 0, "emotion": "{emotion_choices}", "intensity": 0.6}}
    ],
    "emotions": [
        {{
            "speaker": "Speaker label exactly as written in the transcript",
            "dominant_emotion": "{emotion_choices}",
            "speaking_style": "assertive | collaborative | hesitant | formal | authoritative",
            "confidence": 0.9,
            "evidence": "Verbatim words from this speaker that show the emotion"
        }}
    ],
    "sentiment_analysis": {{
        "key_emotional_moments": ["[#segment] what happened, e.g. '[#14] SPEAKER_01 frustrated about the delayed release'"]
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
1. SPEAKERS: Refer to speakers exactly as labelled in the transcript (e.g. "SPEAKER_01", "Unknown Speaker", or a name). Never rename a speaker label to a person's name, and never attribute words to someone who does not appear as a label.
2. Map tasks and owners to the speaker who committed to or was assigned the task. If the owner is not stated, use "Unassigned".
3. EVIDENCE REQUIREMENT: For every Decision and Action Item, "evidence" MUST be copied word-for-word from the transcript text (at least a few words, without the [#n mm:ss] prefix). Items whose evidence cannot be found in the transcript are automatically discarded, so do not paraphrase and do not invent items that were not said.
4. CONFIDENCE SCORING: Assign a confidence score (0.0-1.0). Use 1.0 for verbatim commitments, lower for inferred agreements.
5. EMOTIONS: Return one "segment_emotions" entry for EVERY numbered segment [#n]. Choose exactly one of: {emotion_list}. Judge from the words and context: e.g. complaints about blockers or repeated problems -> frustration; worry about deadlines or risk -> anxiety; low, hopeless or discouraged remarks -> sadness; unmet expectations -> disappointment; praise, laughter or good news -> joy; plain status updates -> neutral. Do not default to neutral when the words clearly express an emotion, and do not invent emotion when the words are plain.
6. Ensure every speaker label in the transcript has an entry in employee_reports, emotions, and accents.
7. Categorize keywords into appropriate business buckets: product, client, risk, budget, hiring, revenue, deadline, deliverable, custom.
8. Output ONLY valid JSON with no extraneous markdown or conversational filler.

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

            intelligence = self._parse_response(response.text, aligned_segments)

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
                    intelligence = self._parse_response(response.text, aligned_segments)

                    self._client = alt_model
                    self.model_name = candidate_name
                    return intelligence
                except Exception as cand_err:
                    last_exception = cand_err

            raise LLMIntelligenceError(f"Gemini API request failed across all model candidates: {last_exception}")
