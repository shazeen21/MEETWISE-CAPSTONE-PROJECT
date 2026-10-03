"""Master Pipeline Orchestrator for MeetWise AI Enterprise Platform.

Orchestrates the 10-phase enterprise processing pipeline:
1. Audio/Video Ingestion, Container Demuxing (PyAV), & Denoising
2. WhisperX Speech Recognition with Word-Level Alignment
3. pyannote.audio Multi-Speaker Diarization
4. Temporal Intersection Speaker Alignment
5. Acoustic Voice Biometrics & Enrolled Employee Recognition
6. Gemini Enterprise Meeting Intelligence (MoM, Decisions, Priorities, Topics, Keywords, Emotions, Accents)
7. Relational Persistence (PostgreSQL/SQLite via atomic repository)
8. Dense Semantic Chunking, BAAI BGE Embeddings & ChromaDB Indexing
9. Minutes of Meeting (MOM) Corporate PDF Generation
10. Multi-Channel Notification Fan-out (Internal, Email, Slack, Teams)
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
import soundfile as sf
import numpy as np

from .config import settings
from .audio.preprocessor import AudioPreprocessor
from .transcription.whisperx_service import WhisperXService
from .diarization.pyannote_service import PyannoteService
from .diarization.aligner import TranscriptAligner
from .meeting_intelligence.gemini_service import GeminiIntelligenceService
from .meeting_intelligence.schemas import MeetingIntelligence
from .embeddings.chunker import TranscriptChunker
from .embeddings.bge_service import BGEEmbeddingService
from .vector_store.chroma_service import ChromaService
from .database.repository import MeetingRepository
from .speakers.voice_biometrics import VoiceBiometricsService, SpeakerRecognitionEngine
from .reports.pdf_generator import MOMPDFGenerator
from .notifications.dispatcher import NotificationDispatcher

logger = logging.getLogger(__name__)


class PipelineExecutionError(Exception):
    """Exception raised when an error occurs during pipeline execution."""
    pass


class MeetWisePipeline:
    """Master Orchestrator for MeetWise AI Enterprise Meeting Processing."""

    def __init__(
        self,
        audio_preprocessor: Optional[AudioPreprocessor] = None,
        whisperx_service: Optional[WhisperXService] = None,
        pyannote_service: Optional[PyannoteService] = None,
        aligner: Optional[TranscriptAligner] = None,
        gemini_service: Optional[GeminiIntelligenceService] = None,
        chunker: Optional[TranscriptChunker] = None,
        bge_service: Optional[BGEEmbeddingService] = None,
        chroma_service: Optional[ChromaService] = None,
        speaker_recognition_engine: Optional[SpeakerRecognitionEngine] = None,
        pdf_generator: Optional[MOMPDFGenerator] = None,
    ):
        # 1. Audio/Video Preprocessing & Denoising
        self.preprocessor = audio_preprocessor or AudioPreprocessor(
            denoiser_type=settings.DENOISER_TYPE
        )

        # 2. Transcription
        self.whisperx = whisperx_service or WhisperXService(
            model_name=settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )

        # 3. Diarization
        self.pyannote = pyannote_service or PyannoteService(
            hf_token=settings.HF_TOKEN,
            device=settings.WHISPER_DEVICE,
        )

        # 4. Alignment
        self.aligner = aligner or TranscriptAligner()

        # 5. Speaker Biometrics & Recognition
        self.speaker_recognition = speaker_recognition_engine or SpeakerRecognitionEngine()

        # 6. Gemini Meeting Intelligence
        self.gemini = gemini_service or GeminiIntelligenceService(
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_MODEL,
        )

        # 7. Semantic Chunking & Vector Store
        self.chunker = chunker or TranscriptChunker()
        self.bge = bge_service or BGEEmbeddingService(model_name=settings.BGE_MODEL)
        self.chroma = chroma_service or ChromaService(persist_dir=settings.CHROMA_PERSIST_DIR)

        # 8. Document Generation
        self.pdf_generator = pdf_generator or MOMPDFGenerator()

    def _extract_speaker_audio_slices(
        self,
        audio_path: str,
        aligned_segments: List[Dict[str, Any]],
        sample_rate: int = 16000,
    ) -> Dict[str, np.ndarray]:
        """Group and slice audio samples by speaker label for biometric matching."""
        try:
            wav_data, sr = sf.read(audio_path, dtype="float32")
            if wav_data.ndim > 1:
                wav_data = wav_data.mean(axis=1)

            speaker_slices: Dict[str, List[np.ndarray]] = {}
            for seg in aligned_segments:
                spk = seg.get("speaker", "UNKNOWN")
                start_sample = int(float(seg.get("start", 0.0)) * sr)
                end_sample = int(float(seg.get("end", 0.0)) * sr)

                if end_sample > start_sample and start_sample < len(wav_data):
                    slice_data = wav_data[start_sample:min(end_sample, len(wav_data))]
                    if len(slice_data) > 0:
                        speaker_slices.setdefault(spk, []).append(slice_data)

            # Concatenate audio per speaker up to 30 seconds for robust biometrics
            result = {}
            for spk, chunks in speaker_slices.items():
                merged = np.concatenate(chunks)
                # Keep max 30 seconds (480,000 samples)
                if len(merged) > 480000:
                    merged = merged[:480000]
                result[spk] = merged

            return result
        except Exception as e:
            logger.warning(f"Failed to slice speaker audio for biometrics: {e}")
            return {}

    def process_meeting(
        self,
        audio_path: str,
        db: Session,
        meeting_id: Optional[str] = None,
        title: Optional[str] = None,
        source_type: str = "uploaded",
    ) -> Dict[str, Any]:
        """Execute the complete enterprise AI pipeline from raw audio/video to PostgreSQL, ChromaDB, and PDF."""
        meeting_id = meeting_id or str(uuid.uuid4())
        media_file_name = Path(audio_path).name
        logger.info(f"=== Starting MeetWise Enterprise Pipeline for Meeting {meeting_id} ({media_file_name}) ===")
        repo = MeetingRepository(db)

        # ----------------------------------------------------
        # Phase 1: Ingestion, Demuxing (Video/Audio) & Denoising
        # ----------------------------------------------------
        logger.info("[1/10] Ingesting media, demuxing streams, and cleaning audio...")
        try:
            preprocess_res = self.preprocessor.process(
                input_path=audio_path,
                output_dir=settings.PROCESSED_DIR,
            )
            clean_audio = preprocess_res["clean_audio_path"]
            duration_seconds = preprocess_res["metadata"].get("duration_seconds", 0.0)
            media_type = preprocess_res.get("media_type", "audio")
            logger.info(f"Preprocessed {media_type} successfully (Duration: {duration_seconds}s).")
        except Exception as e:
            raise PipelineExecutionError(f"Media preprocessing failed: {e}")

        # ----------------------------------------------------
        # Phase 2: WhisperX Speech Recognition
        # ----------------------------------------------------
        logger.info("[2/10] Transcribing speech with WhisperX / faster-whisper...")
        try:
            asr_result = self.whisperx.transcribe(clean_audio)
            raw_segments = asr_result.get("segments", [])
            logger.info(f"Transcribed {len(raw_segments)} speech segments.")
        except Exception as e:
            raise PipelineExecutionError(f"Speech transcription failed: {e}")

        # ----------------------------------------------------
        # Phase 3: pyannote.audio Multi-Speaker Diarization
        # ----------------------------------------------------
        logger.info("[3/10] Running neural speaker diarization...")
        try:
            speaker_turns = self.pyannote.diarize(clean_audio)
            logger.info(f"pyannote detected {len(speaker_turns)} speaker turns.")
        except Exception as e:
            logger.warning(f"pyannote diarization failed or was skipped: {e}. Falling back to default speaker.")
            speaker_turns = []

        # ----------------------------------------------------
        # Phase 4: Temporal Intersection Alignment
        # ----------------------------------------------------
        logger.info("[4/10] Aligning word timestamps and speaker turns...")
        try:
            aligned_segments = self.aligner.align(raw_segments, speaker_turns)
            logger.info(f"Aligned {len(aligned_segments)} speech turns.")
        except Exception as e:
            raise PipelineExecutionError(f"Transcript alignment failed: {e}")

        # ----------------------------------------------------
        # Phase 5: Voice Biometrics & Speaker Recognition
        # ----------------------------------------------------
        logger.info("[5/10] Identifying speakers against enrolled employee voiceprints...")
        speaker_identifications: Dict[str, Dict[str, Any]] = {}
        try:
            enrolled_emps = repo.get_all_enrolled_employees()
            enrolled_data = [
                {
                    "id": emp.id,
                    "employee_id": emp.employee_id,
                    "name": emp.name,
                    "department": emp.department,
                    "voice_embedding": emp.voice_embedding,
                }
                for emp in enrolled_emps
            ]

            speaker_audio_slices = self._extract_speaker_audio_slices(clean_audio, aligned_segments)
            unique_spks = sorted(list(set(s.get("speaker") for s in aligned_segments if "speaker" in s)))

            for spk_label in unique_spks:
                spk_audio = speaker_audio_slices.get(spk_label)
                if spk_audio is not None and len(spk_audio) > 800:
                    id_res = self.speaker_recognition.identify_speaker(spk_audio, enrolled_data)
                else:
                    id_res = {
                        "name": "Unknown Speaker",
                        "employee_id": None,
                        "confidence": 75.0,
                        "is_unknown": True,
                    }
                speaker_identifications[spk_label] = id_res
                logger.info(
                    f"Speaker '{spk_label}' identified as '{id_res['name']}' "
                    f"(Confidence: {id_res['confidence']}%)"
                )

            # Update speaker names in aligned_segments if identified
            for seg in aligned_segments:
                raw_spk = seg.get("speaker")
                if raw_spk in speaker_identifications:
                    resolved = speaker_identifications[raw_spk]["name"]
                    if resolved != "Unknown Speaker":
                        seg["speaker"] = resolved

        except Exception as bio_err:
            logger.warning(f"Speaker recognition warning: {bio_err}. Proceeding with diarization labels.")

        # Save merged transcript JSON
        transcript_json_path = str(Path(settings.TRANSCRIPTS_DIR) / f"{meeting_id}.json")
        try:
            self.aligner.save_transcript_json(aligned_segments, transcript_json_path)
        except Exception as e:
            logger.warning(f"Could not save transcript JSON: {e}")

        # ----------------------------------------------------
        # Phase 6: Gemini Enterprise Meeting Intelligence
        # ----------------------------------------------------
        logger.info("[6/10] Extracting enterprise intelligence with Gemini...")
        known_names = [e["name"] for e in enrolled_data] if "enrolled_data" in locals() else []
        configured_kws = [k.keyword for k in repo.list_configured_keywords()]

        try:
            intelligence = self.gemini.analyze(
                aligned_segments=aligned_segments,
                known_employees=known_names,
                custom_keywords=configured_kws,
            )
        except Exception as e:
            logger.warning(f"Gemini intelligence extraction error: {e}. Building fallback intelligence.")
            inferred_title = title or f"Meeting on {media_file_name}"
            intelligence = MeetingIntelligence(
                title=inferred_title,
                summary="Meeting recording processed. Automatic LLM summarization generated a fallback brief.",
                agenda=["General Discussion"],
                discussion_points=["Reviewed meeting recording items."],
                topics=[],
                decisions=[],
                action_items=[],
                keywords=[],
                emotions=[],
                accents=[],
                employee_reports=[],
                pending_issues=[],
                next_steps=[],
            )

        meeting_title = title or intelligence.title

        # Enforce emotion and style assignments on transcript segments
        emotion_map = {e.speaker: e.dominant_emotion for e in intelligence.emotions}
        style_map = {e.speaker: e.speaking_style for e in intelligence.emotions}
        accent_map = {a.speaker: a.detected_accent for a in intelligence.accents}

        for seg in aligned_segments:
            spk_name = seg.get("speaker")
            seg["emotion"] = emotion_map.get(spk_name, "neutral")
            seg["speaking_style"] = style_map.get(spk_name, "professional")

        for spk_lbl, id_info in speaker_identifications.items():
            spk_name = id_info.get("name", spk_lbl)
            if spk_name in accent_map:
                id_info["accent"] = accent_map[spk_name]
            if spk_name in emotion_map:
                id_info["dominant_emotion"] = emotion_map[spk_name]

        # ----------------------------------------------------
        # Phase 7: PostgreSQL / SQLite Relational Persistence
        # ----------------------------------------------------
        logger.info("[7/10] Persisting enterprise meeting records to database...")
        try:
            meeting_record = repo.save_meeting_pipeline_results(
                meeting_id=meeting_id,
                title=meeting_title,
                audio_file_name=media_file_name,
                duration_seconds=duration_seconds,
                source_type=source_type,
                aligned_segments=aligned_segments,
                intelligence=intelligence,
                speaker_identifications=speaker_identifications,
                media_type=media_type,
            )
        except Exception as e:
            raise PipelineExecutionError(f"Database persistence failed: {e}")

        # ----------------------------------------------------
        # Phase 8: Semantic Chunking & ChromaDB Vector Indexing
        # ----------------------------------------------------
        logger.info("[8/10] Indexing transcripts and meeting intelligence into ChromaDB RAG...")
        try:
            chunks = self.chunker.chunk_transcript(
                meeting_id=meeting_id,
                meeting_title=meeting_title,
                aligned_segments=aligned_segments,
            )
            if chunks:
                chunk_texts = [c.text for c in chunks]
                embeddings = self.bge.embed_documents(chunk_texts)
                self.chroma.add_chunks(chunks=chunks, embeddings=embeddings)
                logger.info(f"Indexed {len(chunks)} transcript chunks in ChromaDB.")
        except Exception as e:
            logger.warning(f"Vector store indexing warning: {e}")

        # ----------------------------------------------------
        # Phase 9: Minutes of Meeting (MOM) PDF Generation
        # ----------------------------------------------------
        logger.info("[9/10] Generating downloadable Minutes of Meeting (MOM) PDF...")
        pdf_path = None
        try:
            mom_payload = {
                "id": meeting_id,
                "title": meeting_title,
                "meeting_date": str(meeting_record.meeting_date),
                "duration_seconds": duration_seconds,
                "audio_file_name": media_file_name,
                "media_type": media_type,
                "summary": intelligence.summary,
                "speakers": [
                    {
                        "speaker_label": s.speaker_label,
                        "speaker_name": s.speaker_name,
                        "confidence_score": s.confidence_score,
                        "total_speaking_time": s.total_speaking_time,
                        "detected_accent": s.detected_accent or "International English",
                    }
                    for s in meeting_record.speakers
                ],
                "decisions": [d.model_dump() for d in intelligence.decisions],
                "action_items": [a.model_dump() for a in intelligence.action_items],
                "pending_issues": intelligence.pending_issues,
            }
            pdf_path = self.pdf_generator.generate(mom_payload)
            logger.info(f"MOM PDF generated at: {pdf_path}")
        except Exception as pdf_err:
            logger.warning(f"MOM PDF generation warning: {pdf_err}")

        # ----------------------------------------------------
        # Phase 10: Multi-Channel Notification Fan-Out
        # ----------------------------------------------------
        logger.info("[10/10] Dispatching participant summaries and manager alerts...")
        try:
            dispatcher = NotificationDispatcher(repository=repo)
            participants = [s.speaker_name for s in meeting_record.speakers if s.speaker_name and s.speaker_name != "Unknown Speaker"]
            dispatcher.dispatch_meeting_summary(
                meeting_id=meeting_id,
                meeting_title=meeting_title,
                intelligence_data=intelligence.model_dump(),
                participant_names=participants or ["Organizer"],
            )
        except Exception as notif_err:
            logger.warning(f"Notification dispatch warning: {notif_err}")

        logger.info(f"=== Successfully completed enterprise pipeline for Meeting {meeting_id} ===")
        return {
            "meeting_id": meeting_id,
            "title": meeting_title,
            "duration_seconds": duration_seconds,
            "media_type": media_type,
            "source_type": source_type,
            "audio_file_name": media_file_name,
            "pdf_report_path": pdf_path,
            "transcript_json_path": transcript_json_path,
            "intelligence": intelligence.model_dump(),
            "speaker_count": len(meeting_record.speakers),
            "transcript_segment_count": len(aligned_segments),
            "chroma_chunks_indexed": len(chunks) if "chunks" in locals() and chunks else 0,
        }
