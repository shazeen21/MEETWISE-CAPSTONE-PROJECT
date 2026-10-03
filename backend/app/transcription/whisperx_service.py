"""WhisperX Speech Recognition Service for MeetWise AI.

Loads WhisperX models with configurable size, device (CPU/GPU), and compute precision.
Transcribes audio into timestamped segments and runs phoneme alignment.
Includes native SoundFile audio loading (zero FFmpeg dependency) and faster-whisper fallback.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Exception raised when speech recognition fails."""
    pass


def _load_audio_array(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file as a float32 1D numpy array normalized to target sample rate.
    
    Uses soundfile directly so that external ffmpeg binaries are not required on Windows.
    """
    try:
        import soundfile as sf
        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            import scipy.signal
            num_samples = int(len(wav) * target_sr / sr)
            wav = scipy.signal.resample(wav, num_samples).astype(np.float32)
        return wav.astype(np.float32)
    except Exception as sf_err:
        logger.warning(f"SoundFile audio loader failed: {sf_err}. Trying whisperx loader...")
        try:
            import whisperx
            return whisperx.load_audio(audio_path)
        except Exception as wx_err:
            raise TranscriptionError(f"Could not load audio file '{audio_path}': {sf_err}; {wx_err}")


class WhisperXService:
    """Encapsulates WhisperX ASR loading, transcription, and alignment."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self._model = None
        self._align_model = None
        self._align_metadata = None

    def _get_model(self):
        """Lazy load the WhisperX model."""
        if self._model is None:
            try:
                import whisperx

                logger.info(
                    f"Loading WhisperX model '{self.model_name}' on device '{self.device}' "
                    f"with compute_type '{self.compute_type}'..."
                )
                self._model = whisperx.load_model(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info("WhisperX model loaded successfully.")
            except ImportError:
                logger.warning("WhisperX import failed. Will attempt faster-whisper fallback.")
                self._model = None
            except Exception as e:
                logger.warning(f"Failed to load WhisperX model: {e}. Will attempt faster-whisper fallback.")
                self._model = None
        return self._model

    def _fallback_faster_whisper(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fallback transcription using faster-whisper directly."""
        logger.info("Using faster-whisper direct fallback...")
        try:
            from faster_whisper import WhisperModel

            fw_model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            segments_gen, info = fw_model.transcribe(audio, language=language)
            cleaned_segments: List[Dict[str, Any]] = []

            for seg in segments_gen:
                text = seg.text.strip()
                if not text:
                    continue
                cleaned_segments.append({
                    "start": round(float(seg.start), 2),
                    "end": round(float(seg.end), 2),
                    "text": text,
                })

            detected_lang = getattr(info, "language", language or "en")
            logger.info(f"faster-whisper completed with {len(cleaned_segments)} segments.")
            return {
                "language": detected_lang,
                "segments": cleaned_segments,
            }
        except Exception as e:
            raise TranscriptionError(f"Both WhisperX and faster-whisper fallback failed: {e}")

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        align: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe an audio file using WhisperX (with faster-whisper fallback).

        Args:
            audio_path: Path to preprocessed 16kHz mono audio file.
            language: Language code (default: 'en').
            align: Whether to run word-level alignment.

        Returns:
            Dictionary containing:
                - language: Detected/specified language.
                - segments: List of dicts with 'start', 'end', 'text', and optional 'words'.
        """
        try:
            # 1. Load audio safely into float32 numpy array (no ffmpeg CLI dependency)
            audio = _load_audio_array(audio_path)

            # 2. Attempt WhisperX transcription
            model = self._get_model()
            if model is None:
                return self._fallback_faster_whisper(audio, language=language)

            import whisperx

            logger.info(f"Transcribing audio with WhisperX ({len(audio)} samples)...")
            result = model.transcribe(audio, batch_size=self.batch_size, language=language)

            raw_segments = result.get("segments", [])
            detected_lang = result.get("language", language)

            # 3. Optional Word-level Alignment
            aligned_segments = raw_segments
            if align and raw_segments:
                try:
                    logger.info(f"Aligning transcript segments for language '{detected_lang}'...")
                    align_model, metadata = whisperx.load_align_model(
                        language_code=detected_lang,
                        device=self.device,
                    )
                    alignment_res = whisperx.align(
                        raw_segments,
                        align_model,
                        metadata,
                        audio,
                        self.device,
                        return_char_alignments=False,
                    )
                    aligned_segments = alignment_res.get("segments", raw_segments)
                except Exception as align_err:
                    logger.warning(f"WhisperX alignment skipped due to warning/error: {align_err}")
                    aligned_segments = raw_segments

            # 4. Standardize output segments
            cleaned_segments: List[Dict[str, Any]] = []
            for seg in aligned_segments:
                start_time = round(float(seg.get("start", 0.0)), 2)
                end_time = round(float(seg.get("end", start_time)), 2)
                text = seg.get("text", "").strip()

                if not text:
                    continue

                item: Dict[str, Any] = {
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                }
                if "words" in seg:
                    item["words"] = seg["words"]

                cleaned_segments.append(item)

            logger.info(f"Transcription completed with {len(cleaned_segments)} segments.")
            return {
                "language": detected_lang,
                "segments": cleaned_segments,
            }

        except TranscriptionError:
            raise
        except Exception as e:
            logger.warning(f"WhisperX transcription encountered error: {e}. Trying fallback...")
            try:
                audio = _load_audio_array(audio_path)
                return self._fallback_faster_whisper(audio, language=language)
            except Exception as fb_err:
                raise TranscriptionError(f"WhisperX transcription failed: {e}; Fallback failed: {fb_err}")
