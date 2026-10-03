"""Speaker Diarization Service using pyannote.audio.

Uses Hugging Face token authentication to run pyannote speaker diarization pipelines.
Identifies distinct speakers (SPEAKER_00, SPEAKER_01...) and their temporal active intervals.
Preloads audio into memory via SoundFile to eliminate FFmpeg and TorchCodec binary dependencies.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DiarizationError(Exception):
    """Exception raised when speaker diarization fails."""
    pass


class PyannoteService:
    """Encapsulates pyannote.audio diarization pipeline."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        pipeline_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "cpu",
    ):
        self.hf_token = hf_token
        self.pipeline_name = pipeline_name
        self.device = device
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy load the pyannote pipeline."""
        if self._pipeline is None:
            if not self.hf_token:
                raise DiarizationError(
                    "HF_TOKEN is missing. Hugging Face token is required for pyannote.audio. "
                    "Please set HF_TOKEN in your environment or .env file and accept the model terms on Hugging Face."
                )

            try:
                from pyannote.audio import Pipeline

                logger.info(f"Loading pyannote pipeline '{self.pipeline_name}'...")
                self._pipeline = Pipeline.from_pretrained(
                    self.pipeline_name,
                    token=self.hf_token,
                )

                import torch
                if self.device == "cuda" and torch.cuda.is_available():
                    self._pipeline.to(torch.device("cuda"))
                    logger.info("pyannote pipeline moved to CUDA.")
                else:
                    self._pipeline.to(torch.device("cpu"))
                    logger.info("pyannote pipeline running on CPU.")

            except ImportError:
                raise DiarizationError(
                    "pyannote.audio is not installed. Please install it via 'pip install pyannote.audio'."
                )
            except Exception as e:
                raise DiarizationError(
                    f"Failed to load pyannote pipeline '{self.pipeline_name}': {e}. "
                    "Ensure your HF_TOKEN has access to the pyannote model on Hugging Face."
                )
        return self._pipeline

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run speaker diarization on audio file.

        Args:
            audio_path: Path to preprocessed 16kHz mono audio file.
            num_speakers: Exact number of speakers if known.
            min_speakers: Minimum number of speakers.
            max_speakers: Maximum number of speakers.

        Returns:
            List of dicts:
                [
                    {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.5},
                    ...
                ]
        """
        pipeline = self._get_pipeline()

        params = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers

        try:
            logger.info(f"Running pyannote diarization on: {audio_path}...")
            # Preload in-memory waveform via soundfile to avoid torchcodec / ffmpeg dll issues
            try:
                import soundfile as sf
                data, sr = sf.read(audio_path, dtype="float32")
                if data.ndim == 1:
                    waveform = torch.from_numpy(data).unsqueeze(0)
                else:
                    waveform = torch.from_numpy(data.T)
                audio_input = {"waveform": waveform, "sample_rate": sr}
                diarization_output = pipeline(audio_input, **params)
            except Exception as io_err:
                logger.warning(f"In-memory audio load for pyannote failed ({io_err}), falling back to filepath...")
                diarization_output = pipeline(audio_path, **params)

            speaker_turns: List[Dict[str, Any]] = []

            # pyannote diarization output iterator
            for turn, _, speaker in diarization_output.itertracks(yield_label=True):
                speaker_turns.append({
                    "speaker": str(speaker),
                    "start": round(float(turn.start), 2),
                    "end": round(float(turn.end), 2),
                })

            # Sort speaker turns chronologically
            speaker_turns.sort(key=lambda x: x["start"])
            logger.info(f"Diarization finished with {len(speaker_turns)} turns.")
            return speaker_turns

        except Exception as e:
            raise DiarizationError(f"pyannote diarization execution failed: {e}")
