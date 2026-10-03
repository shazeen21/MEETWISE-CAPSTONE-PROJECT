"""Audio & Video Ingestion and Preprocessing Module for MeetWise AI.

Handles audio and video validation, demuxing from video containers (MP4, WebM),
format conversion to 16kHz mono PCM WAV via PyAV/SoundFile, metadata extraction,
and applies noise gating / speech enhancement.
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import soundfile as sf
import numpy as np

from .denoiser import get_denoiser, BaseAudioDenoiser

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
SUPPORTED_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
TARGET_SAMPLE_RATE = 16000


class AudioPreprocessingError(Exception):
    """Exception raised when audio/video preprocessing fails."""
    pass


class AudioPreprocessor:
    """Standardizes input audio/video for WhisperX and pyannote.audio."""

    def __init__(self, denoiser: Optional[BaseAudioDenoiser] = None, denoiser_type: str = "spectral"):
        self.denoiser = denoiser or get_denoiser(denoiser_type)

    @staticmethod
    def validate_file(file_path: str) -> None:
        """Validate that file exists and has a supported extension."""
        path = Path(file_path)
        if not path.exists():
            raise AudioPreprocessingError(f"Media file not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise AudioPreprocessingError(
                f"Unsupported format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """Check whether input file is a video container."""
        return Path(file_path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS

    @staticmethod
    def _decode_with_pyav(file_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
        """Demux and decode audio stream from audio or video file using PyAV."""
        try:
            import av

            container = av.open(file_path)
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if not audio_stream:
                raise AudioPreprocessingError(f"No audio stream found in container: {file_path}")

            resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
            chunks = []

            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    resampled = resampler.resample(frame)
                    for rf in resampled:
                        # shape (1, samples)
                        chunks.append(rf.to_ndarray().flatten())

            container.close()

            if not chunks:
                raise AudioPreprocessingError(f"No audio samples decoded from {file_path}")

            raw_int16 = np.concatenate(chunks)
            audio_float32 = raw_int16.astype(np.float32) / 32768.0
            return audio_float32

        except Exception as e:
            raise AudioPreprocessingError(f"PyAV audio decoding failed for {file_path}: {e}")

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract duration, sample rate, channels, file size, and media type."""
        path = Path(file_path)
        file_size = path.stat().st_size
        is_video = self.is_video_file(file_path)
        media_type = "video" if is_video else "audio"

        # If it's a video file, decode metadata via PyAV
        if is_video:
            try:
                import av
                container = av.open(file_path)
                duration_sec = float(container.duration / av.time_base) if container.duration else 0.0
                audio_stream = next((s for s in container.streams if s.type == "audio"), None)
                sr = audio_stream.rate if audio_stream else TARGET_SAMPLE_RATE
                channels = audio_stream.channels if audio_stream else 1
                container.close()
                return {
                    "file_name": path.name,
                    "file_size_bytes": file_size,
                    "duration_seconds": round(duration_sec, 2),
                    "sample_rate": sr,
                    "channels": channels,
                    "format": path.suffix.lstrip(".").upper(),
                    "media_type": media_type,
                }
            except Exception as av_err:
                logger.warning(f"PyAV metadata extraction warning: {av_err}")

        # Try standard soundfile info for audio
        try:
            info = sf.info(file_path)
            return {
                "file_name": path.name,
                "file_size_bytes": file_size,
                "duration_seconds": round(info.duration, 2),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "format": info.format,
                "media_type": media_type,
            }
        except Exception:
            try:
                audio_arr = self._decode_with_pyav(file_path)
                dur = len(audio_arr) / float(TARGET_SAMPLE_RATE)
                return {
                    "file_name": path.name,
                    "file_size_bytes": file_size,
                    "duration_seconds": round(dur, 2),
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "channels": 1,
                    "format": path.suffix.lstrip(".").upper(),
                    "media_type": media_type,
                }
            except Exception as e:
                logger.warning(f"Could not read full metadata: {e}")
                return {
                    "file_name": path.name,
                    "file_size_bytes": file_size,
                    "duration_seconds": 0.0,
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "channels": 1,
                    "format": path.suffix.lstrip(".").upper(),
                    "media_type": media_type,
                }

    def convert_to_16k_mono_wav(self, input_path: str, output_path: str) -> str:
        """Convert any audio/video file to 16kHz mono 16-bit PCM WAV."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        is_video = self.is_video_file(input_path)

        # 1. For video containers or any complex formats, use native PyAV demuxer
        if is_video:
            try:
                logger.info(f"Extracting and demuxing audio from video container: {input_path}")
                audio_data = self._decode_with_pyav(input_path, target_sr=TARGET_SAMPLE_RATE)
                sf.write(output_path, audio_data, TARGET_SAMPLE_RATE, subtype="PCM_16")
                logger.info(f"Successfully extracted video audio -> {output_path}")
                return output_path
            except Exception as av_err:
                logger.warning(f"PyAV video extraction failed: {av_err}. Trying fallback...")

        # 2. Try ffmpeg CLI if available
        ffmpeg_cmd = shutil.which("ffmpeg")
        if ffmpeg_cmd:
            cmd = [
                ffmpeg_cmd,
                "-y",
                "-i", input_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(TARGET_SAMPLE_RATE),
                "-ac", "1",
                output_path,
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                logger.info(f"FFmpeg converted {input_path} -> {output_path}")
                return output_path
            except subprocess.CalledProcessError as e:
                logger.warning(f"FFmpeg conversion failed: {e.stderr.decode('utf-8', errors='ignore')}")

        # 3. Try soundfile + scipy resample for standard audio
        try:
            data, sr = sf.read(input_path, dtype="float32")
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            if sr != TARGET_SAMPLE_RATE:
                from scipy.signal import resample
                num_target_samples = int(len(data) * float(TARGET_SAMPLE_RATE) / sr)
                data = resample(data, num_target_samples).astype(np.float32)

            sf.write(output_path, data, TARGET_SAMPLE_RATE, subtype="PCM_16")
            logger.info(f"soundfile/scipy converted {input_path} -> {output_path}")
            return output_path
        except Exception as sf_err:
            logger.info(f"soundfile audio read failed ({sf_err}), trying PyAV decoder...")

        # 4. PyAV fallback for any remaining audio formats
        try:
            audio_data = self._decode_with_pyav(input_path, target_sr=TARGET_SAMPLE_RATE)
            sf.write(output_path, audio_data, TARGET_SAMPLE_RATE, subtype="PCM_16")
            logger.info(f"PyAV converted {input_path} -> {output_path}")
            return output_path
        except Exception as e:
            raise AudioPreprocessingError(f"Failed to convert media file to 16kHz mono WAV: {e}")

    def process(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """Execute complete preprocessing pipeline: validate -> metadata -> standardize -> denoise."""
        self.validate_file(input_path)
        metadata = self.extract_metadata(input_path)

        input_filename = Path(input_path).stem
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        standardized_path = str(out_dir / f"{input_filename}_16k_mono.wav")
        clean_path = str(out_dir / f"{input_filename}_clean.wav")

        # Step 1: Standardize to 16kHz mono WAV (extracting audio if video)
        self.convert_to_16k_mono_wav(input_path, standardized_path)

        # Step 2: Apply denoising/enhancement
        self.denoiser.enhance(standardized_path, clean_path)

        # Recompute accurate duration from cleaned audio
        clean_meta = self.extract_metadata(clean_path)
        metadata["duration_seconds"] = clean_meta.get("duration_seconds", metadata.get("duration_seconds", 0.0))

        return {
            "metadata": metadata,
            "raw_audio_path": input_path,
            "standardized_audio_path": standardized_path,
            "clean_audio_path": clean_path,
            "media_type": metadata.get("media_type", "audio"),
        }
