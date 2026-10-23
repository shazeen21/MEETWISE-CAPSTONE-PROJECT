"""Voice Biometrics and Speaker Recognition Service for MeetWise AI Enterprise.

Extracts neural voice embeddings using a pretrained ECAPA-TDNN model via SpeechBrain.
Performs cosine similarity matching with calibrated thresholds to identify speakers.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:  # speechbrain < 1.0
    from speechbrain.pretrained import EncoderClassifier
try:
    from speechbrain.utils.fetching import LocalStrategy
except ImportError:  # older speechbrain always symlinks
    LocalStrategy = None

logger = logging.getLogger(__name__)

# ECAPA-TDNN embeddings from SpeechBrain typically have 192 dimensions
EMBEDDING_DIM = 192
DEFAULT_CONFIDENCE_THRESHOLD = 0.25  # Calibrated cosine similarity threshold for ECAPA-TDNN
# Top match must beat the runner-up by this much, otherwise the speaker is ambiguous
DEFAULT_AMBIGUITY_MARGIN = 0.05
MIN_SPEECH_DURATION = 1.0  # Minimum seconds of speech required for robust embedding
TARGET_SAMPLE_RATE = 16000


class VoiceEmbeddingError(Exception):
    """Raised when a reliable voice embedding cannot be extracted from audio.

    Never substitute a placeholder vector: identical placeholders make every
    speaker look like every enrolled employee (cosine similarity 1.0).
    """
    pass

class VoiceBiometricsService:
    """Extracts neural voice embeddings using a pretrained ECAPA-TDNN model via SpeechBrain."""

    embedding_dim = EMBEDDING_DIM

    def __init__(self, model_path: str = "speechbrain/spkrec-ecapa-voxceleb"):
        self.model_path = model_path
        self._classifier = None

    def _get_classifier(self):
        """Lazy load the SpeechBrain EncoderClassifier."""
        if self._classifier is None:
            try:
                logger.info(f"Loading SpeechBrain speaker encoder: {self.model_path}...")
                # Pin to CPU: on Apple Silicon SpeechBrain otherwise picks MPS and
                # tensor.numpy() fails, which previously broke every extraction.
                # Copy instead of symlink: Windows refuses symlinks without admin or
                # Developer Mode (WinError 1314).
                extra = {}
                if LocalStrategy is not None:
                    extra["local_strategy"] = LocalStrategy.COPY
                self._classifier = EncoderClassifier.from_hparams(
                    source=self.model_path,
                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"},
                    **extra,
                )
                logger.info("Speaker encoder loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load SpeechBrain model: {e}")
                raise RuntimeError(f"Could not initialize speaker encoder: {e}")
        return self._classifier

    def _load_audio(
        self,
        audio: Union[str, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Load audio into a float32 numpy array and return with sample rate."""
        if isinstance(audio, np.ndarray):
            data = audio.astype(np.float32)
            if data.ndim > 1:
                data = data.mean(axis=1)
            sr = sample_rate or TARGET_SAMPLE_RATE
        else:
            try:
                data, sr = sf.read(str(audio), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
            except Exception:
                # libsndfile cannot read browser recordings (.webm/opus, .m4a, .mp3 on
                # some builds), so decode those with PyAV straight to 16 kHz mono.
                from ..audio.preprocessor import AudioPreprocessor
                data = AudioPreprocessor._decode_with_pyav(str(audio), target_sr=TARGET_SAMPLE_RATE)
                sr = TARGET_SAMPLE_RATE

        if sr != TARGET_SAMPLE_RATE:
            from scipy.signal import resample
            num_samples = int(len(data) * float(TARGET_SAMPLE_RATE) / sr)
            data = resample(data, num_samples).astype(np.float32)
            sr = TARGET_SAMPLE_RATE

        return data, sr

    def extract_embedding(
        self,
        audio: Union[str, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """Extract an L2-normalized neural voice embedding from speech audio.

        Raises:
            VoiceEmbeddingError: if the audio cannot be decoded, is too short,
                is silent, or the model fails. Callers must treat the speaker
                as unidentifiable rather than use a substitute vector.
        """
        try:
            samples, sr = self._load_audio(audio, sample_rate=sample_rate)
        except Exception as e:
            raise VoiceEmbeddingError(f"Could not decode audio: {e}") from e

        duration = len(samples) / sr
        if duration < MIN_SPEECH_DURATION:
            raise VoiceEmbeddingError(
                f"Audio too short ({duration:.2f}s < {MIN_SPEECH_DURATION}s) for a reliable voiceprint."
            )

        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < 1e-4:
            raise VoiceEmbeddingError("Audio is near-silent; no voice to fingerprint.")

        try:
            classifier = self._get_classifier()
            # SpeechBrain expects torch tensor [batch, time]
            tensor_audio = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                embeddings = classifier.encode_batch(tensor_audio)
            # embeddings shape: [batch, 1, dim] -> [dim]
            embedding = embeddings.squeeze().detach().cpu().numpy()
        except Exception as e:
            raise VoiceEmbeddingError(f"Neural embedding extraction failed: {e}") from e

        if embedding.shape != (EMBEDDING_DIM,) or not np.all(np.isfinite(embedding)):
            raise VoiceEmbeddingError(f"Model returned an invalid embedding of shape {embedding.shape}.")

        norm = np.linalg.norm(embedding)
        if norm < 1e-9:
            raise VoiceEmbeddingError("Model returned a zero embedding.")
        return (embedding / norm).astype(np.float32)

    def check_speech_like(self, audio: Union[str, np.ndarray], sample_rate: Optional[int] = None) -> None:
        """Reject enrollment audio that is not human speech (test tones, hum, beeps).

        A voiceprint built from a pure tone matches any other tone and none of the
        person's real speech. Speech spreads its energy over many harmonics; a tone
        puts nearly all of it in one narrow band.

        Raises:
            VoiceEmbeddingError: if the audio is a near-pure tone.
        """
        try:
            samples, sr = self._load_audio(audio, sample_rate=sample_rate)
        except Exception as e:
            raise VoiceEmbeddingError(f"Could not decode audio: {e}") from e
        power = np.abs(np.fft.rfft(samples)) ** 2
        total = power.sum()
        if total <= 0:
            raise VoiceEmbeddingError("Audio is silent.")
        freqs = np.fft.rfftfreq(len(samples), 1.0 / sr)
        peak = freqs[np.argmax(power)]
        band_share = power[np.abs(freqs - peak) <= 20.0].sum() / total
        if band_share > 0.5:
            raise VoiceEmbeddingError(
                f"Audio is a {peak:.0f} Hz tone ({band_share:.0%} of energy in one band), not human speech."
            )

    def aggregate_centroid(self, embeddings: List[Union[List[float], np.ndarray]]) -> List[float]:
        """Compute the L2-normalized centroid vector across multiple voice samples."""
        if not embeddings:
            return []

        arrays = [np.array(e, dtype=np.float32) for e in embeddings]
        stacked = np.stack(arrays, axis=0)
        mean_vec = np.mean(stacked, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-9:
            centroid = mean_vec / norm
        else:
            centroid = mean_vec

        return [round(float(x), 6) for x in centroid]


class SpeakerRecognitionEngine:
    """Matches diarized speaker audio segments against enrolled employee voiceprints."""

    def __init__(
        self,
        biometrics_service: Optional[VoiceBiometricsService] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    ):
        self.biometrics = biometrics_service or VoiceBiometricsService()
        self.confidence_threshold = confidence_threshold
        self.ambiguity_margin = ambiguity_margin

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @staticmethod
    def _unknown(reason: str, similarity: float = 0.0, matches: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return {
            "name": "Unknown Speaker",
            "employee_id": None,
            "confidence": round(float(similarity), 4),
            "is_unknown": True,
            "raw_similarity": round(float(similarity), 4),
            "all_matches": matches or [],
            "reason": reason,
        }

    def score_speaker(
        self,
        speaker_audio: Union[str, np.ndarray],
        enrolled_employees: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return cosine similarity of the speaker against every enrolled voiceprint, best first.

        Raises VoiceEmbeddingError if the speaker audio is unusable.
        """
        turn_embedding = self.biometrics.extract_embedding(speaker_audio)

        matches = []
        for emp in enrolled_employees:
            emb_data = emp.get("voice_embedding")
            if not emb_data:
                continue
            try:
                emp_vec = np.array(json.loads(emb_data) if isinstance(emb_data, str) else emb_data, dtype=np.float32)
            except Exception:
                continue
            if emp_vec.shape != turn_embedding.shape:
                logger.warning(
                    f"Skipping voiceprint for {emp.get('name')}: dimension {emp_vec.shape} != {turn_embedding.shape}. "
                    "Re-enroll this employee."
                )
                continue

            sim = self.cosine_similarity(turn_embedding, emp_vec)
            matches.append({
                "id": emp.get("id"),
                "employee_id": emp.get("employee_id"),
                "name": emp.get("name"),
                "similarity": round(float(sim), 4),
            })

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches

    def decide(self, matches: List[Dict[str, Any]], excluded_ids: Optional[set] = None) -> Dict[str, Any]:
        """Apply threshold + ambiguity margin to scored matches."""
        candidates = [m for m in matches if m["id"] not in (excluded_ids or set())]
        if not candidates:
            return self._unknown("no enrolled voiceprints to compare", matches=matches)

        top = candidates[0]
        if top["similarity"] < self.confidence_threshold:
            return self._unknown(
                f"best match {top['name']} ({top['similarity']:.2f}) is below threshold {self.confidence_threshold}",
                top["similarity"], matches,
            )

        if len(candidates) > 1 and top["similarity"] - candidates[1]["similarity"] < self.ambiguity_margin:
            return self._unknown(
                f"ambiguous: {top['name']} ({top['similarity']:.2f}) vs "
                f"{candidates[1]['name']} ({candidates[1]['similarity']:.2f})",
                top["similarity"], matches,
            )

        return {
            "name": top["name"],
            "employee_id": top["id"],
            "confidence": top["similarity"],
            "is_unknown": False,
            "raw_similarity": top["similarity"],
            "all_matches": matches,
            "reason": "matched",
        }

    def identify_speaker(
        self,
        speaker_audio: Union[str, np.ndarray],
        enrolled_employees: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Identify an unknown speaker against enrolled employee voice profiles."""
        if not enrolled_employees:
            return self._unknown("no employees enrolled")
        try:
            matches = self.score_speaker(speaker_audio, enrolled_employees)
        except VoiceEmbeddingError as e:
            logger.warning(f"Speaker cannot be identified: {e}")
            return self._unknown(str(e))
        return self.decide(matches)
