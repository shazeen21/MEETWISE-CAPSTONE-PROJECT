"""Audio Denoising & Enhancement Module for MeetWise AI.

Supports pluggable denoising strategies:
- Pretrained Deep Denoising Autoencoder (Meta Denoiser / dns64)
- Spectral Gating (noisereduce / scipy spectral filtering)
- Normalization / Passthrough
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class BaseAudioDenoiser(ABC):
    """Abstract interface for audio denoising algorithms."""

    @abstractmethod
    def enhance(self, input_path: str, output_path: str) -> str:
        """Enhance audio file and save the output.

        Args:
            input_path: Path to input audio file.
            output_path: Destination path for enhanced audio.

        Returns:
            Path to the enhanced audio file.
        """
        pass


class PassthroughDenoiser(BaseAudioDenoiser):
    """Normalization and format standardization without aggressive filtering."""

    def enhance(self, input_path: str, output_path: str) -> str:
        import soundfile as sf

        data, sr = sf.read(input_path)
        # Peak normalization (-1.0 to 1.0 with 3dB headroom)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val * 0.95

        sf.write(output_path, data, sr, subtype="PCM_16")
        logger.info(f"Normalized audio saved to: {output_path}")
        return output_path


class SpectralDenoiser(BaseAudioDenoiser):
    """Spectral gating denoising using noisereduce or scipy spectral filter."""

    def enhance(self, input_path: str, output_path: str) -> str:
        import soundfile as sf

        data, sr = sf.read(input_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        try:
            import noisereduce as nr

            logger.info("Applying noisereduce stationary spectral gating...")
            reduced_noise = nr.reduce_noise(
                y=data,
                sr=sr,
                prop_decrease=0.75,
                stationary=True,
                n_fft=1024,
                win_length=1024,
                hop_length=512,
            )
        except Exception as e:
            logger.warning(
                f"noisereduce package not available or error ({e}), applying gentle highpass/normalization..."
            )
            from scipy.signal import butter, lfilter

            # Gentle highpass filter (80 Hz cut-off to eliminate mic rumble/hum)
            b, a = butter(2, 80 / (sr / 2), btype="high")
            reduced_noise = lfilter(b, a, data)

        # Normalize
        max_val = np.max(np.abs(reduced_noise))
        if max_val > 0:
            reduced_noise = (reduced_noise / max_val) * 0.95

        sf.write(output_path, reduced_noise, sr, subtype="PCM_16")
        logger.info(f"Denoised audio saved to: {output_path}")
        return output_path


class DeepAutoencoderDenoiser(BaseAudioDenoiser):
    """Deep Pretrained Denoising Autoencoder (Meta Denoiser / dns64).

    Follows Section 2.5 of the MeetWise AI Build Guide.
    """

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                import torch
                from denoiser import pretrained

                logger.info("Loading Meta Denoiser (dns64) model...")
                self._model = pretrained.dns64()
                self._model.eval()
            except Exception as e:
                logger.warning(
                    f"Meta denoiser not installed or failed to load: {e}. Falling back to SpectralDenoiser."
                )
                self._model = False

    def enhance(self, input_path: str, output_path: str) -> str:
        self._load_model()
        if not self._model:
            # Fallback to spectral denoiser
            return SpectralDenoiser().enhance(input_path, output_path)

        import torch
        import torchaudio
        from denoiser.dsp import convert_audio

        wav, sr = torchaudio.load(input_path)
        wav = convert_audio(wav, sr, self._model.sample_rate, self._model.chin)
        with torch.no_grad():
            denoised = self._model(wav.unsqueeze(0))[0]
        torchaudio.save(output_path, denoised.cpu(), self._model.sample_rate)
        logger.info(f"Deep autoencoder enhanced audio saved to: {output_path}")
        return output_path


def get_denoiser(denoiser_type: str = "spectral") -> BaseAudioDenoiser:
    """Factory function returning the configured audio denoiser."""
    dtype = (denoiser_type or "spectral").lower()
    if dtype == "deep":
        return DeepAutoencoderDenoiser()
    elif dtype == "passthrough" or dtype == "none":
        return PassthroughDenoiser()
    else:
        return SpectralDenoiser()

