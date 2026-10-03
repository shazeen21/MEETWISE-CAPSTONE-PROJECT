"""Generate Synthetic Sample Meeting Audio for MeetWise AI Testing.

Generates a test WAV file with alternating multi-frequency audio segments
simulating a multi-speaker conversation.
"""

import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 16000
DURATION_PER_SPEAKER = 3.0  # seconds per speaker turn


def generate_tone(freq: float, duration: float, volume: float = 0.5) -> bytes:
    """Generate PCM 16-bit mono sine wave tone."""
    n_samples = int(SAMPLE_RATE * duration)
    audio_bytes = bytearray()
    for i in range(n_samples):
        # Apply amplitude envelope (fade in/out) to prevent clicks
        envelope = min(1.0, i / (0.05 * SAMPLE_RATE), (n_samples - i) / (0.05 * SAMPLE_RATE))
        val = int(volume * envelope * 32767.0 * math.sin(2.0 * math.pi * freq * (i / SAMPLE_RATE)))
        audio_bytes.extend(struct.pack("<h", max(-32768, min(32767, val))))
    return bytes(audio_bytes)


def create_sample_meeting_wav(output_path: str) -> str:
    """Generate a multi-turn multi-speaker synthetic audio file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Simulate 3 speakers with distinct harmonic frequencies:
    # Speaker 0: 220 Hz (A3)
    # Speaker 1: 330 Hz (E4)
    # Speaker 2: 440 Hz (A4)
    speaker_turns = [
        (220, "SPEAKER_00", 3.0),
        (330, "SPEAKER_01", 3.5),
        (220, "SPEAKER_00", 2.5),
        (440, "SPEAKER_02", 3.0),
    ]

    combined_pcm = bytearray()
    for freq, spk, dur in speaker_turns:
        tone_bytes = generate_tone(freq, dur)
        combined_pcm.extend(tone_bytes)
        # Add 0.3s pause
        silence = struct.pack("<h", 0) * int(SAMPLE_RATE * 0.3)
        combined_pcm.extend(silence)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(combined_pcm)

    print(f"Generated sample synthetic meeting audio at: {path}")
    return str(path)


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "data" / "audio" / "sample_meeting.wav"
    create_sample_meeting_wav(str(target))

