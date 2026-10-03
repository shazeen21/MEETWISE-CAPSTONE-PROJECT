"""Transcript and Speaker Alignment Module for MeetWise AI.

Aligns WhisperX speech recognition segments/words with pyannote speaker turns
using temporal intersection maximization. Produces a unified speaker-attributed transcript.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AlignedSegment(BaseModel):
    """Unified speaker-attributed transcript segment."""
    speaker: str = Field(..., description="Speaker identifier (e.g., SPEAKER_00)")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Spoken transcript text")


class TranscriptAligner:
    """Merges WhisperX transcript segments and pyannote speaker turns."""

    def __init__(self, max_merge_gap_seconds: float = 1.2):
        self.max_merge_gap_seconds = max_merge_gap_seconds

    @staticmethod
    def _compute_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
        """Calculate temporal overlap duration between two intervals."""
        overlap = max(0.0, min(end1, end2) - max(start1, start2))
        return overlap

    def align(
        self,
        asr_segments: List[Dict[str, Any]],
        speaker_turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Align ASR segments to speaker turns based on maximum temporal overlap.

        Args:
            asr_segments: Output from WhisperX with 'start', 'end', 'text', and optional 'words'.
            speaker_turns: Output from pyannote with 'speaker', 'start', 'end'.

        Returns:
            List of aligned speaker segments sorted chronologically.
        """
        if not asr_segments:
            return []

        # If no speaker turns were detected, default to SPEAKER_00
        if not speaker_turns:
            logger.warning("No speaker turns provided. Assigning all segments to SPEAKER_00.")
            return [
                {
                    "speaker": "SPEAKER_00",
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"].strip(),
                }
                for seg in asr_segments
                if seg.get("text", "").strip()
            ]

        raw_aligned: List[Dict[str, Any]] = []

        # Check if word-level timestamps are present across segments
        has_word_timestamps = any(
            "words" in seg and isinstance(seg["words"], list) and len(seg["words"]) > 0
            for seg in asr_segments
        )

        if has_word_timestamps:
            raw_aligned = self._align_with_words(asr_segments, speaker_turns)
        else:
            raw_aligned = self._align_with_segments(asr_segments, speaker_turns)

        # Merge consecutive turns with the same speaker that are close in time
        merged = self._merge_consecutive_speaker_turns(raw_aligned)
        return merged

    def _align_with_segments(
        self,
        asr_segments: List[Dict[str, Any]],
        speaker_turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Assign speaker to each segment by highest overlap."""
        aligned = []
        last_known_speaker = speaker_turns[0]["speaker"] if speaker_turns else "SPEAKER_00"

        for seg in asr_segments:
            s_start = float(seg["start"])
            s_end = float(seg["end"])
            text = seg.get("text", "").strip()

            if not text:
                continue

            best_speaker = None
            max_overlap = 0.0

            for turn in speaker_turns:
                overlap = self._compute_overlap(s_start, s_end, turn["start"], turn["end"])
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = turn["speaker"]

            # Fallback if segment didn't overlap any turn: find the temporally closest turn
            if best_speaker is None:
                min_dist = float("inf")
                for turn in speaker_turns:
                    dist = min(abs(s_start - turn["end"]), abs(turn["start"] - s_end))
                    if dist < min_dist:
                        min_dist = dist
                        best_speaker = turn["speaker"]

                if best_speaker is None:
                    best_speaker = last_known_speaker

            last_known_speaker = best_speaker

            aligned.append({
                "speaker": best_speaker,
                "start": round(s_start, 2),
                "end": round(s_end, 2),
                "text": text,
            })

        return aligned

    def _align_with_words(
        self,
        asr_segments: List[Dict[str, Any]],
        speaker_turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Assign speaker to individual words and group into turns."""
        word_list = []
        for seg in asr_segments:
            words = seg.get("words", [])
            if words:
                for w in words:
                    w_text = w.get("word", "").strip()
                    w_start = float(w.get("start", seg["start"]))
                    w_end = float(w.get("end", seg["end"]))
                    if w_text:
                        word_list.append({"word": w_text, "start": w_start, "end": w_end})
            else:
                text = seg.get("text", "").strip()
                if text:
                    word_list.append({
                        "word": text,
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                    })

        if not word_list:
            return []

        # Assign speaker to each word
        last_speaker = speaker_turns[0]["speaker"] if speaker_turns else "SPEAKER_00"
        assigned_words = []

        for item in word_list:
            w_start = item["start"]
            w_end = item["end"]
            best_speaker = None
            max_overlap = 0.0

            for turn in speaker_turns:
                overlap = self._compute_overlap(w_start, w_end, turn["start"], turn["end"])
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = turn["speaker"]

            if best_speaker is None:
                # Find closest turn
                min_dist = float("inf")
                for turn in speaker_turns:
                    dist = min(abs(w_start - turn["end"]), abs(turn["start"] - w_end))
                    if dist < min_dist:
                        min_dist = dist
                        best_speaker = turn["speaker"]
                if best_speaker is None:
                    best_speaker = last_speaker

            last_speaker = best_speaker
            assigned_words.append({
                "speaker": best_speaker,
                "word": item["word"],
                "start": w_start,
                "end": w_end,
            })

        # Group contiguous words by same speaker
        turns: List[Dict[str, Any]] = []
        curr_turn = None

        for w in assigned_words:
            if curr_turn is None:
                curr_turn = {
                    "speaker": w["speaker"],
                    "start": w["start"],
                    "end": w["end"],
                    "words": [w["word"]],
                }
            elif (
                curr_turn["speaker"] == w["speaker"]
                and (w["start"] - curr_turn["end"]) <= self.max_merge_gap_seconds
            ):
                curr_turn["end"] = w["end"]
                curr_turn["words"].append(w["word"])
            else:
                turns.append({
                    "speaker": curr_turn["speaker"],
                    "start": round(curr_turn["start"], 2),
                    "end": round(curr_turn["end"], 2),
                    "text": " ".join(curr_turn["words"]).strip(),
                })
                curr_turn = {
                    "speaker": w["speaker"],
                    "start": w["start"],
                    "end": w["end"],
                    "words": [w["word"]],
                }

        if curr_turn:
            turns.append({
                "speaker": curr_turn["speaker"],
                "start": round(curr_turn["start"], 2),
                "end": round(curr_turn["end"], 2),
                "text": " ".join(curr_turn["words"]).strip(),
            })

        return turns

    def _merge_consecutive_speaker_turns(
        self,
        aligned_turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge adjacent turns belonging to the same speaker separated by small silence."""
        if not aligned_turns:
            return []

        merged: List[Dict[str, Any]] = []
        current = aligned_turns[0].copy()

        for next_turn in aligned_turns[1:]:
            gap = next_turn["start"] - current["end"]
            if current["speaker"] == next_turn["speaker"] and gap <= self.max_merge_gap_seconds:
                # Merge text and extend end timestamp
                current["text"] = f"{current['text']} {next_turn['text']}".strip()
                current["end"] = round(max(current["end"], next_turn["end"]), 2)
            else:
                merged.append(current)
                current = next_turn.copy()

        merged.append(current)
        return merged

    def save_transcript_json(
        self,
        aligned_turns: List[Dict[str, Any]],
        output_file_path: str,
    ) -> str:
        """Save aligned transcript segments to JSON file."""
        out_path = Path(output_file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(aligned_turns, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved aligned transcript to {output_file_path}")
        return str(out_path)

