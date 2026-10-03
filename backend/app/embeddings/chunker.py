"""Semantic Transcript Chunker for MeetWise AI.

Splits multi-speaker transcripts into meaningful chunks for vector embedding,
preserving speaker attribution, timestamp boundaries, and sentence integrity.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TranscriptChunk(BaseModel):
    """Semantic chunk containing formatted text and metadata for vector indexing."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    meeting_id: str = Field(..., description="Meeting ID")
    meeting_title: str = Field(..., description="Meeting Title")
    text: str = Field(..., description="Structured chunk text with timestamps and speaker attribution")
    start_time: float = Field(..., description="Earliest timestamp in seconds")
    end_time: float = Field(..., description="Latest timestamp in seconds")
    speakers: List[str] = Field(default_factory=list, description="Speakers present in this chunk")


class TranscriptChunker:
    """Chunking engine designed for conversational meeting transcripts."""

    def __init__(self, target_chunk_words: int = 150, overlap_turns: int = 1):
        self.target_chunk_words = target_chunk_words
        self.overlap_turns = overlap_turns

    @staticmethod
    def _format_turn(seg: Dict[str, Any]) -> str:
        speaker = seg.get("speaker", "UNKNOWN")
        start = float(seg.get("start", 0.0))
        m_s = f"{int(start // 60):02d}:{int(start % 60):02d}"
        return f"[{m_s}] {speaker}: {seg.get('text', '').strip()}"

    def chunk_transcript(
        self,
        meeting_id: str,
        meeting_title: str,
        aligned_segments: List[Dict[str, Any]],
    ) -> List[TranscriptChunk]:
        """Group transcript turns into coherent overlapping semantic chunks.

        Args:
            meeting_id: Meeting UUID.
            meeting_title: Title of the meeting.
            aligned_segments: Speaker-attributed transcript segments.

        Returns:
            List of TranscriptChunk objects ready for BGE embedding and ChromaDB indexing.
        """
        if not aligned_segments:
            return []

        chunks: List[TranscriptChunk] = []
        n_segs = len(aligned_segments)
        i = 0
        chunk_idx = 0

        while i < n_segs:
            current_segs = []
            word_count = 0
            j = i

            while j < n_segs:
                seg = aligned_segments[j]
                text = seg.get("text", "").strip()
                if not text:
                    j += 1
                    continue

                words_in_seg = len(text.split())
                current_segs.append(seg)
                word_count += words_in_seg
                j += 1

                # If we've reached the target chunk word count, stop extending
                if word_count >= self.target_chunk_words:
                    break

            if current_segs:
                # Format chunk content
                formatted_lines = [self._format_turn(s) for s in current_segs]
                chunk_text = f"Meeting: {meeting_title}\n" + "\n".join(formatted_lines)

                start_time = round(float(current_segs[0].get("start", 0.0)), 2)
                end_time = round(float(current_segs[-1].get("end", start_time)), 2)
                unique_speakers = sorted(list(set(s.get("speaker", "UNKNOWN") for s in current_segs)))

                chunk_id = f"{meeting_id}_chunk_{chunk_idx:04d}"
                chunks.append(
                    TranscriptChunk(
                        chunk_id=chunk_id,
                        meeting_id=meeting_id,
                        meeting_title=meeting_title,
                        text=chunk_text,
                        start_time=start_time,
                        end_time=end_time,
                        speakers=unique_speakers,
                    )
                )
                chunk_idx += 1

            # Advance with overlap
            if j >= n_segs:
                break
            # Step forward by (j - i - overlap_turns), ensuring at least 1 turn advance
            step = max(1, (j - i) - self.overlap_turns)
            i += step

        logger.info(f"Created {len(chunks)} semantic chunks for meeting '{meeting_title}' ({meeting_id}).")
        return chunks

