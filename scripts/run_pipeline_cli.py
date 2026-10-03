"""MeetWise AI — Command Line Pipeline Runner.

Allows testing the entire pipeline end-to-end directly from the terminal
without a web browser or frontend.
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import settings
from backend.app.database.connection import SessionLocal, init_db
from backend.app.pipeline import MeetWisePipeline
from backend.app.embeddings.bge_service import BGEEmbeddingService
from backend.app.vector_store.chroma_service import ChromaService
from backend.app.rag.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="MeetWise AI Complete Pipeline Runner")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav, .mp3, .m4a)")
    parser.add_argument("--title", type=str, default=None, help="Optional meeting title")
    parser.add_argument("--query", type=str, default="What action items were assigned?", help="RAG question to ask after processing")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file '{args.audio}' does not exist.")
        sys.exit(1)

    print("=" * 60)
    print("       MEETWISE AI — COMPLETE PIPELINE RUNNER")
    print("=" * 60)
    print(f"Audio File: {audio_path}")
    print(f"Whisper Model: {settings.WHISPER_MODEL} ({settings.WHISPER_DEVICE})")
    print(f"Embedding Model: {settings.BGE_MODEL}")
    print(f"Database: {settings.DATABASE_URL.split('@')[-1]}")
    print("-" * 60)

    # Initialize DB tables
    init_db()

    # Create pipeline and DB session
    pipeline = MeetWisePipeline()
    db = SessionLocal()

    try:
        # Step 1: Run complete pipeline
        print("\n>>> Executing MeetWise AI Pipeline...")
        result = pipeline.process_meeting(
            audio_path=str(audio_path),
            db=db,
            title=args.title,
            source_type="uploaded",
        )

        print("\n" + "=" * 60)
        print("          PIPELINE RESULTS")
        print("=" * 60)
        print(f"Meeting ID:      {result['meeting_id']}")
        print(f"Title:           {result['title']}")
        print(f"Duration:        {result['duration_seconds']}s")
        print(f"Transcript JSON: {result['transcript_json_path']}")
        print(f"Segments:        {result['transcript_segment_count']}")
        print(f"Chunks Indexed:  {result['chroma_chunks_indexed']}")

        intel = result.get("intelligence", {})
        print("\n--- Executive Summary ---")
        print(intel.get("summary", "N/A"))

        print("\n--- Decisions Reached ---")
        decisions = intel.get("decisions", [])
        if decisions:
            for d in decisions:
                print(f"• {d.get('decision')} (Timestamp: {d.get('timestamp')})")
        else:
            print("None recorded.")

        print("\n--- Action Items ---")
        actions = intel.get("action_items", [])
        if actions:
            for a in actions:
                print(f"• [{a.get('status')}] {a.get('task')} (Owner: {a.get('owner')}, Deadline: {a.get('deadline')})")
        else:
            print("None recorded.")

        # Step 2: Test Grounded RAG Query
        print("\n" + "=" * 60)
        print(f"          RAG QUESTION ANSWERING: '{args.query}'")
        print("=" * 60)

        bge = BGEEmbeddingService(model_name=settings.BGE_MODEL)
        chroma = ChromaService(persist_dir=settings.CHROMA_PERSIST_DIR)
        rag = RAGPipeline(
            bge_service=bge,
            chroma_service=chroma,
            gemini_api_key=settings.GEMINI_API_KEY,
        )

        rag_res = rag.query(
            question=args.query,
            top_k=3,
            meeting_id=result["meeting_id"],
        )

        print(f"\n[Answer]:\n{rag_res.answer}\n")
        print(f"[Citations ({len(rag_res.sources)})]:")
        for s in rag_res.sources:
            print(f"• Meeting: '{s.meeting_title}' | Speaker: {s.speaker} | Time: {s.start_time}s - {s.end_time}s")

    finally:
        db.close()


if __name__ == "__main__":
    main()

