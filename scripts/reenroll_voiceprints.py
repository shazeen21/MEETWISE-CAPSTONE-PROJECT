"""Rebuild every stored voiceprint from the original enrollment audio.

Earlier builds could not decode browser (.webm) recordings and silently stored
placeholder vectors, so enrolled employees looked identical to each other.
This re-extracts each sample with the ECAPA-TDNN model, removes samples that
contain no usable speech, and recomputes each employee's centroid.

Usage (from the project root):
    python scripts/reenroll_voiceprints.py            # dry run, prints report
    python scripts/reenroll_voiceprints.py --apply    # writes to the database
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.database.connection import SessionLocal  # noqa: E402
from app.database.models import Employee, VoiceSample  # noqa: E402
from app.speakers.voice_biometrics import VoiceBiometricsService, VoiceEmbeddingError  # noqa: E402


def main(apply: bool) -> None:
    svc = VoiceBiometricsService()
    voice_dir = Path(settings.DATA_DIR) / "voice_samples"
    db = SessionLocal()
    centroids = {}
    try:
        for emp in db.query(Employee).all():
            samples = db.query(VoiceSample).filter(VoiceSample.employee_id == emp.id).all()
            vectors = []
            for s in samples:
                path = voice_dir / s.audio_file_name
                try:
                    svc.check_speech_like(str(path))
                    emb = svc.extract_embedding(str(path))
                except VoiceEmbeddingError as e:
                    print(f"  [{emp.name}] REMOVE {s.audio_file_name}: {e}")
                    if apply:
                        db.delete(s)
                    continue
                vec = [round(float(x), 6) for x in emb]
                vectors.append(vec)
                if apply:
                    s.embedding_vector = json.dumps(vec)

            if vectors:
                centroid = svc.aggregate_centroid(vectors)
                centroids[emp.name] = np.array(centroid)
            else:
                centroid = None
            print(f"{emp.name}: {len(vectors)} usable sample(s)" + ("" if vectors else " -> NOT ENROLLED, please re-record"))
            if apply:
                emp.voice_embedding = json.dumps(centroid) if centroid else None
                emp.voice_samples_count = len(vectors)

        print("\nCentroid similarity between employees (should be well below 0.25):")
        for a, b in combinations(centroids, 2):
            print(f"  {a} vs {b}: {float(centroids[a] @ centroids[b]):.3f}")

        if apply:
            db.commit()
            print("\nDatabase updated.")
        else:
            print("\nDry run only. Re-run with --apply to save.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the rebuilt voiceprints to the database")
    main(parser.parse_args().apply)
