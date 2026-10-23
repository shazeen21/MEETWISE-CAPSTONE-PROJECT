# MeetWise AI — Intelligent Meeting Intelligence & Organizational Memory Platform

> **Final-Year Capstone Project (6-Month Implementation)**  
> **Domain:** Speech Processing, Neural Speaker Diarization, Large Language Models, and Dense Vector Retrieval (RAG).  
> **Objective:** Convert in-room physical meeting recordings into structured, auditable, and semantically searchable organizational memory.

---

## 1. Project Vision & Problem Statement

### The Problem
Physical meetings are high-friction information silos. Although organizations spend up to 40% of working hours in physical conference rooms and huddle spaces:
1. **Information Evaporation:** Decisions, rationale, and specific commitments made verbally evaporate once participants leave the room.
2. **Unreliable Manual Note-Taking:** Meeting minutes are subjective, incomplete, and frequently omit who committed to what deadline.
3. **No Organizational Memory:** New team members or absent stakeholders cannot search across historical meetings for technical decisions or context without re-asking colleagues.
4. **Speaker Attribution Failure:** Standard automatic speech recognition (ASR) tools produce a continuous wall of text without distinguishing between participants or attributing action items accurately.

### The Solution: MeetWise AI
MeetWise AI provides an end-to-end, privacy-conscious pipeline that ingests physical meeting audio, enhances and denoises it, identifies individual speakers, extracts structured business intelligence (Minutes of Meeting, action items with owners and deadlines, decisions), stores records across relational and vector databases, and provides a strictly grounded Retrieval-Augmented Generation (RAG) assistant for querying historical meetings with precise speaker and timestamp citations.

---

## 2. System Architecture & Pipeline Flow

The following diagram illustrates the complete end-to-end dataflow from raw acoustic input to relational persistence and vector retrieval:

```
                            [ INPUT AUDIO ]
              (WAV / MP3 / M4A / MP4 / WebM from Upload or Mic)
                                   │
                                   ▼
                   ┌───────────────────────────────────┐
                   │   Phase 1: Audio Preprocessor     │
                   │   - 16 kHz, 16-bit Mono Conversion│
                   │   - Peak Normalization            │
                   │   - Spectral Noise Gating (dns64) │
                   └─────────────────┬─────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │ Phase 2: WhisperX ASR   │             │ Phase 3: pyannote.audio │
    │ - CTranslate2 Faster-W. │             │ - Neural Segmentation   │
    │ - Phoneme Alignment     │             │ - Speaker Embeddings    │
    │ - Word-level Timestamps │             │ - VBx / PLDA Clustering │
    └────────────┬────────────┘             └────────────┬────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │ Phase 4: Aligner & Combiner       │
                   │ - Temporal Intersection Max.      │
                   │ - Word Boundary Slicing           │
                   │ - Merged Transcript JSON          │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │ Phase 5: Speaker Identification   │
                   │ - ECAPA-TDNN voiceprints (192-d)  │
                   │ - Cosine match vs enrolled staff  │
                   │ - Threshold + ambiguity margin    │
                   │ - Else "Unknown Speaker"          │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │ Phase 6: Gemini LLM Intelligence  │
                   │ - MoM & Executive Summary         │
                   │ - Action Items (Owner, Deadline)  │
                   │ - Verbatim evidence check         │
                   │ - Per-segment emotion labels      │
                   └─────────────────┬─────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │ Phase 7: Relational DB  │             │ Phase 8: Vector Store   │
    │ - PostgreSQL / SQLite   │             │ - Semantic Chunker      │
    │ - SQLAlchemy ORM        │             │ - BAAI BGE Embeddings   │
    │ - meetings, speakers,   │             │ - ChromaDB Collection   │
    │   transcripts, actions  │             │   (metadata enriched)   │
    └─────────────────────────┘             └────────────┬────────────┘
                                                         │
                                                         ▼
                                            ┌─────────────────────────┐
                                            │ Phase 9: RAG Assistant  │
                                            │ - Dense Query Embedding │
                                            │ - Similarity Search     │
                                            │ - Grounded Gemini Q&A   │
                                            │ - Speaker/Time Citation │
                                            └─────────────────────────┘
```

---

## 3. Technology Stack & Technical Reasoning

Every framework, library, and model in MeetWise AI was chosen to optimize accuracy, offline resilience, low latency, and cost-effectiveness:

| Component | Technology Selected | Version / Model | Why Chosen Over Alternatives |
| :--- | :--- | :--- | :--- |
| **API Gateway** | **FastAPI** | `>=0.110.0` | Asynchronous I/O natively supports large audio uploads and long-running AI pipelines without worker starvation; automatic OpenAPI/Swagger documentation. |
| **ASGI Server** | **Uvicorn** | `>=0.28.0` | High-throughput ASGI implementation based on `uvloop` and `httptools`. |
| **Configuration**| **Pydantic Settings**| `>=2.2.0` | Type-safe environment variable parsing with validation, zero runtime overhead, and IDE autocomplete. |
| **Audio Processing** | **SoundFile & SciPy**| `>=0.12.1` | Native C-level libsndfile bindings provide fast, lossless decoding and sample rate conversion to 16 kHz mono. |
| **Audio Denoising** | **noisereduce / Meta DNS64** | `>=3.0.0` | **noisereduce:** Stationary spectral gating removes continuous HVAC and electrical noise with zero GPU overhead. **Meta DNS64:** Deep learning autoencoder handles non-stationary room reverberation. |
| **Speech Recognition** | **WhisperX** | `>=3.1.1` (CTranslate2) | **Selected over standard Whisper:** WhisperX uses forced phoneme alignment (Wav2Vec2) to produce true **word-level timestamps** with 60–70% lower VRAM and 4x speed via CTranslate2 quantization. Standard Whisper only provides coarse 30s chunk timestamps. |
| **Speaker Diarization** | **pyannote.audio** | `4.x` with `speaker-diarization-community-1` | Industry benchmark for "who spoke when". Neural segmentation finds speech and speaker changes; speaker embeddings + VBx clustering group turns by voice. Accepts an exact speaker count for higher accuracy. |
| **Speaker Identification** | **SpeechBrain ECAPA-TDNN** | `speechbrain/spkrec-ecapa-voxceleb` | Pretrained on VoxCeleb (7,000+ speakers). Converts speech into a 192-d voiceprint capturing pitch, timbre and vocal-tract characteristics. No training on company audio: enrollment only stores voiceprints. |
| **Meeting Intelligence**| **Google Gemini** | `gemini-3.6-flash` (with fallbacks) | Selected for its massive 1M+ token context window (capable of processing 5+ hours of continuous transcript in a single prompt without chunk fragmentation) and near-zero cost/token. |
| **Relational Storage** | **PostgreSQL** + **SQLAlchemy** | `15+` / `>=2.0.28` | Full ACID compliance, foreign key cascades (`ON DELETE CASCADE`), index-backed timestamp queries. Abstracted via SQLAlchemy ORM so SQLite can be used for instant local development with zero setup. |
| **Dense Embeddings** | **BAAI/bge-small-en-v1.5** | via `sentence-transformers` | Top-ranked compact embedding model on Hugging Face MTEB benchmark. 384-dimensional vectors run quickly on CPU while outperforming older 1536-dim models in retrieval accuracy. Prepend instruction format optimizes asymmetric question-passage similarity. |
| **Vector Database** | **ChromaDB** | `>=0.4.24` | Embedded, in-process vector store requiring zero external daemon or cloud subscription. Persists locally in Parquet/DuckDB format with cosine distance indexing. |
| **Frontend** | **React + Vite + Tailwind** | React 18 | Dashboard, meeting detail (transcript, MoM, analytics charts), employee voice enrollment, knowledge-base search. |
| **Testing** | **Pytest + HTTPX** | `>=8.0.0` | Isolated in-memory database and temporary storage per test; ASR, LLM and RAG are mocked so the suite never touches real data or the network. |

---

## 4. Methodology & Engineering Design

### 4.1. Audio Ingestion & Normalization
Physical meetings capture audio in varying formats (lossy MP3, M4A from smartphones, or uncompressed WAV). The `AudioPreprocessor` normalizes all incoming audio:
- Enforces single-channel **mono** (averaging stereo channels if present).
- Resamples to **16,000 Hz** (the native sample rate expected by Whisper and pyannote models).
- Scales bit depth to signed 16-bit PCM.
- Calculates and stores exact metadata (`duration_seconds`, `sample_rate`, `channels`).

### 4.2. Denoising Strategies
Physical meeting rooms suffer from HVAC drone, projector fans, and distant chatter. The `AudioDenoiser` supports three pluggable strategies:
1. **Spectral Gating (`spectral`):** Computes a short-time Fourier transform (STFT), estimates the stationary noise profile from quiet segments, and gates frequency bins below the noise floor.
2. **Deep Autoencoder (`deep`):** Employs Meta's DNS64 (`demucs`/`torchaudio`) pre-trained neural speech enhancement model to separate speech from complex non-stationary acoustics.
3. **Passthrough (`passthrough`):** Zero-latency bypass for studio-quality or already-clean inputs.

### 4.3. The Alignment Problem & Algorithm
WhisperX generates textual segments with word timestamps, but has no innate knowledge of who spoke. Pyannote generates speaker intervals (`SPEAKER_00`, `SPEAKER_01`) with no knowledge of words.

MeetWise AI uses a custom **Temporal Intersection Maximization Algorithm** in `TranscriptAligner`:
1. For every transcription segment $S_i$ with start time $T_{start}(S_i)$ and end time $T_{end}(S_i)$:
2. It iterates through all pyannote speaker intervals $D_j$.
3. It computes the overlap interval:
   $$\text{Overlap}(S_i, D_j) = \max(0, \min(T_{end}(S_i), T_{end}(D_j)) - \max(T_{start}(S_i), T_{start}(D_j)))$$
4. The speaker $D_j$ that maximizes $\text{Overlap}(S_i, D_j)$ is assigned to segment $S_i$.
5. If no speaker turn overlaps (e.g. background voice during silence), it falls back to the chronologically closest speaker or default `SPEAKER_00`.
6. Generates a structured JSON transcript with precise millisecond boundaries for every phrase.

### 4.4. Structured Intelligence Extraction (Gemini)
Rather than requesting an unstructured summary, the pipeline enforces strict Pydantic contract validation (`MeetingIntelligence` schema):
- **Executive Summary:** High-level narrative of the meeting.
- **Key Discussion Topics:** Categorized bullet points.
- **Decisions Made:** Agreed-upon conclusions with context timestamps.
- **Action Items:** Granular tasks assigned to a specific **Owner**, with an explicit **Deadline** and **Status** (`pending`).
- **Unresolved/Pending Issues:** Questions raised but not answered during the session.
- **Evidence-Based Verification:** Every decision and action item must carry a verbatim quote from the transcript. After the LLM responds, Python checks that the quote really occurs in the spoken text (ignoring case and punctuation). Items whose quote is missing or fabricated are **removed**, so they never reach the database, UI or PDF.
- **Speaker grounding:** The LLM must keep speaker labels exactly as in the transcript and only receives the names of employees verified by voice, so it cannot attribute words to people who did not speak.

### 4.5. Context-Preserving Semantic Chunking
Naïve fixed-character chunking splits words across boundaries and severs speaker attribution. MeetWise uses **Semantic Transcript Chunking**:
- Groups contiguous utterances by speaker.
- Respects speaker turn shifts as natural semantic breakpoints.
- Accumulates chunks up to a target size (400 words) with a sliding overlap (2 turns).
- Enriches every chunk with metadata: `meeting_id`, `meeting_title`, `speaker`, `start_time`, `end_time`, and `chunk_index`.

### 4.6. Grounded Retrieval-Augmented Generation (RAG)
When a user queries the system via `POST /search`:
1. The question is formatted with BGE query instruction: `"Represent this sentence for searching relevant passages: " + query`.
2. ChromaDB runs cosine similarity to fetch the top-k most relevant chunks.
3. The retrieved chunks are formatted into an auditable prompt for Gemini.
4. Gemini is instructed via strict system prompt to act as an objective auditor:
   - Answer **only** from the supplied context.
   - For every claim, cite the speaker and timestamp (e.g., `[SPEAKER_01 at 04:15]`).
   - If the transcript does not contain the answer, explicitly declare that it was not discussed.

### 4.7. Speaker Identification (Voice Biometrics)
Diarization only says *SPEAKER_00 vs SPEAKER_01*. To put real names on those labels, MeetWise matches each diarized speaker's voice against enrolled employees.

**No model training happens.** The ECAPA-TDNN model is already trained; it works like a fingerprint scanner:
1. **Enrollment:** Each employee records 4–5 short clips (browser `.webm` or uploaded audio). Every clip is decoded (PyAV handles `.webm`/`.m4a`), checked to be real speech (pure tones, silence and clips under 1 s are rejected), and converted to a 192-d voiceprint. The employee's profile is the L2-normalised average (centroid) of their clips.
2. **Matching:** For each diarized speaker, up to 30 s of their audio is embedded and compared with every centroid by **cosine similarity** (a distance, not a probability).
3. **Decision rules:**
   - best similarity must be **≥ 0.25** (SpeechBrain's verification threshold), and
   - it must beat the runner-up by **≥ 0.05**, otherwise the speaker is ambiguous, and
   - one employee can be assigned to at most one diarized speaker per meeting.
   If any rule fails, or the audio cannot produce a reliable voiceprint, the speaker stays **"Unknown Speaker"**. The UI shows the raw score as "voice match 0.71".

Measured on the project's enrollment data: same-person clips score **0.47–0.80**, different people **≤ 0.19**. Pitch alone is not used for decisions because it overlaps between people and shifts with microphone and mood (one speaker measured 196 Hz at enrollment but 155 Hz in a meeting while still matching at 0.80); the voiceprint encodes pitch together with timbre and vocal-tract shape.

`scripts/reenroll_voiceprints.py` rebuilds every voiceprint from the stored audio (dry run by default, `--apply` to save).

### 4.8. Emotion & Sentiment Analysis
Each numbered transcript segment receives exactly one emotion from a fixed taxonomy:

| Positive | Neutral | Negative |
| :--- | :--- | :--- |
| joy, excitement, satisfaction | neutral, confusion | anxiety, frustration, anger, sadness, disappointment |

Synonyms the LLM may produce are normalised (e.g. *frustrated → frustration*, *depressed / down → sadness*, *worried → anxiety*). The **emotion distribution, positive/neutral/negative percentages and overall sentiment are computed in Python** from the per-segment labels, weighted by speaking time, rather than invented by the LLM. A speaker's dominant emotion is the one they expressed for the longest time.

---

## 5. Project Directory & File Structure

```
MEETWISE/
├── backend/
│   ├── .env                        # Active environment variables (API keys, DB URLs)
│   ├── .env.example                # Configuration template
│   ├── requirements.txt            # Python dependencies
│   └── app/
│       ├── main.py                 # FastAPI application gateway, CORS, healthcheck
│       ├── config.py               # Pydantic Settings & path definitions
│       ├── pipeline.py             # Master pipeline orchestrator (Phases 1-7)
│       ├── api/
│       │   ├── meetings.py         # Endpoints: /meetings/upload, /meetings/{id}
│       │   ├── employees.py        # Employee registration & voice enrollment
│       │   ├── admin.py            # Keywords, audit logs, notifications
│       │   └── search.py           # Endpoint: /search (RAG queries)
│       ├── speakers/
│       │   └── voice_biometrics.py # ECAPA-TDNN voiceprints & speaker matching
│       ├── audio/
│       │   ├── preprocessor.py     # 16kHz mono normalization & metadata extraction
│       │   └── denoiser.py         # Spectral gating & Meta DNS64 denoisers
│       ├── transcription/
│       │   └── whisperx_service.py # WhisperX ASR with forced phoneme alignment
│       ├── diarization/
│       │   ├── pyannote_service.py # pyannote.audio speaker turn detection
│       │   └── aligner.py          # Temporal intersection alignment engine
│       ├── meeting_intelligence/
│       │   ├── schemas.py          # Pydantic models (ActionItem, Decision, MoM)
│       │   └── gemini_service.py   # LLM extraction & JSON sanitization
│       ├── embeddings/
│       │   ├── chunker.py          # Semantic speaker-preserving chunker
│       │   └── bge_service.py      # BAAI BGE dense embedding generator
│       ├── vector_store/
│       │   └── chroma_service.py   # ChromaDB collection indexer & searcher
│       ├── rag/
│       │   └── rag_pipeline.py     # Grounded retrieval & citation engine
│       ├── reports/
│       │   └── pdf_generator.py    # Minutes of Meeting PDF
│       ├── notifications/
│       │   └── dispatcher.py       # Participant & manager notifications
│       └── database/
│           ├── models.py           # SQLAlchemy declarative ORM models
│           ├── connection.py       # Engine, SessionLocal, init_db + column upgrades
│           └── repository.py       # Atomic multi-table database persistence
├── database/
│   └── schema.sql                  # PostgreSQL production DDL with cascades & indexes
├── data/
│   ├── audio/                      # Ingested meeting recordings
│   ├── processed/                  # Normalized 16kHz WAV files
│   ├── transcripts/                # Merged transcript JSON files
│   ├── chromadb/                   # Persistent ChromaDB vector files
│   └── voice_samples/              # Enrollment audio (git-ignored: biometric data)
├── frontend/                       # React + Vite dashboard (proxies /api -> :8000)
├── scripts/
│   ├── generate_sample_audio.py    # Synthetic multi-speaker WAV test generator
│   ├── run_pipeline_cli.py         # Standalone CLI pipeline runner
│   └── reenroll_voiceprints.py     # Rebuild all voiceprints from stored audio
├── tests/
│   ├── conftest.py                 # Pytest mocks & database fixtures
│   ├── test_aligner.py             # Aligner logic tests
│   ├── test_database.py            # ORM CRUD & cascade deletion tests
│   ├── test_chunker.py             # Chunking & overlap tests
│   ├── test_rag.py                 # Grounded RAG query tests
│   ├── test_pipeline.py            # Master orchestrator integration tests
│   ├── test_api.py                 # FastAPI endpoint integration tests
│   ├── test_biometrics_validation.py # Voiceprint rejection, ambiguity, tone checks
│   ├── test_biometrics_and_admin.py  # Enrollment & admin endpoints
│   ├── test_gemini_evidence.py     # Evidence verification & sentiment maths
│   └── test_diarization_regression.py
├── docs/
│   ├── eval_plan_sentiment.md      # Sentiment accuracy evaluation plan
│   ├── security_audit.md           # Platform security analysis
│   └── final_handoff_report.md     # Technical research handoff
├── Dockerfile                      # Container build definition
├── docker-compose.yml              # Multi-container orchestration (App + PostgreSQL)
├── .gitignore                      # Git exclusion rules
└── README.md                       # Comprehensive documentation
```

---

## 6. Database Schema & Data Models

### 6.1. Relational Schema (PostgreSQL / SQLite)

```sql
-- Core meeting session
CREATE TABLE meetings (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    audio_file_name VARCHAR(255) NOT NULL,
    duration_seconds FLOAT NOT NULL DEFAULT 0.0,
    source_type VARCHAR(50) NOT NULL DEFAULT 'uploaded', -- 'uploaded' or 'live'
    status VARCHAR(50) NOT NULL DEFAULT 'completed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Unique speaker labels detected per meeting
CREATE TABLE speakers (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) REFERENCES meetings(id) ON DELETE CASCADE,
    speaker_label VARCHAR(100) NOT NULL,
    total_speaking_time FLOAT NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Aligned transcript utterances
CREATE TABLE transcripts (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) REFERENCES meetings(id) ON DELETE CASCADE,
    speaker_id VARCHAR(36) REFERENCES speakers(id) ON DELETE SET NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    text TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Granular action items extracted by LLM
CREATE TABLE action_items (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) REFERENCES meetings(id) ON DELETE CASCADE,
    task TEXT NOT NULL,
    owner VARCHAR(100) DEFAULT 'Unassigned',
    deadline VARCHAR(100) DEFAULT 'Not specified',
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Formal decisions reached
CREATE TABLE decisions (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) REFERENCES meetings(id) ON DELETE CASCADE,
    decision TEXT NOT NULL,
    timestamp_context VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 6.2. ChromaDB Vector Metadata Contract
Every indexed document in collection `meetwise_transcripts` includes:
```json
{
  "id": "<meeting_id>_chunk_<idx>",
  "document": "[00:12 - 00:45] SPEAKER_01: We decided to deploy the service on Friday.",
  "metadata": {
    "meeting_id": "<uuid>",
    "meeting_title": "Sprint Planning",
    "speaker": "SPEAKER_01",
    "start_time": 12.5,
    "end_time": 45.2,
    "chunk_index": 0
  }
}
```

---

## 7. API Specification & Examples

### 1. Healthcheck
- **Endpoint:** `GET /health`
- **Response:**
  ```json
  {"status": "healthy", "service": "MeetWise AI", "version": "1.0.0"}
  ```

### 2. Upload & Process Meeting Audio
- **Endpoint:** `POST /meetings/upload`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file`: Audio or video file (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.mp4`, `.webm`, `.mov`, `.mkv`)
  - `title` *(optional)*: Meeting title string
  - `num_speakers` *(optional)*: exact number of people speaking, the strongest accuracy hint for diarization
  - `min_speakers` / `max_speakers` *(optional)*: bounds when the exact count is unknown
- **Example cURL:**
  ```bash
  curl -X POST "http://localhost:8000/meetings/upload" \
       -F "file=@meeting_recording.wav" \
       -F "title=Q3 Product Roadmap Review" \
       -F "num_speakers=3"
  ```
- **Reprocess an existing meeting** (replaces previous results): `POST /meetings/{id}/process?num_speakers=3`

### Employee Voice Enrollment
- `POST /employees/register`: create the employee profile.
- `POST /employees/{employee_id}/voice-samples` (`multipart/form-data`, field `file`): add one voice clip. Returns **422** if the clip is unreadable, shorter than 1 s, silent, or not human speech; record again in that case.
- **Response:**
  ```json
  {
    "meeting_id": "8a9f6d72-bc3e-4f11-9a72-882f7e7f1234",
    "title": "Q3 Product Roadmap Review",
    "duration_seconds": 245.8,
    "source_type": "uploaded",
    "audio_file_name": "meeting_recording.wav",
    "intelligence": {
      "title": "Q3 Product Roadmap Review",
      "summary": "The engineering and product teams aligned on Q3 priorities, focusing on multi-speaker meeting intelligence.",
      "topics": ["WhisperX pipeline", "ChromaDB RAG", "Sprint deadlines"],
      "decisions": [
        {"decision": "Deploy WhisperX with int8 quantization for CPU environments", "timestamp": "02:15"}
      ],
      "action_items": [
        {"task": "Integrate WebSocket stream for live microphone mode", "owner": "Alex", "deadline": "Next Tuesday", "status": "pending"}
      ],
      "pending_issues": ["Evaluate GPU cloud instance pricing for pyannote"]
    },
    "speaker_count": 3,
    "transcript_segment_count": 42,
    "chroma_chunks_indexed": 6
  }
  ```

### 3. Retrieve Meeting Record
- **Endpoint:** `GET /meetings/{meeting_id}`
- **Returns:** Core meeting details, action items, decisions, and speaker overview.

### 4. Retrieve Aligned Transcript
- **Endpoint:** `GET /meetings/{meeting_id}/transcript`
- **Returns:** Array of all chronological utterances with speaker labels and millisecond timestamps.

### 5. Semantic Search & RAG Q&A
- **Endpoint:** `POST /search`
- **Request Body:**
  ```json
  {
    "query": "What deadline was set for the WebSocket streaming feature?",
    "meeting_id": "8a9f6d72-bc3e-4f11-9a72-882f7e7f1234"
  }
  ```
- **Response:**
  ```json
  {
    "answer": "Alex was assigned to integrate the WebSocket stream for live microphone mode with a deadline of Next Tuesday [SPEAKER_00 at 02:45].",
    "sources": [
      {
        "meeting_id": "8a9f6d72-bc3e-4f11-9a72-882f7e7f1234",
        "meeting_title": "Q3 Product Roadmap Review",
        "speaker": "SPEAKER_00",
        "start_time": 165.0,
        "end_time": 182.4,
        "text": "Alex, can you take ownership of the WebSocket stream? Let's have it ready by next Tuesday."
      }
    ]
  }
  ```

---

## 8. Gaps Encountered & Engineering Solutions

During the construction and deployment of the MeetWise AI backend, the following technical hurdles were diagnosed and resolved:

### Gap 1: WhisperX C++ Dependencies on Windows
- **Problem:** Running `pip install whisperx` on Windows failed or stalled because WhisperX relies on compiled C++ extensions (`faster-whisper`, `ctranslate2`, and `torchcodec`). Furthermore, the PyPI wheel for WhisperX 3.1 pinned older PyTorch dependencies that collided with existing installations.
- **Root Cause:** CTranslate2 requires pre-compiled binaries matching the specific Python minor version (`cp310-win_amd64`).
- **Solution:** Switched installation to the official GitHub repository head (`git+https://github.com/m-bain/whisperX.git`), which resolved wheels directly compatible with `torch 2.8.0+` on Python 3.10. Added clean fallback error wrapping in `WhisperXService`.

### Gap 2: Executable Console Scripts Missing from PATH
- **Problem:** Running `uvicorn` or `pytest` directly in PowerShell caused: `The term 'uvicorn' is not recognized as the name of a cmdlet`.
- **Root Cause:** In Windows Microsoft Store Python distributions, user site-packages are installed in `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation...`, whose `Scripts` directory is not added to the user's system `PATH`.
- **Solution:** Standardized all execution commands to invoke Python module mode: `python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload` and `python -m pytest tests/`.

### Gap 3: Windows Browser Inability to Resolve `0.0.0.0`
- **Problem:** When uvicorn bound to `0.0.0.0:8000`, users clicking terminal links encountered `ERR_CONNECTION_REFUSED (-102)` in Chrome/Edge.
- **Root Cause:** On Windows, `0.0.0.0` is an inaddr-any binding interface for socket listeners, not a routable loopback address.
- **Solution:** Documented loopback routing to use `http://localhost:8000/docs` or `http://127.0.0.1:8000/docs`.

### Gap 4: Non-Deterministic LLM JSON Output & Markdown Fences
- **Problem:** When prompting Gemini for structured JSON, models occasionally wrap payloads in markdown blocks (` ```json ... ``` `) or include trailing commas, breaking native `json.loads()`.
- **Root Cause:** Temperature fluctuations and system prompt leaks in autoregressive LLMs.
- **Solution:** Built a multi-stage JSON sanitizer in `GeminiIntelligenceService._clean_json()` that strips code fences, balances bracket pairs, removes trailing commas, and validates against Pydantic models. If complete failure occurs, a graceful fallback MoM is generated to prevent pipeline abort.

### Gap 5: Discrepancy Between ASR Segments and Speaker Turns
- **Problem:** WhisperX produces speech segments based on acoustic pauses, while pyannote produces speaker turns based on voice biometric changes. The two boundaries never align cleanly.
- **Root Cause:** Acoustic speech segmentation and speaker diarization are independent temporal tasks.
- **Solution:** Implemented the **Temporal Intersection Maximization Algorithm** in `TranscriptAligner`, which dynamically calculates the mathematical overlap between speech intervals and assigns the speaker turn with the largest intersection volume.

### Gap 6: Hugging Face Model Gating for Diarization
- **Problem:** pyannote models are gated. With pyannote.audio 4.x even `speaker-diarization-3.1` downloads files from `pyannote/speaker-diarization-community-1`; if the token owner has not accepted that model's terms, loading fails with HTTP 403.
- **Earlier behaviour (removed):** failures silently labelled every utterance `SPEAKER_00`, which looked like a one-person meeting.
- **Solution:** Diarization failure now stops processing with an explicit error explaining which model terms to accept. The default pipeline is `pyannote/speaker-diarization-community-1`, configurable via `DIARIZATION_MODEL`.

### Gap 7: SQLite vs. PostgreSQL Dialect Incompatibilities
- **Problem:** PostgreSQL uses native `UUID` and `TIMESTAMP WITH TIME ZONE` column types, which SQLite does not natively support during local development.
- **Solution:** Standardized the SQLAlchemy ORM models in `backend/app/database/models.py` to use generic string identifiers and datetime representations. SQLAlchemy transparently compiles these to SQLite types during testing and native PostgreSQL types in production.

### Gap 8: Every Speaker Matched Every Employee (Placeholder Voiceprints)
- **Problem:** Speaker identification reported similarity 1.0 for whoever was enrolled first; two different employees had voiceprints 0.99 similar.
- **Root Cause:** When embedding extraction failed, the code returned a fixed placeholder vector `[1, 0, 0, …]` instead of an error. Identical placeholders match each other perfectly. Extraction failed constantly because (a) libsndfile cannot decode browser `.webm` recordings and (b) on Apple Silicon SpeechBrain loaded onto the MPS GPU, where `tensor.numpy()` raises.
- **Solution:** Placeholders were removed: extraction raises `VoiceEmbeddingError` and the speaker stays "Unknown Speaker". Audio is decoded with PyAV, the model is pinned to CPU, and enrollment rejects silent, too-short and non-speech (pure-tone) clips. All voiceprints were rebuilt; different people now score ≤ 0.19.

### Gap 9: Identified Speakers Saved With a Fake 1.0 Confidence
- **Problem:** After a speaker was renamed to the employee's name, persistence looked up identification results by the old `SPEAKER_xx` key, missed, and defaulted to confidence 1.0 with no employee link. Speakers with too little audio got a hard-coded 75.0.
- **Solution:** Results are re-keyed by the final label; missing results default to Unknown / 0.0.

### Gap 10: "Failed to load meetings" in the UI
- **Root Causes:** (a) the `Meeting` model gained a `status` column, but `create_all()` never alters existing tables, so every query failed with `no such column: meetings.status`; (b) a relative SQLite path meant starting the server from another folder opened a different, empty database; (c) another program was occupying port 8000, which the Vite proxy targets.
- **Solution:** `init_db()` now adds missing columns automatically, relative database and Chroma paths are resolved from the project root, and the troubleshooting section below covers the port.

### Gap 11: Emotions Did Not Reflect What Was Said
- **Problem:** One emotion per speaker was stamped onto every line they spoke; sentiment percentages were made up by the LLM; backend labels (`joy`) did not match frontend colours (`happy`).
- **Solution:** Per-segment emotion labels from a fixed taxonomy, sentiment computed from those labels, and a shared label set between backend and UI (see §4.8).

### Gap 12: Reprocessing Duplicated Results
- **Problem:** Re-running a meeting appended a second transcript, decision list and speaker set next to the old ones.
- **Solution:** Reprocessing deletes the meeting's previous derived rows before saving new ones.

---

## 9. What Needs to Be Solved (Roadmap wrt Project Goal)

To advance MeetWise AI from a complete backend foundation to a production-ready enterprise solution for physical meetings, the following tasks remain:

| Priority | Area | Feature to Solve | Technical Specification |
| :--- | :--- | :--- | :--- |
| **Critical** | **Environment** | **Complete WhisperX Local Build** | Finalize the local pip installation of `whisperx` / `ctranslate2` in the environment so live transcription can execute without network timeouts. |
| **High** | **Audio Ingestion** | **Live Mode (Microphone Streaming)** | Implement a FastAPI WebSocket route (`/meetings/live/ws`) that accepts streaming PCM audio from USB room microphones or ESP32 audio boards into a circular buffer for rolling transcription. |
| **High** | **Diarization** | **Diarize Before Denoising** | Run pyannote and voiceprint extraction on the 16 kHz audio before spectral gating, and compare accuracy; denoising artifacts can blur differences between voices. |
| **High** | **Speaker ID** | **Confirm & Learn From Meetings** | Let users confirm or correct a speaker in the UI and add that meeting audio to the employee's voiceprint, so profiles match real meeting microphones. |
| **Medium** | **Frontend** | **Waveform Player** | Audio player synced with transcript text and speaker colours. |
| **Medium** | **Pipeline** | **Incremental Long-Meeting Processing** | For 2+ hour board meetings, implement chunked streaming through WhisperX and Gemini to stream partial transcripts to users instead of waiting for post-meeting batch completion. |
| **Medium** | **Multilingual** | **Multilingual Embedding & ASR** | WhisperX supports 99 languages. Upgrade dense retrieval from `bge-small-en` to `BAAI/bge-m3` or `multilingual-e5` to enable cross-lingual RAG queries. |
| **Medium** | **Hardware** | **CUDA / GPU Acceleration** | Configure Docker Compose with NVIDIA Container Toolkit (`runtime: nvidia`) to run WhisperX in `float16` on GPU, reducing processing time from 1x real-time to 0.1x real-time. |
| **Low** | **Integrations** | **Calendar & Task Sync** | Webhook connectors to automatically export action items to Jira, Trello, or Slack, and pull meeting titles and attendees from Google Calendar / Microsoft Outlook. |
| **Low** | **Export** | **Formal Document Generation** | Generate branded PDF, DOCX, and Markdown Minutes of Meeting (MoM) documents for executive sign-off. |

---

## 10. Installation, Configuration & Verification

### Step 1: Clone Repository & Open Directory
```bash
git clone https://github.com/shazeen21/MEETWISE-CAPSTONE-PROJECT.git
cd MEETWISE-CAPSTONE-PROJECT
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```
Create the virtual environment on the machine you run it on; a venv copied from Windows will not work on macOS/Linux.

### Step 2: Configure Environment Variables
Copy the template and configure your API keys:
```powershell
cp backend/.env.example backend/.env      # Windows: copy backend\.env.example backend\.env
```
Edit `backend/.env` with your preferred text editor:
```env
# Google Gemini API key (from https://aistudio.google.com/)
GEMINI_API_KEY=AIzaSy...

# Hugging Face Read Token (from https://huggingface.co/settings/tokens)
# While logged in as the token owner, accept the terms at:
#   https://huggingface.co/pyannote/speaker-diarization-community-1
HF_TOKEN=hf_...
DIARIZATION_MODEL=pyannote/speaker-diarization-community-1

# Default SQLite database for local development
DATABASE_URL=sqlite:///./data/meetwise.db

# Whisper configuration
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Denoising strategy: spectral, deep, or passthrough
DENOISER_TYPE=spectral

# Dense embedding model
BGE_MODEL=BAAI/bge-small-en-v1.5
```

### Step 3: Install Dependencies
```powershell
# Install WhisperX from GitHub
python -m pip install git+https://github.com/m-bain/whisperX.git

# Install all backend requirements
python -m pip install -r backend/requirements.txt
```

### Step 4: Run the Test Suite
Verify that all components (database, aligner, chunker, RAG, API endpoints) are functioning properly:
```powershell
python -m pytest tests/ -v
```
*(Expected: 36 tests passed.)* Tests use an in-memory database and temporary folders, so they never modify `data/`.

### Step 5: Start the Backend and Frontend
```bash
# Terminal 1 - backend (port 8000; the frontend proxy expects this port)
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 - frontend
cd frontend && npm install && npm run dev      # http://localhost:5173
```

### Step 5b: Enroll Employees
In the UI open **Employees → Register**, then record 4–5 clips of 5–10 seconds of natural speech per person, ideally with the same kind of microphone used in meetings. Clips that are silent, too short or not speech are rejected with a message. After upgrading from an older version, rebuild existing voiceprints once:
```bash
python scripts/reenroll_voiceprints.py           # dry run report
python scripts/reenroll_voiceprints.py --apply   # save
```

### Step 6: Access API Documentation & Test
Open your browser and navigate to:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

1. Scroll to `POST /meetings/upload`.
2. Click **Try it out**.
3. Upload an audio file (`.wav`, `.mp3`, `.m4a`) from your machine (or use `data/audio/sample_meeting.wav`).
4. Click **Execute** and review the structured intelligence output.
5. Use `POST /search` to ask questions about the meeting.

### Tips for Accurate Speaker Diarization
1. **Give the number of speakers** when uploading; guessing the count is the most common diarization error.
2. **Record well:** one good room/conference microphone near the speakers, WAV or high-bitrate audio. Compressed phone voice notes from one end of the table are the hardest case.
3. **Avoid crosstalk:** overlapping speech is the hardest part for both diarization and transcription.
4. **Enroll the way you record:** same microphone type and room, natural conversational speech rather than read sentences.

### Troubleshooting
| Symptom | Cause & Fix |
| :--- | :--- |
| *Failed to load meetings* | The backend is not reachable on port 8000. Check `lsof -i :8000` (macOS/Linux): if another program owns the port, stop it, then start the backend. Also open `http://localhost:8000/health`. |
| Upload fails with *Speaker Diarization failed … 403* | The Hugging Face token owner has not accepted the terms of `pyannote/speaker-diarization-community-1`. |
| Everyone is *Unknown Speaker* | No one is enrolled, the enrollment clips are poor, or the meeting audio is too short/noisy. Check the backend log line `Speaker 'SPEAKER_00' -> … (cosine similarity …; reason)`. |
| Voice sample rejected (422) | The clip was empty, under 1 s, silent, or not speech. Record again. |

---

## 11. Authors & Capstone Project Credits

- **Project:** MeetWise AI — Final-Year Capstone Project
- **Specialization:** Artificial Intelligence, Speech Processing & Organizational Intelligence Systems
- **License:** Educational & Research Capstone License
