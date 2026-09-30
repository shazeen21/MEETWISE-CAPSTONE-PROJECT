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
                   (WAV / MP3 / M4A from Upload or Mic)
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
    │ - Word-level Timestamps │             │ - Agglomerative Cluster │
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
                   │ Phase 5: Gemini LLM Intelligence  │
                   │ - MoM & Executive Summary         │
                   │ - Action Items (Owner, Deadline)  │
                   │ - Formal Decisions & Blockers     │
                   │ - Pydantic Schema Validation      │
                   └─────────────────┬─────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │ Phase 6: Relational DB  │             │ Phase 7: Vector Store   │
    │ - PostgreSQL / SQLite   │             │ - Semantic Chunker      │
    │ - SQLAlchemy ORM        │             │ - BAAI BGE Embeddings   │
    │ - meetings, speakers,   │             │ - ChromaDB Collection   │
    │   transcripts, actions  │             │   (metadata enriched)   │
    └─────────────────────────┘             └────────────┬────────────┘
                                                         │
                                                         ▼
                                            ┌─────────────────────────┐
                                            │ Phase 8: RAG Assistant  │
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
| **Speaker Diarization** | **pyannote.audio** | `3.1.1 / 4.0.7` | Industry benchmark for multi-speaker segmentation and speaker turn detection. Provides neural voice embeddings with clustering, outperforming energy-based voice activity detectors. |
| **Meeting Intelligence**| **Google Gemini** | `gemini-2.0-flash-exp` / `gemini-1.5-flash` | Selected for its massive 1M+ token context window (capable of processing 5+ hours of continuous transcript in a single prompt without chunk fragmentation) and near-zero cost/token. |
| **Relational Storage** | **PostgreSQL** + **SQLAlchemy** | `15+` / `>=2.0.28` | Full ACID compliance, foreign key cascades (`ON DELETE CASCADE`), index-backed timestamp queries. Abstracted via SQLAlchemy ORM so SQLite can be used for instant local development with zero setup. |
| **Dense Embeddings** | **BAAI/bge-small-en-v1.5** | via `sentence-transformers` | Top-ranked compact embedding model on Hugging Face MTEB benchmark. 384-dimensional vectors run quickly on CPU while outperforming older 1536-dim models in retrieval accuracy. Prepend instruction format optimizes asymmetric question-passage similarity. |
| **Vector Database** | **ChromaDB** | `>=0.4.24` | Embedded, in-process vector store requiring zero external daemon or cloud subscription. Persists locally in Parquet/DuckDB format with cosine distance indexing. |
| **Testing** | **Pytest + HTTPX** | `>=8.0.0` | Isolated test fixtures for mocked database, ASR, and RAG pipelines; executes entire test suite in under 3 seconds. |

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

### 4.5. Context-Preserving Semantic Chunking
Naïve fixed-character chunking splits words across boundaries and severs speaker attribution. MeetWise uses **Semantic Transcript Chunking**:
- Groups contiguous utterances by speaker.
- Respects speaker turn shifts as natural semantic breakpoints.
- Accumulates chunks up to a target size (400 words) with a sliding overlap (50 words).
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
│       │   └── search.py           # Endpoint: /search (RAG queries)
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
│       └── database/
│           ├── models.py           # SQLAlchemy declarative ORM models
│           ├── connection.py       # Engine, SessionLocal, init_db lifecycle
│           └── repository.py       # Atomic multi-table database persistence
├── database/
│   └── schema.sql                  # PostgreSQL production DDL with cascades & indexes
├── data/
│   ├── audio/                      # Ingested meeting recordings
│   ├── processed/                  # Normalized 16kHz WAV files
│   ├── transcripts/                # Merged transcript JSON files
│   └── chromadb/                   # Persistent ChromaDB vector files
├── scripts/
│   ├── generate_sample_audio.py    # Synthetic multi-speaker WAV test generator
│   └── run_pipeline_cli.py         # Standalone CLI pipeline runner
├── tests/
│   ├── conftest.py                 # Pytest mocks & database fixtures
│   ├── test_aligner.py             # Aligner logic tests
│   ├── test_database.py            # ORM CRUD & cascade deletion tests
│   ├── test_chunker.py             # Chunking & overlap tests
│   ├── test_rag.py                 # Grounded RAG query tests
│   ├── test_pipeline.py            # Master orchestrator integration tests
│   └── test_api.py                 # FastAPI endpoint integration tests
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
  - `file`: Binary audio file (`.wav`, `.mp3`, `.m4a`)
  - `title` *(optional)*: Meeting title string
- **Example cURL:**
  ```bash
  curl -X POST "http://localhost:8000/meetings/upload" \
       -F "file=@meeting_recording.wav" \
       -F "title=Q3 Product Roadmap Review"
  ```
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

During the construction and deployment of the MeetWise AI backend, seven critical technical hurdles were diagnosed and resolved:

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
- **Problem:** `pyannote/speaker-diarization-3.1` is a gated repository requiring user agreement on both segmentation and diarization model cards. Without it, Pyannote throws an HTTP 403 / 401 error.
- **Solution:** Built non-blocking exception handling in `pipeline.py`. If `HF_TOKEN` is absent or unauthorized, the system emits a warning log and automatically assigns all utterances to `SPEAKER_00`, allowing transcription and intelligence extraction to proceed unimpeded.

### Gap 7: SQLite vs. PostgreSQL Dialect Incompatibilities
- **Problem:** PostgreSQL uses native `UUID` and `TIMESTAMP WITH TIME ZONE` column types, which SQLite does not natively support during local development.
- **Solution:** Standardized the SQLAlchemy ORM models in `backend/app/database/models.py` to use generic string identifiers and datetime representations. SQLAlchemy transparently compiles these to SQLite types during testing and native PostgreSQL types in production.

---

## 9. What Needs to Be Solved (Roadmap wrt Project Goal)

To advance MeetWise AI from a complete backend foundation to a production-ready enterprise solution for physical meetings, the following tasks remain:

| Priority | Area | Feature to Solve | Technical Specification |
| :--- | :--- | :--- | :--- |
| **Critical** | **Environment** | **Complete WhisperX Local Build** | Finalize the local pip installation of `whisperx` / `ctranslate2` in the environment so live transcription can execute without network timeouts. |
| **High** | **Audio Ingestion** | **Live Mode (Microphone Streaming)** | Implement a FastAPI WebSocket route (`/meetings/live/ws`) that accepts streaming PCM audio from USB room microphones or ESP32 audio boards into a circular buffer for rolling transcription. |
| **High** | **Diarization** | **Speaker Voice Enrollment (Voiceprints)** | Create a speaker enrollment module (`POST /speakers/enroll`) where team members record a 15-second sample to generate d-vector voice embeddings. Automatically map `SPEAKER_00` to "Alice (Product Lead)". |
| **High** | **Frontend** | **User Interface (React / Next.js)** | Build the user-facing web dashboard: interactive audio waveform player synced with transcript text, speaker color badges, action item checklist, and RAG chat window. |
| **Medium** | **Pipeline** | **Incremental Long-Meeting Processing** | For 2+ hour board meetings, implement chunked streaming through WhisperX and Gemini to stream partial transcripts to users instead of waiting for post-meeting batch completion. |
| **Medium** | **Multilingual** | **Multilingual Embedding & ASR** | WhisperX supports 99 languages. Upgrade dense retrieval from `bge-small-en` to `BAAI/bge-m3` or `multilingual-e5` to enable cross-lingual RAG queries. |
| **Medium** | **Hardware** | **CUDA / GPU Acceleration** | Configure Docker Compose with NVIDIA Container Toolkit (`runtime: nvidia`) to run WhisperX in `float16` on GPU, reducing processing time from 1x real-time to 0.1x real-time. |
| **Low** | **Integrations** | **Calendar & Task Sync** | Webhook connectors to automatically export action items to Jira, Trello, or Slack, and pull meeting titles and attendees from Google Calendar / Microsoft Outlook. |
| **Low** | **Export** | **Formal Document Generation** | Generate branded PDF, DOCX, and Markdown Minutes of Meeting (MoM) documents for executive sign-off. |

---

## 10. Installation, Configuration & Verification

### Step 1: Clone Repository & Open Directory
```powershell
cd c:\Users\hp\Desktop\MEETWISE
```

### Step 2: Configure Environment Variables
Copy the template and configure your API keys:
```powershell
copy backend\.env.example backend\.env
```
Edit `backend/.env` with your preferred text editor:
```env
# Google Gemini API key (from https://aistudio.google.com/)
GEMINI_API_KEY=AIzaSy...

# Hugging Face Read Token (from https://huggingface.co/settings/tokens)
# Must have accepted access to:
# 1. https://hf.co/pyannote/speaker-diarization-3.1
# 2. https://hf.co/pyannote/segmentation-3.0
HF_TOKEN=hf_...

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
*(Expected: 12 tests passed)*

### Step 5: Start the Backend Server
```powershell
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 6: Access API Documentation & Test
Open your browser and navigate to:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

1. Scroll to `POST /meetings/upload`.
2. Click **Try it out**.
3. Upload an audio file (`.wav`, `.mp3`, `.m4a`) from your machine (or use `data/audio/sample_meeting.wav`).
4. Click **Execute** and review the structured intelligence output.
5. Use `POST /search` to ask questions about the meeting.

---

## 11. Authors & Capstone Project Credits

- **Project:** MeetWise AI — Final-Year Capstone Project
- **Specialization:** Artificial Intelligence, Speech Processing & Organizational Intelligence Systems
- **License:** Educational & Research Capstone License
# MEETWISE-CAPSTONE-PROJECT
