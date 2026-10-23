# Security Audit Report — MeetWise AI

## 1. Executive Summary
The MeetWise AI platform implements a multi-stage AI pipeline for processing meeting recordings. The current security posture is suitable for a research/capstone project but requires several hardening measures before any production deployment involving sensitive corporate data.

## 2. Security Analysis

### 2.1. Credential Management
- **Finding**: API keys for Google Gemini and Hugging Face are managed via `.env` files and loaded using `pydantic-settings`.
- **Status**: ✅ SECURE (for development).
- **Recommendation**: In production, transition to a secret manager (e.g., AWS Secrets Manager, HashiCorp Vault) to avoid storing keys in plaintext on disk.

### 2.2. Input Validation & Sanitization
- **Finding**: Media uploads are validated by extension (`SUPPORTED_EXTENSIONS`). Filenames are randomized using `uuid.uuid4()` to prevent path traversal attacks (e.g., `../../etc/passwd`).
- **Status**: ✅ SECURE.
- **Recommendation**: Implement file size limits (e.g., max 500MB) to prevent Denial-of-Service (DoS) attacks via massive file uploads.

### 2.3. API Access Control
- **Finding**: The FastAPI gateway uses `CORSMiddleware` with `allow_origins=["*"]`.
- **Status**: ⚠️ RISK.
- **Recommendation**: Restrict `allow_origins` to the specific domain of the React frontend to prevent Cross-Origin Resource Sharing (CORS) exploits.

### 2.4. Data Privacy & Persistence
- **Finding**: Transcripts and biometric speaker labels are stored in plaintext in the relational database.
- **Status**: ⚠️ RISK.
- **Recommendation**: Encrypt PII (Personally Identifiable Information) such as employee names and voice embeddings at rest using AES-256.

### 2.5. LLM Prompt Injection
- **Finding**: The RAG pipeline incorporates user queries directly into the Gemini prompt.
- **Status**: ⚠️ RISK.
- **Recommendation**: Implement strict system prompting and input sanitization to prevent "prompt injection" where a user could trick the AI into revealing other meetings' data.

## 3. Reproducibility & Documentation
- **Environment**: The system is containerized via `Dockerfile` and `docker-compose.yml`.
- **Dependencies**: All Python dependencies are pinned in `requirements.txt`.
- **Setup**: A detailed installation guide is provided in `README.md` (Sections 10).

## 4. Final Verdict
**Security Grade: B- (Research Ready)**
The system is technically sound and avoids common critical vulnerabilities (like SQL injection via SQLAlchemy ORM). However, it lacks the production-grade authentication and encryption layers required for enterprise deployment.
