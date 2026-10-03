-- ========================================================
-- MeetWise AI Enterprise Platform: PostgreSQL Database Schema
-- Multi-Speaker Meeting Intelligence & Organizational Memory
-- ========================================================

-- 1. EMPLOYEES TABLE (Voice Profiles & Registration)
CREATE TABLE IF NOT EXISTS employees (
    id VARCHAR(36) PRIMARY KEY,
    employee_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    department VARCHAR(100) NOT NULL DEFAULT 'General',
    team VARCHAR(100) NOT NULL DEFAULT 'Core',
    designation VARCHAR(100) NOT NULL DEFAULT 'Team Member',
    voice_embedding TEXT,
    accent_metadata TEXT,
    voice_samples_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_employees_emp_id ON employees (employee_id);
CREATE INDEX IF NOT EXISTS idx_employees_email ON employees (email);
CREATE INDEX IF NOT EXISTS idx_employees_dept ON employees (department);

-- 2. VOICE SAMPLES TABLE (Enrolled Speech Audio)
CREATE TABLE IF NOT EXISTS voice_samples (
    id VARCHAR(36) PRIMARY KEY,
    employee_id VARCHAR(36) NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    audio_file_name VARCHAR(255) NOT NULL,
    duration_seconds DOUBLE PRECISION DEFAULT 0.0,
    prompt_text TEXT,
    embedding_vector TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_voice_samples_emp_id ON voice_samples (employee_id);

-- 3. MEETINGS TABLE
CREATE TABLE IF NOT EXISTS meetings (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    meeting_date TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    duration_seconds DOUBLE PRECISION DEFAULT 0.0,
    audio_file_name VARCHAR(255) NOT NULL,
    media_type VARCHAR(20) NOT NULL DEFAULT 'audio',
    source_type VARCHAR(50) NOT NULL DEFAULT 'uploaded' CHECK (source_type IN ('uploaded', 'live')),
    summary TEXT,
    manager_summary TEXT,
    overall_sentiment TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_meetings_date ON meetings (meeting_date);
CREATE INDEX IF NOT EXISTS idx_meetings_source ON meetings (source_type);

-- 4. SPEAKERS TABLE
CREATE TABLE IF NOT EXISTS speakers (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    employee_id VARCHAR(36) REFERENCES employees(id) ON DELETE SET NULL,
    speaker_label VARCHAR(64) NOT NULL,
    speaker_name VARCHAR(255),
    confidence_score DOUBLE PRECISION DEFAULT 1.0,
    is_unknown INTEGER DEFAULT 0,
    detected_accent VARCHAR(100),
    dominant_emotion VARCHAR(100),
    total_speaking_time DOUBLE PRECISION DEFAULT 0.0,
    CONSTRAINT uq_meeting_speaker_label UNIQUE (meeting_id, speaker_label)
);

CREATE INDEX IF NOT EXISTS idx_speakers_meeting_id ON speakers (meeting_id);
CREATE INDEX IF NOT EXISTS idx_speakers_emp_id ON speakers (employee_id);
CREATE INDEX IF NOT EXISTS idx_speakers_label ON speakers (speaker_label);

-- 5. TRANSCRIPTS TABLE
CREATE TABLE IF NOT EXISTS transcripts (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker_id VARCHAR(36) REFERENCES speakers(id) ON DELETE SET NULL,
    start_time DOUBLE PRECISION NOT NULL,
    end_time DOUBLE PRECISION NOT NULL,
    text TEXT NOT NULL,
    emotion VARCHAR(100) DEFAULT 'neutral',
    speaking_style VARCHAR(100) DEFAULT 'professional',
    segment_index INTEGER DEFAULT 0,
    CONSTRAINT chk_transcript_time CHECK (end_time >= start_time)
);

CREATE INDEX IF NOT EXISTS idx_transcripts_meeting_id ON transcripts (meeting_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_speaker_id ON transcripts (speaker_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_start_time ON transcripts (meeting_id, start_time);

-- 6. ACTION ITEMS TABLE
CREATE TABLE IF NOT EXISTS action_items (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    task TEXT NOT NULL,
    owner VARCHAR(255),
    deadline VARCHAR(100),
    priority VARCHAR(20) DEFAULT 'Medium',
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed')),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_action_items_meeting_id ON action_items (meeting_id);
CREATE INDEX IF NOT EXISTS idx_action_items_owner ON action_items (owner);
CREATE INDEX IF NOT EXISTS idx_action_items_status ON action_items (status);

-- 7. DECISIONS TABLE
CREATE TABLE IF NOT EXISTS decisions (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    decision TEXT NOT NULL,
    timestamp VARCHAR(50),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_decisions_meeting_id ON decisions (meeting_id);

-- 8. TOPICS TABLE
CREATE TABLE IF NOT EXISTS topics (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    topic_name VARCHAR(255) NOT NULL,
    start_time DOUBLE PRECISION DEFAULT 0.0,
    end_time DOUBLE PRECISION DEFAULT 0.0,
    duration_seconds DOUBLE PRECISION DEFAULT 0.0,
    participants TEXT DEFAULT '[]',
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_topics_meeting_id ON topics (meeting_id);

-- 9. KEYWORDS TABLE
CREATE TABLE IF NOT EXISTS keywords (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    keyword VARCHAR(255) NOT NULL,
    category VARCHAR(50) DEFAULT 'general',
    relevance_score DOUBLE PRECISION DEFAULT 1.0,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_keywords_meeting_id ON keywords (meeting_id);

-- 10. EMPLOYEE REPORTS TABLE
CREATE TABLE IF NOT EXISTS employee_reports (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    employee_id VARCHAR(36) REFERENCES employees(id) ON DELETE CASCADE,
    employee_name VARCHAR(255) NOT NULL,
    topics_discussed TEXT DEFAULT '[]',
    decisions_affecting TEXT DEFAULT '[]',
    assigned_tasks TEXT DEFAULT '[]',
    mentioned_deadlines TEXT DEFAULT '[]',
    follow_ups TEXT DEFAULT '[]',
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_emp_reports_meeting ON employee_reports (meeting_id);
CREATE INDEX IF NOT EXISTS idx_emp_reports_emp ON employee_reports (employee_id);

-- 11. CONFIGURED KEYWORDS TABLE
CREATE TABLE IF NOT EXISTS configured_keywords (
    id VARCHAR(36) PRIMARY KEY,
    keyword VARCHAR(255) UNIQUE NOT NULL,
    category VARCHAR(50) DEFAULT 'custom',
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 12. NOTIFICATIONS TABLE
CREATE TABLE IF NOT EXISTS notifications (
    id VARCHAR(36) PRIMARY KEY,
    meeting_id VARCHAR(36) REFERENCES meetings(id) ON DELETE CASCADE,
    recipient VARCHAR(255) NOT NULL,
    channel VARCHAR(50) DEFAULT 'internal',
    status VARCHAR(50) DEFAULT 'sent',
    subject VARCHAR(255),
    payload TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 13. AUDIT LOGS TABLE
CREATE TABLE IF NOT EXISTS audit_logs (
    id VARCHAR(36) PRIMARY KEY,
    action VARCHAR(100) NOT NULL,
    user_id VARCHAR(100) DEFAULT 'system',
    details TEXT,
    ip_address VARCHAR(50),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
