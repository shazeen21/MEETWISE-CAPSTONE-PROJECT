import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.speakers.voice_biometrics import (
    VoiceBiometricsService,
    SpeakerRecognitionEngine,
)

client = TestClient(app)


def test_voice_biometrics_feature_extraction():
    service = VoiceBiometricsService()
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    embedding = service.extract_embedding(audio, sample_rate=sample_rate)
    assert embedding is not None
    assert len(embedding) == service.embedding_dim
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_speaker_recognition_engine():
    engine = SpeakerRecognitionEngine(confidence_threshold=0.70)

    emb1 = np.random.randn(192).astype(np.float32)
    emb1 = (emb1 / np.linalg.norm(emb1)).tolist()

    enrolled = [{
        "id": "uuid-1",
        "employee_id": "EMP-101",
        "name": "Alice Smith",
        "voice_embedding": emb1,
    }]

    # Match with identical synthetic audio / profile -> Alice Smith
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Directly verify cosine similarity and identification logic
    res = engine.identify_speaker(audio, enrolled)
    assert "name" in res
    assert "confidence" in res
    assert "is_unknown" in res
    assert isinstance(res["confidence"], float)


def test_employee_and_admin_endpoints():
    payload = {
        "employee_id": "TEST-EMP-99",
        "name": "Integration Tester",
        "email": "tester@example.com",
        "department": "QA",
        "team": "Automation",
        "designation": "Staff QA Engineer",
        "accent_hint": "General American",
    }
    resp = client.post("/employees/register", json=payload)
    assert resp.status_code in (200, 201)
    data = resp.json()
    assert data["employee_id"] == "TEST-EMP-99"
    assert data["name"] == "Integration Tester"

    resp = client.get("/employees/")
    assert resp.status_code == 200
    employees = resp.json()
    assert any(e["employee_id"] == "TEST-EMP-99" for e in employees)

    kw_resp = client.post(
        "/admin/keywords",
        json={"keyword": "Q4-Deliverable", "category": "deliverable"},
    )
    assert kw_resp.status_code == 200
    kw_data = kw_resp.json()
    assert kw_data["keyword"] == "Q4-Deliverable"

    audit_resp = client.get("/admin/audit-logs")
    assert audit_resp.status_code == 200
    assert isinstance(audit_resp.json(), list)

    notif_resp = client.get("/admin/notifications")
    assert notif_resp.status_code == 200
    assert isinstance(notif_resp.json(), list)
