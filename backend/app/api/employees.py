"""Employee Registration and Voice Biometrics API Router for MeetWise AI.

Handles:
- Employee onboarding & profile management
- Voice sample enrollment via browser recording or audio file upload
- Acoustic voiceprint centroid generation & profile updates
- Employee roster listing
"""

import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from ..config import settings
from ..database.connection import get_db
from ..database.repository import MeetingRepository
from ..speakers.voice_biometrics import VoiceBiometricsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/employees", tags=["Employee Voice Onboarding"])
biometrics_service = VoiceBiometricsService()


# --- Pydantic Request & Response Schemas ---

class EmployeeRegisterRequest(BaseModel):
    employee_id: str = Field(..., description="Corporate Employee ID")
    name: str = Field(..., description="Full Name")
    email: str = Field(..., description="Corporate Email")
    department: str = Field(default="Engineering", description="Department")
    team: str = Field(default="Backend Core", description="Team")
    designation: str = Field(default="Senior Software Engineer")
    accent_hint: Optional[str] = Field(default="American")


class EmployeeResponse(BaseModel):
    id: str
    employee_id: str
    name: str
    email: str
    department: str
    team: str
    designation: str
    voice_samples_count: int
    has_enrolled_voice: bool
    created_at: str


class VoiceEnrollmentResponse(BaseModel):
    message: str
    employee_id: str
    employee_name: str
    samples_enrolled: int
    voice_enrolled: bool
    sample_duration_seconds: float


@router.post("/register", response_model=EmployeeResponse, status_code=status.HTTP_201_CREATED)
def register_employee(
    payload: EmployeeRegisterRequest,
    db: Session = Depends(get_db),
):
    """Register a new employee for voice identification onboarding."""
    repo = MeetingRepository(db)

    # Check for existing employee ID or email
    if repo.get_employee(payload.employee_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Employee with ID '{payload.employee_id}' already exists.",
        )
    if repo.get_employee_by_email(payload.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Employee with email '{payload.email}' already exists.",
        )

    accent_meta = {"primary_accent": payload.accent_hint} if payload.accent_hint else None
    emp = repo.create_employee(
        employee_id=payload.employee_id,
        name=payload.name,
        email=payload.email,
        department=payload.department,
        team=payload.team,
        designation=payload.designation,
        accent_metadata=accent_meta,
    )

    return EmployeeResponse(
        id=emp.id,
        employee_id=emp.employee_id,
        name=emp.name,
        email=emp.email,
        department=emp.department,
        team=emp.team,
        designation=emp.designation,
        voice_samples_count=emp.voice_samples_count,
        has_enrolled_voice=emp.voice_embedding is not None,
        created_at=emp.created_at.isoformat(),
    )


@router.get("", response_model=List[EmployeeResponse])
def list_employees(
    department: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List all registered employees with their voice enrollment status."""
    repo = MeetingRepository(db)
    employees = repo.list_employees(department=department, limit=limit)
    return [
        EmployeeResponse(
            id=emp.id,
            employee_id=emp.employee_id,
            name=emp.name,
            email=emp.email,
            department=emp.department,
            team=emp.team,
            designation=emp.designation,
            voice_samples_count=emp.voice_samples_count,
            has_enrolled_voice=emp.voice_embedding is not None,
            created_at=emp.created_at.isoformat(),
        )
        for emp in employees
    ]


@router.get("/{id_or_emp_id}", response_model=EmployeeResponse)
def get_employee(
    id_or_emp_id: str,
    db: Session = Depends(get_db),
):
    """Fetch employee profile details by UUID or corporate Employee ID."""
    repo = MeetingRepository(db)
    emp = repo.get_employee(id_or_emp_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found.")

    return EmployeeResponse(
        id=emp.id,
        employee_id=emp.employee_id,
        name=emp.name,
        email=emp.email,
        department=emp.department,
        team=emp.team,
        designation=emp.designation,
        voice_samples_count=emp.voice_samples_count,
        has_enrolled_voice=emp.voice_embedding is not None,
        created_at=emp.created_at.isoformat(),
    )


@router.post("/{id_or_emp_id}/voice-samples", response_model=VoiceEnrollmentResponse)
async def upload_voice_sample(
    id_or_emp_id: str,
    file: UploadFile = File(..., description="Audio recording of prompted speech (.wav, .mp3, .webm, .m4a)"),
    prompt_text: Optional[str] = Form(None, description="The prompted sentence read by employee"),
    db: Session = Depends(get_db),
):
    """Enroll a voice sample for an employee.

    Extracts acoustic voiceprint embedding, updates the employee's running centroid vector,
    and stores the audio sample for continuous speaker recognition.
    """
    repo = MeetingRepository(db)
    emp = repo.get_employee(id_or_emp_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found.")

    # Save uploaded voice sample
    sample_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    dest_name = f"voice_sample_{emp.employee_id}_{sample_id}{suffix}"
    voice_dir = Path(settings.DATA_DIR) / "voice_samples"
    voice_dir.mkdir(parents=True, exist_ok=True)
    dest_path = voice_dir / dest_name

    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)

        # Extract 192-d acoustic voiceprint embedding
        sample_emb = biometrics_service.extract_embedding(str(dest_path))
        sample_emb_list = [round(float(x), 6) for x in sample_emb]

        # Calculate duration
        try:
            import soundfile as sf
            info = sf.info(str(dest_path))
            duration = round(info.duration, 2)
        except Exception:
            duration = 5.0

        # Save sample to database
        repo.add_voice_sample(
            employee_id=emp.id,
            audio_file_name=dest_name,
            duration_seconds=duration,
            prompt_text=prompt_text or "Voice enrollment sample",
            embedding_vector=sample_emb_list,
        )

        # Re-aggregate all samples for this employee to compute centroid vector
        all_samples = [sample for sample in emp.voice_samples if sample.id != sample_id]
        sample_vectors = []
        for s in all_samples:
            if s.embedding_vector:
                import json
                sample_vectors.append(json.loads(s.embedding_vector))
        sample_vectors.append(sample_emb_list)

        new_centroid = biometrics_service.aggregate_centroid(sample_vectors)
        repo.update_employee_voice_profile(emp.id, new_centroid)

        logger.info(f"Successfully enrolled voice sample for employee {emp.name} ({emp.employee_id})")
        return VoiceEnrollmentResponse(
            message=f"Voice sample enrolled successfully for {emp.name}.",
            employee_id=emp.employee_id,
            employee_name=emp.name,
            samples_enrolled=emp.voice_samples_count,
            voice_enrolled=True,
            sample_duration_seconds=duration,
        )

    except Exception as e:
        logger.error(f"Voice enrollment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice enrollment failed: {e}")
