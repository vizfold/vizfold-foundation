from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from app.db.database import get_db
from app.db.models import Protein, InferenceJob
from app.tasks.openfold_tasks import run_openfold_inference

router = APIRouter()


# Pydantic schemas
class InferenceRequest(BaseModel):
    protein_id: int
    extract_attention: bool = True


class InferenceJobResponse(BaseModel):
    id: int
    job_id: str
    protein_id: int
    status: str
    progress: float
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_pdb: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


@router.post("/run", response_model=InferenceJobResponse)
async def run_inference(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start OpenFold inference job"""

    # Check if protein exists
    protein = db.query(Protein).filter(Protein.id == request.protein_id).first()
    if not protein:
        raise HTTPException(status_code=404, detail="Protein not found")

    # Create job
    job_id = str(uuid.uuid4())
    job = InferenceJob(
        protein_id=request.protein_id,
        job_id=job_id,
        status="pending",
        metadata={"extract_attention": request.extract_attention}
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start inference task asynchronously
    background_tasks.add_task(
        run_openfold_inference,
        job_id=job_id,
        protein_sequence=protein.sequence,
        extract_attention=request.extract_attention
    )

    return job


@router.get("/{job_id}/status", response_model=InferenceJobResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """Get inference job status"""
    job = db.query(InferenceJob).filter(InferenceJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/protein/{protein_id}", response_model=list[InferenceJobResponse])
async def get_protein_jobs(
    protein_id: int,
    db: Session = Depends(get_db)
):
    """Get all inference jobs for a protein"""
    jobs = db.query(InferenceJob).filter(
        InferenceJob.protein_id == protein_id
    ).order_by(InferenceJob.created_at.desc()).all()
    return jobs


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """Cancel/delete inference job"""
    job = db.query(InferenceJob).filter(InferenceJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # TODO: Implement actual job cancellation in Celery
    if job.status in ["pending", "running"]:
        job.status = "cancelled"
        db.commit()

    return {"message": "Job cancelled"}
