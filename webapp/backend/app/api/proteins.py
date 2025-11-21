from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import os
import shutil

from app.db.database import get_db
from app.db.models import Protein
from app.core.config import settings
from app.services.pdb_fetch_service import PDBFetchService

router = APIRouter()


def fetch_pdb_background(protein_id: int, protein_name: str, protein_dir: str, db_url: str):
    """Background task to fetch PDB file without blocking the response"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    pdb_fetch = PDBFetchService()
    if not pdb_fetch.is_valid_pdb_id(protein_name):
        return

    try:
        pdb_path = pdb_fetch.fetch_pdb(protein_name, protein_dir)

        # Update database in background
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        try:
            protein = db.query(Protein).filter(Protein.id == protein_id).first()
            if protein:
                protein.pdb_file = pdb_path
                db.commit()
            print(f"✓ PDB fetched for {protein_name}")
        finally:
            db.close()
    except Exception as e:
        print(f"✗ PDB fetch failed for {protein_name}: {e}")


# Pydantic schemas
class ProteinBase(BaseModel):
    name: str
    sequence: str
    description: Optional[str] = None


class ProteinCreate(ProteinBase):
    pass


class ProteinResponse(ProteinBase):
    id: int
    sequence_length: int
    pdb_file: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


@router.post("/", response_model=ProteinResponse)
async def create_protein(
    protein: ProteinCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new protein from sequence (FAST: PDB fetch in background)"""
    sequence = protein.sequence.replace(" ", "").replace("\n", "").upper()

    if len(sequence) > settings.MAX_SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence length exceeds maximum of {settings.MAX_SEQUENCE_LENGTH}"
        )

    db_protein = Protein(
        name=protein.name,
        sequence=sequence,
        sequence_length=len(sequence),
        description=protein.description
    )
    db.add(db_protein)
    db.commit()
    db.refresh(db_protein)

    # Create output directory structure for this protein
    protein_dir = os.path.join(settings.OUTPUT_DIR, f"protein_{db_protein.id}")
    os.makedirs(os.path.join(protein_dir, "attention"), exist_ok=True)
    os.makedirs(os.path.join(protein_dir, "visualizations"), exist_ok=True)

    # Fetch PDB in background - don't block the response!
    pdb_fetch = PDBFetchService()
    if pdb_fetch.is_valid_pdb_id(protein.name):
        background_tasks.add_task(
            fetch_pdb_background,
            db_protein.id,
            protein.name,
            protein_dir,
            settings.DATABASE_URL
        )
        print(f"→ PDB fetch scheduled in background for {protein.name}")

    # Return immediately (FAST!)
    return db_protein


@router.post("/upload", response_model=ProteinResponse)
async def upload_protein(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload protein from FASTA or PDB file"""

    if file.size and file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    # Create upload directory if not exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Save uploaded file temporarily
    temp_file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Parse file based on extension
    sequence = None
    pdb_file_path = None

    if file.filename.endswith(".fasta") or file.filename.endswith(".fa"):
        with open(temp_file_path, "r") as f:
            lines = f.readlines()
            sequence = "".join([line.strip() for line in lines if not line.startswith(">")])
    elif file.filename.endswith(".pdb"):
        # Extract sequence from PDB file
        try:
            from Bio import PDB
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("protein", temp_file_path)
            # Get sequence from first chain
            for model in structure:
                for chain in model:
                    sequence = ""
                    for residue in chain:
                        if residue.id[0] == " ":  # Standard amino acid
                            resname = residue.get_resname()
                            # Convert 3-letter to 1-letter code
                            aa_dict = {
                                'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                                'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                                'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                                'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                                'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                            }
                            sequence += aa_dict.get(resname, 'X')
                    break
                break
        except ImportError:
            # Fallback if BioPython not available - parse manually
            sequence = ""
            with open(temp_file_path, "r") as f:
                for line in f:
                    if line.startswith("SEQRES"):
                        parts = line.split()
                        if len(parts) > 4:
                            sequence += "".join(parts[4:])

        if not sequence:
            raise HTTPException(status_code=400, detail="Could not extract sequence from PDB file")

        pdb_file_path = temp_file_path  # Will be moved after protein creation
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use .fasta, .fa, or .pdb")

    sequence = sequence.upper()

    if len(sequence) > settings.MAX_SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence length exceeds maximum of {settings.MAX_SEQUENCE_LENGTH}"
        )

    db_protein = Protein(
        name=name,
        sequence=sequence,
        sequence_length=len(sequence),
        description=description
    )
    db.add(db_protein)
    db.commit()
    db.refresh(db_protein)

    # Create output directory structure for this protein
    protein_dir = os.path.join(settings.OUTPUT_DIR, f"protein_{db_protein.id}")
    os.makedirs(os.path.join(protein_dir, "attention"), exist_ok=True)
    os.makedirs(os.path.join(protein_dir, "visualizations"), exist_ok=True)

    # Move PDB file to protein directory if it was uploaded
    if pdb_file_path:
        final_pdb_path = os.path.join(protein_dir, f"{name}_relaxed.pdb")
        shutil.move(pdb_file_path, final_pdb_path)
        db_protein.pdb_file = final_pdb_path
        db.commit()

    return db_protein


@router.get("/", response_model=List[ProteinResponse])
async def list_proteins(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all proteins"""
    proteins = db.query(Protein).offset(skip).limit(limit).all()
    return proteins


@router.get("/{protein_id}", response_model=ProteinResponse)
async def get_protein(
    protein_id: int,
    db: Session = Depends(get_db)
):
    """Get protein by ID"""
    protein = db.query(Protein).filter(Protein.id == protein_id).first()
    if not protein:
        raise HTTPException(status_code=404, detail="Protein not found")
    return protein


@router.delete("/{protein_id}")
async def delete_protein(
    protein_id: int,
    db: Session = Depends(get_db)
):
    """Delete protein"""
    protein = db.query(Protein).filter(Protein.id == protein_id).first()
    if not protein:
        raise HTTPException(status_code=404, detail="Protein not found")

    db.delete(protein)
    db.commit()
    return {"message": "Protein deleted successfully"}


@router.get("/{protein_id}/pdb")
async def download_protein_pdb(
    protein_id: int,
    db: Session = Depends(get_db)
):
    """Download the stored PDB file for a protein."""
    protein = db.query(Protein).filter(Protein.id == protein_id).first()
    if not protein or not protein.pdb_file:
        raise HTTPException(status_code=404, detail="PDB file not available for this protein")

    if not os.path.exists(protein.pdb_file):
        raise HTTPException(status_code=404, detail="PDB file missing on server")

    filename = os.path.basename(protein.pdb_file)
    return FileResponse(
        protein.pdb_file,
        media_type="chemical/x-pdb",
        filename=filename
    )
