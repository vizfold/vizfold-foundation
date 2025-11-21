from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os

from app.db.database import get_db
from app.db.models import AttentionData, Protein
from app.core.config import settings

router = APIRouter()


# Pydantic schemas
class AttentionDataResponse(BaseModel):
    id: int
    protein_id: int
    layer: int
    head: int
    attention_type: str
    residue_index: Optional[int] = None
    data_file: str
    top_k: Optional[int] = None

    class Config:
        from_attributes = True


class AttentionWeightsResponse(BaseModel):
    protein_id: int
    layer: int
    head: int
    attention_type: str
    weights: List[Dict[str, Any]]  # List of {source, target, weight}


@router.get("/protein/{protein_id}", response_model=List[AttentionDataResponse])
async def get_protein_attention_data(
    protein_id: int,
    layer: Optional[int] = Query(None),
    head: Optional[int] = Query(None),
    attention_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all attention data for a protein with optional filters"""

    # Check if protein exists
    protein = db.query(Protein).filter(Protein.id == protein_id).first()
    if not protein:
        raise HTTPException(status_code=404, detail="Protein not found")

    query = db.query(AttentionData).filter(AttentionData.protein_id == protein_id)

    if layer is not None:
        query = query.filter(AttentionData.layer == layer)
    if head is not None:
        query = query.filter(AttentionData.head == head)
    if attention_type:
        query = query.filter(AttentionData.attention_type == attention_type)

    attention_data = query.all()
    return attention_data


@router.get("/weights/{protein_id}/{layer}/{head}")
async def get_attention_weights(
    protein_id: int,
    layer: int,
    head: int,
    attention_type: str = Query(...),
    residue_index: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    """Get actual attention weight values"""

    # Find attention data
    query = db.query(AttentionData).filter(
        AttentionData.protein_id == protein_id,
        AttentionData.layer == layer,
        AttentionData.head == head,
        AttentionData.attention_type == attention_type
    )

    if residue_index is not None:
        query = query.filter(AttentionData.residue_index == residue_index)

    attention_data = query.first()
    if not attention_data:
        raise HTTPException(status_code=404, detail="Attention data not found")

    # Load weights from file
    if not os.path.exists(attention_data.data_file):
        raise HTTPException(status_code=404, detail="Attention data file not found")

    with open(attention_data.data_file, 'r') as f:
        weights = json.load(f)

    return AttentionWeightsResponse(
        protein_id=protein_id,
        layer=layer,
        head=head,
        attention_type=attention_type,
        weights=weights
    )


@router.get("/download/{protein_id}/{layer}/{head}")
async def download_attention_data(
    protein_id: int,
    layer: int,
    head: int,
    attention_type: str = Query(...),
    residue_index: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    """Download attention data as file"""

    # Find attention data
    query = db.query(AttentionData).filter(
        AttentionData.protein_id == protein_id,
        AttentionData.layer == layer,
        AttentionData.head == head,
        AttentionData.attention_type == attention_type
    )

    if residue_index is not None:
        query = query.filter(AttentionData.residue_index == residue_index)

    attention_data = query.first()
    if not attention_data:
        raise HTTPException(status_code=404, detail="Attention data not found")

    if not os.path.exists(attention_data.data_file):
        raise HTTPException(status_code=404, detail="Attention data file not found")

    return FileResponse(
        attention_data.data_file,
        media_type="application/json",
        filename=f"attention_layer{layer}_head{head}_{attention_type}.json"
    )


@router.get("/layers/{protein_id}")
async def get_available_layers(
    protein_id: int,
    db: Session = Depends(get_db)
):
    """Get list of available layers for a protein"""

    layers = db.query(AttentionData.layer).filter(
        AttentionData.protein_id == protein_id
    ).distinct().all()

    return {"layers": sorted([layer[0] for layer in layers])}


@router.get("/heads/{protein_id}/{layer}")
async def get_available_heads(
    protein_id: int,
    layer: int,
    attention_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get list of available heads for a protein and layer"""

    query = db.query(AttentionData.head).filter(
        AttentionData.protein_id == protein_id,
        AttentionData.layer == layer
    )

    if attention_type:
        query = query.filter(AttentionData.attention_type == attention_type)

    heads = query.distinct().all()

    return {"heads": sorted([head[0] for head in heads])}
