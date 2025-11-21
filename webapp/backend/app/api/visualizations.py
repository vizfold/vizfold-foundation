from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.db.database import get_db
from app.db.models import Visualization, Protein
from app.services.visualization_service import VisualizationService
from app.services.real_visualization_service import RealVisualizationService

router = APIRouter()


# Pydantic schemas
class VisualizationResponse(BaseModel):
    id: int
    protein_id: int
    viz_type: str
    layer: int
    head: Optional[int] = None
    attention_type: str
    residue_index: Optional[int] = None
    image_path: str
    thumbnail_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class GenerateVisualizationRequest(BaseModel):
    protein_id: int
    viz_type: str  # heatmap, arc_diagram, 3d, combined
    layer: int
    head: Optional[int] = None
    attention_type: str  # msa_row, triangle_start
    residue_index: Optional[int] = None
    top_k: Optional[int] = None


@router.post("/generate", response_model=VisualizationResponse)
async def generate_visualization(
    request: GenerateVisualizationRequest,
    db: Session = Depends(get_db)
):
    """Generate a new visualization"""

    # Check if protein exists
    protein = db.query(Protein).filter(Protein.id == request.protein_id).first()
    if not protein:
        raise HTTPException(status_code=404, detail="Protein not found")

    # Generate visualization using REAL matplotlib utilities
    real_viz_service = RealVisualizationService()
    try:
        # For arc diagrams, use the real utility
        if request.viz_type == "arc_diagram":
            attention_file = f"outputs/protein_{request.protein_id}/attention/{request.attention_type}_attn_layer{request.layer}.txt"
            output_path = f"outputs/protein_{request.protein_id}/visualizations/{request.viz_type}_layer_{request.layer}_head_{request.head}.png"

            image_path = real_viz_service.generate_arc_diagram(
                attention_file=attention_file,
                sequence=protein.sequence,
                layer=request.layer,
                head=request.head or 0,
                output_path=output_path,
                top_k=request.top_k or 50,
                highlight_residue=request.residue_index
            )
            thumbnail_path = None
        elif request.viz_type == "heatmap":
            # Use real heatmap generator
            attention_dir = f"outputs/protein_{request.protein_id}/attention"
            output_dir = f"outputs/protein_{request.protein_id}/visualizations"

            image_path = real_viz_service.generate_heatmap(
                attention_dir=attention_dir,
                protein=f"protein_{request.protein_id}",
                layer=request.layer,
                attention_type=request.attention_type,
                output_dir=output_dir,
                top_k=request.top_k,
                residue_index=request.residue_index
            )
            thumbnail_path = None
        elif request.viz_type == "3d":
            # Use real 3D PyMOL generator
            pdb_file = f"outputs/protein_{request.protein_id}/{protein.name}_relaxed.pdb"
            attention_dir = f"outputs/protein_{request.protein_id}/attention"
            output_dir = f"outputs/protein_{request.protein_id}/visualizations"

            image_path = real_viz_service.generate_3d_visualization(
                pdb_file=pdb_file,
                attention_dir=attention_dir,
                protein=f"protein_{request.protein_id}",
                layer=request.layer,
                head=request.head or 0,
                attention_type=request.attention_type,
                output_dir=output_dir,
                top_k=request.top_k or 50,
                residue_index=request.residue_index
            )
            thumbnail_path = None
        else:
            # Fallback to placeholder service
            viz_service = VisualizationService()
            image_path, thumbnail_path = await viz_service.generate_visualization(
                protein_id=request.protein_id,
                viz_type=request.viz_type,
                layer=request.layer,
                head=request.head,
                attention_type=request.attention_type,
                residue_index=request.residue_index,
                top_k=request.top_k
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

    # Save to database
    viz = Visualization(
        protein_id=request.protein_id,
        viz_type=request.viz_type,
        layer=request.layer,
        head=request.head,
        attention_type=request.attention_type,
        residue_index=request.residue_index,
        image_path=image_path,
        thumbnail_path=thumbnail_path,
        metadata={"top_k": request.top_k}
    )
    db.add(viz)
    db.commit()
    db.refresh(viz)

    return viz


@router.get("/protein/{protein_id}", response_model=List[VisualizationResponse])
async def get_protein_visualizations(
    protein_id: int,
    viz_type: Optional[str] = Query(None),
    layer: Optional[int] = Query(None),
    attention_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all visualizations for a protein with optional filters"""

    query = db.query(Visualization).filter(Visualization.protein_id == protein_id)

    if viz_type:
        query = query.filter(Visualization.viz_type == viz_type)
    if layer is not None:
        query = query.filter(Visualization.layer == layer)
    if attention_type:
        query = query.filter(Visualization.attention_type == attention_type)

    visualizations = query.order_by(Visualization.created_at.desc()).all()
    return visualizations


@router.get("/{viz_id}", response_model=VisualizationResponse)
async def get_visualization(
    viz_id: int,
    db: Session = Depends(get_db)
):
    """Get visualization by ID"""
    viz = db.query(Visualization).filter(Visualization.id == viz_id).first()
    if not viz:
        raise HTTPException(status_code=404, detail="Visualization not found")
    return viz


@router.delete("/{viz_id}")
async def delete_visualization(
    viz_id: int,
    db: Session = Depends(get_db)
):
    """Delete visualization"""
    viz = db.query(Visualization).filter(Visualization.id == viz_id).first()
    if not viz:
        raise HTTPException(status_code=404, detail="Visualization not found")

    # TODO: Delete actual image files
    db.delete(viz)
    db.commit()
    return {"message": "Visualization deleted"}
