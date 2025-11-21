from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Protein(Base):
    """Protein model"""
    __tablename__ = "proteins"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    sequence = Column(Text, nullable=False)
    sequence_length = Column(Integer, nullable=False)
    pdb_file = Column(String(512), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    inference_jobs = relationship("InferenceJob", back_populates="protein", cascade="all, delete-orphan")
    visualizations = relationship("Visualization", back_populates="protein", cascade="all, delete-orphan")


class InferenceJob(Base):
    """Inference job model"""
    __tablename__ = "inference_jobs"

    id = Column(Integer, primary_key=True, index=True)
    protein_id = Column(Integer, ForeignKey("proteins.id"), nullable=False)
    job_id = Column(String(255), unique=True, nullable=False, index=True)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    progress = Column(Float, default=0.0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    result_pdb = Column(String(512), nullable=True)
    attention_data = Column(JSON, nullable=True)
    job_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    protein = relationship("Protein", back_populates="inference_jobs")


class Visualization(Base):
    """Visualization model"""
    __tablename__ = "visualizations"

    id = Column(Integer, primary_key=True, index=True)
    protein_id = Column(Integer, ForeignKey("proteins.id"), nullable=False)
    viz_type = Column(String(50), nullable=False)  # heatmap, arc_diagram, 3d, combined
    layer = Column(Integer, nullable=False)
    head = Column(Integer, nullable=True)
    attention_type = Column(String(50), nullable=False)  # msa_row, triangle_start
    residue_index = Column(Integer, nullable=True)  # for triangle attention
    image_path = Column(String(512), nullable=False)
    thumbnail_path = Column(String(512), nullable=True)
    viz_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    protein = relationship("Protein", back_populates="visualizations")


class AttentionData(Base):
    """Attention data model"""
    __tablename__ = "attention_data"

    id = Column(Integer, primary_key=True, index=True)
    protein_id = Column(Integer, ForeignKey("proteins.id"), nullable=False)
    layer = Column(Integer, nullable=False)
    head = Column(Integer, nullable=False)
    attention_type = Column(String(50), nullable=False)
    residue_index = Column(Integer, nullable=True)
    data_file = Column(String(512), nullable=False)  # path to attention weights file
    top_k = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
