"""
Pydantic schemas for AI model management, health status, and embedding generation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ModelStatusItem(BaseModel):
    """
    Status of an individual offline AI model.
    Exposes all fields required by SIH 2026 Phase 4 specification.
    """
    model_name: str = Field(..., description="Canonical model name, e.g. RemoteCLIP, DINOv2")
    version: str = Field(..., description="Model version or architecture identifier")
    loaded: bool = Field(..., description="Whether model weights are currently resident in memory")
    loaded_status: bool = Field(..., description="Explicit loaded status boolean")
    local_path: str = Field(..., description="Configured local filesystem path to checkpoint / cache")
    local_path_valid: bool = Field(..., description="Whether the local checkpoint exists on the filesystem")
    device: str = Field(..., description="Inference device (e.g. 'cpu', 'cuda')")
    embedding_dimension: int = Field(..., description="Output vector dimension (e.g. 512, 384)")
    embedding_dim: int = Field(..., description="Alias for embedding_dimension")
    modalities: List[str] = Field(default_factory=list, description="Supported modalities (e.g. ['text', 'image'])")
    status: Literal["LOADED", "UNLOADED", "CHECKPOINT_MISSING", "ERROR"] = Field(
        "UNLOADED", description="Operational lifecycle status"
    )
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata or diagnostics")


class ModelsStatusResponse(BaseModel):
    """
    Consolidated health and status response for all offline AI models.
    """
    models: List[ModelStatusItem] = Field(..., description="List of registered offline models")
    device: str = Field(..., description="Active default inference device")
    cuda_available: bool = Field(..., description="Hardware acceleration availability")
    offline_mode: bool = Field(..., description="Strict offline enforcement flag")
    timestamp: str = Field(..., description="UTC ISO timestamp of status query")


class TextEmbeddingRequest(BaseModel):
    """Request payload for single text query embedding."""
    text: str = Field(..., min_length=1, max_length=2000, description="Natural language search query")
    model_name: str = Field("RemoteCLIP", description="Model to use for text encoding ('RemoteCLIP')")
    normalize: bool = Field(True, description="Whether to L2-normalize the output embedding vector")


class BatchTextEmbeddingRequest(BaseModel):
    """Request payload for batch text queries."""
    texts: List[str] = Field(..., min_length=1, description="List of query strings to encode")
    model_name: str = Field("RemoteCLIP", description="Model to use ('RemoteCLIP')")
    normalize: bool = Field(True, description="Whether to L2-normalize output vectors")


class TextEmbeddingResponse(BaseModel):
    """Response payload containing generated text embedding."""
    model_name: str
    text: str
    embedding: List[float]
    dimension: int
    device: str


class BatchTextEmbeddingResponse(BaseModel):
    """Response payload containing batch embeddings."""
    model_name: str
    count: int
    embeddings: List[List[float]]
    dimension: int
    device: str


class ImageEmbeddingRequest(BaseModel):
    """Request payload for generating an embedding from a local image file or tile."""
    image_path: Optional[str] = Field(None, description="Local filesystem path to image")
    tile_id: Optional[str] = Field(None, description="Database tile ID")
    model_name: str = Field("RemoteCLIP", description="Model to use ('RemoteCLIP' or 'DINOv2')")
    normalize: bool = Field(True, description="Whether to L2-normalize the output vector")


class ImageEmbeddingResponse(BaseModel):
    """Response payload containing generated image embedding."""
    model_name: str
    dimension: int
    embedding: List[float]
    device: str
