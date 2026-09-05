"""
API endpoints for AI model status, lifecycle management, and embedding generation.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image

from app.schemas.model import (
    BatchTextEmbeddingRequest,
    BatchTextEmbeddingResponse,
    ImageEmbeddingResponse,
    ModelStatusItem,
    ModelsStatusResponse,
    TextEmbeddingRequest,
    TextEmbeddingResponse,
)
from app.services.embedding_service import embedding_service
from app.services.model_manager import model_manager

router = APIRouter()


@router.get("/status", response_model=ModelsStatusResponse)
async def get_models_status(
    as_list: bool = Query(False, description="If true, return directly as a list of models"),
) -> Any:
    """
    Get operational status for all offline AI models.
    Returns:
    - model name
    - version
    - loaded status
    - local path validity
    - device
    - embedding dimension
    """
    status_report = model_manager.get_all_statuses()
    if as_list:
        return status_report.models
    return status_report


@router.get("/status/{name}", response_model=ModelStatusItem)
async def get_single_model_status(name: str) -> ModelStatusItem:
    """Get operational status for a specific model (e.g., RemoteCLIP, DINOv2)."""
    try:
        return model_manager.get_model_status(name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@router.post("/{name}/load", response_model=ModelStatusItem)
async def load_model_endpoint(name: str) -> ModelStatusItem:
    """
    Explicitly load a model into memory from local filesystem.
    Safe for offline operations.
    """
    try:
        model_manager.load_model(name)
        return model_manager.get_model_status(name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model offline: {exc}",
        )


@router.post("/{name}/unload", response_model=ModelStatusItem)
async def unload_model_endpoint(name: str) -> ModelStatusItem:
    """Unload a model from memory to free system resources."""
    try:
        model_manager.unload_model(name)
        return model_manager.get_model_status(name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@router.post("/embed/text", response_model=TextEmbeddingResponse)
async def embed_text_endpoint(req: TextEmbeddingRequest) -> TextEmbeddingResponse:
    """
    Generate an embedding vector for a natural language text query.
    Used for semantic text-to-image satellite search.
    """
    try:
        embedding = embedding_service.embed_text(
            text=req.text,
            model_name=req.model_name,
            normalize=req.normalize,
        )
        return TextEmbeddingResponse(
            model_name=req.model_name,
            text=req.text,
            embedding=embedding,
            dimension=len(embedding),
            device=model_manager.device,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {exc}",
        )


@router.post("/embed/batch-text", response_model=BatchTextEmbeddingResponse)
async def embed_batch_text_endpoint(req: BatchTextEmbeddingRequest) -> BatchTextEmbeddingResponse:
    """Generate embeddings for a batch of text queries."""
    try:
        embeddings = embedding_service.embed_texts(
            texts=req.texts,
            model_name=req.model_name,
            normalize=req.normalize,
        )
        dim = len(embeddings[0]) if embeddings else 0
        return BatchTextEmbeddingResponse(
            model_name=req.model_name,
            count=len(embeddings),
            embeddings=embeddings,
            dimension=dim,
            device=model_manager.device,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch embedding generation failed: {exc}",
        )


@router.post("/embed/image", response_model=ImageEmbeddingResponse)
async def embed_image_endpoint(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    model_name: str = Form("RemoteCLIP"),
    normalize: bool = Form(True),
) -> ImageEmbeddingResponse:
    """
    Generate an embedding vector from an uploaded image file or local image path.
    Supports RemoteCLIP (512-dim) and DINOv2 (384-dim).
    """
    try:
        if file is not None:
            content = await file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
        elif image_path:
            img = Image.open(image_path).convert("RGB")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either an uploaded file or image_path must be provided.",
            )

        embedding = embedding_service.embed_image(
            image=img,
            model_name=model_name,
            normalize=normalize,
        )

        return ImageEmbeddingResponse(
            model_name=model_name,
            dimension=len(embedding),
            embedding=embedding,
            device=model_manager.device,
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image embedding failed: {exc}",
        )
