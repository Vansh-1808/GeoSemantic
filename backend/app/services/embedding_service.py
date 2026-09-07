"""
EmbeddingService: High-level embedding generation service for offline AI models.
Supports:
1. Text query embedding via RemoteCLIP (512 dimensions).
2. Image embedding via RemoteCLIP (512 dimensions) and DINOv2 (384 dimensions).
3. Batch embedding for indexing workflows.
4. L2 normalization and deterministic reproducibility.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import settings
from app.core.logging import get_logger
from app.services.model_manager import model_manager

logger = get_logger(__name__)


class EmbeddingService:
    """
    Embedding generation service for satellite imagery search and retrieval.
    Guarantees consistent dimensions, L2 normalization, and offline operation.
    """

    _instance: Optional["EmbeddingService"] = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.manager = model_manager
        logger.info("embedding_service_initialized")

    def embed_text(
        self,
        text: str,
        model_name: str = "RemoteCLIP",
        normalize: bool = True,
        use_ensemble: bool = True,
    ) -> List[float]:
        """
        Generate embedding vector for a single natural language text query.
        Applies satellite-domain prompt ensembling to align text with remote sensing features.
        Returns a 512-dimensional float vector for RemoteCLIP.
        """
        if use_ensemble and model_name.lower() in ("remoteclip", "clip"):
            templates = [
                text.strip(),
                f"a satellite photo of {text.strip()}",
                f"satellite imagery showing {text.strip()}",
                f"aerial view of {text.strip()}",
            ]
            embeddings = self.embed_texts(templates, model_name=model_name, normalize=True)
            arr = np.mean(embeddings, axis=0)
            if normalize:
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
            return arr.tolist()

        embeddings = self.embed_texts([text], model_name=model_name, normalize=normalize)
        return embeddings[0]

    def embed_texts(
        self,
        texts: List[str],
        model_name: str = "RemoteCLIP",
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of text queries.
        """
        if not texts:
            return []

        canonical = self.manager._canonical_name(model_name)
        status = self.manager.get_model_status(canonical)

        if "text" not in status.modalities:
            raise ValueError(
                f"Model '{canonical}' does not support text modality. Supported: {status.modalities}"
            )

        model, _, tokenizer = self.manager.get_model(canonical)
        if tokenizer is None:
            raise RuntimeError(f"Tokenizer not initialized for model '{canonical}'")

        device = self.manager.device

        with torch.no_grad():
            tokens = tokenizer(texts).to(device)
            text_features = model.encode_text(tokens)

            if normalize:
                text_features = F.normalize(text_features, p=2, dim=-1)

            result = text_features.cpu().numpy().tolist()

        return result

    def embed_image(
        self,
        image: Union[Image.Image, np.ndarray, Path, str, bytes],
        model_name: str = "RemoteCLIP",
        normalize: bool = True,
    ) -> List[float]:
        """
        Generate embedding vector for a single satellite image or tile.
        Returns:
        - 512 dimensions for RemoteCLIP
        - 384 dimensions for DINOv2
        """
        embeddings = self.embed_images([image], model_name=model_name, normalize=normalize)
        return embeddings[0]

    def embed_images(
        self,
        images: List[Union[Image.Image, np.ndarray, Path, str, bytes]],
        model_name: str = "RemoteCLIP",
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of images or tiles.
        """
        if not images:
            return []

        canonical = self.manager._canonical_name(model_name)
        model, preprocess, _ = self.manager.get_model(canonical)
        if preprocess is None:
            raise RuntimeError(f"Preprocessor not initialized for model '{canonical}'")

        device = self.manager.device

        # Preprocess all images into torch tensors
        tensors = []
        for img_item in images:
            pil_img = self._to_pil_image(img_item)
            tensor = preprocess(pil_img)
            tensors.append(tensor)

        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            if canonical == "RemoteCLIP":
                features = model.encode_image(batch)
            elif canonical == "DINOv2":
                features = model(batch)
            else:
                raise ValueError(f"Unsupported model for image embedding: '{canonical}'")

            if normalize:
                features = F.normalize(features, p=2, dim=-1)

            result = features.cpu().numpy().tolist()

        return result

    def _to_pil_image(self, item: Union[Image.Image, np.ndarray, Path, str, bytes]) -> Image.Image:
        """Convert various image inputs to an RGB PIL Image."""
        if isinstance(item, Image.Image):
            return item.convert("RGB") if item.mode != "RGB" else item

        if isinstance(item, np.ndarray):
            # Handle float arrays scaled 0-1
            if item.dtype in (np.float32, np.float64):
                if item.max() <= 1.0:
                    item = (item * 255).astype(np.uint8)
                else:
                    item = item.astype(np.uint8)

            # Handle (C, H, W) vs (H, W, C)
            if item.ndim == 3 and item.shape[0] in (1, 3, 4) and item.shape[0] < item.shape[2]:
                item = np.transpose(item, (1, 2, 0))

            if item.ndim == 2:
                return Image.fromarray(item).convert("RGB")
            elif item.ndim == 3:
                if item.shape[2] == 1:
                    return Image.fromarray(item.squeeze(2)).convert("RGB")
                elif item.shape[2] in (3, 4):
                    return Image.fromarray(item[:, :, :3]).convert("RGB")

        if isinstance(item, (str, Path)):
            p = Path(item)
            if not p.exists():
                raise FileNotFoundError(f"Image file not found: {p}")
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                try:
                    import rasterio
                    with rasterio.open(p) as src:
                        if src.count >= 3:
                            rgb = np.stack([src.read(1), src.read(2), src.read(3)], axis=-1)
                        else:
                            gray = src.read(1)
                            rgb = np.stack([gray, gray, gray], axis=-1)
                        if rgb.dtype != np.uint8:
                            p2, p98 = np.percentile(rgb, (2, 98))
                            rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-5) * 255.0, 0, 255).astype(np.uint8)
                        return Image.fromarray(rgb, mode="RGB")
                except Exception as exc:
                    raise ValueError(f"Failed to read image at {p}: {exc}")

        if isinstance(item, bytes):
            try:
                return Image.open(io.BytesIO(item)).convert("RGB")
            except Exception:
                try:
                    import rasterio
                    from rasterio.io import MemoryFile
                    with MemoryFile(item) as memfile:
                        with memfile.open() as src:
                            if src.count >= 3:
                                rgb = np.stack([src.read(1), src.read(2), src.read(3)], axis=-1)
                            else:
                                gray = src.read(1)
                                rgb = np.stack([gray, gray, gray], axis=-1)
                            if rgb.dtype != np.uint8:
                                p2, p98 = np.percentile(rgb, (2, 98))
                                rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-5) * 255.0, 0, 255).astype(np.uint8)
                            return Image.fromarray(rgb, mode="RGB")
                except Exception as exc:
                    raise ValueError(f"Failed to decode image bytes: {exc}")

        raise TypeError(f"Unsupported image input type: {type(item)}")


# Module-level singleton
embedding_service = EmbeddingService()
