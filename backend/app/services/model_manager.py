"""
ModelManager: Offline-first AI model management layer.
Handles local lifecycle (registration, offline loading, unloading, health checks, device management)
for RemoteCLIP (Vision-Language) and DINOv2 (Visual Similarity).

Strict Offline Guarantee:
- Zero runtime downloads.
- Enforces HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.
- All weights and architectures load strictly from local filesystem or local in-memory initialization.
"""
from __future__ import annotations

import gc
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Enforce offline mode in environment before importing ML libraries
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.model import ModelStatusItem, ModelsStatusResponse

logger = get_logger(__name__)


class ModelMetadata:
    """Internal registry entry for an offline AI model."""

    def __init__(
        self,
        name: str,
        model_id: str,
        version: str,
        embedding_dimension: int,
        modalities: List[str],
        checkpoint_filename: str,
        default_checkpoint_path: Path,
    ):
        self.name = name
        self.model_id = model_id
        self.version = version
        self.embedding_dimension = embedding_dimension
        self.modalities = modalities
        self.checkpoint_filename = checkpoint_filename
        self.default_checkpoint_path = default_checkpoint_path

        # Runtime state
        self.loaded: bool = False
        self.status: str = "UNLOADED"
        self.model_instance: Optional[nn.Module] = None
        self.preprocess: Optional[Callable] = None
        self.tokenizer: Optional[Callable] = None
        self.last_loaded_path: Optional[Path] = None
        self.last_error: Optional[str] = None
        self.active_device: str = "cpu"


class ModelManager:
    """
    Central manager for offline AI models in the GeoSemantic platform.
    Ensures zero runtime network calls, CPU & hardware acceleration support,
    and reliable status reporting.
    """

    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self.device = self._resolve_device()
        self._registry: Dict[str, ModelMetadata] = {}
        self._register_models()
        logger.info(
            "model_manager_initialized",
            device=self.device,
            offline_mode=settings.offline_mode,
        )

    def _resolve_device(self) -> str:
        """Resolve target device with fallback to CPU."""
        requested = (settings.inference_device or "cpu").lower()
        if requested in ("cuda", "gpu") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _register_models(self) -> None:
        """Register the supported offline models."""
        # 1. RemoteCLIP (Vision-Language Remote Sensing Model)
        self._registry["RemoteCLIP"] = ModelMetadata(
            name="RemoteCLIP",
            model_id=settings.remoteclip_model_name,
            version="ViT-B-32",
            embedding_dimension=512,
            modalities=["text", "image"],
            checkpoint_filename="RemoteCLIP-ViT-B-32.pt",
            default_checkpoint_path=settings.remoteclip_checkpoint_path,
        )

        # 2. DINOv2 (Visual Similarity Model)
        self._registry["DINOv2"] = ModelMetadata(
            name="DINOv2",
            model_id=settings.dino_model_name,
            version="vit_small_patch14_dinov2.lvd142m",
            embedding_dimension=384,
            modalities=["image"],
            checkpoint_filename="dinov2_vits14.pt",
            default_checkpoint_path=settings.dino_weights_path,
        )

    def _find_checkpoint(self, meta: ModelMetadata) -> Optional[Path]:
        """Check if local checkpoint file or directory exists on filesystem."""
        # Check explicit path
        if meta.default_checkpoint_path.exists():
            return meta.default_checkpoint_path.resolve()

        # Check candidate locations via config helper
        found = settings.find_local_checkpoint(meta.checkpoint_filename)
        if found:
            return found

        # For DINOv2, also check dino_cache directory
        if meta.name == "DINOv2" and settings.dino_checkpoint_path.exists():
            return settings.dino_checkpoint_path.resolve()

        return None

    def load_model(self, name: str) -> bool:
        """
        Load an AI model into memory strictly offline from the local filesystem.
        If no weights checkpoint exists yet on disk, initializes a local offline
        deterministic architecture for testing and pipeline validation without network calls.
        """
        # Canonicalize name
        key = self._canonical_name(name)
        meta = self._registry.get(key)
        if not meta:
            raise ValueError(f"Unknown model '{name}'. Available: {list(self._registry.keys())}")

        if meta.loaded and meta.model_instance is not None:
            return True

        device = self.device
        meta.active_device = device

        try:
            checkpoint_path = self._find_checkpoint(meta)

            if key == "RemoteCLIP":
                self._load_remoteclip(meta, checkpoint_path, device)
            elif key == "DINOv2":
                self._load_dinov2(meta, checkpoint_path, device)

            meta.loaded = True
            meta.status = "LOADED"
            meta.last_error = None
            logger.info(
                "model_loaded_successfully",
                model=meta.name,
                device=device,
                checkpoint_present=checkpoint_path is not None,
            )
            return True

        except Exception as exc:
            meta.loaded = False
            meta.status = "ERROR"
            meta.last_error = str(exc)
            logger.error("model_load_failed", model=meta.name, error=str(exc))
            raise RuntimeError(f"Failed to load model '{name}' offline: {exc}") from exc

    def _load_remoteclip(
        self, meta: ModelMetadata, checkpoint_path: Optional[Path], device: str
    ) -> None:
        """Load RemoteCLIP model and tokenizer offline."""
        import open_clip

        # Fix seed for reproducible offline architecture initialization
        torch.manual_seed(42)

        # Create model architecture offline with pretrained=None (ZERO network calls)
        model, _, preprocess = open_clip.create_model_and_transforms(
            meta.model_id, pretrained=None
        )

        tokenizer = open_clip.get_tokenizer(meta.model_id)

        if checkpoint_path and checkpoint_path.is_file():
            meta.last_loaded_path = checkpoint_path
            logger.info("loading_remoteclip_weights_from_file", path=str(checkpoint_path))
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            # Support various checkpoint format envelopes
            if isinstance(state_dict, dict):
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]

                # Clean any 'module.' prefixes from distributed training
                clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(clean_sd, strict=False)
        else:
            meta.last_loaded_path = meta.default_checkpoint_path
            logger.warning(
                "remoteclip_checkpoint_not_found_using_offline_base",
                expected_path=str(meta.default_checkpoint_path),
            )

        model.eval()
        model.to(device)

        meta.model_instance = model
        meta.preprocess = preprocess
        meta.tokenizer = tokenizer

    def _load_dinov2(
        self, meta: ModelMetadata, checkpoint_path: Optional[Path], device: str
    ) -> None:
        """Load DINOv2 model and image transform offline."""
        import timm

        torch.manual_seed(42)

        # Create model architecture offline (pretrained=False, ZERO network calls)
        model = timm.create_model(
            meta.model_id,
            pretrained=False,
            num_classes=0,
        )

        data_cfg = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_cfg, is_training=False)

        if checkpoint_path and checkpoint_path.is_file():
            meta.last_loaded_path = checkpoint_path
            logger.info("loading_dinov2_weights_from_file", path=str(checkpoint_path))
            if str(checkpoint_path).endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(str(checkpoint_path))
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_sd, strict=False)
        else:
            meta.last_loaded_path = meta.default_checkpoint_path
            logger.warning(
                "dinov2_checkpoint_not_found_using_offline_base",
                expected_path=str(meta.default_checkpoint_path),
            )

        model.eval()
        model.to(device)

        meta.model_instance = model
        meta.preprocess = transform
        meta.tokenizer = None

    def unload_model(self, name: str) -> bool:
        """Unload model from memory to free system resources."""
        key = self._canonical_name(name)
        meta = self._registry.get(key)
        if not meta:
            raise ValueError(f"Unknown model '{name}'")

        if not meta.loaded:
            return True

        # Clear references
        meta.model_instance = None
        meta.preprocess = None
        meta.tokenizer = None
        meta.loaded = False
        meta.status = "UNLOADED"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("model_unloaded", model=meta.name)
        return True

    def get_model(self, name: str) -> Tuple[nn.Module, Optional[Callable], Optional[Callable]]:
        """
        Get model instance, preprocess transform, and tokenizer.
        Loads model automatically if not yet resident.
        """
        key = self._canonical_name(name)
        meta = self._registry.get(key)
        if not meta:
            raise ValueError(f"Unknown model '{name}'")

        if not meta.loaded or meta.model_instance is None:
            self.load_model(key)

        return meta.model_instance, meta.preprocess, meta.tokenizer

    def get_model_status(self, name: str) -> ModelStatusItem:
        """Return standardized health and status information for an individual model."""
        key = self._canonical_name(name)
        meta = self._registry.get(key)
        if not meta:
            raise ValueError(f"Unknown model '{name}'")

        checkpoint = self._find_checkpoint(meta)
        path_str = str(checkpoint if checkpoint else meta.default_checkpoint_path)
        path_valid = checkpoint is not None and checkpoint.exists()

        status_code = meta.status
        if not meta.loaded:
            status_code = "UNLOADED" if path_valid else "CHECKPOINT_MISSING"

        return ModelStatusItem(
            model_name=meta.name,
            version=meta.version,
            loaded=meta.loaded,
            loaded_status=meta.loaded,
            local_path=path_str,
            local_path_valid=path_valid,
            device=meta.active_device if meta.loaded else self.device,
            embedding_dimension=meta.embedding_dimension,
            embedding_dim=meta.embedding_dimension,
            modalities=meta.modalities,
            status=status_code,
            details={
                "model_id": meta.model_id,
                "has_weights_file": path_valid,
                "last_error": meta.last_error,
            },
        )

    def get_all_statuses(self) -> ModelsStatusResponse:
        """Return consolidated status report across all registered offline models."""
        items = [self.get_model_status(name) for name in self._registry]
        return ModelsStatusResponse(
            models=items,
            device=self.device,
            cuda_available=torch.cuda.is_available(),
            offline_mode=settings.offline_mode,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def check_health(self) -> Dict[str, Any]:
        """Perform non-destructive health checks on offline models."""
        results: Dict[str, Any] = {}
        for name, meta in self._registry.items():
            checkpoint = self._find_checkpoint(meta)
            results[name] = {
                "registered": True,
                "version": meta.version,
                "loaded": meta.loaded,
                "local_path": str(checkpoint or meta.default_checkpoint_path),
                "local_path_valid": checkpoint is not None and checkpoint.exists(),
                "embedding_dimension": meta.embedding_dimension,
                "modalities": meta.modalities,
                "device": meta.active_device if meta.loaded else self.device,
            }
        return results

    def _canonical_name(self, name: str) -> str:
        """Map user input string to canonical model key."""
        clean = name.strip().lower().replace("-", "").replace("_", "")
        if "clip" in clean:
            return "RemoteCLIP"
        if "dino" in clean:
            return "DINOv2"
        # Exact match attempt
        for registered in self._registry:
            if registered.lower() == name.strip().lower():
                return registered
        raise ValueError(f"Unknown model '{name}'. Expected 'RemoteCLIP' or 'DINOv2'.")


# Module-level singleton
model_manager = ModelManager()
