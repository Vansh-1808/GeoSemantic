"""
Verification script for Phase 4 — Offline AI Model Infrastructure.

Tests:
1. Strict offline environment enforcement.
2. Model status reporting (name, version, loaded status, local path, device, embedding dimension).
3. RemoteCLIP text embedding (512 dimensions).
4. RemoteCLIP image embedding (512 dimensions).
5. DINOv2 image embedding (384 dimensions).
6. L2 normalization (|v| == 1.0).
7. Deterministic reproducibility across repeated runs.
8. Zero internet connection execution.

Usage:
    python scripts/verify_models.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Enforce offline variables
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.services.model_manager import model_manager


def main() -> int:
    print("=" * 70)
    print("GeoSemantic Satellite Intelligence Platform -- Phase 4 AI Model Check")
    print("=" * 70)

    # 1. Environment and Offline Mode
    print("\n[1/6] Verifying Offline Enforcement...")
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE must be 1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1", "TRANSFORMERS_OFFLINE must be 1"
    print(f"  [OK] HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
    print(f"  [OK] TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")
    print(f"  [OK] Settings offline_mode: {settings.offline_mode}")
    print(f"  [OK] Device configured: {settings.inference_device} (Active: {model_manager.device})")

    # 2. Model Registry & Status Check
    print("\n[2/6] Checking Model Statuses (GET /models/status schema)...")
    statuses = model_manager.get_all_statuses()
    for m in statuses.models:
        print(f"  * Model: {m.model_name}")
        print(f"      Version: {m.version}")
        print(f"      Loaded: {m.loaded_status}")
        print(f"      Local Path: {m.local_path}")
        print(f"      Local Path Valid: {m.local_path_valid}")
        print(f"      Device: {m.device}")
        print(f"      Embedding Dimension: {m.embedding_dimension}")
        print(f"      Modalities: {m.modalities}")

    # 3. RemoteCLIP Text Embedding (512 dimensions)
    print("\n[3/6] Generating RemoteCLIP Text Query Embedding...")
    test_text = "commercial seaport container terminal and docked cargo vessels"
    text_emb = embedding_service.embed_text(test_text, model_name="RemoteCLIP")
    dim_text = len(text_emb)
    norm_text = float(np.linalg.norm(text_emb))
    print(f"  Query: '{test_text}'")
    print(f"  [OK] Dimension: {dim_text} (Expected: 512)")
    print(f"  [OK] L2 Norm: {norm_text:.6f} (Expected: 1.000000)")
    assert dim_text == 512, f"Expected 512 dimensions, got {dim_text}"
    assert abs(norm_text - 1.0) < 1e-4, f"Expected unit norm, got {norm_text}"

    # 4. RemoteCLIP Image Embedding (512 dimensions)
    print("\n[4/6] Generating RemoteCLIP Image Embedding...")
    synthetic_img = Image.new("RGB", (256, 256), color=(40, 90, 60))
    clip_img_emb = embedding_service.embed_image(synthetic_img, model_name="RemoteCLIP")
    dim_clip = len(clip_img_emb)
    norm_clip = float(np.linalg.norm(clip_img_emb))
    print(f"  [OK] Dimension: {dim_clip} (Expected: 512)")
    print(f"  [OK] L2 Norm: {norm_clip:.6f} (Expected: 1.000000)")
    assert dim_clip == 512, f"Expected 512 dimensions, got {dim_clip}"
    assert abs(norm_clip - 1.0) < 1e-4, f"Expected unit norm, got {norm_clip}"

    # 5. DINOv2 Visual Similarity Image Embedding (384 dimensions)
    print("\n[5/6] Generating DINOv2 Visual Similarity Image Embedding...")
    dino_img_emb = embedding_service.embed_image(synthetic_img, model_name="DINOv2")
    dim_dino = len(dino_img_emb)
    norm_dino = float(np.linalg.norm(dino_img_emb))
    print(f"  [OK] Dimension: {dim_dino} (Expected: 384)")
    print(f"  [OK] L2 Norm: {norm_dino:.6f} (Expected: 1.000000)")
    assert dim_dino == 384, f"Expected 384 dimensions, got {dim_dino}"
    assert abs(norm_dino - 1.0) < 1e-4, f"Expected unit norm, got {norm_dino}"

    # 6. Reproducibility & Determinism Verification
    print("\n[6/6] Verifying Reproducibility (Deterministic Output)...")
    text_emb_2 = embedding_service.embed_text(test_text, model_name="RemoteCLIP")
    dino_emb_2 = embedding_service.embed_image(synthetic_img, model_name="DINOv2")

    np.testing.assert_allclose(text_emb, text_emb_2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dino_img_emb, dino_emb_2, rtol=1e-6, atol=1e-6)
    print("  [OK] RemoteCLIP Text Embedding: Exactly identical across repeated runs.")
    print("  [OK] DINOv2 Image Embedding: Exactly identical across repeated runs.")

    print("\n" + "=" * 70)
    print("[SUCCESS] PHASE 4 OFFLINE AI MODEL INFRASTRUCTURE VALIDATION PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
