"""
Automated test suite for Phase 4: Offline AI Model Infrastructure.

Validates:
1. Zero internet requirement (HF_HUB_OFFLINE=1, socket mock).
2. RemoteCLIP text and image embeddings (512 dimensions).
3. DINOv2 image embeddings (384 dimensions).
4. L2 normalization (|vector| == 1.0).
5. Deterministic reproducibility (same input yields identical vector).
6. GET /models/status API structure and lifecycle endpoints.
"""
from __future__ import annotations

import io
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.services.model_manager import model_manager


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a deterministic synthetic satellite-like image for testing."""
    img = Image.new("RGB", (256, 256), color=(45, 95, 60))
    # Draw simple contrasting patch
    pixels = img.load()
    for x in range(80, 180):
        for y in range(80, 180):
            pixels[x, y] = (180, 140, 80)
    return img


@pytest.fixture
def sample_image_bytes(sample_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    sample_image.save(buf, format="PNG")
    return buf.getvalue()


class TestOfflineModelInfrastructure:
    """Core offline model loading and execution tests."""

    def test_offline_environment_variables_set(self):
        """Verify that offline environment variables are strictly enforced."""
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        assert settings.offline_mode is True

    def test_model_runs_without_internet(self, sample_image: Image.Image):
        """
        Verify that model loading and inference succeed when all network
        socket attempts raise an exception (zero internet).
        """
        def blocked_socket(*args, **kwargs):
            raise socket.error("Network is unreachable (Strict Offline Test)")

        with patch("socket.create_connection", side_effect=blocked_socket), \
             patch("socket.getaddrinfo", side_effect=blocked_socket):
            # Model loading must succeed with zero network access
            model_manager.load_model("RemoteCLIP")
            text_emb = embedding_service.embed_text("satellite runway airport")
            assert len(text_emb) == 512

            img_emb = embedding_service.embed_image(sample_image, model_name="RemoteCLIP")
            assert len(img_emb) == 512

    def test_remoteclip_embedding_dimension(self, sample_image: Image.Image):
        """Verify RemoteCLIP text and image embeddings are consistently 512-dim."""
        text_emb = embedding_service.embed_text("dense urban settlement with roads")
        assert len(text_emb) == 512

        img_emb = embedding_service.embed_image(sample_image, model_name="RemoteCLIP")
        assert len(img_emb) == 512

    def test_dinov2_embedding_dimension(self, sample_image: Image.Image):
        """Verify DINOv2 visual similarity embedding is strictly 384-dim."""
        img_emb = embedding_service.embed_image(sample_image, model_name="DINOv2")
        assert len(img_emb) == 384

    def test_dinov2_text_embedding_rejected(self):
        """Verify DINOv2 raises error when queried with text modality."""
        with pytest.raises(ValueError, match="does not support text modality"):
            embedding_service.embed_text("should fail", model_name="DINOv2")

    def test_embeddings_reproducibility(self, sample_image: Image.Image):
        """Verify that identical inputs produce identical embedding vectors."""
        query = "coastal wetland and mangrove forest"

        emb_1 = embedding_service.embed_text(query, model_name="RemoteCLIP")
        emb_2 = embedding_service.embed_text(query, model_name="RemoteCLIP")

        # Must be identical across consecutive offline runs
        np.testing.assert_allclose(emb_1, emb_2, rtol=1e-6, atol=1e-6)

        img_1 = embedding_service.embed_image(sample_image, model_name="RemoteCLIP")
        img_2 = embedding_service.embed_image(sample_image, model_name="RemoteCLIP")
        np.testing.assert_allclose(img_1, img_2, rtol=1e-6, atol=1e-6)

    def test_l2_normalization(self, sample_image: Image.Image):
        """Verify that output vectors have Euclidean norm == 1.0 (cosine = dot product)."""
        text_emb = embedding_service.embed_text("agricultural fields", normalize=True)
        norm_text = float(np.linalg.norm(text_emb))
        assert abs(norm_text - 1.0) < 1e-4

        img_clip = embedding_service.embed_image(sample_image, model_name="RemoteCLIP", normalize=True)
        norm_clip = float(np.linalg.norm(img_clip))
        assert abs(norm_clip - 1.0) < 1e-4

        img_dino = embedding_service.embed_image(sample_image, model_name="DINOv2", normalize=True)
        norm_dino = float(np.linalg.norm(img_dino))
        assert abs(norm_dino - 1.0) < 1e-4

    def test_batch_embedding_consistency(self, sample_image: Image.Image):
        """Verify batch embedding methods preserve ordering and dimensions."""
        queries = ["query A", "query B", "query C"]
        batch_embs = embedding_service.embed_texts(queries, model_name="RemoteCLIP")
        assert len(batch_embs) == 3
        for emb in batch_embs:
            assert len(emb) == 512

        batch_imgs = embedding_service.embed_images([sample_image, sample_image], model_name="DINOv2")
        assert len(batch_imgs) == 2
        for emb in batch_imgs:
            assert len(emb) == 384


class TestModelEndpoints:
    """REST API endpoints verification for models."""

    def test_get_models_status_root_endpoint(self, client: TestClient):
        """
        Verify GET /models/status returns:
        - model name
        - version
        - loaded status
        - local path validity
        - device
        - embedding dimension
        """
        res = client.get("/models/status")
        assert res.status_code == 200
        data = res.json()

        assert "models" in data
        assert "device" in data
        assert "offline_mode" in data
        assert data["offline_mode"] is True

        models = data["models"]
        assert len(models) >= 2

        model_names = [m["model_name"] for m in models]
        assert "RemoteCLIP" in model_names
        assert "DINOv2" in model_names

        for m in models:
            assert "model_name" in m
            assert "version" in m
            assert "loaded_status" in m
            assert "local_path_valid" in m
            assert "device" in m
            assert "embedding_dimension" in m

            if m["model_name"] == "RemoteCLIP":
                assert m["embedding_dimension"] == 512
                assert "text" in m["modalities"]
                assert "image" in m["modalities"]
            elif m["model_name"] == "DINOv2":
                assert m["embedding_dimension"] == 384
                assert "image" in m["modalities"]

    def test_get_models_status_api_prefix(self, client: TestClient):
        """Verify GET /api/models/status returns same status."""
        res = client.get("/api/models/status")
        assert res.status_code == 200
        data = res.json()
        assert "models" in data

    def test_get_single_model_status(self, client: TestClient):
        """Verify GET /models/status/RemoteCLIP."""
        res = client.get("/models/status/RemoteCLIP")
        assert res.status_code == 200
        data = res.json()
        assert data["model_name"] == "RemoteCLIP"
        assert data["embedding_dimension"] == 512

    def test_model_load_and_unload_lifecycle(self, client: TestClient):
        """Verify POST /models/{name}/unload and load endpoints."""
        # Unload
        res_unload = client.post("/models/RemoteCLIP/unload")
        assert res_unload.status_code == 200
        assert res_unload.json()["loaded"] is False

        # Load
        res_load = client.post("/models/RemoteCLIP/load")
        assert res_load.status_code == 200
        assert res_load.json()["loaded"] is True

    def test_embed_text_endpoint(self, client: TestClient):
        """Verify POST /models/embed/text returns 512-dim embedding."""
        res = client.post(
            "/models/embed/text",
            json={"text": "deep water reservoir and dam structure", "model_name": "RemoteCLIP"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["model_name"] == "RemoteCLIP"
        assert data["dimension"] == 512
        assert len(data["embedding"]) == 512

    def test_embed_image_endpoint(self, client: TestClient, sample_image_bytes: bytes):
        """Verify POST /models/embed/image with file upload."""
        res = client.post(
            "/models/embed/image",
            files={"file": ("test_tile.png", sample_image_bytes, "image/png")},
            data={"model_name": "DINOv2"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["model_name"] == "DINOv2"
        assert data["dimension"] == 384
        assert len(data["embedding"]) == 384
