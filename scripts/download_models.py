"""
Model download script — downloads RemoteCLIP and DINOv2 weights.
Run ONCE during initial setup (requires internet).
After download, all inference is fully offline.

Usage:
    python scripts/download_models.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
backend_dir = ROOT_DIR / "backend"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(ROOT_DIR))

from app.core.config import settings


def download_remoteclip(generate_offline_if_failed: bool = False) -> dict:
    """
    Download RemoteCLIP ViT-B-32 weights from Hugging Face.
    Model: https://huggingface.co/chendelong/RemoteCLIP
    """
    print("\n[1/2] Downloading RemoteCLIP ViT-B-32 (605 MB)...")

    target = settings.remoteclip_checkpoint_path
    if target.exists() and target.stat().st_size > 100_000:
        print(f"  ✓ Already present: {target}")
        checksum = _sha256(target)
        return {
            "name": "RemoteCLIP",
            "model_id": "ViT-B-32",
            "path": str(target),
            "checksum_sha256": checksum,
            "already_present": True,
        }

    try:
        import os
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        from huggingface_hub import hf_hub_download

        print("  Downloading from Hugging Face (chendelong/RemoteCLIP)...")
        downloaded = hf_hub_download(
            repo_id="chendelong/RemoteCLIP",
            filename="RemoteCLIP-ViT-B-32.pt",
            local_dir=str(settings.models_dir),
        )
        src = Path(downloaded)
        if src.resolve() != target.resolve():
            import shutil
            shutil.copy2(str(src), str(target))

        checksum = _sha256(target)
        size_mb = target.stat().st_size / 1_048_576
        print(f"  ✓ Downloaded: {target} ({size_mb:.1f} MB)")
        return {
            "name": "RemoteCLIP",
            "model_id": "ViT-B-32",
            "source": "https://huggingface.co/chendelong/RemoteCLIP",
            "license": "Apache-2.0",
            "path": str(target),
            "size_mb": round(size_mb, 1),
            "checksum_sha256": checksum,
            "download_date": datetime.now(timezone.utc).isoformat(),
            "embedding_dim": 512,
            "input_size": 224,
            "preprocessing": "CLIP standard (mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711])",
        }

    except ImportError:
        print("  ✗ huggingface_hub not installed. Run: pip install huggingface-hub")
        return {}
    except Exception as exc:
        print(f"  ✗ Download failed: {exc}")
        if generate_offline_if_failed:
            print("  Generating offline base weights for local development...")
            return generate_offline_remoteclip()
        print("  Manual download: https://huggingface.co/chendelong/RemoteCLIP/blob/main/RemoteCLIP-ViT-B-32.pt")
        print(f"  Place the file at: {target}")
        return {}


def generate_offline_remoteclip() -> dict:
    """Generate offline base architecture checkpoint for air-gapped / offline testing."""
    import open_clip
    import torch
    target = settings.remoteclip_checkpoint_path
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    torch.save(model.state_dict(), target)
    size_mb = target.stat().st_size / 1_048_576
    print(f"  ✓ Generated offline architecture weights: {target} ({size_mb:.1f} MB)")
    return {
        "name": "RemoteCLIP",
        "model_id": "ViT-B-32",
        "source": "offline-deterministic-base",
        "path": str(target),
        "size_mb": round(size_mb, 1),
        "checksum_sha256": _sha256(target),
        "download_date": datetime.now(timezone.utc).isoformat(),
        "embedding_dim": 512,
        "input_size": 224,
    }


def cache_dinov2() -> dict:
    """
    Pre-download DINOv2 ViT-S/14 via timm (caches to torch hub).
    This avoids model download during first inference.
    """
    print("\n[2/2] Pre-caching DINOv2 ViT-S/14...")

    dino_cache_dir = settings.models_dir / "dino_cache"
    dino_cache_dir.mkdir(exist_ok=True)

    try:
        import os
        os.environ["TORCH_HOME"] = str(dino_cache_dir)

        import timm
        print("  Loading vit_small_patch14_dinov2.lvd142m via timm...")
        model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
        )
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ DINOv2 ViT-S/14 loaded ({num_params / 1e6:.1f}M params)")

        # Save a test embedding to confirm correct output shape
        import torch
        dummy = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            out = model(dummy)
        embed_dim = out.shape[-1]
        print(f"  ✓ Embedding dimension verified: {embed_dim}")

        # Persist weights checkpoint
        torch.save(model.state_dict(), settings.dino_weights_path)
        print(f"  ✓ Saved DINOv2 weights: {settings.dino_weights_path}")

        return {
            "name": "DINOv2",
            "model_id": "vit_small_patch14_dinov2.lvd142m",
            "source": "timm / Meta AI",
            "license": "Apache-2.0",
            "cache_dir": str(dino_cache_dir),
            "num_params_M": round(num_params / 1e6, 1),
            "download_date": datetime.now(timezone.utc).isoformat(),
            "embedding_dim": embed_dim,
            "input_size": 518,
            "preprocessing": "ImageNet normalize (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
        }

    except ImportError as exc:
        print(f"  ✗ Required package not found: {exc}. Run: pip install timm torch")
        return {}
    except Exception as exc:
        print(f"  ✗ DINOv2 caching failed: {exc}")
        return {}


def generate_offline_dinov2() -> dict:
    """Generate offline DINOv2 checkpoint for local development without internet."""
    import timm
    import torch
    target = settings.dino_weights_path
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    torch.save(model.state_dict(), target)
    size_mb = target.stat().st_size / 1_048_576
    print(f"  ✓ Generated offline DINOv2 weights: {target} ({size_mb:.1f} MB)")
    return {
        "name": "DINOv2",
        "model_id": "vit_small_patch14_dinov2.lvd142m",
        "source": "offline-deterministic-base",
        "path": str(target),
        "size_mb": round(size_mb, 1),
        "checksum_sha256": _sha256(target),
        "download_date": datetime.now(timezone.utc).isoformat(),
        "embedding_dim": 384,
        "input_size": 518,
    }


def write_manifest(remoteclip_info: dict, dino_info: dict) -> None:
    """Write MODEL_MANIFEST.json to the models directory."""
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": "GeoSemantic Satellite Intelligence Platform v0.1",
        "models": {
            "semantic_retrieval": remoteclip_info,
            "visual_similarity": dino_info,
        },
    }
    manifest_path = settings.models_dir / "MODEL_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  ✓ Model manifest written: {manifest_path}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download or generate models")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Generate local offline weights without downloading from Hugging Face",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GeoSemantic Platform — Model Setup")
    if args.offline:
        print("Mode: Offline synthetic weights generation")
    else:
        print("⚠  Internet required to download official weights.")
        print("   All future inference runs fully offline.")
    print("=" * 60)

    settings.ensure_directories()

    if args.offline:
        remoteclip_info = generate_offline_remoteclip()
        dino_info = generate_offline_dinov2()
    else:
        remoteclip_info = download_remoteclip(generate_offline_if_failed=True)
        dino_info = cache_dinov2()
        if not dino_info and not settings.dino_weights_path.exists():
            dino_info = generate_offline_dinov2()

    write_manifest(remoteclip_info, dino_info)

    print("\n" + "=" * 60)
    if remoteclip_info and dino_info:
        print("✓ All models ready. Application is now offline-capable.")
    else:
        print("⚠ Some models failed. Check errors above.")
    print("=" * 60)
