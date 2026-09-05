"""
Satellite image tiling and preprocessing service.
Pipeline:
  Source Scene → Read Raster (Windowed) → Validate bands → Normalize →
  Generate tiles → Generate thumbnails & previews → Calculate tile geometry →
  Store tiles → Store metadata in PostGIS
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
import rasterio
from rasterio import windows
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.provenance import ProcessingJob, ProvenanceRecord
from app.models.scene import Scene
from app.models.tile import Tile

logger = get_logger(__name__)


@dataclass
class TilingConfig:
    tile_size: int = 512
    overlap_px: int = 0
    normalize: bool = True
    min_valid_pixel_ratio: float = 0.3
    generate_preview: bool = True
    preview_size: int = 512
    thumbnail_size: int = 128


@dataclass
class TilingResult:
    scene_id: uuid.UUID
    tile_count: int
    processing_time_seconds: float
    total_windows_inspected: int
    skipped_nodata_count: int
    tiles: list[Tile]


async def process_scene_tiling(
    scene_id: uuid.UUID,
    session: AsyncSession,
    config: TilingConfig | None = None,
    job_id: uuid.UUID | None = None,
    progress_callback: Callable[[float, str], Any] | None = None,
) -> TilingResult:
    """
    Process a scene into analysis tiles with windowed reading and normalization.
    Original source raster remains completely untouched.
    """
    if config is None:
        config = TilingConfig(tile_size=settings.default_tile_size)

    start_time = time.perf_counter()

    scene = await session.get(Scene, scene_id)
    if not scene:
        raise ValueError(f"Scene {scene_id} not found in database")

    source_path = Path(scene.source_path)
    if not source_path.exists():
        if (settings.data_dir / source_path).exists():
            source_path = settings.data_dir / source_path
        elif (settings.imagery_dir / source_path.name).exists():
            source_path = settings.imagery_dir / source_path.name
        else:
            raise FileNotFoundError(f"Source raster file not found at: {source_path}")

    logger.info(
        "tiling_started",
        scene_id=str(scene_id),
        source_path=str(source_path),
        tile_size=config.tile_size,
        overlap=config.overlap_px,
    )

    async def report_progress(pct: float, msg: str) -> None:
        if progress_callback:
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(pct, msg)
            else:
                progress_callback(pct, msg)
        if job_id:
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.progress_pct = pct
                job.progress_message = msg
                await session.flush()

    await report_progress(5.0, "Validating source raster bands and geometry...")

    # Step 1: Validate bands and dimensions (Read-only)
    band_info = validate_bands(source_path)

    # Prepare output directories
    tiles_dir = settings.tiles_dir / str(scene_id)
    thumbnails_dir = settings.thumbnails_dir / str(scene_id)
    previews_dir = settings.previews_dir / str(scene_id)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    # Optional clean-up of previous tiles for this scene if re-processing
    await session.execute(
        text("DELETE FROM tiles WHERE scene_id = :scene_id"),
        {"scene_id": str(scene_id)},
    )
    await session.flush()

    await report_progress(10.0, "Calculating grid and reading windows...")

    tile_size = config.tile_size
    overlap = max(0, min(config.overlap_px, tile_size - 16))
    stride = tile_size - overlap

    created_tiles: list[Tile] = []
    tile_wkt_map: list[tuple[Tile, str]] = []
    skipped_nodata = 0
    total_inspected = 0

    # Step 2: Open raster read-only and generate tiles using Windowed reading
    with rasterio.open(source_path, "r") as ds:
        raster_w, raster_h = ds.width, ds.height
        crs: CRS = ds.crs or CRS.from_epsg(4326)
        nodata_val = ds.nodata
        band_count = ds.count
        read_bands = list(range(1, band_count + 1))
        rgb_indices = band_info["rgb_indices"]

        # Calculate total windows expected
        cols_count = max(1, (raster_w - overlap + stride - 1) // stride) if raster_w > tile_size else 1
        rows_count = max(1, (raster_h - overlap + stride - 1) // stride) if raster_h > tile_size else 1
        total_expected = cols_count * rows_count

        tile_row = 0
        for row_off in range(0, raster_h, stride):
            tile_col = 0
            actual_h = min(tile_size, raster_h - row_off)
            if actual_h <= 0:
                break

            for col_off in range(0, raster_w, stride):
                actual_w = min(tile_size, raster_w - col_off)
                if actual_w <= 0:
                    break

                total_inspected += 1
                win = windows.Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=actual_w,
                    height=actual_h,
                )

                # WINDOWED READ: Only load the specific tile window into RAM
                tile_raw = ds.read(indexes=read_bands, window=win)

                # Check no-data pixel ratio
                nodata_mask = _compute_nodata_mask(tile_raw, nodata_val)
                nodata_ratio = float(nodata_mask.mean()) if nodata_mask.size > 0 else 0.0

                if nodata_ratio > (1.0 - config.min_valid_pixel_ratio):
                    skipped_nodata += 1
                    tile_col += 1
                    continue

                # Calculate tile geotransform and geographic bounds
                tile_transform = windows.transform(win, ds.transform)
                tile_bounds = windows.bounds(win, ds.transform)

                try:
                    tile_bbox_4326 = transform_bounds(crs, "EPSG:4326", *tile_bounds)
                except Exception:
                    tile_bbox_4326 = tile_bounds

                w4326, s4326, e4326, n4326 = tile_bbox_4326
                center_lon = float((w4326 + e4326) / 2.0)
                center_lat = float((s4326 + n4326) / 2.0)
                footprint_wkt = box(w4326, s4326, e4326, n4326).wkt

                # Step 3: Normalize bands (if enabled)
                if config.normalize:
                    tile_processed = normalize_raster_bands(tile_raw, nodata_mask)
                else:
                    tile_processed = tile_raw.copy()

                # Step 4: Write GeoTIFF tile to disk (preserving CRS and native transform)
                tile_filename = f"tile_{tile_col:04d}_{tile_row:04d}.tif"
                tile_file_path = tiles_dir / tile_filename
                out_meta = ds.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "width": actual_w,
                    "height": actual_h,
                    "transform": tile_transform,
                    "count": tile_processed.shape[0],
                    "dtype": str(tile_processed.dtype),
                    "nodata": nodata_val,
                })
                with rasterio.open(tile_file_path, "w", **out_meta) as out_ds:
                    out_ds.write(tile_processed)

                # Step 5: Generate RGB Preview and Thumbnail
                preview_rel = f"preview_{tile_col:04d}_{tile_row:04d}.png"
                preview_path = previews_dir / preview_rel
                thumb_rel = f"thumb_{tile_col:04d}_{tile_row:04d}.png"
                thumb_path = thumbnails_dir / thumb_rel

                rgb_vis = extract_rgb_visualization(
                    tile_raw, rgb_indices=rgb_indices, nodata_mask=nodata_mask
                )
                _save_image_preview(
                    rgb_vis,
                    preview_path,
                    target_size=(config.preview_size, config.preview_size),
                )
                _save_image_preview(
                    rgb_vis,
                    thumb_path,
                    target_size=(config.thumbnail_size, config.thumbnail_size),
                )

                # Compute quality metrics
                cloud_pct = estimate_cloud_cover(rgb_vis, nodata_mask)
                quality_score = max(0.0, min(1.0, 1.0 - (nodata_ratio * 0.5) - ((cloud_pct / 100.0) * 0.4)))

                tile_id = uuid.uuid4()
                tile_record = Tile(
                    id=tile_id,
                    scene_id=scene.id,
                    tile_col=tile_col,
                    tile_row=tile_row,
                    tile_size=tile_size,
                    pixel_x_off=col_off,
                    pixel_y_off=row_off,
                    pixel_width=actual_w,
                    pixel_height=actual_h,
                    center_lon=center_lon,
                    center_lat=center_lat,
                    bbox_west=w4326,
                    bbox_south=s4326,
                    bbox_east=e4326,
                    bbox_north=n4326,
                    tile_path=str(tile_file_path),
                    thumbnail_path=str(thumb_path),
                    acquisition_date=scene.acquisition_date,
                    sensor=scene.sensor,
                    resolution_m=scene.resolution_x_m,
                    cloud_cover_pct=cloud_pct,
                    quality_score=round(quality_score, 4),
                    nodata_ratio=round(nodata_ratio, 4),
                    is_valid=True,
                    processing_history=[
                        {
                            "operation": "tiling",
                            "pipeline": "phase3_production",
                            "tile_size": tile_size,
                            "overlap": overlap,
                            "normalized": config.normalize,
                            "stride": stride,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                )

                session.add(tile_record)
                created_tiles.append(tile_record)
                tile_wkt_map.append((tile_record, footprint_wkt))

                tile_col += 1

                # Flush in chunks to keep memory low and report progress
                if len(created_tiles) % 20 == 0:
                    await session.flush()
                    pct = 10.0 + (total_inspected / max(total_expected, 1)) * 80.0
                    await report_progress(
                        min(90.0, pct),
                        f"Generated {len(created_tiles)} tiles ({total_inspected}/{total_expected})...",
                    )
                    await asyncio.sleep(0)

            tile_row += 1

    # Step 6: Flush all tiles and update PostGIS geometry
    await report_progress(92.0, "Updating PostGIS spatial footprints...")
    await session.flush()

    for tile_obj, wkt in tile_wkt_map:
        await session.execute(
            text("UPDATE tiles SET footprint = ST_GeomFromText(:wkt, 4326) WHERE id = :id"),
            {"wkt": wkt, "id": str(tile_obj.id)},
        )
    await session.flush()

    elapsed = round(time.perf_counter() - start_time, 3)

    # Step 7: Update Scene metadata and add Provenance record
    scene.tile_count = len(created_tiles)
    scene.ingestion_status = "COMPLETED"
    scene.updated_at = datetime.now(timezone.utc)

    provenance = ProvenanceRecord(
        entity_id=scene.id,
        entity_type="scene",
        operation="tiling_and_preprocessing",
        details={
            "pipeline": "phase3_tiling",
            "tile_size": tile_size,
            "overlap_px": overlap,
            "normalize": config.normalize,
            "tiles_generated": len(created_tiles),
            "skipped_nodata": skipped_nodata,
            "processing_time_seconds": elapsed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    session.add(provenance)

    # Update job if one was supplied
    if job_id:
        job = await session.get(ProcessingJob, job_id)
        if job:
            job.status = "COMPLETED"
            job.progress_pct = 100.0
            job.progress_message = f"Completed in {elapsed}s: {len(created_tiles)} tiles generated"
            job.completed_at = datetime.now(timezone.utc)

    await session.commit()

    logger.info(
        "tiling_completed",
        scene_id=str(scene_id),
        tile_count=len(created_tiles),
        elapsed_sec=elapsed,
    )

    return TilingResult(
        scene_id=scene_id,
        tile_count=len(created_tiles),
        processing_time_seconds=elapsed,
        total_windows_inspected=total_inspected,
        skipped_nodata_count=skipped_nodata,
        tiles=created_tiles,
    )


# ─────────────────────────────────────────────────────────────
# Preprocessing and Validation Helpers
# ─────────────────────────────────────────────────────────────

def validate_bands(path: Path) -> dict[str, Any]:
    """
    Validate raster bands, data types, and determine RGB band assignment.
    """
    with rasterio.open(path) as ds:
        band_count = ds.count
        if band_count == 0:
            raise ValueError(f"Raster {path.name} contains no bands")
        if ds.width <= 0 or ds.height <= 0:
            raise ValueError(f"Raster {path.name} has zero dimensions ({ds.width}x{ds.height})")

        dtype = ds.dtypes[0]
        descriptions = [ds.descriptions[i] or f"Band_{i+1}" for i in range(band_count)]

        # Determine RGB indices (0-indexed into read array)
        rgb_indices = [0, 0, 0]
        if band_count >= 3:
            lower_descs = [d.lower() for d in descriptions]
            r_idx, g_idx, b_idx = -1, -1, -1
            for idx, name in enumerate(lower_descs):
                if "red" in name:
                    r_idx = idx
                elif "green" in name:
                    g_idx = idx
                elif "blue" in name:
                    b_idx = idx

            if r_idx != -1 and g_idx != -1 and b_idx != -1:
                rgb_indices = [r_idx, g_idx, b_idx]
            elif band_count == 4:
                rgb_indices = [0, 1, 2]
            else:
                rgb_indices = [0, 1, 2]
        elif band_count == 1:
            rgb_indices = [0, 0, 0]
        elif band_count == 2:
            rgb_indices = [0, 1, 0]

        return {
            "band_count": band_count,
            "dtype": dtype,
            "descriptions": descriptions,
            "rgb_indices": rgb_indices,
            "crs_wkt": ds.crs.to_wkt() if ds.crs else None,
            "nodata": ds.nodata,
        }


def normalize_raster_bands(
    data: np.ndarray, nodata_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Robust 2%-98% percentile stretch per band for radiometric normalization.
    Preserves float32 fidelity without modifying original source data.
    """
    out = np.zeros_like(data, dtype=np.float32)
    bands = data.shape[0]

    for b in range(bands):
        band = data[b].astype(np.float32)
        valid_pixels = band[~nodata_mask] if nodata_mask is not None else band.flatten()

        if valid_pixels.size > 0:
            valid_pixels = valid_pixels[np.isfinite(valid_pixels)]

        if valid_pixels.size > 10:
            p2, p98 = np.percentile(valid_pixels, [2.0, 98.0])
            if p98 > p2:
                stretched = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
            else:
                max_val = float(np.max(valid_pixels)) or 1.0
                stretched = np.clip(band / max_val, 0.0, 1.0)
        else:
            stretched = np.clip(band, 0.0, 1.0)

        if nodata_mask is not None:
            stretched[nodata_mask] = 0.0

        out[b] = stretched

    return out


def extract_rgb_visualization(
    data: np.ndarray,
    rgb_indices: list[int] = [0, 1, 2],
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Constructs an 8-bit RGB array (H, W, 3) suitable for display,
    normalizing contrast via 2nd-98th percentile stretch.
    """
    bands, height, width = data.shape
    rgb_stack = []

    for idx in rgb_indices:
        safe_idx = min(idx, bands - 1)
        band = data[safe_idx].astype(np.float32)

        valid = band[~nodata_mask] if nodata_mask is not None else band.flatten()
        if valid.size > 0:
            valid = valid[np.isfinite(valid)]

        if valid.size > 10:
            p2, p98 = np.percentile(valid, [2.0, 98.0])
            if p98 > p2:
                scaled = np.clip((band - p2) / (p98 - p2) * 255.0, 0, 255)
            else:
                scaled = np.clip(band, 0, 255)
        else:
            scaled = np.clip(band, 0, 255)

        if nodata_mask is not None:
            scaled[nodata_mask] = 0.0

        rgb_stack.append(scaled.astype(np.uint8))

    rgb_hwc = np.stack(rgb_stack, axis=-1)
    return rgb_hwc


def estimate_cloud_cover(rgb_hwc: np.ndarray, nodata_mask: np.ndarray) -> float:
    """
    Fast RGB brightness and spectral neutrality heuristic for cloud cover estimate (0-100%).
    """
    if rgb_hwc.shape[-1] < 3:
        return 0.0

    r = rgb_hwc[:, :, 0].astype(np.float32) / 255.0
    g = rgb_hwc[:, :, 1].astype(np.float32) / 255.0
    b = rgb_hwc[:, :, 2].astype(np.float32) / 255.0

    brightness = (r + g + b) / 3.0
    whiteness = 1.0 - (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0
    cloud_score = brightness * whiteness

    cloud_mask = (cloud_score > 0.72) & (~nodata_mask)
    valid_count = np.count_nonzero(~nodata_mask)
    if valid_count == 0:
        return 0.0

    cloud_pct = (np.count_nonzero(cloud_mask) / valid_count) * 100.0
    return round(float(cloud_pct), 2)


def _compute_nodata_mask(data: np.ndarray, nodata_val: float | None) -> np.ndarray:
    """Mask of pixels where all bands match nodata or are 0/NaN."""
    if nodata_val is not None and not np.isnan(nodata_val):
        mask = np.all(data == nodata_val, axis=0)
    else:
        nan_mask = np.any(np.isnan(data), axis=0)
        zero_mask = np.all(data == 0, axis=0)
        mask = nan_mask | zero_mask
    return mask


def _save_image_preview(
    rgb_hwc: np.ndarray,
    out_path: Path,
    target_size: tuple[int, int] = (512, 512),
) -> None:
    """Resize and save RGB image preview as PNG."""
    img = Image.fromarray(rgb_hwc, mode="RGB")
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    img.save(out_path, format="PNG", optimize=True)
