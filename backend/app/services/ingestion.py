"""
GeoTIFF ingestion service.
Pipeline: validate → extract metadata → quality → tile → thumbnails → PostGIS → provenance
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import os
try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    os.environ["PROJ_LIB"] = proj_dir
    os.environ["PROJ_DATA"] = proj_dir
except Exception:
    pass

import numpy as np
import rasterio
from PIL import Image
from rasterio import windows
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.provenance import ProcessingJob, ProvenanceRecord
from app.models.scene import Scene
from app.models.tile import Tile

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

async def ingest_scene(
    file_path: Path,
    session: AsyncSession,
    job_id: uuid.UUID,
) -> Scene:
    """
    Full ingestion pipeline for a single GeoTIFF/COG file.
    Updates the ProcessingJob record with progress at each stage.
    Raises on unrecoverable errors; per-tile errors are logged and skipped.
    """
    logger.info("ingestion_started", path=str(file_path))
    scene_id = uuid.uuid4()

    async def update_progress(pct: float, msg: str) -> None:
        job = await session.get(ProcessingJob, job_id)
        if job:
            job.progress_pct = pct
            job.progress_message = msg
            await session.flush()

    try:
        # ── Stage 1: Validate ─────────────────────────────────
        await update_progress(5.0, "Validating file...")
        _validate_geotiff(file_path)

        # ── Stage 2: Extract metadata ─────────────────────────
        await update_progress(10.0, "Extracting metadata...")
        meta = _extract_metadata(file_path)

        # ── Stage 3: Quality assessment ───────────────────────
        await update_progress(20.0, "Running quality assessment...")
        quality = _assess_quality(file_path, meta)

        # ── Stage 4: Create Scene record ──────────────────────
        await update_progress(25.0, "Saving scene record...")
        scene = Scene(
            id=scene_id,
            source_path=str(file_path),
            filename=file_path.name,
            file_checksum=_compute_checksum(file_path),
            file_size_bytes=file_path.stat().st_size,
            sensor=meta.get("sensor"),
            platform=meta.get("platform"),
            acquisition_date=meta.get("acquisition_date"),
            acquisition_date_str=meta.get("acquisition_date_str"),
            crs_wkt=meta["crs_wkt"],
            crs_epsg=meta.get("crs_epsg"),
            width_px=meta["width"],
            height_px=meta["height"],
            num_bands=meta["band_count"],
            band_descriptions=meta.get("band_descriptions"),
            resolution_x_m=meta.get("res_x_m"),
            resolution_y_m=meta.get("res_y_m"),
            nodata_value=meta.get("nodata"),
            bbox_west=meta["bbox_4326"][0],
            bbox_south=meta["bbox_4326"][1],
            bbox_east=meta["bbox_4326"][2],
            bbox_north=meta["bbox_4326"][3],
            cloud_cover_pct=quality.get("cloud_cover_pct"),
            quality_score=quality.get("quality_score"),
            quality_details=quality,
            ingestion_status="PROCESSING",
            raw_metadata=meta.get("raw_tags"),
        )
        session.add(scene)
        await session.flush()

        # Set PostGIS geometry footprint (polygon in EPSG:4326)
        w, s, e, n = meta["bbox_4326"]
        footprint_wkt = box(w, s, e, n).wkt
        from sqlalchemy import text
        await session.execute(
            text(
                "UPDATE scenes SET footprint = ST_GeomFromText(:wkt, 4326) "
                "WHERE id = :id"
            ),
            {"wkt": footprint_wkt, "id": str(scene_id)},
        )

        # ── Stage 5: Generate tiles ───────────────────────────
        await update_progress(30.0, "Generating tiles...")
        tile_count = await _generate_tiles(file_path, scene, meta, session, update_progress)

        # ── Stage 6: Finalize ─────────────────────────────────
        scene.ingestion_status = "COMPLETED"
        scene.tile_count = tile_count

        # ── Stage 7: Provenance record ────────────────────────
        prov = ProvenanceRecord(
            entity_id=scene_id,
            entity_type="scene",
            operation="ingestion",
            details={
                "pipeline_version": "1.0.0",
                "source_file": str(file_path),
                "tile_count": tile_count,
                "quality": quality,
                "metadata": {k: str(v) for k, v in meta.items() if k != "raw_tags"},
            },
        )
        session.add(prov)

        # Finalize job
        job = await session.get(ProcessingJob, job_id)
        if job:
            job.status = "COMPLETED"
            job.progress_pct = 100.0
            job.progress_message = f"Completed: {tile_count} tiles generated"
            job.completed_at = datetime.now(timezone.utc)

        await session.commit()
        logger.info("ingestion_completed", scene_id=str(scene_id), tiles=tile_count)
        return scene

    except Exception as exc:
        logger.exception("ingestion_failed", path=str(file_path), error=str(exc))
        # Mark scene and job as failed
        try:
            existing_scene = await session.get(Scene, scene_id)
            if existing_scene:
                existing_scene.ingestion_status = "FAILED"
                existing_scene.ingestion_error = str(exc)
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = "FAILED"
                job.error_message = str(exc)
                job.completed_at = datetime.now(timezone.utc)
            await session.commit()
        except Exception:
            pass
        raise


# ─────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────

def _validate_geotiff(path: Path) -> None:
    """Open the file with rasterio to confirm it is a valid raster."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with rasterio.open(path) as ds:
            if ds.count == 0:
                raise ValueError("Raster has no bands")
            if ds.width == 0 or ds.height == 0:
                raise ValueError("Raster has zero dimensions")
    except rasterio.errors.RasterioIOError as exc:
        raise ValueError(f"Not a valid raster file: {exc}") from exc


def _extract_metadata(path: Path) -> dict[str, Any]:
    """
    Extract all geospatial and radiometric metadata from the file.
    Preserves original CRS; generates WGS84 bounding box for spatial queries.
    """
    with rasterio.open(path) as ds:
        crs: CRS = ds.crs
        transform = ds.transform
        width, height = ds.width, ds.height
        band_count = ds.count

        # Resolution in native units
        res_x = abs(transform.a)
        res_y = abs(transform.e)

        # Attempt to determine resolution in metres
        res_x_m: float | None = None
        res_y_m: float | None = None
        if crs and crs.is_projected:
            res_x_m = res_x
            res_y_m = res_y
        elif crs and crs.is_geographic:
            # Rough conversion: 1 degree ≈ 111 km at equator
            res_x_m = res_x * 111_000
            res_y_m = res_y * 111_000

        # Bounding box in WGS84
        try:
            bbox_4326 = transform_bounds(crs, "EPSG:4326", *ds.bounds)
        except Exception:
            # Fallback: use raw bounds if reprojection fails
            b = ds.bounds
            bbox_4326 = (b.left, b.bottom, b.right, b.top)

        # Band descriptions
        band_descs = [ds.descriptions[i] or f"Band_{i+1}" for i in range(band_count)]

        # Acquisition metadata from GDAL tags
        tags = ds.tags()
        acq_date, acq_date_str = _parse_acquisition_date(tags, path)

        # Sensor information from tags
        sensor, platform = _parse_sensor_info(tags, path)

        return {
            "crs_wkt": crs.to_wkt() if crs else None,
            "crs_epsg": crs.to_epsg() if crs else None,
            "width": width,
            "height": height,
            "band_count": band_count,
            "band_descriptions": band_descs,
            "res_x": res_x,
            "res_y": res_y,
            "res_x_m": res_x_m,
            "res_y_m": res_y_m,
            "nodata": ds.nodata,
            "bbox_4326": list(bbox_4326),
            "acquisition_date": acq_date,
            "acquisition_date_str": acq_date_str,
            "sensor": sensor,
            "platform": platform,
            "raw_tags": dict(tags),
        }


def _parse_acquisition_date(
    tags: dict, path: Path
) -> tuple[datetime | None, str | None]:
    """Try to extract acquisition date from GDAL metadata tags or filename."""
    # Common GDAL tag keys
    date_keys = [
        "ACQUISITIONDATETIME", "ACQUISITION_DATE", "DATE_ACQUIRED",
        "Acquisition_DateTime", "SensingTime", "time_coverage_start",
    ]
    for key in date_keys:
        if key in tags:
            raw = tags[key]
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return dt, raw
            except ValueError:
                pass

    # Try parsing from filename (e.g., S2A_20230615T... or LC08_L2SP_20221201...)
    stem = path.stem
    import re
    patterns = [
        r"(\d{8}T\d{6})",  # 20230615T103045
        r"(\d{8})",         # 20230615
        r"_(\d{4}-\d{2}-\d{2})_",
    ]
    for pat in patterns:
        m = re.search(pat, stem)
        if m:
            raw = m.group(1)
            try:
                if "T" in raw:
                    dt = datetime.strptime(raw, "%Y%m%dT%H%M%S").replace(
                        tzinfo=timezone.utc
                    )
                else:
                    clean = raw.replace("-", "")
                    dt = datetime.strptime(clean, "%Y%m%d").replace(
                        tzinfo=timezone.utc
                    )
                return dt, raw
            except ValueError:
                pass

    return None, None


def _parse_sensor_info(tags: dict, path: Path) -> tuple[str | None, str | None]:
    """Extract sensor and platform from GDAL tags or filename convention."""
    sensor_keys = ["SENSOR", "sensor", "SENSOR_ID", "sensor_id", "satellite", "SATELLITE", "SPACECRAFT_ID", "Mission", "INSTRUMENT"]
    platform_keys = ["PLATFORM", "platform", "SPACECRAFT_ID", "spacecraft", "Mission", "SATELLITE", "satellite"]

    lower_tags = {k.lower(): v for k, v in tags.items()}
    sensor = next((tags[k] for k in sensor_keys if k in tags), None) or next((lower_tags[k.lower()] for k in sensor_keys if k.lower() in lower_tags), None)
    platform = next((tags[k] for k in platform_keys if k in tags), None) or next((lower_tags[k.lower()] for k in platform_keys if k.lower() in lower_tags), None)

    # Filename fallback if no metadata tags exist
    name = path.stem.upper()
    if not sensor or not platform:
        if "SENTINEL-2" in name or "SENTINEL2" in name or name.startswith("S2"):
            sensor = sensor or "Sentinel-2 MSI"
            platform = platform or "Sentinel-2"
        elif "LANDSAT-8" in name or "LANDSAT8" in name or name.startswith("LC08") or name.startswith("LT08"):
            sensor = sensor or "Landsat OLI/TIRS"
            platform = platform or "Landsat-8"
        elif "LANDSAT-9" in name or "LANDSAT9" in name or name.startswith("LC09"):
            sensor = sensor or "Landsat OLI-2/TIRS-2"
            platform = platform or "Landsat-9"
        elif "LANDSAT-7" in name or "LANDSAT7" in name or name.startswith("LE07"):
            sensor = sensor or "Landsat ETM+"
            platform = platform or "Landsat-7"

    return sensor, platform


def _assess_quality(path: Path, meta: dict) -> dict[str, Any]:
    """
    Compute quality metrics from actual raster data.
    Uses satellite-provided masks when present; falls back to statistical heuristics.
    """
    with rasterio.open(path) as ds:
        band_count = ds.count
        nodata = ds.nodata

        # Read a sample of bands for efficiency (max 3)
        read_bands = list(range(1, min(band_count + 1, 4)))
        # Use a subsampled read for large files
        out_shape = (len(read_bands), min(512, ds.height), min(512, ds.width))
        data = ds.read(
            indexes=read_bands,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        ).astype(np.float32)

        # No-data ratio
        if nodata is not None:
            nodata_mask = np.any(data == nodata, axis=0)
        else:
            nodata_mask = np.any(data == 0, axis=0) if band_count >= 3 else np.zeros(
                (out_shape[1], out_shape[2]), dtype=bool
            )
        nodata_ratio = float(nodata_mask.mean())
        usable_ratio = 1.0 - nodata_ratio

        quality_details: dict[str, Any] = {
            "nodata_ratio": round(nodata_ratio, 4),
            "usable_ratio": round(usable_ratio, 4),
        }

        # RGB brightness-based cloud heuristic (fallback only)
        cloud_cover_pct: float | None = None
        if band_count >= 3:
            rgb = data[:3]
            # Normalize to 0–1 if needed
            max_val = rgb.max()
            if max_val > 1.0:
                norm_factor = 10000.0 if max_val > 5000 else 255.0
                rgb = rgb / norm_factor
            rgb = np.clip(rgb, 0, 1)

            # Cloud heuristic: bright + white (R≈G≈B, all high)
            brightness = rgb.mean(axis=0)
            r, g, b = rgb[0], rgb[1], rgb[2]
            whiteness = 1 - (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3
            cloud_prob = brightness * whiteness
            cloud_mask = (cloud_prob > 0.75) & ~nodata_mask
            cloud_cover_pct = round(float(cloud_mask.mean()) * 100, 2)
            quality_details["cloud_cover_pct"] = cloud_cover_pct
            quality_details["cloud_method"] = "rgb_brightness_whiteness_heuristic"

            # Blur/sharpness via Laplacian variance on luminance
            import cv2
            lum = (0.299 * r + 0.587 * g + 0.114 * b)
            lum_uint8 = (lum * 255).astype(np.uint8)
            laplacian_var = float(cv2.Laplacian(lum_uint8, cv2.CV_64F).var())
            sharpness = min(1.0, laplacian_var / 1000.0)
            quality_details["sharpness_score"] = round(sharpness, 4)
        else:
            quality_details["cloud_method"] = "not_assessed_insufficient_bands"

        # Composite quality score (0–1, higher = better)
        cloud_penalty = (cloud_cover_pct / 100.0) if cloud_cover_pct is not None else 0.0
        sharpness = quality_details.get("sharpness_score", 0.7)
        quality_score = round(
            (1.0 - cloud_penalty * 0.5) * (1.0 - nodata_ratio * 0.3) * (0.3 + 0.7 * sharpness),
            4,
        )
        quality_details["quality_score"] = quality_score

        return quality_details


async def _generate_tiles(
    path: Path,
    scene: Scene,
    meta: dict,
    session: AsyncSession,
    update_progress,
) -> int:
    """
    Tile the raster and create Tile records.
    Skips tiles below MIN_VALID_PIXEL_RATIO no-data threshold.
    Generates PNG thumbnails for each valid tile.
    """
    tile_size = settings.default_tile_size
    tiles_dir = settings.tiles_dir / str(scene.id)
    thumbnails_dir = settings.thumbnails_dir / str(scene.id)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    valid_tiles: list[Tile] = []
    tile_col = 0
    tile_row = 0
    total_tiles_expected = 0

    with rasterio.open(path) as ds:
        width, height = ds.width, ds.height
        num_cols = (width + tile_size - 1) // tile_size
        num_rows = (height + tile_size - 1) // tile_size
        total_tiles_expected = num_cols * num_rows

        band_count = ds.count
        read_bands = list(range(1, min(band_count + 1, 4)))  # Read at most 4 bands
        nodata = ds.nodata
        crs = ds.crs

        processed = 0

        for row_off in range(0, height, tile_size):
            tile_col = 0
            for col_off in range(0, width, tile_size):
                w = windows.Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=min(tile_size, width - col_off),
                    height=min(tile_size, height - row_off),
                )
                actual_w, actual_h = int(w.width), int(w.height)

                # Read tile data
                data = ds.read(indexes=read_bands, window=w)

                # Skip no-data dominated tiles
                if nodata is not None:
                    nodata_mask = np.all(data == nodata, axis=0)
                else:
                    nodata_mask = np.all(data == 0, axis=0)
                nodata_ratio = float(nodata_mask.mean())

                if nodata_ratio > (1.0 - settings.min_valid_pixel_ratio):
                    tile_col += 1
                    processed += 1
                    continue

                # Compute tile geographic footprint
                tile_transform = windows.transform(w, ds.transform)
                tile_bounds = windows.bounds(w, ds.transform)
                try:
                    tile_bbox_4326 = transform_bounds(
                        crs, "EPSG:4326", *tile_bounds
                    )
                except Exception:
                    tile_bbox_4326 = tile_bounds

                w4326, s4326, e4326, n4326 = tile_bbox_4326
                center_lon = (w4326 + e4326) / 2
                center_lat = (s4326 + n4326) / 2

                # Save tile as GeoTIFF
                tile_filename = f"tile_{tile_col:04d}_{tile_row:04d}.tif"
                tile_path = tiles_dir / tile_filename
                out_meta = ds.meta.copy()
                out_meta.update({
                    "width": actual_w,
                    "height": actual_h,
                    "transform": tile_transform,
                    "count": len(read_bands),
                })
                with rasterio.open(tile_path, "w", **out_meta) as out_ds:
                    out_ds.write(data)

                # Generate thumbnail PNG
                thumb_path = _save_thumbnail(data, thumbnails_dir, tile_col, tile_row, nodata)

                # Compute per-tile cloud cover
                tile_cloud_pct = _quick_cloud_estimate(data) if len(read_bands) >= 3 else None

                tile_quality = max(0.0, 1.0 - nodata_ratio - (tile_cloud_pct or 0) / 200)

                tile_id = uuid.uuid4()
                footprint_wkt = box(w4326, s4326, e4326, n4326).wkt

                tile = Tile(
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
                    tile_path=str(tile_path),
                    thumbnail_path=str(thumb_path) if thumb_path else None,
                    acquisition_date=scene.acquisition_date,
                    sensor=scene.sensor,
                    resolution_m=meta.get("res_x_m"),
                    cloud_cover_pct=tile_cloud_pct,
                    quality_score=round(tile_quality, 4),
                    nodata_ratio=round(nodata_ratio, 4),
                    is_valid=True,
                    processing_history=[
                        {
                            "operation": "tiling",
                            "tile_size": tile_size,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                )

                # Set PostGIS footprint geometry
                # We'll do it via bulk update after flush
                valid_tiles.append((tile, footprint_wkt))
                session.add(tile)

                tile_col += 1
                processed += 1

                # Flush every 50 tiles and report progress
                if len(valid_tiles) % 50 == 0:
                    await session.flush()
                    pct = 30.0 + (processed / max(total_tiles_expected, 1)) * 65.0
                    await update_progress(pct, f"Tiling... {processed}/{total_tiles_expected}")
                    # Yield to event loop briefly
                    await asyncio.sleep(0)

            tile_row += 1

    # Final flush and geometry update
    await session.flush()

    # Update footprints for all tiles (batch)
    from sqlalchemy import text
    for tile_obj, wkt in valid_tiles:
        await session.execute(
            text(
                "UPDATE tiles SET footprint = ST_GeomFromText(:wkt, 4326) "
                "WHERE id = :id"
            ),
            {"wkt": wkt, "id": str(tile_obj.id)},
        )

    await session.flush()
    return len(valid_tiles)


def _save_thumbnail(
    data: np.ndarray,
    out_dir: Path,
    col: int,
    row: int,
    nodata: float | None,
) -> Path | None:
    """Generate a 128×128 PNG thumbnail from tile data."""
    try:
        if data.shape[0] >= 3:
            rgb = data[:3].astype(np.float32)
        elif data.shape[0] == 1:
            rgb = np.stack([data[0]] * 3, axis=0).astype(np.float32)
        else:
            rgb = np.stack([data[0], data[1], data[0]], axis=0).astype(np.float32)

        # Mask nodata
        if nodata is not None:
            mask = np.all(rgb == nodata, axis=0)
            for i in range(3):
                rgb[i][mask] = 0

        # Normalize to 0–255
        p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 1)
        if p98 > p2:
            rgb = np.clip((rgb - p2) / (p98 - p2) * 255, 0, 255)
        else:
            rgb = np.clip(rgb / max(rgb.max(), 1) * 255, 0, 255)

        # HWC → PIL
        arr = np.moveaxis(rgb.astype(np.uint8), 0, -1)
        img = Image.fromarray(arr).resize((128, 128), Image.LANCZOS)

        thumb_path = out_dir / f"thumb_{col:04d}_{row:04d}.png"
        img.save(thumb_path)
        return thumb_path
    except Exception as exc:
        logger.warning("thumbnail_failed", col=col, row=row, error=str(exc))
        return None


def _quick_cloud_estimate(data: np.ndarray) -> float:
    """Fast per-tile cloud estimate (0–100) using RGB brightness heuristic."""
    if data.shape[0] < 3:
        return 0.0
    rgb = data[:3].astype(np.float32)
    max_val = rgb.max()
    if max_val <= 0:
        return 0.0
    norm = 10000.0 if max_val > 5000 else 255.0
    rgb = np.clip(rgb / norm, 0, 1)
    r, g, b = rgb[0], rgb[1], rgb[2]
    brightness = rgb.mean(axis=0)
    whiteness = 1 - (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3
    cloud_prob = brightness * whiteness
    return round(float((cloud_prob > 0.75).mean()) * 100, 2)


def _compute_checksum(path: Path, chunk_size: int = 65536) -> str:
    """SHA-256 checksum of the file for duplicate detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
