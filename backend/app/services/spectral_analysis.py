"""
SpectralAnalysisService: Deterministic physical landcover analysis and cross-modal fusion.
Extracts spectral ratios (Water, Vegetation, Urban) directly from satellite raster imagery
and performs constraint fusion with vision-language embeddings (RemoteCLIP / DINOv2).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# Keywords for dynamic intent detection (zero hardcoding)
WATER_KEYWORDS = {
    "water", "ocean", "sea", "river", "coast", "coastal", "lake", "harbor",
    "port", "canal", "stream", "bay", "reservoir", "waterway", "shore", "beach",
    "creek", "estuary", "lagoon", "dock", "docks", "marina", "vessel", "vessels",
    "ship", "ships", "boat", "boats", "ferry"
}

WATER_PHRASES = [
    "water area", "water body", "water bodies", "near water", "along river",
    "coastal area", "by the ocean", "near ocean", "near sea", "waterfront"
]

VEGETATION_KEYWORDS = {
    "agricultural", "agriculture", "farmland", "farm", "farms", "crop", "crops",
    "irrigation", "forest", "forestry", "vegetation", "trees", "woodland", "orchard",
    "fields", "greenery", "pasture", "paddy"
}

VEGETATION_PHRASES = [
    "crop field", "agricultural land", "green area", "dense forest", "irrigation canal"
]

URBAN_KEYWORDS = {
    "residential", "residents", "houses", "housing", "buildings", "apartments",
    "neighborhood", "suburb", "urban", "city", "downtown", "commercial", "industrial"
}

URBAN_PHRASES = [
    "residential area", "housing colony", "road grid", "dense buildings", "urban grid"
]

AVIATION_KEYWORDS = {
    "airport", "airports", "runway", "runways", "apron", "aprons",
    "airfield", "airfields", "airstrip", "airstrips", "tarmac",
    "hangar", "hangars", "aerodrome", "aircraft", "airplane", "airplanes", "aviation"
}


class SpectralAnalysisService:
    """
    Computes deterministic optical landcover fractions and reranks semantic
    results to prevent compositionality failures in remote sensing foundation models.
    """

    def __init__(self) -> None:
        # In-memory LRU cache: image_path -> {"water_pct": float, "veg_pct": float, "urban_pct": float}
        self._cache: Dict[str, Dict[str, float]] = {}
        self._quality_cache: Dict[str, Dict[str, Any]] = {}

    def analyze_image(self, image_path: Union[str, Path]) -> Dict[str, float]:
        """
        Extract physical landcover percentages (water_pct, veg_pct, urban_pct)
        from a tile raster thumbnail using vectorized OpenCV operations (< 0.5 ms).
        """
        path_str = str(image_path)
        if path_str in self._cache:
            return self._cache[path_str]

        if not os.path.exists(path_str):
            default_res = {"water_pct": 0.0, "veg_pct": 0.0, "urban_pct": 100.0}
            return default_res

        try:
            img = cv2.imread(path_str)
            if img is None or img.size == 0:
                default_res = {"water_pct": 0.0, "veg_pct": 0.0, "urban_pct": 100.0}
                return default_res

            # In OpenCV, default image channels are BGR
            b = img[:, :, 0].astype(np.float32)
            g = img[:, :, 1].astype(np.float32)
            r = img[:, :, 2].astype(np.float32)

            # 1. Optical Water Index (RGB Normalized Difference + absorption gating)
            # Water reflects more in blue/green and strongly absorbs red.
            # Deep ocean in Sentinel-2 true-color can be very dark (B ~10-25, nearly black), so we
            # require blue spectral DOMINANCE (b > r * 1.05) rather than a fixed brightness floor (b > 25)
            # which incorrectly excluded dark deep-ocean pixels.
            ndwi_rgb = (b - r) / (b + r + 1e-5)
            water_mask = (
                # Primary: NDWI-based ocean/water with blue dominance (excludes asphalt where B ≈ R)
                ((ndwi_rgb > 0.08) & (r < 95) & (b > r * 1.05)) |
                # Secondary: Blue-green tinted shallow/mid water (requires modest total brightness)
                ((b > r * 1.1) & (g > r * 1.1) & (r < 65) & (b + g > 25))
            )
            # Land pixels where vegetation and urban can exist (mutually exclusive from water)
            land_mask = ~water_mask

            # 2. Peer-Reviewed Optical Vegetation Index:
            # Green Leaf Index (GLI): (2G - R - B) / (2G + R + B)
            # Standard remote sensing index for true-color satellite / aerial imagery (Louhaichi et al., 2001)
            denom = 2.0 * g + r + b + 1e-5
            gli = (2.0 * g - r - b) / denom

            # HSV Chromatic Analysis for natural chlorophyll signature
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0]  # In OpenCV: 0-179 (22-85 represents yellow-green, olive, emerald, and forest green)
            s = hsv[:, :, 1]  # 0-255 saturation (excludes achromatic greys/concrete where S < 20)
            v = hsv[:, :, 2]  # 0-255 brightness

            # Vegetation classification:
            # A. Chromatic chlorophyll hue (yellow-green to dark emerald) with adequate saturation on land
            veg_hsv = land_mask & (h >= 22) & (h <= 85) & (s >= 20) & (v >= 20)
            # B. Strong optical green excess (positive GLI with green dominance)
            veg_gli = land_mask & (gli > 0.025) & (g > r) & (g > b) & (g > 18)

            veg_mask = veg_hsv | veg_gli

            water_frac = float(water_mask.mean())
            veg_frac = float(veg_mask.mean())
            urban_frac = max(0.0, 1.0 - water_frac - veg_frac)

            water_pct = round(water_frac * 100.0, 1)
            veg_pct = round(veg_frac * 100.0, 1)
            urban_pct = max(0.0, round(100.0 - water_pct - veg_pct, 1))

            result = {
                "water_pct": water_pct,
                "veg_pct": veg_pct,
                "urban_pct": urban_pct,
            }
            self._cache[path_str] = result
            return result

        except Exception as exc:
            logger.warning("spectral_analysis_failed", path=path_str, error=str(exc))
            return {"water_pct": 0.0, "veg_pct": 0.0, "urban_pct": 100.0}

    def parse_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Parses physical landcover requirements from natural language queries.
        """
        q_lower = query.lower()
        cleaned_words = set(
            q_lower.replace("&", " ")
            .replace("/", " ")
            .replace(",", " ")
            .replace("-", " ")
            .split()
        )

        requires_water = (
            bool(cleaned_words & WATER_KEYWORDS) or
            any(phrase in q_lower for phrase in WATER_PHRASES)
        )
        requires_vegetation = (
            bool(cleaned_words & VEGETATION_KEYWORDS) or
            any(phrase in q_lower for phrase in VEGETATION_PHRASES)
        )
        is_aviation = bool(cleaned_words & AVIATION_KEYWORDS)
        requires_urban = (
            (
                bool(cleaned_words & URBAN_KEYWORDS) or
                any(phrase in q_lower for phrase in URBAN_PHRASES)
            )
            and not is_aviation
        )

        pure_water = requires_water and not (requires_urban or requires_vegetation)

        return {
            "requires_water": requires_water,
            "requires_vegetation": requires_vegetation,
            "requires_urban": requires_urban,
            "pure_water": pure_water,
            "is_aviation": is_aviation,
        }

    def compute_fused_score(
        self,
        raw_similarity: float,
        spectral: Dict[str, float],
        intent: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        Applies Spectral-Semantic Fusion constraints:
        - Penalizes tiles lacking required physical features (e.g. water query on dry tile).
        - Synergistically boosts tiles that satisfy both semantic and physical constraints.
        - Returns (fused_score, calibrated_similarity_score) where calibrated is in [0.05, 0.98].
        """
        fused = float(raw_similarity)
        water_pct = spectral.get("water_pct", 0.0)
        veg_pct = spectral.get("veg_pct", 0.0)
        urban_pct = spectral.get("urban_pct", 0.0)

        # 1. Water Constraint Logic
        if intent.get("requires_water"):
            if water_pct < 1.5:
                # Heavy penalty: user explicitly asked for water/water-adjacent, but tile has 0% water
                fused -= 0.15
            else:
                # Boost based on water presence & semantic synergy
                water_factor = min(1.0, water_pct / 20.0)
                fused += 0.04 * water_factor
                if intent.get("pure_water"):
                    # For pure water queries, higher water fraction receives proportional reward
                    fused += 0.03 * (water_pct / 100.0)

        # 2. Vegetation Constraint Logic
        if intent.get("requires_vegetation"):
            if veg_pct < 3.0:
                fused -= 0.12
            else:
                veg_factor = min(1.0, veg_pct / 25.0)
                fused += 0.04 * veg_factor

        # 3. Urban / Built-up Constraint Logic
        if intent.get("requires_urban"):
            if urban_pct < 15.0 and not intent.get("requires_water"):
                fused -= 0.10
            else:
                urban_factor = min(1.0, urban_pct / 60.0)
                fused += 0.02 * urban_factor

        # 4. Score Calibration: map raw/fused scores to realistic match percentages [0.05, 0.98]
        # Prevents artificial score inflation where unrelated queries (airports or arctic in Chennai) show 80%+.
        # - Negative/unrelated (fused <= -0.015): 5% - 25%
        # - Borderline/weak overlap (fused ≈ 0.000): 30% - 45%
        # - Moderate match (fused ≈ 0.010): 55% - 65%
        # - Strong match (fused >= 0.020): 75% - 90%
        # - High match (fused >= 0.035): 88% - 98%
        k = 65.0
        f0 = 0.007
        exponent = max(-50.0, min(50.0, -k * (fused - f0)))
        calibrated = round(min(0.98, max(0.05, 1.0 / (1.0 + math.exp(exponent)))), 4)

        return round(fused, 4), calibrated

    def analyze_quality(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Deterministic image quality assessment and confounding factors detection.
        Detects cloud/haze, shadows (distinguishing water vs land), no-data pixels,
        sensor saturation/burn, and row-to-row noise.
        Returns full quality scores, suitability, usability, and honest caveats.
        """
        path_str = str(image_path)
        if path_str in self._quality_cache:
            return dict(self._quality_cache[path_str])

        default_poor = {
            "quality_score": 0.0,
            "usable_for_change_analysis": False,
            "suitability": "Poor Quality",
            "cloud_score": 0.0,
            "shadow_score": 0.0,
            "nodata_score": 1.0,
            "noise_score": 0.0,
            "saturation_score": 0.0,
            "cloud_cover_pct": 0.0,
            "nodata_ratio": 1.0,
            "analysis_method": "deterministic_rgb_photometry",
            "satellite_mask_available": False,
            "caveats": [
                "Raster file unavailable or corrupted on disk.",
                "Defaulting to zero-quality safety fallback.",
            ],
        }

        if not os.path.exists(path_str):
            return default_poor

        try:
            img = cv2.imread(path_str)
            if img is None or img.size == 0:
                return default_poor

            b = img[:, :, 0].astype(np.float32)
            g = img[:, :, 1].astype(np.float32)
            r = img[:, :, 2].astype(np.float32)

            # 1. No-Data Detection (near-black boundary or unmapped sensor pixels)
            nodata_mask = (r < 8) & (g < 8) & (b < 8)
            valid_mask = ~nodata_mask
            nodata_score = float(nodata_mask.mean())

            # 2. Water Mask (Absorption Gating to prevent oceans being flagged as shadows)
            ndwi_rgb = (b - r) / (b + r + 1e-5)
            water_mask = (
                ((ndwi_rgb > 0.08) & (r < 95) & (b > r * 1.05)) |
                ((b > r * 1.1) & (g > r * 1.1) & (r < 65) & (b + g > 25))
            )
            land_mask = (~water_mask) & valid_mask

            # 3. Cloud / Haze Detection
            # True clouds are high reflectance across all optical bands with near-zero chromaticity (whiteness)
            max_rgb = np.maximum(np.maximum(r, g), b)
            min_rgb = np.minimum(np.minimum(r, g), b)
            whiteness = (max_rgb - min_rgb) / (max_rgb + 1e-5)
            cloud_mask = valid_mask & (min_rgb > 195) & (whiteness < 0.15)
            cloud_score = float(cloud_mask.mean())

            # 4. Shadow Detection
            # Shadows are dark on land surfaces (excluding water bodies and no-data)
            shadow_mask = land_mask & (max_rgb < 40)
            shadow_score = float(shadow_mask.mean())

            # 5. Saturation / Overexposure Detection
            # Extremely bright land pixels that blow out sensor dynamic range (excluding clouds)
            sat_mask = valid_mask & (min_rgb > 240) & (~cloud_mask)
            sat_score = float(sat_mask.mean())

            # 6. Sensor Noise / Banding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            row_means = np.mean(gray, axis=1)
            row_diffs = np.abs(np.diff(row_means))
            noise_score = min(1.0, float(np.mean(row_diffs) / (np.mean(row_means) + 1e-5)))

            # 7. Composite Quality Score
            # Occlusions (nodata + cloud) reduce the observable surface directly
            occlusion = min(1.0, nodata_score + cloud_score)
            impairment = shadow_score * 0.25 + noise_score * 0.20 + sat_score * 0.15
            quality_score = max(0.0, min(1.0, round(max(0.0, 1.0 - occlusion) * max(0.0, 1.0 - impairment), 4)))

            # 8. Change Suitability Classification
            # High-confidence change detection requires clear, well-exposed land
            if quality_score >= 0.75 and cloud_score < 0.20 and nodata_score < 0.15:
                suitability = "Suitable"
                usable_for_change_analysis = True
            elif quality_score >= 0.40 and cloud_score < 0.50 and nodata_score < 0.40:
                suitability = "Caution"
                usable_for_change_analysis = False
            else:
                suitability = "Poor Quality"
                usable_for_change_analysis = False

            caveats = [
                "No satellite quality mask (SCL/QA60) available -- cloud/haze detected using RGB spectral whiteness index.",
                "Thin haze cannot be distinguished from high-altitude cirrus without SWIR/thermal bands.",
                "Shadow detection distinguishes dark land from deep ocean using NDWI absorption gating.",
            ]

            result = {
                "quality_score": quality_score,
                "usable_for_change_analysis": usable_for_change_analysis,
                "suitability": suitability,
                "cloud_score": round(cloud_score, 4),
                "shadow_score": round(shadow_score, 4),
                "nodata_score": round(nodata_score, 4),
                "noise_score": round(noise_score, 4),
                "saturation_score": round(sat_score, 4),
                "cloud_cover_pct": round(cloud_score * 100.0, 2),
                "nodata_ratio": round(nodata_score, 4),
                "analysis_method": "deterministic_rgb_photometry",
                "satellite_mask_available": False,
                "caveats": caveats,
            }

            self._quality_cache[path_str] = result
            return dict(result)

        except Exception as exc:
            logger.warning("quality_analysis_failed", path=path_str, error=str(exc))
            return default_poor


# Module singleton
spectral_analysis_service = SpectralAnalysisService()
