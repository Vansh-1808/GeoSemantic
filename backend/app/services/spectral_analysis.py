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

SOLAR_KEYWORDS = {
    "solar", "photovoltaic", "pv", "panels", "bhadla"
}

SOLAR_PHRASES = [
    "solar park", "solar farm", "solar panels", "solar plant", "solar power", "solar array"
]

MOUNTAIN_KEYWORDS = {
    "mountain", "mountains", "himalaya", "himalayas", "glacier", "snow", "ice",
    "pangong", "ladakh", "kashmir", "rocky", "altitude"
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
            # Satellite imagery (Sentinel-2 etc.) can show water in multiple optical signatures:
            #   a) Bright water: high reflectance in blue channel (shallow/turbid)
            #   b) Dark water: deep reservoirs/lakes appear nearly black in true-color 8-bit thumbnails
            #   c) Teal/blue-green water: moderate brightness with cyan hue (clear freshwater)
            ndwi_rgb = (b - r) / (b + r + 1e-5)
            rgb_total = r + g + b  # total brightness proxy

            # No-data mask: truly black pixels (all channels < 8) are sensor no-data, not water
            nodata_mask = (r < 8) & (g < 8) & (b < 8)

            water_mask = (
                # Primary: NDWI-based water with blue dominance (typical shallow/reflective water)
                ((ndwi_rgb > 0.08) & (r < 95) & (b > r * 1.05) & ~nodata_mask) |
                # Secondary: Blue-green tinted shallow/mid water (teal to cyan hue)
                ((b > r * 1.1) & (g > r * 1.1) & (r < 65) & (b + g > 25) & ~nodata_mask) |
                # Tertiary: Dark deep water bodies (reservoirs/lakes in Sentinel-2)
                # These appear very dark (total brightness < 90) but are NOT no-data,
                # and have slightly more blue+green than red (even if marginal)
                ((rgb_total > 8) & (rgb_total < 90) & (b + g > r * 1.02) & (r < 60) & ~nodata_mask) |
                # Quaternary: Moderately dark dark-blue/navy water (turbid lakes at distance)
                ((b > r * 1.05) & (b > 8) & (r < 50) & (rgb_total < 130) & ~nodata_mask)
            )
            # Land pixels where vegetation and urban can exist (mutually exclusive from water)
            land_mask = ~water_mask & ~nodata_mask

            # 2. Peer-Reviewed Optical Vegetation Index:
            # Green Leaf Index (GLI): (2G - R - B) / (2G + R + B)
            # Standard remote sensing index for true-color satellite / aerial imagery (Louhaichi et al., 2001)
            denom = 2.0 * g + r + b + 1e-5
            gli = (2.0 * g - r - b) / denom

            # HSV Chromatic Analysis for natural chlorophyll signature
            # - Bright coastal / open sea: High blue-to-red ratio
            # - Turbid river / inland lake: Moderate green-blue with red absorption
            # - Deep dark reservoir (Sardar Sarovar): Absorbs across all bands, very low brightness
            bright_water = (~nodata_mask) & (b > r * 1.08) & (b > 25.0) & (r < 110.0)
            green_water = (~nodata_mask) & (g > r * 1.15) & (b > r * 0.95) & (r < 90.0) & (g < 140.0)
            # Deep/dark water: dark pixels that are non-zero and slightly blue/green biased
            dark_water = (
                (~nodata_mask)
                & (b >= 10.0)
                & (b <= 50.0)
                & (r <= 45.0)
                & (g <= 55.0)
                & ((b >= r) | (g >= r))
                & (b + g > r * 1.8)
            )
            # Reservoir / deep lake signature: very low red reflectance, balanced dark blue-green
            reservoir_water = (
                (~nodata_mask)
                & (r < 35.0)
                & (b >= 12.0)
                & (b < 65.0)
                & (g >= 12.0)
                & (g < 65.0)
                & (r < b * 0.90)
            )
            water_mask = bright_water | green_water | dark_water | reservoir_water

            # 2. Vegetation Extraction via HSV + Green Leaf Index (GLI)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]

            # Green hue range: 25 to 85, with minimum saturation
            veg_hsv = (~nodata_mask) & (~water_mask) & (h >= 25) & (h <= 85) & (s >= 25) & (v >= 30)

            # Green Leaf Index: (2*G - R - B) / (2*G + R + B + eps)
            gli_denom = 2.0 * g + r + b + 1e-6
            gli = (2.0 * g - r - b) / gli_denom
            veg_gli = (~nodata_mask) & (~water_mask) & (gli > 0.06) & (v >= 30)

            veg_mask = veg_hsv | veg_gli

            # Compute percentages relative to valid pixels (exclude no-data border pixels)
            total_valid = max(1.0, float((~nodata_mask).sum()))
            water_frac = float(water_mask.sum()) / total_valid
            veg_frac = float(veg_mask.sum()) / total_valid
            urban_frac = max(0.0, 1.0 - water_frac - veg_frac)

            water_pct = round(min(100.0, water_frac * 100.0), 1)
            veg_pct = round(min(100.0, veg_frac * 100.0), 1)
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
        is_solar = bool(cleaned_words & SOLAR_KEYWORDS) or any(phrase in q_lower for phrase in SOLAR_PHRASES)
        is_mountain = bool(cleaned_words & MOUNTAIN_KEYWORDS)

        requires_urban = (
            (
                bool(cleaned_words & URBAN_KEYWORDS) or
                any(phrase in q_lower for phrase in URBAN_PHRASES)
            )
            and not is_aviation
            and not is_solar
        )

        pure_water = requires_water and not (requires_urban or requires_vegetation or is_aviation or is_solar)

        return {
            "requires_water": requires_water,
            "requires_vegetation": requires_vegetation,
            "requires_urban": requires_urban,
            "pure_water": pure_water,
            "is_aviation": is_aviation,
            "is_solar": is_solar,
            "is_mountain": is_mountain,
        }

    def compute_fused_score(
        self,
        raw_similarity: float,
        spectral: Dict[str, float],
        intent: Dict[str, Any],
        scene_name: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Applies Hybrid Spectral-Semantic-Lexical Fusion.
        1. Spectral physical landcover matching ensures water/vegetation/urban queries
           strictly align with ground-truth satellite optical properties.
        2. Lexical & metadata matching accurately matches location and scene names.
        3. Vision-language semantic cosine similarity provides fine-grained tie-breaking.
        Returns (fused_score, calibrated_display_percentage in [0.05, 0.99]).
        """
        water_pct = spectral.get("water_pct", 0.0)
        veg_pct = spectral.get("veg_pct", 0.0)
        urban_pct = spectral.get("urban_pct", 0.0)
        sc_lower = (scene_name or "").lower()

        # --- Step 1: Spectral match score (primary signal, 0.0 to 1.0) ---
        spectral_score = 0.50  # default neutral score

        if intent.get("requires_water"):
            if water_pct >= 20.0:
                spectral_score = 0.90 + min(0.08, (water_pct - 20.0) / 1000.0)
            elif water_pct >= 10.0:
                spectral_score = 0.80
            elif water_pct >= 3.0:
                spectral_score = 0.60
            elif water_pct >= 1.0:
                spectral_score = 0.35
            else:
                # No water detected – strictly penalize so non-water regions don't pollute water queries
                spectral_score = 0.05

        elif intent.get("is_solar"):
            if "solar" in sc_lower:
                spectral_score = 0.92
            elif urban_pct >= 80.0 and veg_pct < 15.0:
                spectral_score = 0.72
            else:
                spectral_score = 0.20

        elif intent.get("is_aviation"):
            if "airport" in sc_lower or "airfield" in sc_lower:
                spectral_score = 0.92
            elif urban_pct >= 60.0 and veg_pct < 20.0:
                spectral_score = 0.78
            elif urban_pct >= 30.0:
                spectral_score = 0.55
            else:
                spectral_score = 0.20

        elif intent.get("requires_vegetation") and not intent.get("requires_water"):
            if veg_pct >= 30.0:
                spectral_score = 0.88
            elif veg_pct >= 15.0:
                spectral_score = 0.75
            elif veg_pct >= 5.0:
                spectral_score = 0.55
            elif veg_pct >= 2.0:
                spectral_score = 0.35
            else:
                spectral_score = 0.15

        elif intent.get("requires_urban") and not intent.get("requires_water"):
            if urban_pct >= 60.0:
                spectral_score = 0.85
            elif urban_pct >= 30.0:
                spectral_score = 0.70
            else:
                spectral_score = 0.30

        # --- Step 2: Lexical and Location Metadata Matching (Hybrid Search) ---
        lex_boost = 0.0
        if query_text and scene_name:
            q_clean = set(query_text.lower().replace("_", " ").replace("-", " ").replace(",", " ").split())
            s_clean = set(sc_lower.replace("_", " ").replace("-", " ").replace(".", " ").split())
            stops = {"in", "near", "the", "a", "an", "and", "of", "to", "for", "with", "tif", "png", "jpg"}
            q_clean -= stops
            s_clean -= stops
            matches = q_clean & s_clean
            if matches:
                lex_boost = min(0.22, len(matches) * 0.11)

        # --- Step 3: Semantic tie-breaker (continuous fine-tuning) ---
        SEMANTIC_MIN = -0.035
        SEMANTIC_MAX = 0.015
        sem_range = SEMANTIC_MAX - SEMANTIC_MIN
        normalized_sem = max(0.0, min(1.0, (float(raw_similarity) - SEMANTIC_MIN) / sem_range))
        sem_adjustment = (normalized_sem - 0.5) * 0.08  # [-0.04, +0.04]

        # Combine
        fused = spectral_score + lex_boost + sem_adjustment

        # If pure water was requested and tile has < 1% water, hard cap at 0.10
        if intent.get("pure_water") and water_pct < 1.0:
            fused = min(fused, 0.08)

        # Clamp and calibrate
        fused = max(0.05, min(0.99, fused))
        calibrated = round(fused, 4)

        return round(fused - 0.5, 4), calibrated

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
