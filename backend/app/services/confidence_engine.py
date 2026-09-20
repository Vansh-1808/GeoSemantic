"""
Multi-Factor Change Confidence Engine.
Core novelty of the change detection system:
Combines 10 distinct physical, semantic, and radiometric signals into a transparent,
explainable weighted scoring model with configurable weights, suppressive false-alarm filters,
and human-readable decision explanations.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
import numpy as np
import cv2

from app.core.logging import get_logger

logger = get_logger(__name__)


class ConfidenceWeights(BaseModel):
    """Configurable weights for the 10 input signals."""
    # Positive Change Signals (Sum to ~ 0.70 default)
    weight_visual: float = Field(0.25, ge=0.0, le=1.0, description="Weight of radiometric/structural visual change (RMSE)")
    weight_semantic: float = Field(0.25, ge=0.0, le=1.0, description="Weight of foundation model semantic distance (RemoteCLIP)")
    weight_spectral: float = Field(0.15, ge=0.0, le=1.0, description="Weight of physical landcover shift (NDVI/NDWI/NDBI)")
    weight_spatial: float = Field(0.15, ge=0.0, le=1.0, description="Weight of spatial cluster compactness and coherence")
    
    # Quality & Compatibility Modulation Signals (Sum to ~ 0.20 default)
    weight_quality: float = Field(0.08, ge=0.0, le=1.0, description="Weight of composite image sharpness and data fidelity")
    weight_registration: float = Field(0.08, ge=0.0, le=1.0, description="Weight of OpenCV sub-pixel alignment quality")
    weight_sensor: float = Field(0.02, ge=0.0, le=1.0, description="Weight of cross-sensor resolution and band compatibility")
    weight_seasonal: float = Field(0.02, ge=0.0, le=1.0, description="Weight of phenological calendar compatibility")

    # Penalties Multipliers
    cloud_penalty_multiplier: float = Field(1.2, ge=0.0, le=3.0, description="Multiplier for cloud/haze penalty")
    shadow_penalty_multiplier: float = Field(0.8, ge=0.0, le=3.0, description="Multiplier for terrain/illumination shadow penalty")
    misalignment_penalty_multiplier: float = Field(1.5, ge=0.0, le=3.0, description="Multiplier for misregistration penalty")
    seasonal_veg_penalty_multiplier: float = Field(1.2, ge=0.0, le=3.0, description="Multiplier for seasonal phenology false-alarm penalty")


class ConfidenceThresholds(BaseModel):
    """Configurable decision thresholds for confidence tiers and false-alarm suppression."""
    high_threshold: float = Field(0.70, ge=0.0, le=1.0, description="Minimum confidence for High Confidence tier")
    medium_threshold: float = Field(0.45, ge=0.0, le=1.0, description="Minimum confidence for Medium Confidence tier")
    low_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Minimum confidence for Low Confidence tier (below is Suppressed)")
    
    # Specific Hard Suppression Triggers
    cloud_suppression_limit: float = Field(0.25, ge=0.0, le=1.0, description="Cloud score above which change is suppressed as cloud artifact")
    registration_suppression_limit: float = Field(0.40, ge=0.0, le=1.0, description="Registration score below which change is suppressed as alignment error")
    seasonal_veg_ratio_limit: float = Field(0.60, ge=0.0, le=1.0, description="Vegetation change ratio above which seasonal vegetation is suppressed")
    shadow_suppression_limit: float = Field(0.40, ge=0.0, le=1.0, description="Shadow score above which change is suppressed as shadow artifact")


class FactorDetail(BaseModel):
    """Detailed breakdown for a single input signal."""
    key: str
    name: str
    raw_value: float
    weight: float
    contribution: float
    status: str  # "optimal", "moderate", "caution", "concerning", "suppressed"
    description: str


class ConfidenceEvaluationResult(BaseModel):
    """Complete transparent evaluation result from the Multi-Factor Confidence Engine."""
    final_confidence: float
    confidence_tier: str  # "High Confidence", "Medium Confidence", "Low Confidence", "Suppressed"
    is_suppressed: bool
    reasons: List[str]
    positive_factors: List[str]
    factor_breakdown: Dict[str, FactorDetail]
    penalties_applied: Dict[str, float]
    raw_signals: Dict[str, float]
    weights_applied: Dict[str, float]


class MultiFactorConfidenceEngine:
    """
    State-of-the-art Multi-Factor Change Confidence Engine.
    Suppresses false alarms from seasons, clouds, shadows, sensor differences,
    illumination, registration errors, and temporary anomalies.
    """

    def __init__(
        self,
        weights: Optional[ConfidenceWeights] = None,
        thresholds: Optional[ConfidenceThresholds] = None,
    ) -> None:
        self.weights = weights or ConfidenceWeights()
        self.thresholds = thresholds or ConfidenceThresholds()

    def update_config(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """Dynamically update weights and thresholds at runtime."""
        if weights:
            current_w = self.weights.model_dump()
            current_w.update(weights)
            self.weights = ConfidenceWeights(**current_w)
        if thresholds:
            current_t = self.thresholds.model_dump()
            current_t.update(thresholds)
            self.thresholds = ConfidenceThresholds(**current_t)
        logger.info("Updated Multi-Factor Confidence Engine configuration.")

    def get_config(self) -> Dict[str, Any]:
        """Return the current engine configuration."""
        return {
            "weights": self.weights.model_dump(),
            "thresholds": self.thresholds.model_dump(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Signal Calculators & Extractors
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_spatial_consistency(diff_mask: Optional[np.ndarray]) -> float:
        """
        Signal 10: Spatial Consistency.
        Computes spatial clustering and blob compactness using Connected Components Analysis.
        Real structural changes (buildings, roads, clearing) form contiguous spatial clusters.
        Salt-and-pepper sensor noise and registration edge errors form isolated single-pixel scatter.
        Returns score in [0.0, 1.0].
        """
        if diff_mask is None or diff_mask.size == 0:
            return 0.5

        # Binary mask of changed pixels
        mask_u8 = (diff_mask > 0).astype(np.uint8)
        changed_count = int(np.count_nonzero(mask_u8))
        if changed_count < 10:
            return 0.2  # Too few pixels to establish spatial consistency

        # Connected component labeling with 8-connectivity
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels <= 1:
            return 0.2

        # Filter out background (index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return 0.2

        # 1. Cluster area concentration: ratio of changed pixels in top-5 largest clusters
        areas_sorted = sorted(areas, reverse=True)
        top_clusters_area = sum(areas_sorted[:5])
        concentration_ratio = top_clusters_area / float(changed_count)

        # 2. Mean component size: real changes have large clusters (>50 pixels)
        mean_area = float(np.mean(areas))
        size_score = min(1.0, mean_area / 120.0)

        # 3. Compactness of the largest component: Area / (Bounding Box Area)
        max_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        w = stats[max_idx, cv2.CC_STAT_WIDTH]
        h = stats[max_idx, cv2.CC_STAT_HEIGHT]
        bbox_area = max(1, w * h)
        compactness = min(1.0, float(stats[max_idx, cv2.CC_STAT_AREA]) / bbox_area)

        # Fused spatial score
        spatial_score = (concentration_ratio * 0.45) + (size_score * 0.35) + (compactness * 0.20)
        return float(np.clip(spatial_score, 0.05, 0.99))

    @staticmethod
    def compute_sensor_compatibility(
        sensor_before: Optional[str],
        res_before: Optional[float],
        sensor_after: Optional[str],
        res_after: Optional[float],
    ) -> float:
        """
        Signal 7: Sensor Compatibility.
        Evaluates Ground Sampling Distance (GSD) resolution ratio and sensor platform match.
        Identical sensors = 1.0. Moderate resolution ratio (e.g. 10m vs 15m) = 0.85.
        High resolution mismatch (e.g. 10m vs 30m) = 0.60. Severe mismatch (< 0.2 ratio) = 0.30.
        """
        r1 = float(res_before or 10.0)
        r2 = float(res_after or 10.0)
        
        # GSD Resolution ratio (always <= 1.0)
        gsd_ratio = min(r1, r2) / max(r1, r2)

        # Sensor name compatibility
        s1 = (sensor_before or "optical").strip().lower()
        s2 = (sensor_after or "optical").strip().lower()

        if s1 == s2:
            sensor_name_match = 1.0
        elif any(k in s1 and k in s2 for k in ["sentinel", "landsat", "planet", "aerial"]):
            sensor_name_match = 0.90
        else:
            sensor_name_match = 0.70

        score = (gsd_ratio * 0.70) + (sensor_name_match * 0.30)
        return float(np.clip(score, 0.1, 1.0))

    @staticmethod
    def compute_seasonal_compatibility(
        date_before: Optional[Union[datetime, str]],
        date_after: Optional[Union[datetime, str]],
    ) -> float:
        """
        Signal 8: Seasonal Compatibility.
        Computes cyclic day-of-year distance to evaluate whether images are from
        the same season (anniversary acquisition) vs different seasons (e.g. wet monsoon vs dry winter).
        Same season = 1.0. 6 months apart (opposite season) = 0.10.
        """
        if not date_before or not date_after:
            return 0.75  # Moderate default when dates are unrecorded

        try:
            d1 = datetime.fromisoformat(str(date_before).replace("Z", "+00:00")) if isinstance(date_before, str) else date_before
            d2 = datetime.fromisoformat(str(date_after).replace("Z", "+00:00")) if isinstance(date_after, str) else date_after
            
            doy1 = d1.timetuple().tm_yday
            doy2 = d2.timetuple().tm_yday
            
            # Cyclic day-of-year difference (max 182.5 days)
            diff_days = abs(doy1 - doy2)
            if diff_days > 182:
                diff_days = 365 - diff_days

            # Cyclic cosine similarity: 0 days diff -> 1.0, 182 days diff -> 0.1
            norm_diff = diff_days / 182.5
            seasonal_sim = 0.5 * (1.0 + math.cos(math.pi * norm_diff))
            return float(np.clip(seasonal_sim * 0.9 + 0.1, 0.10, 1.0))
        except Exception as e:
            logger.warning(f"Could not compute seasonal compatibility: {e}")
            return 0.75

    @staticmethod
    def compute_spectral_evidence(
        delta_urban: float,
        delta_veg: float,
        delta_water: float,
        change_type: Optional[str] = None,
    ) -> Tuple[float, bool]:
        """
        Signal 9: Spectral Evidence.
        Verifies whether radiometric changes correspond to permanent physical material transitions
        (e.g., impervious built-up gain, persistent clearance) vs ephemeral phenological shifts (greening/browning).
        Returns:
          - spectral_score: [0.0, 1.0]
          - is_seasonal_vegetation_shift: bool
        """
        abs_urban = abs(delta_urban)
        abs_veg = abs(delta_veg)
        abs_water = abs(delta_water)

        is_seasonal_veg = False
        
        # If vegetation shifted dramatically (+/- 8%) but urban shift is tiny (< 1.5%),
        # it strongly indicates seasonal agricultural greening or winter drying!
        if abs_veg >= 6.0 and abs_urban < 2.0 and abs_water < 2.0:
            is_seasonal_veg = True
            # Spectral evidence for permanent structural change is low
            spectral_score = 0.20
        elif abs_urban >= 3.0:
            # High evidence of permanent impervious built-up change
            spectral_score = min(0.98, 0.60 + (abs_urban * 0.05))
        elif abs_water >= 4.0:
            # Physical water surface shift
            spectral_score = min(0.95, 0.55 + (abs_water * 0.04))
        elif abs_veg >= 4.0 and delta_veg < 0:
            # Vegetation clearance / deforestation
            spectral_score = min(0.90, 0.50 + (abs_veg * 0.04))
        else:
            # Moderate structural shift
            spectral_score = 0.50

        return float(np.clip(spectral_score, 0.05, 0.99)), is_seasonal_veg

    # ──────────────────────────────────────────────────────────────────────────
    # Multi-Factor Scoring & Decision Logic
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_signals(
        self,
        visual_change: float,
        semantic_change: float,
        image_quality: float,
        cloud_score: float,
        shadow_score: float,
        registration_confidence: float,
        sensor_compatibility: float,
        seasonal_compatibility: float,
        spectral_evidence: float,
        spatial_consistency: float,
        is_seasonal_vegetation: bool = False,
        change_type: Optional[str] = None,
        query_match_score: Optional[float] = None,
    ) -> ConfidenceEvaluationResult:
        """
        Core scoring equation combining all 10 signals into a transparent,
        explainable decision with honest reasons and configurable weights.
        """
        # Ensure clean bounded values [0.0, 1.0]
        v_change = float(np.clip(visual_change, 0.0, 1.0))
        s_change = float(np.clip(semantic_change, 0.0, 1.0))
        img_qual = float(np.clip(image_quality, 0.0, 1.0))
        c_score = float(np.clip(cloud_score, 0.0, 1.0))
        sh_score = float(np.clip(shadow_score, 0.0, 1.0))
        reg_conf = float(np.clip(registration_confidence, 0.0, 1.0))
        sens_comp = float(np.clip(sensor_compatibility, 0.0, 1.0))
        seas_comp = float(np.clip(seasonal_compatibility, 0.0, 1.0))
        spec_ev = float(np.clip(spectral_evidence, 0.0, 1.0))
        spat_cons = float(np.clip(spatial_consistency, 0.0, 1.0))

        raw_signals = {
            "visual_change_score": v_change,
            "semantic_change_score": s_change,
            "image_quality_score": img_qual,
            "cloud_score": c_score,
            "shadow_score": sh_score,
            "registration_confidence": reg_conf,
            "sensor_compatibility": sens_comp,
            "seasonal_compatibility": seas_comp,
            "spectral_evidence": spec_ev,
            "spatial_consistency": spat_cons,
        }

        # 1. Base Weighted Score (Linear combination of positive & quality signals)
        w = self.weights
        total_pos_weight = (
            w.weight_visual +
            w.weight_semantic +
            w.weight_spectral +
            w.weight_spatial +
            w.weight_quality +
            w.weight_registration +
            w.weight_sensor +
            w.weight_seasonal
        ) or 1.0

        pos_score = (
            (v_change * w.weight_visual) +
            (s_change * w.weight_semantic) +
            (spec_ev * w.weight_spectral) +
            (spat_cons * w.weight_spatial) +
            (img_qual * w.weight_quality) +
            (reg_conf * w.weight_registration) +
            (sens_comp * w.weight_sensor) +
            (seas_comp * w.weight_seasonal)
        ) / total_pos_weight

        # 2. Calculate Honest Confounder Penalties
        penalties = {}
        total_penalty = 0.0

        # Cloud Penalty (scales up rapidly if clouds are detected on either observation)
        if c_score > 0.05:
            cloud_pen = min(0.60, (c_score ** 1.3) * w.cloud_penalty_multiplier)
            penalties["cloud_penalty"] = round(cloud_pen, 4)
            total_penalty += cloud_pen

        # Shadow Penalty (illumination & terrain cast shadow shifts)
        if sh_score > 0.08:
            shadow_pen = min(0.35, (sh_score ** 1.2) * w.shadow_penalty_multiplier)
            penalties["shadow_penalty"] = round(shadow_pen, 4)
            total_penalty += shadow_pen

        # Misalignment Penalty (penalize if sub-pixel ECC correlation is weak)
        if reg_conf < 0.70:
            misalign_deficit = max(0.0, 0.70 - reg_conf)
            misalign_pen = min(0.50, (misalign_deficit * 1.5) * w.misalignment_penalty_multiplier)
            penalties["misalignment_penalty"] = round(misalign_pen, 4)
            total_penalty += misalign_pen

        # Seasonal Vegetation Penalty (phenological greening/drying masquerading as change)
        if is_seasonal_vegetation:
            veg_pen = min(0.45, 0.30 * w.seasonal_veg_penalty_multiplier)
            penalties["seasonal_vegetation_penalty"] = round(veg_pen, 4)
            total_penalty += veg_pen
        elif seas_comp < 0.40:
            # Minor seasonal timing penalty
            season_pen = round((0.40 - seas_comp) * 0.35, 4)
            penalties["seasonal_timing_penalty"] = season_pen
            total_penalty += season_pen

        # Sensor Disparity Penalty
        if sens_comp < 0.60:
            sensor_pen = round((0.60 - sens_comp) * 0.30, 4)
            penalties["sensor_mismatch_penalty"] = sensor_pen
            total_penalty += sensor_pen

        # 3. Compute Net Confidence
        raw_confidence = pos_score - total_penalty

        # If natural language query was provided and matched, boost confidence accordingly
        if query_match_score is not None:
            raw_confidence = (raw_confidence * 0.70) + (query_match_score * 0.30)

        final_confidence = float(np.clip(raw_confidence, 0.04, 0.99))

        # 4. Suppression Logic & Reason Generation
        is_suppressed = False
        reasons: List[str] = []
        positive_factors: List[str] = []

        th = self.thresholds

        # Hard Suppression Triggers
        if c_score >= th.cloud_suppression_limit:
            is_suppressed = True
            reasons.append(f"high cloud coverage ({c_score * 100.0:.1f}%)")
        elif c_score >= 0.12:
            reasons.append(f"moderate cloud/haze interference ({c_score * 100.0:.1f}%)")

        if reg_conf <= th.registration_suppression_limit:
            is_suppressed = True
            reasons.append(f"poor image alignment (OpenCV ECC quality: {reg_conf:.2f})")
        elif reg_conf < 0.65:
            reasons.append(f"sub-optimal image co-registration ({reg_conf:.2f})")

        if is_seasonal_vegetation and spec_ev <= th.seasonal_veg_ratio_limit:
            is_suppressed = True
            reasons.append("seasonal vegetation variation (phenological shift without structural change)")
        elif is_seasonal_vegetation:
            reasons.append("vegetation index fluctuations detected across seasons")

        if sh_score >= th.shadow_suppression_limit:
            is_suppressed = True
            reasons.append(f"excessive shadow occlusion ({sh_score * 100.0:.1f}%)")
        elif sh_score >= 0.15:
            reasons.append(f"illumination and cast shadow disparity ({sh_score * 100.0:.1f}%)")

        if sens_comp < 0.45:
            reasons.append(f"sensor mismatch (resolution/band disparity score: {sens_comp:.2f})")

        if img_qual < 0.30:
            reasons.append(f"poor image quality or data fidelity ({img_qual:.2f})")

        # Fallback suppression if confidence fell below low threshold
        if final_confidence < th.low_threshold and not is_suppressed:
            is_suppressed = True
            if not reasons:
                reasons.append("insufficient physical and semantic change evidence")

        # Positive Factors Documentation
        if reg_conf >= 0.85:
            positive_factors.append(f"high sub-pixel registration accuracy ({reg_conf * 100.0:.0f}%)")
        if c_score < 0.05 and img_qual >= 0.75:
            positive_factors.append("clear atmospheric conditions and optimal image fidelity")
        if spat_cons >= 0.70:
            positive_factors.append("spatially coherent, contiguous change cluster")
        if s_change >= 0.65:
            positive_factors.append("strong semantic transition confirmed by vision model")
        if spec_ev >= 0.70:
            positive_factors.append("verified physical material / built-up expansion")
        if sens_comp >= 0.90:
            positive_factors.append("homogeneous sensor observation pair")
        if seas_comp >= 0.85:
            positive_factors.append("consistent seasonal acquisition window")

        # 5. Tier Determination
        if is_suppressed or final_confidence < th.low_threshold:
            confidence_tier = "Suppressed"
            is_suppressed = True
        elif final_confidence >= th.high_threshold:
            confidence_tier = "High Confidence"
        elif final_confidence >= th.medium_threshold:
            confidence_tier = "Medium Confidence"
        else:
            confidence_tier = "Low Confidence"

        # 6. Build Detailed Factor Breakdown
        factor_breakdown = {
            "visual_change_score": FactorDetail(
                key="visual_change_score",
                name="Visual Change Score",
                raw_value=round(v_change, 3),
                weight=w.weight_visual,
                contribution=round((v_change * w.weight_visual) / total_pos_weight, 3),
                status="optimal" if v_change > 0.40 else "moderate",
                description="Radiometric RMSE and structural edge differences between aligned observations.",
            ),
            "semantic_change_score": FactorDetail(
                key="semantic_change_score",
                name="Semantic Change Score",
                raw_value=round(s_change, 3),
                weight=w.weight_semantic,
                contribution=round((s_change * w.weight_semantic) / total_pos_weight, 3),
                status="optimal" if s_change > 0.50 else "moderate",
                description="RemoteCLIP foundation embedding distance reflecting real landcover transition.",
            ),
            "image_quality_score": FactorDetail(
                key="image_quality_score",
                name="Image Quality Score",
                raw_value=round(img_qual, 3),
                weight=w.weight_quality,
                contribution=round((img_qual * w.weight_quality) / total_pos_weight, 3),
                status="optimal" if img_qual >= 0.75 else "caution" if img_qual >= 0.40 else "concerning",
                description="Composite photometric quality, sharpness, and lack of nodata/corruption.",
            ),
            "cloud_score": FactorDetail(
                key="cloud_score",
                name="Cloud Score",
                raw_value=round(c_score, 3),
                weight=w.cloud_penalty_multiplier,
                contribution=round(penalties.get("cloud_penalty", 0.0), 3),
                status="optimal" if c_score < 0.08 else "caution" if c_score < th.cloud_suppression_limit else "suppressed",
                description="Atmospheric haze and cloud reflectance whiteness index across observations.",
            ),
            "shadow_score": FactorDetail(
                key="shadow_score",
                name="Shadow Score",
                raw_value=round(sh_score, 3),
                weight=w.shadow_penalty_multiplier,
                contribution=round(penalties.get("shadow_penalty", 0.0), 3),
                status="optimal" if sh_score < 0.10 else "caution" if sh_score < th.shadow_suppression_limit else "suppressed",
                description="Cast shadow and low-luminance non-water land surface occlusion.",
            ),
            "registration_confidence": FactorDetail(
                key="registration_confidence",
                name="Registration Confidence",
                raw_value=round(reg_conf, 3),
                weight=w.weight_registration,
                contribution=round((reg_conf * w.weight_registration) / total_pos_weight, 3),
                status="optimal" if reg_conf >= 0.80 else "caution" if reg_conf > th.registration_suppression_limit else "suppressed",
                description="OpenCV ECC sub-pixel geometric co-registration correlation coefficient.",
            ),
            "sensor_compatibility": FactorDetail(
                key="sensor_compatibility",
                name="Sensor Compatibility",
                raw_value=round(sens_comp, 3),
                weight=w.weight_sensor,
                contribution=round((sens_comp * w.weight_sensor) / total_pos_weight, 3),
                status="optimal" if sens_comp >= 0.80 else "moderate",
                description="Spatial resolution GSD ratio and satellite sensor platform equivalence.",
            ),
            "seasonal_compatibility": FactorDetail(
                key="seasonal_compatibility",
                name="Seasonal Compatibility",
                raw_value=round(seas_comp, 3),
                weight=w.weight_seasonal,
                contribution=round((seas_comp * w.weight_seasonal) / total_pos_weight, 3),
                status="optimal" if seas_comp >= 0.70 else "moderate" if seas_comp >= 0.40 else "caution",
                description="Cyclic day-of-year acquisition alignment to mitigate phenological season drift.",
            ),
            "spectral_evidence": FactorDetail(
                key="spectral_evidence",
                name="Spectral Evidence",
                raw_value=round(spec_ev, 3),
                weight=w.weight_spectral,
                contribution=round((spec_ev * w.weight_spectral) / total_pos_weight, 3),
                status="optimal" if spec_ev >= 0.60 else "moderate",
                description="Physical index transitions (NDBI, NDVI, NDWI) confirming permanent material change.",
            ),
            "spatial_consistency": FactorDetail(
                key="spatial_consistency",
                name="Spatial Consistency",
                raw_value=round(spat_cons, 3),
                weight=w.weight_spatial,
                contribution=round((spat_cons * w.weight_spatial) / total_pos_weight, 3),
                status="optimal" if spat_cons >= 0.65 else "moderate",
                description="Connected component cluster compactness vs salt-and-pepper noise.",
            ),
        }

        return ConfidenceEvaluationResult(
            final_confidence=round(final_confidence, 3),
            confidence_tier=confidence_tier,
            is_suppressed=is_suppressed,
            reasons=reasons,
            positive_factors=positive_factors,
            factor_breakdown=factor_breakdown,
            penalties_applied=penalties,
            raw_signals=raw_signals,
            weights_applied=w.model_dump(),
        )

    def evaluate_pair(
        self,
        img1: np.ndarray,
        img2_aligned: np.ndarray,
        diff_mask: np.ndarray,
        visual_score: float,
        semantic_score: float,
        reg_quality: float,
        before_tile: Any,
        after_tile: Any,
        spec_before: Dict[str, float],
        spec_after: Dict[str, float],
        query_match_score: Optional[float] = None,
    ) -> ConfidenceEvaluationResult:
        """
        Evaluate full multi-factor confidence from raw image arrays and tile metadata.
        Extracts all 10 signals and invokes the transparent scoring engine.
        """
        # 1. Visual change score (already computed from normalized RMSE)
        # 2. Semantic change score (RemoteCLIP cosine distance)
        # 3. Image quality score (average quality or fallback)
        q1 = float(getattr(before_tile, "quality_score", None) or 0.80)
        q2 = float(getattr(after_tile, "quality_score", None) or 0.80)
        composite_quality = min(q1, q2)

        # 4. Cloud score (max cloud detected across both observations)
        c1 = float(getattr(before_tile, "cloud_cover_pct", None) or 0.0) / 100.0
        c2 = float(getattr(after_tile, "cloud_cover_pct", None) or 0.0) / 100.0
        cloud_score = max(c1, c2)

        # 5. Shadow score (estimated from low-reflectance non-water pixels)
        shadow_score = 0.0
        try:
            gray = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
            shadow_pixels = np.count_nonzero(gray < 35)
            shadow_score = float(shadow_pixels / gray.size)
        except Exception:
            shadow_score = 0.05

        # 6. Registration confidence (OpenCV ECC cc)
        registration_confidence = float(reg_quality)

        # 7. Sensor compatibility
        sensor_compat = self.compute_sensor_compatibility(
            getattr(before_tile, "sensor", None),
            getattr(before_tile, "resolution_m", None),
            getattr(after_tile, "sensor", None),
            getattr(after_tile, "resolution_m", None),
        )

        # 8. Seasonal compatibility
        seasonal_compat = self.compute_seasonal_compatibility(
            getattr(before_tile, "acquisition_date", None),
            getattr(after_tile, "acquisition_date", None),
        )

        # 9. Spectral evidence & seasonal vegetation detection
        delta_urban = spec_after.get("urban_pct", 0.0) - spec_before.get("urban_pct", 0.0)
        delta_veg = spec_after.get("veg_pct", 0.0) - spec_before.get("veg_pct", 0.0)
        delta_water = spec_after.get("water_pct", 0.0) - spec_before.get("water_pct", 0.0)

        spectral_evidence, is_seasonal_veg = self.compute_spectral_evidence(
            delta_urban=delta_urban,
            delta_veg=delta_veg,
            delta_water=delta_water,
        )

        # 10. Spatial consistency via connected components
        spatial_consistency = self.compute_spatial_consistency(diff_mask)

        # Evaluate and return complete transparent result
        return self.evaluate_signals(
            visual_change=visual_score,
            semantic_change=semantic_score,
            image_quality=composite_quality,
            cloud_score=cloud_score,
            shadow_score=shadow_score,
            registration_confidence=registration_confidence,
            sensor_compatibility=sensor_compat,
            seasonal_compatibility=seasonal_compat,
            spectral_evidence=spectral_evidence,
            spatial_consistency=spatial_consistency,
            is_seasonal_vegetation=is_seasonal_veg,
            query_match_score=query_match_score,
        )


# Singleton instance
confidence_engine = MultiFactorConfidenceEngine()
