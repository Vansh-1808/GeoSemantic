import pytest
import numpy as np
import cv2
from datetime import datetime, timezone

from app.services.confidence_engine import (
    MultiFactorConfidenceEngine,
    ConfidenceWeights,
    ConfidenceThresholds,
    confidence_engine,
)
from app.models.tile import Tile


def test_scenario_1_real_structural_change():
    """
    Scenario 1: Real structural change.
    New infrastructure development on clear, co-registered same-season imagery.
    Expectation: High Confidence tier, not suppressed, positive evidence generated.
    """
    engine = MultiFactorConfidenceEngine()

    result = engine.evaluate_signals(
        visual_change=0.75,
        semantic_change=0.72,
        image_quality=0.88,
        cloud_score=0.02,
        shadow_score=0.03,
        registration_confidence=0.94,
        sensor_compatibility=1.0,
        seasonal_compatibility=0.92,
        spectral_evidence=0.85,
        spatial_consistency=0.80,
        is_seasonal_vegetation=False,
    )

    assert result.confidence_tier == "High Confidence"
    assert not result.is_suppressed
    assert result.final_confidence >= 0.70
    assert len(result.positive_factors) > 0
    # Must have positive confirmation
    assert any("registration" in p.lower() for p in result.positive_factors)
    assert any("spatially coherent" in p.lower() for p in result.positive_factors)


def test_scenario_2_seasonal_vegetation_variation():
    """
    Scenario 2: Seasonal vegetation variation.
    Monsoon greening or winter drying causes high visual difference,
    but spectral ratio shift is purely phenological with low seasonal calendar compatibility.
    Expectation: Suppressed / Low Confidence, reasons include 'seasonal vegetation variation'.
    """
    engine = MultiFactorConfidenceEngine()

    result = engine.evaluate_signals(
        visual_change=0.65,
        semantic_change=0.45,
        image_quality=0.85,
        cloud_score=0.03,
        shadow_score=0.04,
        registration_confidence=0.90,
        sensor_compatibility=1.0,
        seasonal_compatibility=0.20,  # 6 months apart
        spectral_evidence=0.20,        # No impervious built-up shift
        spatial_consistency=0.50,
        is_seasonal_vegetation=True,
    )

    assert result.is_suppressed
    assert result.confidence_tier in ["Suppressed", "Low Confidence"]
    assert any("seasonal vegetation" in r.lower() for r in result.reasons)


def test_scenario_3_cloud_artifact():
    """
    Scenario 3: Cloud artifact.
    High radiometric change triggered by bright cloud or haze cover.
    Expectation: Suppressed, reasons explicitly cite 'high cloud coverage'.
    """
    engine = MultiFactorConfidenceEngine()

    result = engine.evaluate_signals(
        visual_change=0.82,  # Misleading high visual difference
        semantic_change=0.60,
        image_quality=0.45,
        cloud_score=0.38,   # Exceeds cloud suppression limit (0.25)
        shadow_score=0.10,
        registration_confidence=0.88,
        sensor_compatibility=1.0,
        seasonal_compatibility=0.80,
        spectral_evidence=0.40,
        spatial_consistency=0.50,
        is_seasonal_vegetation=False,
    )

    assert result.is_suppressed
    assert result.confidence_tier == "Suppressed"
    assert any("cloud" in r.lower() for r in result.reasons)
    assert "cloud_penalty" in result.penalties_applied


def test_scenario_4_misalignment_artifact():
    """
    Scenario 4: Misalignment artifact.
    High pixel-difference caused by shifted building edges from poor registration.
    Expectation: Suppressed, reasons explicitly cite 'poor image alignment'.
    """
    engine = MultiFactorConfidenceEngine()

    result = engine.evaluate_signals(
        visual_change=0.78,  # High edge differences
        semantic_change=0.50,
        image_quality=0.80,
        cloud_score=0.02,
        shadow_score=0.04,
        registration_confidence=0.25,  # Severe misregistration (< 0.40 limit)
        sensor_compatibility=1.0,
        seasonal_compatibility=0.85,
        spectral_evidence=0.45,
        spatial_consistency=0.20,      # Edges are scattered and fragmented
        is_seasonal_vegetation=False,
    )

    assert result.is_suppressed
    assert result.confidence_tier == "Suppressed"
    assert any("alignment" in r.lower() or "registration" in r.lower() for r in result.reasons)
    assert "misalignment_penalty" in result.penalties_applied


def test_scenario_5_water_variation():
    """
    Scenario 5: Water surface variation.
    Distinguishes hydrological reservoir level shift from construction false alarms.
    """
    engine = MultiFactorConfidenceEngine()

    # Hydrological shift: delta water = +8%, delta urban = 0%
    spec_ev, is_seasonal_veg = engine.compute_spectral_evidence(
        delta_urban=0.2,
        delta_veg=-1.0,
        delta_water=8.4,
    )

    assert spec_ev >= 0.70  # High spectral evidence for water shift
    assert not is_seasonal_veg  # Not vegetation shift

    result = engine.evaluate_signals(
        visual_change=0.68,
        semantic_change=0.65,
        image_quality=0.85,
        cloud_score=0.02,
        shadow_score=0.04,
        registration_confidence=0.92,
        sensor_compatibility=1.0,
        seasonal_compatibility=0.75,
        spectral_evidence=spec_ev,
        spatial_consistency=0.70,
        is_seasonal_vegetation=False,
        change_type="water_change",
    )

    assert result.final_confidence >= 0.50
    assert result.confidence_tier in ["High Confidence", "Medium Confidence"]
    assert not result.is_suppressed


def test_configurable_weights_transparency():
    """
    Verify weights are dynamically configurable and the engine is NOT a black box:
    Adjusting weights shifts the final score deterministically.
    """
    # Base configuration
    custom_w1 = ConfidenceWeights(weight_visual=0.50, weight_semantic=0.10)
    engine1 = MultiFactorConfidenceEngine(weights=custom_w1)

    # Alternate configuration prioritizing semantic over visual
    custom_w2 = ConfidenceWeights(weight_visual=0.10, weight_semantic=0.50)
    engine2 = MultiFactorConfidenceEngine(weights=custom_w2)

    # Input where visual change is high (0.9) but semantic change is low (0.2)
    signals = dict(
        visual_change=0.90,
        semantic_change=0.20,
        image_quality=0.90,
        cloud_score=0.0,
        shadow_score=0.0,
        registration_confidence=0.95,
        sensor_compatibility=1.0,
        seasonal_compatibility=1.0,
        spectral_evidence=0.60,
        spatial_consistency=0.70,
    )

    res1 = engine1.evaluate_signals(**signals)
    res2 = engine2.evaluate_signals(**signals)

    # Engine 1 weights visual much higher, so its confidence must be significantly higher than Engine 2
    assert res1.final_confidence > res2.final_confidence
    assert res1.weights_applied["weight_visual"] == 0.50
    assert res2.weights_applied["weight_visual"] == 0.10


def test_ten_signals_completeness():
    """
    Verify that all 10 input signals are present in factor_breakdown with honest metrics.
    """
    engine = MultiFactorConfidenceEngine()
    result = engine.evaluate_signals(
        visual_change=0.5,
        semantic_change=0.5,
        image_quality=0.8,
        cloud_score=0.1,
        shadow_score=0.05,
        registration_confidence=0.85,
        sensor_compatibility=0.9,
        seasonal_compatibility=0.75,
        spectral_evidence=0.6,
        spatial_consistency=0.7,
    )

    expected_signals = [
        "visual_change_score",
        "semantic_change_score",
        "image_quality_score",
        "cloud_score",
        "shadow_score",
        "registration_confidence",
        "sensor_compatibility",
        "seasonal_compatibility",
        "spectral_evidence",
        "spatial_consistency",
    ]

    for sig in expected_signals:
        assert sig in result.factor_breakdown
        factor = result.factor_breakdown[sig]
        assert 0.0 <= factor.raw_value <= 1.0
        assert factor.weight >= 0.0
        assert len(factor.description) > 0


def test_spatial_consistency_connected_components():
    """
    Verify connected component spatial analysis:
    Compact clustered mask vs scattered noise.
    """
    engine = MultiFactorConfidenceEngine()

    # 1. Compact block of 50x50 pixels in 256x256 image (Real structural change)
    mask_cluster = np.zeros((256, 256), dtype=np.uint8)
    mask_cluster[100:150, 100:150] = 255
    score_cluster = engine.compute_spatial_consistency(mask_cluster)

    # 2. Random salt-and-pepper noise of equivalent pixel count
    mask_noise = np.zeros((256, 256), dtype=np.uint8)
    np.random.seed(42)
    noise_coords = np.random.choice(256 * 256, 2500, replace=False)
    mask_noise.ravel()[noise_coords] = 255
    score_noise = engine.compute_spatial_consistency(mask_noise)

    # Compact block must have significantly higher spatial consistency than noise
    assert score_cluster > score_noise
    assert score_cluster > 0.60


def test_seasonal_compatibility_cyclic_dates():
    """
    Verify cyclic day-of-year seasonal compatibility.
    """
    engine = MultiFactorConfidenceEngine()

    # Same date across different years (Anniversary observation)
    d1 = datetime(2021, 8, 15, tzinfo=timezone.utc)
    d2 = datetime(2024, 8, 17, tzinfo=timezone.utc)
    same_season_score = engine.compute_seasonal_compatibility(d1, d2)

    # Opposite seasons: August (summer/monsoon) vs February (dry winter)
    d3 = datetime(2022, 2, 15, tzinfo=timezone.utc)
    opposite_season_score = engine.compute_seasonal_compatibility(d1, d3)

    assert same_season_score > 0.90
    assert opposite_season_score < 0.35


def test_sensor_compatibility_gsd_ratios():
    """
    Verify GSD resolution ratio and sensor platform matching.
    """
    engine = MultiFactorConfidenceEngine()

    # Identical sensors (Sentinel-2 10m vs Sentinel-2 10m)
    identical = engine.compute_sensor_compatibility("Sentinel-2", 10.0, "Sentinel-2", 10.0)

    # Cross-sensor with different GSD (Sentinel-2 10m vs Landsat-8 30m)
    mismatch = engine.compute_sensor_compatibility("Sentinel-2", 10.0, "Landsat-8", 30.0)

    assert identical == 1.0
    assert mismatch < identical
    assert mismatch > 0.40  # Both optical multispectral, reasonable compatibility
