"""
Unit and integration tests for Phase 12 Query Understanding Foundation.
Verifies query normalization, singular/plural handling, domain synonym substitution,
intent classification, entity/relationship extraction, and structured query generation
across 10+ natural language queries.
"""
from datetime import datetime, timezone
import pytest

from app.services.query_understanding import (
    QueryNormalizer,
    QueryUnderstandingEngine,
    query_understanding_service,
)


# ── 1. Query Normalization Tests ───────────────────────────────────────────────

def test_normalization_whitespace_punctuation_lowercasing():
    """Verify whitespace collapsing, lowercasing, and punctuation cleanup."""
    raw = "   NEW   Construction...  near   ROADS,  after  2023!  "
    norm = QueryNormalizer.normalize(raw)
    assert norm == "new construction near road after 2023"


def test_normalization_singular_plural_lemmatization():
    """Verify satellite and geographical domain lemmatization."""
    raw = "airports runways terminals buildings roads bridges reservoirs rivers trees"
    norm = QueryNormalizer.normalize(raw)
    assert norm == "airport runway terminal building road bridge reservoir river forest"


def test_normalization_synonym_substitution():
    """Verify domain synonym mapping to canonical terms."""
    # photovoltaic -> solar, aerodrome -> airport, expressway -> road, deforestation -> vegetation_clearance
    raw = "photovoltaic panels near aerodrome and expressways with deforestation"
    norm = QueryNormalizer.normalize(raw)
    assert "solar" in norm
    assert "airport" in norm
    assert "road" in norm
    assert "vegetation_clearance" in norm


# ── 2. The 10+ Required Natural Language Queries ──────────────────────────────

def test_query_1_new_construction_near_roads_after_2023():
    """
    Query 1 (Explicit from User Prompt):
    'new construction near roads after 2023'
    -> intent: change_search, target: construction, relationship: near_road, start_date: 2023
    """
    res = query_understanding_service.parse("new construction near roads after 2023")

    assert res.intent == "change_search"
    assert res.target == "construction"
    assert res.relationship == "near_road"
    assert res.start_date is not None
    assert res.start_date.year == 2023
    assert res.start_date.month == 1
    assert res.start_date.day == 1


def test_query_2_rajasthan_desert():
    """
    Query 2 (Explicit from User Prompt):
    'Rajasthan desert'
    -> location: Rajasthan, concept: desert, intent: semantic_search
    """
    res = query_understanding_service.parse("Rajasthan desert")

    assert res.location == "Rajasthan"
    assert res.concept == "desert"
    assert res.intent == "semantic_search"
    assert res.location_bbox is not None
    assert len(res.location_bbox) == 4  # [west, south, east, north]


def test_query_3_solar_farms_bhadla_after_2020():
    """
    Query 3:
    'solar farms in Bhadla after 2020'
    -> concept: solar, location: Bhadla, start_date: 2020
    """
    res = query_understanding_service.parse("solar farms in Bhadla after 2020")

    assert res.concept == "solar"
    assert res.location == "Bhadla"
    assert res.start_date is not None
    assert res.start_date.year == 2020


def test_query_4_cloud_free_airports_noida():
    """
    Query 4:
    'cloud-free satellite imagery of airports in Noida'
    -> concept: airport, location: Noida, max_cloud_cover: 5.0, intent: semantic_search
    """
    res = query_understanding_service.parse("cloud-free satellite imagery of airports in Noida")

    assert res.intent == "semantic_search"
    assert res.concept == "airport"
    assert res.location == "Noida"
    assert res.max_cloud_cover == 5.0


def test_query_5_vegetation_clearance_near_water_between_dates():
    """
    Query 5:
    'vegetation clearance near water bodies between 2018 and 2022'
    -> intent: change_search, target: vegetation_clearance, relationship: near_water, start_date: 2018, end_date: 2022
    """
    res = query_understanding_service.parse("vegetation clearance near water bodies between 2018 and 2022")

    assert res.intent == "change_search"
    assert res.target == "vegetation_clearance"
    assert res.relationship == "near_water"
    assert res.start_date is not None
    assert res.start_date.year == 2018
    assert res.end_date is not None
    assert res.end_date.year == 2022


def test_query_6_urban_expansion_along_highways():
    """
    Query 6:
    'urban expansion and buildings along highways'
    -> intent: change_search, target: urban_expansion, relationship: near_road
    """
    res = query_understanding_service.parse("urban expansion and buildings along highways")

    assert res.intent == "change_search"
    assert res.target in ["urban_expansion", "construction"]
    assert res.relationship == "near_road"


def test_query_7_water_reservoir_decrease_since_2021():
    """
    Query 7:
    'water reservoir decrease since 2021'
    -> intent: change_search, target: water, start_date: 2021, requires_water: True
    """
    res = query_understanding_service.parse("water reservoir decrease since 2021")

    assert res.intent == "change_search"
    assert res.target in ["water", "reservoir"]
    assert res.start_date is not None
    assert res.start_date.year == 2021
    assert res.requires_water is True


def test_query_8_road_development_jewar():
    """
    Query 8:
    'highways and road development in Jewar'
    -> intent: change_search, target: road_expansion, location: Jewar
    """
    res = query_understanding_service.parse("highways and road development in Jewar")

    assert res.intent == "change_search"
    assert res.target in ["road_expansion", "road"]
    assert res.location == "Jewar"


def test_query_9_dense_agricultural_crop_fields():
    """
    Query 9:
    'dense agricultural crop fields'
    -> intent: semantic_search, concept: farmland, requires_vegetation: True
    """
    res = query_understanding_service.parse("dense agricultural crop fields")

    assert res.intent == "semantic_search"
    assert res.concept == "farmland"
    assert res.requires_vegetation is True


def test_query_10_pangong_lake_water_coastline():
    """
    Query 10:
    'Pangong lake water coastline'
    -> intent: semantic_search, concept: lake/water, location: Pangong, requires_water: True
    """
    res = query_understanding_service.parse("Pangong lake water coastline")

    assert res.intent == "semantic_search"
    assert res.location == "Pangong"
    assert res.concept in ["lake", "water"]
    assert res.requires_water is True


def test_query_11_demolition_residential_before_2020():
    """
    Query 11:
    'demolition of residential buildings before 2020'
    -> intent: change_search, target: demolition, concept: residential, end_date: 2020
    """
    res = query_understanding_service.parse("demolition of residential buildings before 2020")

    assert res.intent == "change_search"
    assert res.target == "demolition"
    assert res.concept == "residential"
    assert res.end_date is not None
    assert res.end_date.year == 2020


def test_query_12_sentinel2_coastal_harbor_no_clouds():
    """
    Query 12:
    'Sentinel-2 imagery of coastal harbor with no clouds'
    -> sensor: Sentinel-2, concept: port/harbor, max_cloud_cover: 5.0, requires_water: True
    """
    res = query_understanding_service.parse("Sentinel-2 imagery of coastal harbor with no clouds")

    assert res.sensor_constraints == ["Sentinel-2"]
    assert res.concept in ["port", "harbor"]
    assert res.max_cloud_cover == 5.0
    assert res.requires_water is True


def test_canonical_embedding_text_quality():
    """
    Verify that canonical_embedding_text strips temporal prepositions and stop words
    to maximize RemoteCLIP vision-language embedding cosine similarity.
    """
    res = query_understanding_service.parse("new construction near roads after 2023")
    # Should not contain temporal word 'after' or year '2023'
    assert "after" not in res.canonical_embedding_text
    assert "2023" not in res.canonical_embedding_text
    assert "construction" in res.canonical_embedding_text
    assert "road" in res.canonical_embedding_text
