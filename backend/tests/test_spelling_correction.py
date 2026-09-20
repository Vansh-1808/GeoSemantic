"""
Unit & integration tests for Phase 13: Spelling Correction & Query Robustness.
Validates:
  1. Required evaluator typos:
     - dessert -> desert
     - constrution -> construction
     - buildng -> building
     - developement -> development
     - vegitaiton -> vegetation
  2. General satellite domain typos:
     - airpot -> airport, resorvoir -> reservoir, satelite -> satellite, highwy -> highway
  3. Comparative analysis:
     - 'Rajasthan desert' vs 'rajasthan dessert' produce identical interpretation and search results.
  4. 'Did you mean: ...' suggestions.
"""
import pytest
from app.services.spelling_correction import (
    SpellingCorrectionEngine,
    damerau_levenshtein_distance,
    spelling_correction_service,
)
from app.services.query_understanding import query_understanding_service


# ── 1. Algorithm Unit Tests ───────────────────────────────────────────────────

def test_damerau_levenshtein_transposition():
    """Verify Damerau-Levenshtein handles transpositions, insertions, deletions."""
    # Transposition: 'ei' -> 'ie'
    assert damerau_levenshtein_distance("vegitaiton", "vegetation") == 2
    # Deletion: missing 'c'
    assert damerau_levenshtein_distance("constrution", "construction") == 1
    # Insertion: extra 'e'
    assert damerau_levenshtein_distance("developement", "development") == 1
    # Deletion: missing 'i'
    assert damerau_levenshtein_distance("buildng", "building") == 1


# ── 2. Five Required Evaluator Spelling Errors ────────────────────────────────

def test_evaluator_error_1_dessert_to_desert():
    """Verify homophone/typo 'dessert' is corrected to 'desert'."""
    res = spelling_correction_service.correct_query("rajasthan dessert")
    assert res.is_corrected is True
    assert res.corrected_query == "rajasthan desert"
    assert res.did_you_mean == "Did you mean: Rajasthan desert?"
    assert len(res.corrections) == 1
    assert res.corrections[0].original_word == "dessert"
    assert res.corrections[0].corrected_word == "desert"


def test_evaluator_error_2_constrution_to_construction():
    """Verify missing 'c' in 'constrution' is corrected to 'construction'."""
    res = spelling_correction_service.correct_query("new constrution near roads")
    assert res.is_corrected is True
    assert "construction" in res.corrected_query
    assert any(c.corrected_word == "construction" for c in res.corrections)


def test_evaluator_error_3_buildng_to_building():
    """Verify missing 'i' in 'buildng' is corrected to 'building'."""
    res = spelling_correction_service.correct_query("urban buildng expansion")
    assert res.is_corrected is True
    assert "building" in res.corrected_query
    assert any(c.corrected_word == "building" for c in res.corrections)


def test_evaluator_error_4_developement_to_development():
    """Verify extra 'e' in 'developement' is corrected to 'development'."""
    res = spelling_correction_service.correct_query("road developement in Jewar")
    assert res.is_corrected is True
    assert "development" in res.corrected_query
    assert any(c.corrected_word == "development" for c in res.corrections)


def test_evaluator_error_5_vegitaiton_to_vegetation():
    """Verify transposition/typo in 'vegitaiton' is corrected to 'vegetation'."""
    res = spelling_correction_service.correct_query("vegitaiton clearance near water")
    assert res.is_corrected is True
    assert "vegetation" in res.corrected_query
    assert any(c.corrected_word == "vegetation" for c in res.corrections)


# ── 3. General Satellite Domain Typos ─────────────────────────────────────────

def test_general_domain_typos():
    """Verify general domain typos across vocabulary without specific hardcoding."""
    cases = [
        ("cloud-free satelite imagery", "satellite"),
        ("airpot runway and terminals", "airport"),
        ("water resorvoir decrease", "reservoir"),
        ("highwy expansion", "highway"),
        ("dense farmlnd fields", "farmland"),
    ]
    for prompt, expected_word in cases:
        res = spelling_correction_service.correct_query(prompt)
        assert res.is_corrected is True
        assert expected_word in res.corrected_query


# ── 4. End-to-End Comparative Test: 'Rajasthan desert' vs 'rajasthan dessert' ─

def test_comparative_rajasthan_desert_vs_rajasthan_dessert():
    """
    CRITICAL EVALUATOR REQUIREMENT:
    Compare:
      1. 'Rajasthan desert'
      2. 'rajasthan dessert'
    Both should produce substantially similar search interpretations/results.
    """
    correct_parsed = query_understanding_service.parse("Rajasthan desert")
    typo_parsed = query_understanding_service.parse("rajasthan dessert")

    # 1. Intent must match
    assert correct_parsed.intent == "semantic_search"
    assert typo_parsed.intent == "semantic_search"

    # 2. Location must match
    assert correct_parsed.location == "Rajasthan"
    assert typo_parsed.location == "Rajasthan"

    # 3. Location Bounding Box must match exactly
    assert correct_parsed.location_bbox == typo_parsed.location_bbox
    assert correct_parsed.location_bbox is not None

    # 4. Extracted Concept must match
    assert correct_parsed.concept == "desert"
    assert typo_parsed.concept == "desert"

    # 5. Canonical embedding text passed to RemoteCLIP must match
    assert correct_parsed.canonical_embedding_text == typo_parsed.canonical_embedding_text
    assert typo_parsed.canonical_embedding_text == "desert"

    # 6. Normalized query must match
    assert correct_parsed.normalized_query == typo_parsed.normalized_query

    # 7. Typo query should include the 'Did you mean' suggestion
    assert typo_parsed.did_you_mean == "Did you mean: Rajasthan desert?"
    assert typo_parsed.corrected_query == "rajasthan desert"
    assert len(typo_parsed.corrections) == 1
    assert typo_parsed.corrections[0].original_word == "dessert"
    assert typo_parsed.corrections[0].corrected_word == "desert"

    # 8. Correct query should have no spelling corrections needed
    assert correct_parsed.did_you_mean is None
    assert correct_parsed.corrected_query is None
    assert len(correct_parsed.corrections) == 0


def test_no_false_corrections_on_valid_queries():
    """Verify valid queries do not trigger false spelling corrections."""
    queries = [
        "new construction near roads after 2023",
        "Rajasthan desert",
        "solar farms in Bhadla after 2020",
        "cloud-free satellite imagery of airports in Noida",
        "vegetation clearance near water bodies between 2018 and 2022",
    ]
    for q in queries:
        res = spelling_correction_service.correct_query(q)
        assert res.is_corrected is False
        assert len(res.corrections) == 0
        assert res.did_you_mean is None
