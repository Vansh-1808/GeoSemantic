"""
Unit & Integration Tests for Phase 14: Geographic Entity Understanding.
Validates:
  1. Separation of geographic entities (countries, states, cities, regions) from visual concepts.
  2. Four required test queries:
     - 'desert in Rajasthan'
     - 'industrial areas in Tamil Nadu'
     - 'urban development in Chennai'
     - 'water bodies in Kerala'
  3. Gazetteer hierarchical entity resolution (country, state, city, region).
  4. PostGIS geographic boundary filtering and candidate ranking scoping.
"""
import pytest
from app.services.gazetteer import gazetteer_service, GeographicEntity
from app.services.query_understanding import query_understanding_service
from app.schemas.search import SemanticSearchRequest
from app.db.database import AsyncSessionLocal
from app.services.semantic_search import semantic_search_service


# ── 1. The Four Required Test Queries ─────────────────────────────────────────

def test_query_1_desert_in_rajasthan():
    """
    Query 1: 'desert in Rajasthan'
    Must mean: 'Find desert/arid imagery geographically within Rajasthan state boundary.'
    - Location: Rajasthan (type: state)
    - Concept: desert
    - RemoteCLIP text: 'desert' (Rajasthan and 'in' stripped)
    - PostGIS boundary: [69.4, 23.0, 78.3, 30.2]
    """
    res = query_understanding_service.parse("desert in Rajasthan")

    assert res.location == "Rajasthan"
    assert res.location_type == "state"
    assert res.location_bbox is not None
    assert res.location_bbox == [69.4, 23.0, 78.3, 30.2]
    assert res.concept == "desert"
    assert res.intent == "semantic_search"

    # CRITICAL: RemoteCLIP text must NOT have geographic proper noun or preposition
    assert "rajasthan" not in res.canonical_embedding_text.lower()
    assert "in" not in res.canonical_embedding_text.split()
    assert "desert" in res.canonical_embedding_text


def test_query_2_industrial_areas_in_tamil_nadu():
    """
    Query 2: 'industrial areas in Tamil Nadu'
    Must mean: 'Find industrial visual concept geographically within Tamil Nadu state boundary.'
    - Location: Tamil Nadu (type: state)
    - Concept: industrial
    - RemoteCLIP text: 'industrial' or 'industrial area' (Tamil Nadu and 'in' stripped)
    - PostGIS boundary: [76.2, 8.0, 80.4, 13.6]
    """
    res = query_understanding_service.parse("industrial areas in Tamil Nadu")

    assert res.location == "Tamil Nadu"
    assert res.location_type == "state"
    assert res.location_bbox is not None
    assert res.location_bbox == [76.2, 8.0, 80.4, 13.6]
    assert res.concept == "industrial"

    assert "tamil" not in res.canonical_embedding_text.lower()
    assert "nadu" not in res.canonical_embedding_text.lower()
    assert "in" not in res.canonical_embedding_text.split()
    assert "industrial" in res.canonical_embedding_text


def test_query_3_urban_development_in_chennai():
    """
    Query 3: 'urban development in Chennai'
    Must mean: 'Find urban development/expansion within Chennai city boundary.'
    - Location: Chennai (type: city)
    - Target: urban_expansion
    - Concept: urban
    - RemoteCLIP text: 'urban development' (Chennai and 'in' stripped)
    - PostGIS boundary: [80.1, 12.9, 80.38, 13.25]
    """
    res = query_understanding_service.parse("urban development in Chennai")

    assert res.location == "Chennai"
    assert res.location_type == "city"
    assert res.location_bbox is not None
    assert res.location_bbox[0] >= 80.0 and res.location_bbox[2] <= 80.5
    assert res.concept in ["urban", "development"]
    assert res.target in ["urban_expansion", "construction"]

    assert "chennai" not in res.canonical_embedding_text.lower()
    assert "in" not in res.canonical_embedding_text.split()
    assert "urban" in res.canonical_embedding_text


def test_query_4_water_bodies_in_kerala():
    """
    Query 4: 'water bodies in Kerala'
    Must mean: 'Find water/hydrological features within Kerala state boundary.'
    - Location: Kerala (type: state)
    - Concept: water
    - Requires Water: True
    - RemoteCLIP text: 'water' (Kerala and 'in' stripped)
    - PostGIS boundary: [74.8, 8.2, 77.6, 12.8]
    """
    res = query_understanding_service.parse("water bodies in Kerala")

    assert res.location == "Kerala"
    assert res.location_type == "state"
    assert res.location_bbox is not None
    assert res.location_bbox == [74.8, 8.2, 77.6, 12.8]
    assert res.concept == "water"
    assert res.requires_water is True

    assert "kerala" not in res.canonical_embedding_text.lower()
    assert "in" not in res.canonical_embedding_text.split()
    assert "water" in res.canonical_embedding_text


# ── 2. Gazetteer Hierarchical Classification Tests ────────────────────────────

def test_gazetteer_hierarchy_types():
    """Verify gazetteer correctly distinguishes countries, states, cities, and regions."""
    # Country
    ent_country = gazetteer_service.resolve_entity("satellite imagery over India")
    assert ent_country is not None
    assert ent_country.canonical_name == "India"
    assert ent_country.entity_type == "country"

    # State
    ent_state = gazetteer_service.resolve_entity("forest loss across Rajasthan")
    assert ent_state is not None
    assert ent_state.canonical_name == "Rajasthan"
    assert ent_state.entity_type == "state"

    # City
    ent_city = gazetteer_service.resolve_entity("airports in Chennai")
    assert ent_city is not None
    assert ent_city.canonical_name == "Chennai"
    assert ent_city.entity_type == "city"

    # Region
    ent_region = gazetteer_service.resolve_entity("sand dunes in Thar desert")
    assert ent_region is not None
    assert ent_region.canonical_name == "Thar"
    assert ent_region.entity_type == "region"


# ── 3. PostGIS Geographic Filtering in Semantic Search ─────────────────────────

@pytest.mark.asyncio
async def test_postgis_filtering_desert_in_rajasthan():
    """
    Verifies that 'desert in Rajasthan' enforces PostGIS spatial envelope bounding
    and only returns candidate tiles geographically within Rajasthan.
    """
    req = SemanticSearchRequest(
        query="desert in Rajasthan",
        top_k=10,
    )
    async with AsyncSessionLocal() as session:
        response = await semantic_search_service.search(req=req, session=session)

    # PostGIS spatial filter applied
    assert "geographic_entity" in response.filters_applied
    geo_filter = response.filters_applied["geographic_entity"]
    assert geo_filter["name"] == "Rajasthan"
    assert geo_filter["type"] == "state"

    # All returned tiles must geographically lie inside Rajasthan's bounding envelope
    rajasthan_bbox = (69.4, 23.0, 78.3, 30.2)
    for item in response.results:
        center = item.center_coordinates
        assert rajasthan_bbox[0] <= center["lon"] <= rajasthan_bbox[2], (
            f"Tile {item.tile_id} lon {center['lon']} outside Rajasthan west/east"
        )
        assert rajasthan_bbox[1] <= center["lat"] <= rajasthan_bbox[3], (
            f"Tile {item.tile_id} lat {center['lat']} outside Rajasthan south/north"
        )


@pytest.mark.asyncio
async def test_postgis_filtering_urban_development_chennai():
    """
    Verifies that 'urban development in Chennai' enforces PostGIS spatial filtering
    and returns tiles strictly within Chennai metropolitan coordinates.
    """
    req = SemanticSearchRequest(
        query="urban development in Chennai",
        top_k=10,
    )
    async with AsyncSessionLocal() as session:
        response = await semantic_search_service.search(req=req, session=session)

    assert "geographic_entity" in response.filters_applied
    geo_filter = response.filters_applied["geographic_entity"]
    assert geo_filter["name"] == "Chennai"
    assert geo_filter["type"] == "city"

    # All returned tiles must be within Chennai boundary
    for item in response.results:
        center = item.center_coordinates
        assert 80.0 <= center["lon"] <= 80.5
        assert 12.8 <= center["lat"] <= 13.4
