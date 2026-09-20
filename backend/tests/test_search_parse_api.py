"""
End-to-end API test for Phase 12 Query Understanding HTTP endpoints.
Uses FastAPI TestClient to test POST /api/search/parse-query and GET /api/search/parse-query.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_parse_query_post_construction(client):
    """Test POST /api/search/parse-query with 'new construction near roads after 2023'"""
    response = client.post(
        "/api/search/parse-query",
        json={"query": "new construction near roads after 2023"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "parsed" in data
    parsed = data["parsed"]
    assert parsed["intent"] == "change_search"
    assert parsed["target"] == "construction"
    assert parsed["relationship"] == "near_road"
    assert parsed["start_date"] is not None
    assert "2023" in parsed["start_date"]
    assert "after" not in parsed["canonical_embedding_text"]
    assert "2023" not in parsed["canonical_embedding_text"]


def test_api_parse_query_get_rajasthan(client):
    """Test GET /api/search/parse-query?q=Rajasthan+desert"""
    response = client.get("/api/search/parse-query?q=Rajasthan desert")
    assert response.status_code == 200
    data = response.json()
    parsed = data["parsed"]
    assert parsed["location"] == "Rajasthan"
    assert parsed["concept"] == "desert"
    assert parsed["intent"] == "semantic_search"
    assert parsed["location_bbox"] is not None
    assert len(parsed["location_bbox"]) == 4


def test_api_parse_query_bhadla_solar(client):
    """Test POST /api/search/parse-query with 'solar farms in Bhadla after 2020'"""
    response = client.post(
        "/api/search/parse-query",
        json={"query": "solar farms in Bhadla after 2020"},
    )
    assert response.status_code == 200
    data = response.json()
    parsed = data["parsed"]
    assert parsed["location"] == "Bhadla"
    assert parsed["concept"] == "solar"
    assert parsed["start_date"] is not None
    assert "2020" in parsed["start_date"]


def test_api_parse_query_cloud_free_airports(client):
    """Test POST /api/search/parse-query with 'cloud-free satellite imagery of airports in Noida'"""
    response = client.post(
        "/api/search/parse-query",
        json={"query": "cloud-free satellite imagery of airports in Noida"},
    )
    assert response.status_code == 200
    data = response.json()
    parsed = data["parsed"]
    assert parsed["concept"] == "airport"
    assert parsed["location"] == "Noida"
    assert parsed["max_cloud_cover"] == 5.0


def test_api_parse_query_spelling_correction_rajasthan_dessert(client):
    """Test POST /api/search/parse-query with 'rajasthan dessert' returns Did you mean suggestion"""
    response = client.post(
        "/api/search/parse-query",
        json={"query": "rajasthan dessert"},
    )
    assert response.status_code == 200
    data = response.json()
    parsed = data["parsed"]
    assert parsed["location"] == "Rajasthan"
    assert parsed["concept"] == "desert"
    assert parsed["did_you_mean"] == "Did you mean: Rajasthan desert?"
    assert parsed["corrected_query"] == "rajasthan desert"
    assert len(parsed["corrections"]) == 1
    assert parsed["corrections"][0]["original_word"] == "dessert"
    assert parsed["corrections"][0]["corrected_word"] == "desert"
