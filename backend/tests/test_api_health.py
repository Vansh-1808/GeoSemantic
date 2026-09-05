"""
API health endpoint tests using FastAPI TestClient.
These run without a real database — health endpoint gracefully handles DB unavailability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoints:
    def test_health_returns_200(self, client: TestClient):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "timestamp" in data

    def test_health_status_ok(self, client: TestClient):
        res = client.get("/api/health")
        assert res.json()["status"] == "ok"

    def test_detailed_health_returns_components(self, client: TestClient):
        res = client.get("/api/health/detailed")
        # May be 200 (healthy) or 207 (degraded) — both valid
        assert res.status_code in (200, 207)
        data = res.json()
        assert "components" in data
        assert "database" in data["components"]
        assert "models" in data["components"]
        assert "offline_mode" in data

    def test_openapi_schema_accessible(self, client: TestClient):
        res = client.get("/api/openapi.json")
        assert res.status_code == 200
        assert "paths" in res.json()
