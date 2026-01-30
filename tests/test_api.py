"""Basic API tests for LEXe."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from lexe_api.main import app
    return TestClient(app)


def test_root(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "LEXe API"
    assert "version" in data
    assert "tools" in data


def test_liveness(client):
    """Test liveness probe."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.skip(reason="Requires database connection")
def test_tools_status(client):
    """Test tools status endpoint.

    Note: This test requires database connection.
    Run with Docker services: docker compose -f docker-compose.lexe.yml up -d
    """
    response = client.get("/api/v1/tools/status")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert "normattiva" in data["tools"]
    assert "eurlex" in data["tools"]
    assert "infolex" in data["tools"]
