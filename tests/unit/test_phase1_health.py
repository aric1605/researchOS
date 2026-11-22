"""Unit test for Phase 1: Health check endpoint and config."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings


def test_config_base_url_strips_slash():
    assert not settings.BASE_URL.endswith("/")


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "services" in data
    assert "postgres" in data["services"]
    assert "qdrant" in data["services"]
