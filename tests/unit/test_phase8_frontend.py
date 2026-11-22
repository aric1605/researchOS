"""Unit tests for Phase 8: Frontend static asset serving and root endpoint."""

from fastapi.testclient import TestClient
from app.main import app


def test_root_serves_html():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "ResearchOS" in response.text
    assert "Agent Timeline" in response.text or "timeline" in response.text


def test_static_assets_serve():
    client = TestClient(app)
    css_res = client.get("/static/styles.css")
    assert css_res.status_code == 200
    assert "ResearchOS Styling System" in css_res.text

    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200
    assert "ResearchOS Frontend Application" in js_res.text
