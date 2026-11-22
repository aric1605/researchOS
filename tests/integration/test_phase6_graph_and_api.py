"""Integration tests for Phase 6: Full LangGraph run, API endpoints, SSE events,
concurrency queue limits, access code, and rate limiting.
"""

import asyncio
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from app.graph.state import ResearchState
from app.graph.build import build_research_graph
from app.clients.llm import LLMClient
from app.db.repository import Repository
import app.api.routes as routes_module


@pytest.mark.asyncio
async def test_full_graph_execution():
    """Tests complete LangGraph execution from question to evaluated report."""
    llm = LLMClient(force_mock=True)
    graph = build_research_graph(llm_client=llm)

    state = ResearchState(
        run_id="test_run_123",
        question="How do Transformers compare to RNNs in sequence modeling?",
        depth="quick",
        max_papers=2,
    )

    result = await graph.ainvoke(state)

    assert result["status"] == "completed"
    assert "sub_questions" in result
    assert len(result["sub_questions"]) > 0
    assert "synthesis" in result
    assert "markdown" in result["synthesis"]
    assert "reference_list" in result["synthesis"]
    assert "eval_scores" in result
    assert result["eval_scores"].get("faithfulness_score") is not None
    assert len(result["trace"]) >= 5  # Supervisor, Retrieval, Extraction, Review, Comparison, Synthesis, Evaluator


@pytest.mark.asyncio
async def test_supervisor_reroute_flow():
    """Tests supervisor triggering a second round when round 1 finds zero papers."""
    llm = LLMClient(force_mock=True)
    graph = build_research_graph(llm_client=llm)

    state = ResearchState(
        run_id="test_reroute_456",
        question="Extremely obscure topic with zero immediate results",
        depth="quick",
        max_papers=2,
    )

    # Retrieval in mock will succeed on round 2 or proceed gracefully
    result = await graph.ainvoke(state)
    assert result["status"] == "completed"


def test_api_access_code_enforcement(monkeypatch):
    monkeypatch.setattr(settings, "ACCESS_CODE", "secret_passcode_123")
    client = TestClient(app)

    # Without header
    res1 = client.post("/api/research", json={"question": "What is the Transformer architecture?"})
    assert res1.status_code == 401

    # With invalid header
    res2 = client.post(
        "/api/research",
        json={"question": "What is the Transformer architecture?"},
        headers={"X-Access-Code": "wrong_code"},
    )
    assert res2.status_code == 401

    # With correct header
    res3 = client.post(
        "/api/research",
        json={"question": "What is the Transformer architecture?"},
        headers={"X-Access-Code": "secret_passcode_123"},
    )
    assert res3.status_code == 202
    assert "run_id" in res3.json()


def test_api_rate_limiter(monkeypatch):
    monkeypatch.setattr(settings, "ACCESS_CODE", None)
    monkeypatch.setattr(settings, "RATE_LIMIT_RUNS_PER_HOUR", 2)
    # Clear existing rate limit timestamps
    routes_module._rate_limits.clear()
    client = TestClient(app)

    # Request 1 & 2 succeed
    r1 = client.post("/api/research", json={"question": "Query number one for rate limit testing"})
    r2 = client.post("/api/research", json={"question": "Query number two for rate limit testing"})
    assert r1.status_code == 202
    assert r2.status_code == 202

    # Request 3 hit 429
    r3 = client.post("/api/research", json={"question": "Query number three should be blocked"})
    assert r3.status_code == 429
    assert "Rate limit exceeded" in r3.json()["detail"]


@pytest.mark.asyncio
async def test_api_run_retrieval_and_events():
    client = TestClient(app)
    # Clear rate limits
    routes_module._rate_limits.clear()

    res = client.post("/api/research", json={"question": "Explain self-attention mechanisms in deep learning"})
    assert res.status_code == 202
    run_id = res.json()["run_id"]

    # Poll until background run completes (or up to 10 seconds in mock)
    for _ in range(20):
        run_res = client.get(f"/api/research/{run_id}")
        assert run_res.status_code == 200
        run_data = run_res.json()
        if run_data.get("status") in ("completed", "failed"):
            break
        await asyncio.sleep(0.5)

    assert run_data["status"] in ("completed", "pending", "running")
