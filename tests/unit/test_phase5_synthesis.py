"""Unit tests for Phase 5: Comparison and synthesis agents with citation enforcement."""

import pytest
from app.agents.synthesis import (
    extract_sentences,
    check_and_enforce_citations,
    SynthesisAgent,
)
from app.agents.comparison import ComparisonAgent
from app.clients.llm import LLMClient


def test_sentence_extraction_and_citation_check():
    text = (
        "# Title Header\n\n"
        "Transformers rely entirely on self-attention mechanisms [1].\n"
        "Multi-head attention projects queries and keys into multiple subspaces [1].\n"
        "This sentence has no citation and makes an unsupported assertion.\n\n"
        "## References\n"
        "- [1] Vaswani et al. Attention Is All You Need."
    )

    sentences = extract_sentences(text)
    assert len(sentences) == 3

    check = check_and_enforce_citations(text, valid_citation_indices={1})
    assert check["total_sentences"] == 3
    assert check["cited_sentences"] == 2
    assert len(check["uncited_sentences"]) == 1
    assert "This sentence has no citation" in check["uncited_sentences"][0]
    assert round(check["citation_coverage"], 2) == 0.67


def test_invalid_citation_detection():
    text = "Recent models scale context windows efficiently [99]."
    check = check_and_enforce_citations(text, valid_citation_indices={1, 2})
    assert 99 in check["invalid_citations"]
    assert check["cited_sentences"] == 0
    assert check["citation_coverage"] == 0.0


@pytest.mark.asyncio
async def test_comparison_agent_mock():
    agent = ComparisonAgent(llm_client=LLMClient(force_mock=True))
    claims = [
        {"claim_id": "c1", "claim": "Self-attention reduces computation", "paper_id": "p1", "stance": "supports"},
        {"claim_id": "c2", "claim": "Attention has quadratic memory", "paper_id": "p1", "stance": "supports"},
    ]
    res = await agent.execute("What is self-attention?", claims, papers=[])
    assert "themes" in res
    assert len(res["themes"]) > 0
    assert "claim_ids" in res["themes"][0]
    assert "limitations_found" in res


@pytest.mark.asyncio
async def test_synthesis_agent_mock():
    agent = SynthesisAgent(llm_client=LLMClient(force_mock=True))
    papers = [
        {
            "paper_id": "1706.03762",
            "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."],
            "year": 2017,
            "arxiv_url": "https://arxiv.org/abs/1706.03762",
        }
    ]
    claims = [
        {"claim_id": "c1", "claim": "Self attention", "paper_id": "1706.03762", "quote": "Self-attention reduces"}
    ]
    res = await agent.execute("Explain transformers", papers, claims, comparison={})
    assert "title" in res
    assert "markdown" in res
    assert "reference_list" in res
    assert len(res["reference_list"]) == 1
    assert res["citation_coverage"] > 0.0
