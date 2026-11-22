"""Unit tests for Phase 4: Claim extraction and verbatim quote verification."""

import pytest
from app.agents.extraction import (
    verify_verbatim_quote,
    ExtractionAgent,
    ExtractedClaims,
    RawClaim,
)
from app.clients.llm import LLMClient


def test_verify_verbatim_quote_success():
    chunk = "The Transformer is the first transduction model relying entirely on self-attention to compute representations."
    quote = "relying entirely on self-attention to compute representations"
    assert verify_verbatim_quote(quote, chunk) is True

    # With quotes and whitespace differences
    quote_quoted = '  "relying entirely on self-attention to compute representations." '
    assert verify_verbatim_quote(quote_quoted, chunk) is True


def test_verify_verbatim_quote_rejections():
    chunk = "The Transformer reduces sequential computation to a constant number of operations."

    # Hallucinated / altered wording
    hallucinated = "The Transformer eliminates all sequential computation completely."
    assert verify_verbatim_quote(hallucinated, chunk) is False

    # Overly long quote (> 45 words)
    long_quote = "word " * 50
    assert verify_verbatim_quote(long_quote, long_quote) is False

    # Empty quote
    assert verify_verbatim_quote("", chunk) is False


@pytest.mark.asyncio
async def test_extraction_agent_drops_hallucinations():
    llm = LLMClient(force_mock=True)
    agent = ExtractionAgent(llm_client=llm)

    chunks = [
        {
            "chunk_id": "1706.03762_chunk_0",
            "paper_id": "1706.03762",
            "text": "Self-attention reduces sequential computation to a constant number of operations.",
        },
        {
            "chunk_id": "1706.03762_chunk_1",
            "paper_id": "1706.03762",
            "text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
        },
    ]

    result = await agent.execute(chunks)
    assert result["total_extracted"] == 2
    assert result["total_verified"] == 2
    assert result["quote_verification_rate"] == 1.0
    assert len(result["claims"]) == 2
    assert result["claims"][0]["verified"] is True


@pytest.mark.asyncio
async def test_extraction_agent_detects_mismatch():
    # Provide chunks where chunk 0 does NOT contain the quote
    agent = ExtractionAgent(llm_client=LLMClient(force_mock=True))
    chunks = [
        {
            "chunk_id": "1706.03762_chunk_0",
            "paper_id": "1706.03762",
            "text": "Completely different text that does not have the quote.",
        },
        {
            "chunk_id": "1706.03762_chunk_1",
            "paper_id": "1706.03762",
            "text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
        },
    ]

    result = await agent.execute(chunks)
    # The first claim should be dropped because chunk 0 text doesn't contain it
    assert result["total_verified"] == 1
    assert len(result["dropped_claims"]) == 1
    assert result["dropped_claims"][0]["claim_id"] == "c1"
    assert result["quote_verification_rate"] == 0.5
