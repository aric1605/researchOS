"""Unit tests for Phase 2: LLM client, caching, budget tracking, JSON repair, and embeddings."""

import pytest
from pydantic import BaseModel
from typing import List

from app.config import settings
from app.clients.cache import DiskCache
from app.clients.llm import CallBudgetTracker, BudgetExceededError, LLMClient, _extract_json_block
from app.clients.embeddings import get_embedding, get_embeddings, EMBEDDING_DIM


class SampleItem(BaseModel):
    name: str
    score: float
    tags: List[str]


def test_budget_tracker_enforcement():
    tracker = CallBudgetTracker(max_calls=2)
    assert tracker.call_count == 0

    tracker.record_call(is_cached=False)
    assert tracker.call_count == 1

    tracker.record_call(is_cached=True)
    assert tracker.call_count == 1
    assert tracker.cached_count == 1

    tracker.record_call(is_cached=False)
    assert tracker.call_count == 2

    # Exceeding budget raises BudgetExceededError
    with pytest.raises(BudgetExceededError):
        tracker.record_call(is_cached=False)


def test_disk_cache(tmp_path):
    cache = DiskCache(namespace="test_cache", cache_dir=tmp_path)
    key_payload = {"prompt": "hello world", "model": "test-model"}
    val = {"response": "Hi there!"}

    # Miss
    assert cache.get(key_payload) is None

    # Set and hit
    cache.set(key_payload, val)
    retrieved = cache.get(key_payload)
    assert retrieved == val


def test_extract_json_block():
    markdown_json = 'Here is the JSON:\n```json\n{"name": "test", "score": 1.0}\n```\nEnjoy!'
    extracted = _extract_json_block(markdown_json)
    assert extracted == '{"name": "test", "score": 1.0}'

    raw_json = '  {"name": "raw", "score": 2.0}  '
    assert _extract_json_block(raw_json) == '{"name": "raw", "score": 2.0}'


@pytest.mark.asyncio
async def test_mock_embedding_generation():
    text1 = "Transformers in natural language processing"
    text2 = "Attention mechanisms for sequence modeling"

    emb1 = await get_embedding(text1, force_mock=True)
    emb2 = await get_embedding(text2, force_mock=True)

    assert len(emb1) == EMBEDDING_DIM
    assert len(emb2) == EMBEDDING_DIM
    assert emb1 != emb2  # Deterministically different for different texts

    # Determinism: same text yields identical embedding
    emb1_repeat = await get_embedding(text1, force_mock=True)
    assert emb1 == emb1_repeat


@pytest.mark.asyncio
async def test_llm_client_mock_structured():
    client = LLMClient(force_mock=True)
    result = await client.generate_structured(
        prompt="List key architecture elements",
        schema=SampleItem,
    )
    assert isinstance(result, SampleItem)
    assert isinstance(result.name, str)
    assert isinstance(result.score, float)
    assert isinstance(result.tags, list)


@pytest.mark.asyncio
async def test_llm_json_repair_retry(monkeypatch):
    """Verifies that if the initial response is malformed, LLMClient executes exactly 1 repair retry."""
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test_key")
    client = LLMClient(force_mock=False)
    # Monkeypatch execute_with_retry to fail once with bad json, then return valid json
    call_count = 0

    async def mock_execute(prompt, system_instruction, temperature=0.2):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "This is malformed, not valid JSON at all!"
        return '{"name": "Repaired Object", "score": 9.9, "tags": ["repaired"]}'

    monkeypatch.setattr(client, "_execute_with_retry", mock_execute)

    import uuid
    result = await client.generate_structured(
        prompt=f"Test repair {uuid.uuid4()}",
        schema=SampleItem,
    )
    assert call_count == 2  # exactly 1 initial + 1 repair retry
    assert result.name == "Repaired Object"
    assert result.score == 9.9
