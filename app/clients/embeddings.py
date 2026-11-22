"""Embedding client abstraction with on-disk caching and deterministic mock fallback."""

import hashlib
import logging
import math
import numpy as np
from typing import List, Optional

from app.config import settings
from app.clients.cache import embedding_cache

logger = logging.getLogger("researchos.embeddings")
EMBEDDING_DIM = 768


def _get_mock_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """Generates a deterministic, normalized vector based on SHA-256 hash of text."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand 32 bytes to dim floats using deterministic pseudo-random sequence
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


async def get_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    force_mock: bool = False,
) -> List[List[float]]:
    """Generates embeddings for a list of texts.
    Uses on-disk cache to avoid unnecessary API calls and conserve quota.
    Falls back to deterministic mock embeddings if GEMINI_API_KEY is not configured or force_mock=True.
    """
    if not texts:
        return []

    target_model = model or settings.EMBEDDING_MODEL
    results: List[Optional[List[float]]] = [None] * len(texts)
    missing_indices: List[int] = []

    # Check cache first
    for idx, text in enumerate(texts):
        cached = embedding_cache.get({"text": text, "model": target_model})
        if cached is not None and isinstance(cached, list):
            results[idx] = cached
        else:
            missing_indices.append(idx)

    if not missing_indices:
        return [r for r in results if r is not None]

    # If force_mock or no API key, generate mock embeddings
    if force_mock or not settings.GEMINI_API_KEY:
        logger.debug(f"Using mock embeddings for {len(missing_indices)} texts.")
        for idx in missing_indices:
            emb = _get_mock_embedding(texts[idx], dim=EMBEDDING_DIM)
            embedding_cache.set({"text": texts[idx], "model": target_model}, emb)
            results[idx] = emb
        return [r for r in results if r is not None]

    # Real call via google-genai SDK
    try:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        for idx in missing_indices:
            text = texts[idx]
            resp = await client.aio.models.embed_content(
                model=target_model,
                contents=text,
            )
            # Response contains embedding.values
            if hasattr(resp, "embedding") and hasattr(resp.embedding, "values"):
                emb = list(resp.embedding.values)
            elif hasattr(resp, "embeddings") and len(resp.embeddings) > 0:
                emb = list(resp.embeddings[0].values)
            else:
                logger.warning("Unexpected embed_content response structure; falling back to mock.")
                emb = _get_mock_embedding(text, dim=EMBEDDING_DIM)

            embedding_cache.set({"text": text, "model": target_model}, emb)
            results[idx] = emb
    except Exception as e:
        logger.error(f"Error calling Gemini Embedding API: {e}. Falling back to mock embeddings.")
        for idx in missing_indices:
            if results[idx] is None:
                emb = _get_mock_embedding(texts[idx], dim=EMBEDDING_DIM)
                results[idx] = emb

    return [r for r in results if r is not None]


async def get_embedding(
    text: str,
    model: Optional[str] = None,
    force_mock: bool = False,
) -> List[float]:
    """Generates embedding for a single text."""
    res = await get_embeddings([text], model=model, force_mock=force_mock)
    return res[0]
