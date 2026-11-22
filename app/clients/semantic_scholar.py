"""Semantic Scholar Graph API client (optional enrichment for citation metrics).
Works without an API key with safe timeouts and caching.
"""

import logging
from typing import Dict, Any, Optional
import httpx

from app.clients.cache import DiskCache

logger = logging.getLogger("researchos.s2")
s2_cache = DiskCache(namespace="semantic_scholar")


class SemanticScholarClient:
    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self._client = client

    async def get_paper_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetches paper citationCount and influentialCitationCount from Semantic Scholar."""
        cache_key = {"s2_arxiv_id": arxiv_id}
        cached = s2_cache.get(cache_key)
        if cached is not None and isinstance(cached, dict):
            return cached

        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
        params = {"fields": "citationCount,influentialCitationCount,year,title"}

        try:
            async with (self._client or httpx.AsyncClient(timeout=8.0)) as http_client:
                resp = await http_client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    res = {
                        "citation_count": data.get("citationCount", 0),
                        "influential_citation_count": data.get("influentialCitationCount", 0),
                    }
                    s2_cache.set(cache_key, res)
                    return res
        except Exception as e:
            logger.debug(f"Semantic Scholar lookup skipped for {arxiv_id}: {e}")

        default_res = {"citation_count": 0, "influential_citation_count": 0}
        return default_res
