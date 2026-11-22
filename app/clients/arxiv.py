"""arXiv API client with rate limiting (max 1 request per 3 seconds),
disk caching, XML parsing, and size-capped PDF extraction.
"""

import asyncio
import io
import logging
import re
import time
from typing import Any, Dict, List, Optional
from pathlib import Path
import httpx
import xmltodict
from pypdf import PdfReader

from app.config import settings
from app.clients.cache import arxiv_cache

logger = logging.getLogger("researchos.arxiv")

_last_request_time = 0.0
_rate_limit_lock = asyncio.Lock()
MIN_REQUEST_INTERVAL = 3.0  # arXiv policy: max 1 request every 3s


def _clean_paper_id(raw_id: str) -> str:
    """Extracts normalized arXiv ID (e.g., '1706.03762' from URL or versioned string)."""
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", raw_id)
    if match:
        return match.group(1)
    # Older format (e.g. cs/0101001)
    match_old = re.search(r"([a-z\-]+/\d{7})", raw_id)
    if match_old:
        return match_old.group(1)
    return raw_id.split("/")[-1].replace(".pdf", "")


def _clean_text(text: str) -> str:
    """Normalizes whitespace in abstracts and titles."""
    return re.sub(r"\s+", " ", text or "").strip()


class ArxivClient:
    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self._client = client

    async def _enforce_rate_limit(self) -> None:
        """Ensures at least 3 seconds between arXiv calls."""
        global _last_request_time
        async with _rate_limit_lock:
            now = time.time()
            elapsed = now - _last_request_time
            if elapsed < MIN_REQUEST_INTERVAL:
                sleep_time = MIN_REQUEST_INTERVAL - elapsed
                logger.debug(f"arXiv rate limit: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
            _last_request_time = time.time()

    async def search(
        self,
        query: str,
        max_results: int = 5,
        start: int = 0,
        sort_by: str = "relevance",
    ) -> List[Dict[str, Any]]:
        """Searches arXiv API with disk caching and rate limiting."""
        cache_key = {
            "query": query,
            "max_results": max_results,
            "start": start,
            "sort_by": sort_by,
        }
        cached = arxiv_cache.get(cache_key)
        if cached is not None and isinstance(cached, list):
            logger.debug(f"arXiv cache hit for query: '{query}'")
            return cached

        # Clean query: replace special characters
        formatted_query = query.strip()
        url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{formatted_query}",
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        await self._enforce_rate_limit()

        try:
            async with (self._client or httpx.AsyncClient(timeout=15.0)) as http_client:
                resp = await http_client.get(url, params=params)
                resp.raise_for_status()
                data = xmltodict.parse(resp.text)
        except Exception as e:
            logger.error(f"Failed to query arXiv for '{query}': {e}")
            return []

        feed = data.get("feed", {})
        entries = feed.get("entry", [])
        if isinstance(entries, dict):
            entries = [entries]

        papers = []
        for entry in entries:
            raw_id = entry.get("id", "")
            paper_id = _clean_paper_id(raw_id)
            title = _clean_text(entry.get("title", ""))
            summary = _clean_text(entry.get("summary", ""))

            # Authors
            authors_data = entry.get("author", [])
            if isinstance(authors_data, dict):
                authors_data = [authors_data]
            authors = [a.get("name", "") for a in authors_data if isinstance(a, dict) and "name" in a]

            # Published date / year
            published = entry.get("published", "")
            year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else 2024

            # PDF link
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            links = entry.get("link", [])
            if isinstance(links, dict):
                links = [links]
            for link in links:
                if link.get("@title") == "pdf" or link.get("@type") == "application/pdf":
                    pdf_url = link.get("@href", pdf_url)

            papers.append({
                "paper_id": paper_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "year": year,
                "published": published,
                "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
                "pdf_url": pdf_url,
            })

        arxiv_cache.set(cache_key, papers)
        return papers

    async def fetch_pdf_text(self, paper_id: str, max_pages: int = 6) -> str:
        """Downloads and extracts text from first max_pages of paper PDF.
        Capped to prevent memory exhaustion on small instances.
        """
        cache_key = {"pdf_paper_id": paper_id, "max_pages": max_pages}
        cached = arxiv_cache.get(cache_key)
        if cached is not None and isinstance(cached, str):
            return cached

        url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        pdf_dir = Path(settings.CACHE_DIR) / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        local_pdf_path = pdf_dir / f"{paper_id}.pdf"

        try:
            if not local_pdf_path.exists():
                await self._enforce_rate_limit()
                async with (self._client or httpx.AsyncClient(timeout=20.0, follow_redirects=True)) as http_client:
                    resp = await http_client.get(url)
                    resp.raise_for_status()
                    local_pdf_path.write_bytes(resp.content)

            # Read PDF text
            reader = PdfReader(str(local_pdf_path))
            pages_text = []
            num_pages = min(len(reader.pages), max_pages)
            for page_idx in range(num_pages):
                text = reader.pages[page_idx].extract_text() or ""
                pages_text.append(text)

            full_text = "\n\n".join(pages_text)
            arxiv_cache.set(cache_key, full_text)
            return full_text
        except Exception as e:
            logger.warning(f"Could not extract PDF text for {paper_id}: {e}")
            return ""
