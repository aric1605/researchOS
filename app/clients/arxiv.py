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
MIN_REQUEST_INTERVAL = 4.0  # arXiv policy: max 1 request every 3s (4s for safety)


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


def _sanitize_arxiv_query(query: str) -> str:
    """Sanitizes raw query into 2-3 clean keywords to prevent arXiv API timeouts and 429s."""
    stopwords = {
        "what", "is", "the", "foundational", "architecture", "of", "and", "how",
        "does", "replace", "in", "on", "for", "with", "by", "at", "to", "from",
        "about", "why", "when", "where", "which", "whose", "whom", "you", "need",
        "all", "are", "an", "as", "be", "this", "that", "these", "those"
    }
    words = [w for w in re.findall(r"[a-zA-Z0-9\-]+", query) if len(w) > 1]
    filtered = list(dict.fromkeys([w for w in words if w.lower() not in stopwords]))
    if not filtered:
        filtered = words[:3]
    return " ".join(filtered[:3])


ARXIV_HEADERS = {
    "User-Agent": "ResearchOS/1.0 (scientific-research-agent; team@researchos.internal)"
}


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
        """Searches arXiv API with disk caching, rate limiting, and 429 retry backoff."""
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

        formatted_query = _sanitize_arxiv_query(query)
        url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{formatted_query}",
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        data = None
        for attempt in range(3):
            await self._enforce_rate_limit()
            try:
                async with (self._client or httpx.AsyncClient(timeout=45.0, headers=ARXIV_HEADERS)) as http_client:
                    resp = await http_client.get(url, params=params)
                    if resp.status_code == 429:
                        wait_time = 10.0 * (attempt + 1)
                        logger.warning(f"arXiv 429 encountered on '{formatted_query}' (attempt {attempt+1}/3). Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    resp.raise_for_status()
                    data = xmltodict.parse(resp.text)
                    break
            except Exception as e:
                logger.warning(f"arXiv query attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    return []
                await asyncio.sleep(4.0)

        if not data:
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

    async def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Fetches paper metadata for a specific arXiv ID with caching."""
        cache_key = {"paper_id_fetch": paper_id}
        cached = arxiv_cache.get(cache_key)
        if cached is not None and isinstance(cached, dict):
            return cached

        clean_id = _clean_paper_id(paper_id)
        url = "https://export.arxiv.org/api/query"
        params = {"id_list": clean_id}

        await self._enforce_rate_limit()
        try:
            async with (self._client or httpx.AsyncClient(timeout=45.0, headers=ARXIV_HEADERS)) as http_client:
                resp = await http_client.get(url, params=params)
                resp.raise_for_status()
                data = xmltodict.parse(resp.text)
                feed = data.get("feed", {})
                entry = feed.get("entry")
                if not entry:
                    return None
                if isinstance(entry, list):
                    entry = entry[0]

                authors_data = entry.get("author", [])
                if isinstance(authors_data, dict):
                    authors_data = [authors_data]
                authors = [a.get("name", "") for a in authors_data if isinstance(a, dict) and "name" in a]
                published = entry.get("published", "")
                year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else 2024

                p = {
                    "paper_id": clean_id,
                    "title": _clean_text(entry.get("title", "")),
                    "summary": _clean_text(entry.get("summary", "")),
                    "authors": authors,
                    "year": year,
                    "published": published,
                    "arxiv_url": f"https://arxiv.org/abs/{clean_id}",
                    "pdf_url": f"https://arxiv.org/pdf/{clean_id}.pdf",
                }
                arxiv_cache.set(cache_key, p)
                return p
        except Exception as e:
            logger.warning(f"Failed to fetch paper by id {paper_id}: {e}")
            return None

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
                async with (self._client or httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=ARXIV_HEADERS)) as http_client:
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
