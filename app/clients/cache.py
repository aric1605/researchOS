"""Disk-based cache for LLM and Embedding API responses."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger("researchos.cache")


class DiskCache:
    def __init__(self, namespace: str = "general"):
        self.cache_dir = Path(settings.CACHE_DIR) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, payload: Any) -> str:
        """Computes a deterministic SHA256 hex string from any JSON-serializable payload."""
        data_str = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def get(self, payload: Any) -> Optional[Any]:
        """Retrieves cached response if present, else None."""
        key = self._compute_key(payload)
        file_path = self.cache_dir / f"{key}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    logger.debug(f"Cache hit: {namespace_prefix(self.cache_dir)}/{key}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache file {file_path}: {e}")
        return None

    def set(self, payload: Any, value: Any) -> None:
        """Saves a JSON-serializable response to disk."""
        key = self._compute_key(payload)
        file_path = self.cache_dir / f"{key}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Error writing cache file {file_path}: {e}")


def namespace_prefix(path: Path) -> str:
    return path.name


llm_cache = DiskCache(namespace="llm")
embedding_cache = DiskCache(namespace="embeddings")
arxiv_cache = DiskCache(namespace="arxiv")
