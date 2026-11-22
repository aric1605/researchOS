"""Qdrant client abstraction for vector storage, upsert and similarity search."""

import time
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings

logger = logging.getLogger("researchos.vectorstore")

_qdrant_client: Optional[QdrantClient] = None
EMBEDDING_DIM = 768  # Standard for text-embedding-004


def get_qdrant_client() -> QdrantClient:
    """Returns singleton QdrantClient or initializes one.
    Falls back to in-memory instance if QDRANT_URL is not set.
    """
    global _qdrant_client
    if _qdrant_client is None:
        if settings.QDRANT_URL:
            logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")
            _qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=10.0,
            )
        else:
            logger.warning("QDRANT_URL not set; using in-memory Qdrant instance for development/testing.")
            _qdrant_client = QdrantClient(location=":memory:")
    return _qdrant_client


def ensure_collection(collection_name: Optional[str] = None, dim: int = EMBEDDING_DIM) -> None:
    """Ensures the specified Qdrant collection exists with Cosine distance."""
    client = get_qdrant_client()
    coll = collection_name or settings.QDRANT_COLLECTION
    try:
        collections = client.get_collections().collections
        exists = any(c.name == coll for c in collections)
        if not exists:
            logger.info(f"Creating collection '{coll}' with dimension {dim}")
            client.create_collection(
                collection_name=coll,
                vectors_config=qmodels.VectorParams(
                    size=dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
    except Exception as e:
        logger.error(f"Error ensuring collection '{coll}': {e}")
        raise


def upsert_chunks(
    chunks: List[Dict[str, Any]],
    vectors: List[List[float]],
    collection_name: Optional[str] = None,
) -> None:
    """Upserts chunk payloads and vectors to Qdrant.
    Each chunk in chunks must have:
      chunk_id (str), paper_id (str), text (str), section (str), chunk_index (int), etc.
    """
    if not chunks or not vectors:
        return

    client = get_qdrant_client()
    coll = collection_name or settings.QDRANT_COLLECTION
    ensure_collection(coll, dim=len(vectors[0]))

    points = []
    import uuid
    for chunk, vector in zip(chunks, vectors):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.get("chunk_id", str(uuid.uuid4()))))
        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk,
            )
        )

    client.upsert(collection_name=coll, points=points)
    logger.info(f"Upserted {len(points)} chunks into '{coll}'.")


def search_vectors(
    query_vector: List[float],
    limit: int = 8,
    paper_ids: Optional[List[str]] = None,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Performs cosine vector search on Qdrant and returns scored chunk payloads."""
    client = get_qdrant_client()
    coll = collection_name or settings.QDRANT_COLLECTION
    ensure_collection(coll, dim=len(query_vector))

    query_filter = None
    if paper_ids:
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="paper_id",
                    match=qmodels.MatchAny(any=paper_ids),
                )
            ]
        )

    # Use query_points for newer qdrant-client versions
    try:
        response = client.query_points(
            collection_name=coll,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        scored_points = response.points
    except AttributeError:
        # Fallback to search if query_points not present in older versions
        scored_points = client.search(
            collection_name=coll,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

    results = []
    for pt in scored_points:
        payload = pt.payload or {}
        results.append({
            **payload,
            "score": pt.score,
        })
    return results


def check_qdrant_health() -> Dict[str, Any]:
    """Health check for Qdrant service."""
    if not settings.QDRANT_URL:
        return {"status": "unconfigured", "message": "QDRANT_URL not set (in-memory mode)"}

    start_time = time.time()
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "status": "healthy",
            "latency_ms": latency_ms,
            "collections_count": len(collections.collections),
        }
    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {"status": "unhealthy", "latency_ms": latency_ms, "error": str(e)}
