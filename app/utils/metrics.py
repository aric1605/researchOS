"""Evaluation metrics for retrieval quality: Precision@k, Recall@k, MRR, and nDCG@k."""

import math
from typing import List, Set, Union


def precision_at_k(retrieved_ids: List[str], relevant_ids: Union[List[str], Set[str]], k: int = 5) -> float:
    """Computes Precision@k: fraction of top-k retrieved documents that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return round(hits / k, 4)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Union[List[str], Set[str]], k: int = 5) -> float:
    """Computes Recall@k: fraction of all relevant documents that are in top-k."""
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return round(hits / len(relevant_set), 4)


def mrr(retrieved_ids: List[str], relevant_ids: Union[List[str], Set[str]], k: int = 10) -> float:
    """Computes Mean Reciprocal Rank (MRR) within top-k."""
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Union[List[str], Set[str]], k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@k) with binary relevance."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    if not relevant_set or not top_k:
        return 0.0

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        rel = 1.0 if doc_id in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)

    # Ideal DCG (IDCG)
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def reciprocal_rank_fusion(
    vector_ranked_ids: List[str],
    bm25_ranked_ids: List[str],
    k: int = 60,
) -> List[tuple[str, float]]:
    """Combines two rankings using standard Reciprocal Rank Fusion (RRF).
    Returns list of (doc_id, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = {}

    for rank, doc_id in enumerate(vector_ranked_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    for rank, doc_id in enumerate(bm25_ranked_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    sorted_ranks = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_ranks
