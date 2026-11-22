"""Unit tests for Phase 3: Chunking, metrics, fusion, and retrieval agent."""

import pytest
from app.utils.chunking import chunk_paper, _split_into_sections
from app.utils.metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k, reciprocal_rank_fusion
from app.clients.arxiv import _clean_paper_id
from app.clients.llm import LLMClient
from app.agents.retrieval import RetrievalAgent, _tokenize


def test_clean_paper_id():
    assert _clean_paper_id("http://arxiv.org/abs/1706.03762v2") == "1706.03762"
    assert _clean_paper_id("1706.03762") == "1706.03762"
    assert _clean_paper_id("http://arxiv.org/pdf/2005.14165.pdf") == "2005.14165"


def test_metrics_on_known_examples():
    retrieved = ["p1", "p2", "p3", "p4", "p5"]
    relevant = ["p2", "p4", "p9"]

    # Precision@k: p2, p4 are in top 5 -> 2/5 = 0.4
    assert precision_at_k(retrieved, relevant, k=5) == 0.4
    # Precision@2: only p2 is in top 2 -> 1/2 = 0.5
    assert precision_at_k(retrieved, relevant, k=2) == 0.5

    # Recall@k: 2 hits out of 3 total relevant -> 2/3 = 0.6667
    assert recall_at_k(retrieved, relevant, k=5) == 0.6667

    # MRR: first relevant is at rank 2 (p2) -> 1/2 = 0.5
    assert mrr(retrieved, relevant, k=5) == 0.5

    # nDCG@5: rank 2 hit (rel=1/log2(3)), rank 4 hit (rel=1/log2(5))
    # ideal hits: rank 1, rank 2
    ndcg = ndcg_at_k(retrieved, relevant, k=5)
    assert 0.0 < ndcg <= 1.0


def test_reciprocal_rank_fusion():
    vec_ranks = ["d1", "d2", "d3"]
    bm25_ranks = ["d2", "d1", "d4"]

    fused = reciprocal_rank_fusion(vec_ranks, bm25_ranks, k=60)
    top_doc = fused[0][0]
    # Both d1 and d2 have top ranks in both lists, so d1 or d2 must lead
    assert top_doc in ("d1", "d2")


def test_chunking_section_detection():
    sample_paper = {
        "paper_id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017,
        "summary": "We propose the Transformer, a model architecture eschewing recurrence.",
    }
    body = (
        "1. Introduction\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks.\n\n"
        "2. Background\nThe goal of reducing sequential computation also forms the foundation of the Extended Neural GPU.\n\n"
        "3. Model Architecture\nThe Transformer follows this overall architecture using stacked self-attention and point-wise layers."
    )

    chunks = chunk_paper(sample_paper, pdf_text=body, target_chunk_size=300, overlap_size=50)
    assert len(chunks) >= 3
    sections = {c["section"] for c in chunks}
    assert "Introduction" in sections or "Model Architecture" in sections
    for c in chunks:
        assert "chunk_id" in c
        assert "paper_id" in c
        assert "text" in c


@pytest.mark.asyncio
async def test_retrieval_agent_mock():
    llm = LLMClient(force_mock=True)
    agent = RetrievalAgent(llm_client=llm)

    # Mock arXiv search via monkeypatch
    sample_papers = [
        {
            "paper_id": "1706.03762",
            "title": "Attention Is All You Need",
            "summary": "The Transformer is a network architecture based solely on attention mechanisms.",
            "authors": ["Vaswani et al."],
            "year": 2017,
            "arxiv_url": "https://arxiv.org/abs/1706.03762",
            "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
        }
    ]

    async def mock_search(query, max_results=3):
        return sample_papers

    async def mock_pdf(paper_id, max_pages=6):
        return "Self-attention reduces sequential computation to a constant number of operations."

    agent.arxiv.search = mock_search
    agent.arxiv.fetch_pdf_text = mock_pdf

    res = await agent.execute(
        sub_questions=["What is the self-attention mechanism?"],
        max_papers=1,
        depth="quick",
        collection_name="test_retrieval_coll",
    )

    assert "papers" in res
    assert len(res["papers"]) == 1
    assert "retrieved_chunks" in res
    assert len(res["retrieved_chunks"]) > 0
    assert "retrieval_score" in res["retrieved_chunks"][0]
