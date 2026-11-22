"""Retrieval Agent: query generation, arXiv fetching, PDF text extraction,
chunking, Qdrant upsert, and hybrid (Vector + BM25) retrieval with RRF.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from app.clients.arxiv import ArxivClient
from app.clients.semantic_scholar import SemanticScholarClient
from app.clients.embeddings import get_embedding, get_embeddings
from app.clients.vectorstore import upsert_chunks, search_vectors
from app.clients.llm import LLMClient
from app.utils.chunking import chunk_paper
from app.utils.metrics import reciprocal_rank_fusion

logger = logging.getLogger("researchos.retrieval")


class QueryPlan(BaseModel):
    queries: List[str] = Field(
        description="2 targeted academic search queries optimized for arXiv search",
        min_length=1,
        max_length=3,
    )


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class RetrievalAgent:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        arxiv_client: Optional[ArxivClient] = None,
        s2_client: Optional[SemanticScholarClient] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.arxiv = arxiv_client or ArxivClient()
        self.s2 = s2_client or SemanticScholarClient()

    async def generate_queries_for_subquestion(self, sub_question: str) -> List[str]:
        """Generates 2 search queries for a given sub-question."""
        prompt = (
            f"Given this research sub-question:\n\"{sub_question}\"\n\n"
            f"Generate 2 specific, keyword-dense search queries suitable for arXiv title and abstract search. "
            f"Do not use Boolean operators like AND/OR; use key scientific terms."
        )
        try:
            plan = await self.llm.generate_structured(
                prompt=prompt,
                schema=QueryPlan,
                system_instruction="You are an expert scientific search specialist.",
            )
            return plan.queries[:2]
        except Exception as e:
            logger.warning(f"Failed to generate structured queries ({e}); using heuristic query.")
            clean_q = re.sub(r"[^\w\s]", "", sub_question)
            return [clean_q[:60]]

    async def execute(
        self,
        sub_questions: List[str],
        max_papers: int = 6,
        depth: str = "standard",
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the full retrieval pipeline:
        1. Generate queries for each sub-question
        2. Fetch & de-duplicate papers from arXiv
        3. Parse PDF text (for standard depth)
        4. Chunk papers and generate embeddings
        5. Upsert to Qdrant
        6. Perform hybrid retrieval (Vector + BM25) for each sub-question
        """
        all_queries: List[str] = []
        papers_by_id: Dict[str, Dict[str, Any]] = {}

        # 1 & 2: Generate queries and search arXiv
        for sq in sub_questions:
            queries = await self.generate_queries_for_subquestion(sq)
            all_queries.extend(queries)
            for query in queries:
                if len(papers_by_id) >= max_papers:
                    break
                found_papers = await self.arxiv.search(query, max_results=3)
                for p in found_papers:
                    pid = p["paper_id"]
                    if pid not in papers_by_id:
                        papers_by_id[pid] = p
                        if len(papers_by_id) >= max_papers:
                            break

        paper_list = list(papers_by_id.values())
        logger.info(f"Retrieved {len(paper_list)} unique papers across {len(all_queries)} queries.")

        # 3. Enrich with Semantic Scholar & extract PDF text if standard depth
        for p in paper_list:
            pid = p["paper_id"]
            # S2 citation enrichment (non-blocking / safe)
            try:
                s2_data = await self.s2.get_paper_metadata(pid)
                p["citation_count"] = s2_data.get("citation_count", 0)
            except Exception:
                p["citation_count"] = 0

            # PDF text extraction
            pdf_text = ""
            if depth == "standard":
                pdf_text = await self.arxiv.fetch_pdf_text(pid, max_pages=6)
            p["pdf_text"] = pdf_text

        # 4. Chunk papers
        all_chunks: List[Dict[str, Any]] = []
        for p in paper_list:
            chunks = chunk_paper(p, pdf_text=p.get("pdf_text", ""))
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks generated from retrieved papers.")
            return {
                "papers": paper_list,
                "chunks": [],
                "retrieved_chunks": [],
                "queries": all_queries,
            }

        # 5. Embed and upsert to Qdrant
        chunk_texts = [c["text"] for c in all_chunks]
        chunk_vectors = await get_embeddings(chunk_texts, force_mock=self.llm.force_mock)
        upsert_chunks(all_chunks, chunk_vectors, collection_name=collection_name)

        # 6. Hybrid Retrieval for each sub-question
        # Build in-memory BM25 index across all indexed chunks
        tokenized_corpus = [_tokenize(c["text"]) for c in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        chunk_map = {c["chunk_id"]: c for c in all_chunks}

        selected_chunk_ids: set[str] = set()
        retrieved_chunks: List[Dict[str, Any]] = []

        for sq in sub_questions:
            sq_vector = await get_embedding(sq, force_mock=self.llm.force_mock)

            # Vector search
            vector_results = search_vectors(
                query_vector=sq_vector,
                limit=min(12, len(all_chunks)),
                collection_name=collection_name,
            )
            vector_ranked_ids = [v["chunk_id"] for v in vector_results]

            # BM25 search
            sq_tokens = _tokenize(sq)
            bm25_scores = bm25.get_scores(sq_tokens)
            bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            bm25_ranked_ids = [all_chunks[i]["chunk_id"] for i in bm25_ranked_indices[:12]]

            # Reciprocal Rank Fusion (RRF)
            fused_ranks = reciprocal_rank_fusion(vector_ranked_ids, bm25_ranked_ids, k=60)

            # Pick top-k for this subquestion
            for chunk_id, score in fused_ranks[:4]:
                if chunk_id not in selected_chunk_ids and chunk_id in chunk_map:
                    selected_chunk_ids.add(chunk_id)
                    chunk_obj = dict(chunk_map[chunk_id])
                    chunk_obj["retrieval_score"] = round(score, 5)
                    chunk_obj["retrieval_query"] = sq
                    retrieved_chunks.append(chunk_obj)

        # Cap retrieved chunks to top 8-10 chunks
        retrieved_chunks = retrieved_chunks[:8]

        return {
            "papers": paper_list,
            "all_chunks": all_chunks,
            "retrieved_chunks": retrieved_chunks,
            "queries": all_queries,
        }
