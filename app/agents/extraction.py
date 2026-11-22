"""Claim and evidence extraction agent with strict programmatic verbatim quote verification.
Hallucinated evidence or fabricated quotes are dropped automatically.
"""

import logging
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from app.clients.llm import LLMClient

logger = logging.getLogger("researchos.extraction")


class RawClaim(BaseModel):
    claim_id: str = Field(description="Unique identifier, e.g. 'c1', 'c2'")
    claim: str = Field(description="Clear, atomic factual assertion made by the paper")
    paper_id: str = Field(description="arXiv ID of the source paper")
    chunk_id: str = Field(description="ID of the specific chunk containing the evidence")
    quote: str = Field(description="Exact verbatim excerpt under 40 words from the chunk text")
    stance: Literal["supports", "contradicts", "neutral", "inconclusive"] = Field(
        default="supports",
        description="Stance of the paper towards the claim",
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )


class ExtractedClaims(BaseModel):
    claims: List[RawClaim] = Field(default_factory=list)


def normalize_whitespace(text: str) -> str:
    """Normalizes whitespace and removes quotation marks for robust comparison."""
    text = re.sub(r"[\"\'`“”‘’]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_verbatim_quote(quote: str, chunk_text: str, max_words: int = 45) -> bool:
    """Programmatic check: verifies that quote appears verbatim in chunk_text.
    Rejects empty quotes, overly long quotes, and hallucinated quotes.
    """
    if not quote or not chunk_text:
        return False

    words = quote.strip().split()
    if len(words) > max_words:
        logger.debug(f"Quote rejected: exceeds word limit ({len(words)} > {max_words})")
        return False

    norm_quote = normalize_whitespace(quote)
    norm_chunk = normalize_whitespace(chunk_text)

    if not norm_quote:
        return False

    return norm_quote in norm_chunk


class ExtractionAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    async def execute(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extracts atomic claims from retrieved chunks and programmatically verifies quotes.
        Returns:
            {
                "claims": List[Dict],
                "dropped_claims": List[Dict],
                "quote_verification_rate": float,
                "total_extracted": int,
                "total_verified": int
            }
        """
        if not chunks:
            return {
                "claims": [],
                "dropped_claims": [],
                "quote_verification_rate": 1.0,
                "total_extracted": 0,
                "total_verified": 0,
            }

        # Build chunk prompt context
        chunk_context_blocks = []
        chunk_map: Dict[str, str] = {}
        for c in chunks:
            cid = c.get("chunk_id", "")
            pid = c.get("paper_id", "")
            text = c.get("text", "")
            chunk_map[cid] = text
            chunk_context_blocks.append(
                f"--- CHUNK ID: {cid} | PAPER ID: {pid} ---\n{text}\n"
            )

        context_str = "\n".join(chunk_context_blocks)
        prompt = (
            f"From the academic paper excerpts below, extract key atomic factual claims and their supporting evidence.\n\n"
            f"CRITICAL RULES:\n"
            f"1. For each claim, you MUST provide an exact verbatim quote copied directly from the chunk text.\n"
            f"2. The quote MUST be under 40 words and exist character-for-character in the chunk.\n"
            f"3. Do not paraphrase or alter the quote. Fabricated quotes will fail programmatic verification.\n"
            f"4. Map each claim to its exact chunk_id and paper_id.\n\n"
            f"Excerpts:\n{context_str}"
        )

        extracted = await self.llm.generate_structured(
            prompt=prompt,
            schema=ExtractedClaims,
            system_instruction="You are a rigorous scientific evidence extraction agent.",
        )

        verified_claims: List[Dict[str, Any]] = []
        dropped_claims: List[Dict[str, Any]] = []

        for raw_claim in extracted.claims:
            chunk_text = chunk_map.get(raw_claim.chunk_id, "")
            is_valid = verify_verbatim_quote(raw_claim.quote, chunk_text)

            claim_dict = raw_claim.model_dump()
            if is_valid:
                claim_dict["verified"] = True
                verified_claims.append(claim_dict)
            else:
                claim_dict["verified"] = False
                claim_dict["drop_reason"] = "Quote not found verbatim in chunk text or exceeds word limit"
                dropped_claims.append(claim_dict)
                logger.warning(
                    f"Dropped hallucinated claim {raw_claim.claim_id}: quote '{raw_claim.quote[:50]}...' not in chunk {raw_claim.chunk_id}"
                )

        total_extracted = len(extracted.claims)
        total_verified = len(verified_claims)
        rate = round(total_verified / total_extracted, 4) if total_extracted > 0 else 1.0

        return {
            "claims": verified_claims,
            "dropped_claims": dropped_claims,
            "quote_verification_rate": rate,
            "total_extracted": total_extracted,
            "total_verified": total_verified,
        }
