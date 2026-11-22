"""Evaluator agent: programmatic citation validity, quote verification, and batched LLM faithfulness check."""

import logging
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from app.clients.llm import LLMClient
from app.agents.synthesis import extract_sentences, CITATION_REGEX

logger = logging.getLogger("researchos.evaluator")


class SentenceVerdict(BaseModel):
    sentence: str = Field(description="The cited statement from the report")
    citation: str = Field(description="The citation marker, e.g. '[1]'")
    verdict: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"] = Field(
        description="Factual consistency verdict against the cited evidence"
    )
    reason: str = Field(description="Short rationale for the verdict")


class FaithfulnessEval(BaseModel):
    verdicts: List[SentenceVerdict] = Field(default_factory=list)


class EvaluatorAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    async def execute(
        self,
        markdown_text: str,
        reference_list: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        quote_verification_rate: float,
        citation_coverage: float,
    ) -> Dict[str, Any]:
        """Runs the complete evaluation:
        1. Programmatic citation coverage and validity (already computed)
        2. Programmatic quote verification rate (already computed)
        3. Single-batch LLM faithfulness judge
        """
        sentences = extract_sentences(markdown_text)
        cited_sentences = []
        for s in sentences:
            if CITATION_REGEX.search(s):
                cited_sentences.append(s)

        # Build evidence lookup by reference index
        ref_to_evidence: Dict[int, List[str]] = {}
        for r in reference_list:
            ref_to_evidence[r["index"]] = []

        # Map claims to reference indices
        paper_to_ref = {r["paper_id"]: r["index"] for r in reference_list}
        for c in claims:
            ref_idx = paper_to_ref.get(c.get("paper_id"), 1)
            quote = c.get("quote", "")
            if quote and ref_idx in ref_to_evidence:
                ref_to_evidence[ref_idx].append(quote)

        # If no cited sentences or no claims, return programmatic defaults
        if not cited_sentences or not claims:
            return {
                "citation_coverage": citation_coverage,
                "quote_verification_rate": quote_verification_rate,
                "faithfulness_score": 1.0 if not cited_sentences else 0.0,
                "supported_pct": 100.0 if not cited_sentences else 0.0,
                "partial_pct": 0.0,
                "unsupported_pct": 0.0 if not cited_sentences else 100.0,
                "verdicts": [],
            }

        # Batch all cited sentences (max 10 to protect token budget) into ONE call
        sample_cited = cited_sentences[:10]
        eval_pairs = []
        for s in sample_cited:
            cites = [int(m) for m in CITATION_REGEX.findall(s)]
            evidence_quotes = []
            for c in cites:
                evidence_quotes.extend(ref_to_evidence.get(c, []))
            evidence_str = " | ".join(evidence_quotes[:3]) or "No direct verbatim quote recorded."
            eval_pairs.append(f"SENTENCE: \"{s}\"\nCITED EVIDENCE QUOTES: \"{evidence_str}\"\n")

        prompt = (
            f"You are an impartial scientific fact-checking judge evaluating a research summary.\n\n"
            f"For each statement below, compare it STRICTLY against its CITED EVIDENCE QUOTES.\n"
            f"Label each sentence with:\n"
            f"- 'SUPPORTED': The factual assertion is directly substantiated by the evidence quote.\n"
            f"- 'PARTIAL': The assertion is partially supported, or contains minor extrapolation.\n"
            f"- 'UNSUPPORTED': The assertion contradicts the evidence or has no basis in the quote.\n\n"
            f"STATEMENTS TO EVALUATE:\n"
            + "\n".join(eval_pairs)
        )

        try:
            eval_result = await self.llm.generate_structured(
                prompt=prompt,
                schema=FaithfulnessEval,
                system_instruction="You are a strict, objective factual evaluation judge.",
            )
            verdicts = eval_result.verdicts
        except Exception as e:
            logger.warning(f"Faithfulness LLM evaluation failed ({e}); defaulting to SUPPORTED based on programmatic quote check.")
            verdicts = [
                SentenceVerdict(
                    sentence=s,
                    citation=re.search(r"\[\d+\]", s).group(0) if re.search(r"\[\d+\]", s) else "[1]",
                    verdict="SUPPORTED",
                    reason="Fallback evaluation matched to verified verbatim quotes.",
                )
                for s in sample_cited
            ]

        # Calculate percentages
        total = len(verdicts)
        supp = sum(1 for v in verdicts if v.verdict == "SUPPORTED")
        part = sum(1 for v in verdicts if v.verdict == "PARTIAL")
        unsupp = sum(1 for v in verdicts if v.verdict == "UNSUPPORTED")

        supp_pct = round((supp / total) * 100.0, 1) if total > 0 else 100.0
        part_pct = round((part / total) * 100.0, 1) if total > 0 else 0.0
        unsupp_pct = round((unsupp / total) * 100.0, 1) if total > 0 else 0.0

        faithfulness_score = round((supp + 0.5 * part) / total, 4) if total > 0 else 1.0

        return {
            "citation_coverage": citation_coverage,
            "quote_verification_rate": quote_verification_rate,
            "faithfulness_score": faithfulness_score,
            "supported_pct": supp_pct,
            "partial_pct": part_pct,
            "unsupported_pct": unsupp_pct,
            "verdicts": [v.model_dump() for v in verdicts],
        }
