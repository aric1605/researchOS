"""Critical comparison agent: groups claims by theme, identifies agreements/contradictions,
synthesizes methodological trade-offs and stated limitations, referencing claim IDs.
"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.clients.llm import LLMClient

logger = logging.getLogger("researchos.comparison")


class ComparisonTheme(BaseModel):
    theme_name: str = Field(description="Name of the thematic area, e.g. 'Computational Complexity'")
    agreements: List[str] = Field(
        default_factory=list,
        description="Agreed insights across papers supported by evidence",
    )
    contradictions: List[str] = Field(
        default_factory=list,
        description="Contradictions, divergent findings, or conflicting claims across papers",
    )
    claim_ids: List[str] = Field(
        default_factory=list,
        description="Claim IDs supporting this theme, e.g. ['c1', 'c2']",
    )


class CriticalComparison(BaseModel):
    themes: List[ComparisonTheme] = Field(
        default_factory=list,
        description="Thematic groupings of verified claims",
    )
    methodology_differences: List[str] = Field(
        default_factory=list,
        description="Key differences in method, dataset, benchmark, or scale across papers",
    )
    limitations_found: List[str] = Field(
        default_factory=list,
        description="Explicit weaknesses or constraints stated by the papers",
    )
    open_questions: List[str] = Field(
        default_factory=list,
        description="Short list of unresolved technical questions or future research directions",
    )


class ComparisonAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    async def execute(
        self,
        question: str,
        claims: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Performs critical comparative analysis over verified claims."""
        if not claims:
            return {
                "themes": [],
                "methodology_differences": [],
                "limitations_found": [],
                "open_questions": ["Insufficient verified claims extracted to perform comparison."],
            }

        # Build prompt listing verified claims with claim IDs
        claims_text_blocks = []
        valid_claim_ids = set()
        for c in claims:
            cid = c.get("claim_id", "")
            valid_claim_ids.add(cid)
            claims_text_blocks.append(
                f"- [{cid}] (Paper {c.get('paper_id')}): \"{c.get('claim')}\" "
                f"(Stance: {c.get('stance')}, Quote: \"{c.get('quote')}\")"
            )

        claims_summary = "\n".join(claims_text_blocks)
        prompt = (
            f"You are a critical scientific analysis agent evaluating claims for research question:\n"
            f"\"{question}\"\n\n"
            f"Here are verified atomic claims extracted from academic papers:\n"
            f"{claims_summary}\n\n"
            f"TASKS:\n"
            f"1. Group claims into coherent themes. In each theme, specify agreements and contradictions.\n"
            f"2. You MUST cite existing claim_ids in the 'claim_ids' field for each theme.\n"
            f"3. Highlight methodological differences (datasets, model size, evaluation benchmarks).\n"
            f"4. Highlight explicit stated limitations and formulate 2-4 open research questions.\n"
        )

        comparison = await self.llm.generate_structured(
            prompt=prompt,
            schema=CriticalComparison,
            system_instruction="You are an expert scientific comparative analyst.",
        )

        # Sanitize claim_ids to ensure they only refer to existing claims
        sanitized_themes = []
        for theme in comparison.themes:
            filtered_ids = [cid for cid in theme.claim_ids if cid in valid_claim_ids]
            theme_dict = theme.model_dump()
            theme_dict["claim_ids"] = filtered_ids or list(valid_claim_ids)[:2]
            sanitized_themes.append(theme_dict)

        result = comparison.model_dump()
        result["themes"] = sanitized_themes
        return result
