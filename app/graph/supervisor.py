"""Supervisor Agent: research decomposition, dynamic routing, and sufficiency review."""

import logging
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from app.clients.llm import LLMClient

logger = logging.getLogger("researchos.supervisor")


class SupervisorPlan(BaseModel):
    sub_questions: List[str] = Field(
        description="2 to 4 focused scientific sub-questions to comprehensively investigate the topic",
        min_length=1,
        max_length=4,
    )
    reasoning: str = Field(description="Strategic justification for this breakdown")


class SupervisorReview(BaseModel):
    decision: Literal["proceed", "re-route"] = Field(
        description="'proceed' if evidence is sufficient, 're-route' for an additional retrieval round"
    )
    reasoning: str = Field(description="Explanation of sufficiency evaluation")
    refined_subquestions: List[str] = Field(
        default_factory=list,
        description="Optional broadened or alternative queries if re-routing",
    )


class SupervisorAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    async def create_plan(self, question: str) -> SupervisorPlan:
        """Decomposes the core research question into up to 4 structured sub-questions."""
        prompt = (
            f"You are the Research Supervisor coordinating an autonomous scientific research team.\n\n"
            f"RESEARCH QUESTION:\n\"{question}\"\n\n"
            f"TASKS:\n"
            f"Decompose this into 2 to 4 distinct, complementary sub-questions that together cover:\n"
            f"1. Core architecture / foundational method\n"
            f"2. Empirical performance benchmarks and quantitative results\n"
            f"3. Stated limitations, trade-offs, and critical challenges\n"
            f"Provide concise, high-impact scientific questions."
        )

        plan = await self.llm.generate_structured(
            prompt=prompt,
            schema=SupervisorPlan,
            system_instruction="You are an elite AI research director.",
        )
        # Enforce max 4 sub-questions
        plan.sub_questions = plan.sub_questions[:4]
        return plan

    async def review_evidence(
        self,
        question: str,
        papers: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        retrieval_round: int,
        max_rounds: int = 2,
    ) -> SupervisorReview:
        """Evaluates whether retrieved evidence and claims are sufficient or require re-routing."""
        # Programmatic check first: if already reached max rounds, must proceed
        if retrieval_round >= max_rounds:
            return SupervisorReview(
                decision="proceed",
                reasoning=f"Max retrieval rounds ({max_rounds}) reached. Proceeding to comparison.",
            )

        # Check sufficiency: need at least 2 distinct papers and 2 verified claims
        if len(papers) < 2 or len(claims) < 2:
            return SupervisorReview(
                decision="re-route",
                reasoning=(
                    f"Insufficient evidence found ({len(papers)} papers, {len(claims)} verified claims). "
                    f"Initiating second round of targeted retrieval."
                ),
                refined_subquestions=[
                    f"foundational overview of {question[:40]}",
                    f"benchmarks and comparative analysis of {question[:40]}",
                ],
            )

        return SupervisorReview(
            decision="proceed",
            reasoning=f"Sufficient evidence collected: {len(papers)} papers and {len(claims)} verified claims.",
        )
