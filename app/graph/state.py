"""Typed state definition for LangGraph research workflow."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentTraceStep(BaseModel):
    agent: str
    started_at: str
    ended_at: str
    duration_s: float
    llm_calls: int = 0
    tokens_estimate: int = 0
    status: str = "success"  # success, warning, failed
    notes: Optional[str] = None


class ResearchState(BaseModel):
    # Job input
    run_id: str
    question: str
    depth: str = "standard"  # quick, standard
    max_papers: int = 6
    collection_name: Optional[str] = None

    # Supervisor plan
    sub_questions: List[str] = Field(default_factory=list)
    retrieval_round: int = 1
    max_retrieval_rounds: int = 2

    # Retrieved & processed evidence
    papers: List[Dict[str, Any]] = Field(default_factory=list)
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)

    # Claims & evidence
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    dropped_claims: List[Dict[str, Any]] = Field(default_factory=list)
    quote_verification_rate: float = 1.0

    # Comparison & synthesis
    comparison: Dict[str, Any] = Field(default_factory=dict)
    synthesis: Dict[str, Any] = Field(default_factory=dict)

    # Evaluation results
    eval_scores: Dict[str, Any] = Field(default_factory=dict)

    # Telemetry & execution trace
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timings: Dict[str, float] = Field(default_factory=dict)
    call_counts: Dict[str, int] = Field(default_factory=dict)
    status: str = "running"  # running, completed, failed
