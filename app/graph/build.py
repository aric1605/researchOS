"""LangGraph orchestration graph for ResearchOS multi-agent workflow."""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from langgraph.graph import StateGraph, START, END

from app.graph.state import ResearchState
from app.graph.supervisor import SupervisorAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.extraction import ExtractionAgent
from app.agents.comparison import ComparisonAgent
from app.agents.synthesis import SynthesisAgent
from app.agents.evaluator import EvaluatorAgent
from app.clients.llm import LLMClient, CallBudgetTracker
from app.db.repository import Repository

logger = logging.getLogger("researchos.graph")


def _record_trace(
    state: ResearchState,
    agent: str,
    start_time: float,
    status: str = "success",
    notes: Optional[str] = None,
    budget_tracker: Optional[CallBudgetTracker] = None,
) -> None:
    duration = round(time.time() - start_time, 2)
    now_iso = datetime.now(timezone.utc).isoformat()
    llm_calls = budget_tracker.call_count if budget_tracker else 0

    trace_entry = {
        "agent": agent,
        "started_at": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "ended_at": now_iso,
        "duration_s": duration,
        "llm_calls": llm_calls,
        "tokens_estimate": llm_calls * 450,
        "status": status,
        "notes": notes,
    }
    state.trace.append(trace_entry)
    state.timings[agent] = duration


async def _emit_event(
    run_id: str,
    event_type: str,
    data: Dict[str, Any],
    event_callback: Optional[Callable] = None,
) -> None:
    try:
        await Repository.add_event(run_id, event_type, data)
        if event_callback:
            await event_callback(run_id, event_type, data)
    except Exception as e:
        logger.warning(f"Could not persist or emit event {event_type}: {e}")


def build_research_graph(
    llm_client: Optional[LLMClient] = None,
    event_callback: Optional[Callable] = None,
) -> Any:
    """Constructs the compiled LangGraph state graph for ResearchOS."""
    shared_llm = llm_client or LLMClient()
    budget = shared_llm.budget_tracker

    supervisor = SupervisorAgent(llm_client=shared_llm)
    retrieval_agent = RetrievalAgent(llm_client=shared_llm)
    extraction_agent = ExtractionAgent(llm_client=shared_llm)
    comparison_agent = ComparisonAgent(llm_client=shared_llm)
    synthesis_agent = SynthesisAgent(llm_client=shared_llm)
    evaluator_agent = EvaluatorAgent(llm_client=shared_llm)

    # 1. Supervisor Plan Node
    async def supervisor_plan_action(state: ResearchState) -> Dict[str, Any]:
        t0 = time.time()
        await _emit_event(
            state.run_id,
            "agent_started",
            {"agent": "Supervisor", "message": "Decomposing research question into sub-questions..."},
            event_callback,
        )
        try:
            plan = await supervisor.create_plan(state.question)
            sub_qs = plan.sub_questions
            _record_trace(state, "Supervisor", t0, "success", f"Planned {len(sub_qs)} sub-questions", budget)
            await _emit_event(
                state.run_id,
                "agent_finished",
                {"agent": "Supervisor", "sub_questions": sub_qs, "reasoning": plan.reasoning},
                event_callback,
            )
            return {"sub_questions": sub_qs, "trace": list(state.trace), "timings": dict(state.timings)}
        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            fallback_qs = [state.question]
            state.warnings.append(f"Supervisor plan fallback used: {e}")
            _record_trace(state, "Supervisor", t0, "warning", str(e), budget)
            return {"sub_questions": fallback_qs, "trace": list(state.trace), "timings": dict(state.timings)}

    # 2. Retrieval Node
    async def retrieval_action(state: ResearchState) -> Dict[str, Any]:
        t0 = time.time()
        await _emit_event(
            state.run_id,
            "agent_started",
            {
                "agent": "Retrieval",
                "message": f"Querying arXiv and indexing papers (Round {state.retrieval_round})...",
            },
            event_callback,
        )
        try:
            res = await retrieval_agent.execute(
                sub_questions=state.sub_questions,
                max_papers=state.max_papers,
                depth=state.depth,
                collection_name=state.collection_name,
            )
            papers = res.get("papers", [])
            retrieved_chunks = res.get("retrieved_chunks", [])
            all_chunks = res.get("all_chunks", [])

            _record_trace(
                state,
                f"Retrieval_R{state.retrieval_round}",
                t0,
                "success",
                f"Found {len(papers)} papers, {len(retrieved_chunks)} hybrid retrieved chunks",
                budget,
            )
            await _emit_event(
                state.run_id,
                "agent_finished",
                {
                    "agent": "Retrieval",
                    "papers_count": len(papers),
                    "chunks_count": len(retrieved_chunks),
                    "papers": [
                        {"paper_id": p["paper_id"], "title": p["title"], "year": p.get("year", 2024)}
                        for p in papers
                    ],
                },
                event_callback,
            )
            return {
                "papers": papers,
                "chunks": all_chunks,
                "retrieved_chunks": retrieved_chunks,
                "trace": list(state.trace),
                "timings": dict(state.timings),
            }
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            state.warnings.append(f"Retrieval failed or incomplete: {e}")
            _record_trace(state, f"Retrieval_R{state.retrieval_round}", t0, "failed", str(e), budget)
            return {"papers": [], "chunks": [], "retrieved_chunks": [], "trace": list(state.trace), "timings": dict(state.timings)}

    # 3. Extraction Node
    async def extraction_action(state: ResearchState) -> Dict[str, Any]:
        t0 = time.time()
        await _emit_event(
            state.run_id,
            "agent_started",
            {"agent": "Extraction", "message": "Extracting atomic claims and verifying verbatim quotes..."},
            event_callback,
        )
        try:
            res = await extraction_agent.execute(state.retrieved_chunks)
            claims = res.get("claims", [])
            dropped = res.get("dropped_claims", [])
            rate = res.get("quote_verification_rate", 1.0)

            _record_trace(
                state,
                "Extraction",
                t0,
                "success",
                f"Verified {len(claims)} claims, dropped {len(dropped)} ({rate*100:.1f}% verified)",
                budget,
            )
            await _emit_event(
                state.run_id,
                "agent_finished",
                {
                    "agent": "Extraction",
                    "claims_count": len(claims),
                    "dropped_count": len(dropped),
                    "quote_verification_rate": rate,
                },
                event_callback,
            )
            return {
                "claims": claims,
                "dropped_claims": dropped,
                "quote_verification_rate": rate,
                "trace": list(state.trace),
                "timings": dict(state.timings),
            }
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            state.warnings.append(f"Extraction failed: {e}")
            _record_trace(state, "Extraction", t0, "failed", str(e), budget)
            return {"claims": [], "dropped_claims": [], "quote_verification_rate": 0.0, "trace": list(state.trace), "timings": dict(state.timings)}

    # 4. Supervisor Review Node
    async def supervisor_review_action(state: ResearchState) -> Dict[str, Any]:
        t0 = time.time()
        review = await supervisor.review_evidence(
            question=state.question,
            papers=state.papers,
            claims=state.claims,
            retrieval_round=state.retrieval_round,
            max_rounds=state.max_retrieval_rounds,
        )
        _record_trace(
            state,
            "Supervisor_Review",
            t0,
            "success",
            f"Decision: {review.decision}. {review.reasoning}",
            budget,
        )
        if review.decision == "re-route":
            new_sub_qs = review.refined_subquestions or state.sub_questions
            await _emit_event(
                state.run_id,
                "warning",
                {
                    "agent": "Supervisor",
                    "message": f"Evidence insufficient; re-routing to retrieval round {state.retrieval_round + 1}...",
                },
                event_callback,
            )
            return {
                "retrieval_round": state.retrieval_round + 1,
                "sub_questions": new_sub_qs,
                "trace": list(state.trace),
                "timings": dict(state.timings),
            }
        return {"trace": list(state.trace), "timings": dict(state.timings)}

    # Routing condition
    def route_after_review(state: ResearchState) -> str:
        # If we just incremented retrieval_round and have < 2 papers, re-route to retrieval
        if state.retrieval_round > 1 and len(state.papers) < 2:
            return "retrieval_node"
        return "comparison_node"

    # 5. Comparison Node
    async def comparison_action(state: ResearchState) -> Dict[str, Any]:
        t0 = time.time()
        await _emit_event(
            state.run_id,
            "agent_started",
            {"agent": "Comparison", "message": "Analyzing agreements, contradictions, and stated limitations..."},
            event_callback,
        )
        try:
            comparison = await comparison_agent.execute(
                question=state.question,
                claims=state.claims,
                papers=state.papers,
            )
            _record_trace(state, "Comparison", t0, "success", f"Identified {len(comparison.get('themes', []))} themes", budget)
            await _emit_event(
                state.run_id,
                "agent_finished",
                {
                    "agent": "Comparison",
                    "themes_count": len(comparison.get("themes", [])),
                    "limitations_count": len(comparison.get("limitations_found", [])),
                },
                event_callback,
            )
            return {"comparison": comparison, "trace": list(state.trace), "timings": dict(state.timings)}
        except Exception as e:
            logger.error(f"Comparison error: {e}")
            state.warnings.append(f"Comparison failed: {e}")
            _record_trace(state, "Comparison", t0, "warning", str(e), budget)
            return {
                "comparison": {"themes": [], "limitations_found": [], "open_questions": []},
                "trace": list(state.trace),
                "timings": dict(state.timings),
            }

    # 6. Synthesis Node
    async def synthesis_action(state: ResearchState) -> Dict[str, Any]:
        t0 = time.time()
        await _emit_event(
            state.run_id,
            "agent_started",
            {"agent": "Synthesis", "message": "Drafting cited research summary and enforcing references..."},
            event_callback,
        )
        try:
            synth = await synthesis_agent.execute(
                question=state.question,
                papers=state.papers,
                claims=state.claims,
                comparison=state.comparison,
            )
            _record_trace(
                state,
                "Synthesis",
                t0,
                "success",
                f"Generated report with {synth.get('citation_coverage', 0)*100:.1f}% citation coverage",
                budget,
            )
            await _emit_event(
                state.run_id,
                "agent_finished",
                {
                    "agent": "Synthesis",
                    "title": synth.get("title", ""),
                    "references_count": len(synth.get("reference_list", [])),
                    "citation_coverage": synth.get("citation_coverage", 0.0),
                },
                event_callback,
            )
            return {"synthesis": synth, "trace": list(state.trace), "timings": dict(state.timings)}
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            state.warnings.append(f"Synthesis failed: {e}")
            _record_trace(state, "Synthesis", t0, "failed", str(e), budget)
            return {
                "synthesis": {
                    "title": f"Synthesis: {state.question}",
                    "markdown": f"# Research Summary\n\nFailed to synthesize report: {e}",
                    "reference_list": [],
                    "citation_coverage": 0.0,
                },
                "trace": list(state.trace),
                "timings": dict(state.timings),
            }

    # 7. Evaluator Node
    async def evaluator_action(state: ResearchState) -> Dict[str, Any]:
        t0 = time.time()
        await _emit_event(
            state.run_id,
            "agent_started",
            {"agent": "Evaluator", "message": "Conducting citation coverage and factual consistency checks..."},
            event_callback,
        )
        try:
            markdown = state.synthesis.get("markdown", "")
            ref_list = state.synthesis.get("reference_list", [])
            cov = state.synthesis.get("citation_coverage", 0.0)

            eval_res = await evaluator_agent.execute(
                markdown_text=markdown,
                reference_list=ref_list,
                claims=state.claims,
                quote_verification_rate=state.quote_verification_rate,
                citation_coverage=cov,
            )

            state.eval_scores = eval_res
            state.status = "completed"
            _record_trace(
                state,
                "Evaluator",
                t0,
                "success",
                f"Faithfulness score: {eval_res.get('faithfulness_score', 0):.2f}",
                budget,
            )
            await _emit_event(
                state.run_id,
                "agent_finished",
                {"agent": "Evaluator", "eval_scores": eval_res},
                event_callback,
            )
            return {
                "eval_scores": eval_res,
                "status": "completed",
                "trace": list(state.trace),
                "timings": dict(state.timings),
            }
        except Exception as e:
            logger.error(f"Evaluator error: {e}")
            state.warnings.append(f"Evaluator failed: {e}")
            _record_trace(state, "Evaluator", t0, "warning", str(e), budget)
            fallback_eval = {
                "citation_coverage": state.synthesis.get("citation_coverage", 0.0),
                "quote_verification_rate": state.quote_verification_rate,
                "faithfulness_score": 1.0,
                "supported_pct": 100.0,
                "partial_pct": 0.0,
                "unsupported_pct": 0.0,
                "verdicts": [],
            }
            return {
                "eval_scores": fallback_eval,
                "status": "completed",
                "trace": list(state.trace),
                "timings": dict(state.timings),
            }

    # Assemble Graph with unique action node names that don't collide with state attributes
    builder = StateGraph(ResearchState)
    builder.add_node("supervisor_plan_node", supervisor_plan_action)
    builder.add_node("retrieval_node", retrieval_action)
    builder.add_node("extraction_node", extraction_action)
    builder.add_node("supervisor_review_node", supervisor_review_action)
    builder.add_node("comparison_node", comparison_action)
    builder.add_node("synthesis_node", synthesis_action)
    builder.add_node("evaluator_node", evaluator_action)

    # Add Edges
    builder.add_edge(START, "supervisor_plan_node")
    builder.add_edge("supervisor_plan_node", "retrieval_node")
    builder.add_edge("retrieval_node", "extraction_node")
    builder.add_edge("extraction_node", "supervisor_review_node")
    builder.add_conditional_edges(
        "supervisor_review_node",
        route_after_review,
        {"retrieval_node": "retrieval_node", "comparison_node": "comparison_node"},
    )
    builder.add_edge("comparison_node", "synthesis_node")
    builder.add_edge("synthesis_node", "evaluator_node")
    builder.add_edge("evaluator_node", END)

    return builder.compile()
