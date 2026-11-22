"""End-to-end smoke test script for ResearchOS.
Runs a complete research job and asserts report generation, citations, and evaluation scores.
"""

import asyncio
import json
import logging
from app.graph.state import ResearchState
from app.graph.build import build_research_graph
from app.clients.llm import LLMClient
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


async def main():
    print("=== ResearchOS End-to-End Smoke Test ===")
    question = "How does the Transformer architecture replace recurrent networks using self-attention mechanisms?"
    print(f"Research Question: {question}")
    print(f"Provider: {settings.LLM_PROVIDER} | Model: {settings.LLM_MODEL}")

    llm = LLMClient()
    graph = build_research_graph(llm_client=llm)

    state = ResearchState(
        run_id="smoke_test_run",
        question=question,
        depth="standard",
        max_papers=3,
    )

    print("\nExecuting graph workflow...")
    result = await graph.ainvoke(state)

    print("\n=== Result Summary ===")
    print(f"Status: {result.get('status')}")
    print(f"Sub-questions planned: {len(result.get('sub_questions', []))}")
    print(f"Papers retrieved: {len(result.get('papers', []))}")
    print(f"Verified claims: {len(result.get('claims', []))}")
    print(f"Dropped claims: {len(result.get('dropped_claims', []))}")

    synth = result.get("synthesis", {})
    print(f"Report Title: {synth.get('title')}")
    print(f"Citation Coverage: {synth.get('citation_coverage', 0)*100:.1f}%")
    print(f"References Count: {len(synth.get('reference_list', []))}")

    eval_scores = result.get("eval_scores", {})
    print(f"Faithfulness Score: {eval_scores.get('faithfulness_score', 'N/A')}")
    print(f"Supported %: {eval_scores.get('supported_pct', 'N/A')}%")

    print("\n=== Generated Report Preview ===")
    print(synth.get("markdown", "")[:600] + "...\n")

    # Assertions
    assert result.get("status") == "completed", "Status must be completed"
    assert len(synth.get("markdown", "")) > 100, "Markdown must not be empty"
    assert len(synth.get("reference_list", [])) > 0, "Must have references"
    assert len(result.get("trace", [])) >= 5, "Trace must log execution steps"
    print("✓ All smoke test assertions passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
