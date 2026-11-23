import sys
import os
import asyncio
import json

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, os.getcwd())

from app.graph.state import ResearchState
from app.graph.build import build_research_graph
from app.clients.llm import LLMClient
from app.config import settings
from app.db.repository import Repository


async def main():
    print("=== STEP 2: RUNNING ONE REAL QUICK RESEARCH QUESTION END TO END ===")
    question = "What is the foundational architecture of the Transformer model and how does self-attention replace recurrence?"
    print(f"Question: {question}")
    print(f"Provider: {settings.LLM_PROVIDER} | Model: {settings.LLM_MODEL}")
    print(f"Vector Store: Qdrant Cloud ({settings.QDRANT_URL})")
    print(f"Relational DB: Neon PostgreSQL")

    import time
    run_id = f"real_quick_run_{int(time.time())}"
    await Repository.create_run(run_id=run_id, question=question, depth="quick", max_papers=3)

    llm = LLMClient(force_mock=False)
    graph = build_research_graph(llm_client=llm)

    state = ResearchState(
        run_id=run_id,
        question=question,
        depth="quick",
        max_papers=3,
        collection_name=f"papers_{run_id}",
    )

    print("\nExecuting multi-agent LangGraph workflow with REAL services...")
    final_state = await graph.ainvoke(state)

    print("\n" + "="*70)
    print("STEP 2 EXECUTION RESULTS")
    print("="*70)

    synth = final_state.get("synthesis", {})
    claims = final_state.get("claims", [])
    dropped = final_state.get("dropped_claims", [])
    verif_rate = final_state.get("quote_verification_rate", 0.0)
    eval_scores = final_state.get("eval_scores", {})
    trace = final_state.get("trace", [])

    print(f"\n1. STATUS: {final_state.get('status')}")
    print(f"2. PAPERS RETRIEVED ({len(final_state.get('papers', []))}):")
    for p in final_state.get("papers", []):
        print(f"   - [{p.get('paper_id')}] {p.get('title')} ({p.get('year')})")

    print(f"\n3. CLAIMS & VERBATIM QUOTE VERIFICATION:")
    print(f"   - Verified Claims Kept: {len(claims)}")
    print(f"   - Dropped Claims: {len(dropped)}")
    print(f"   - Quote Verification Rate: {verif_rate*100:.1f}%")
    for c in claims[:3]:
        print(f"     * [{c.get('claim_id')}] Claim: {c.get('claim')}")
        print(f"       Quote: \"{c.get('quote')}\" (Verified verbatim: {c.get('verified')})")

    print(f"\n4. CITATIONS & SYNTHESIS:")
    print(f"   - Citation Coverage: {synth.get('citation_coverage', 0)*100:.1f}%")
    print(f"   - Total Sentences: {synth.get('total_sentences', 0)}")
    print(f"   - Cited Sentences: {synth.get('cited_sentences', 0)}")
    print(f"   - References Count: {len(synth.get('reference_list', []))}")
    for ref in synth.get("reference_list", []):
        print(f"     {ref.get('citation_text')}")

    print(f"\n5. LLM CALLS & USAGE BUDGET:")
    print(f"   - Real API LLM Calls: {llm.budget_tracker.call_count} / {llm.budget_tracker.max_calls}")
    print(f"   - Disk-Cached Hits: {llm.budget_tracker.cached_count}")

    print(f"\n6. EVALUATION AUDIT:")
    print(f"   - Faithfulness Score: {eval_scores.get('faithfulness_score', 'N/A')}")
    print(f"   - Supported %: {eval_scores.get('supported_pct', 'N/A')}%")
    print(f"   - Partial %: {eval_scores.get('partial_pct', 'N/A')}%")
    print(f"   - Unsupported %: {eval_scores.get('unsupported_pct', 'N/A')}%")

    print(f"\n7. FULL SYNTHESIS REPORT:")
    print("-" * 70)
    print(synth.get("markdown", ""))
    print("-" * 70)


if __name__ == "__main__":
    asyncio.run(main())
