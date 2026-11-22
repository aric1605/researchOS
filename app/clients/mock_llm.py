"""Deterministic mock LLM response generator for unit and integration testing.
Zero API calls, zero quota consumption, fully predictable outputs.
"""

from typing import Any, Dict, List, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def get_mock_text_response(prompt: str, system_instruction: str = "") -> str:
    """Returns deterministic text based on prompt content."""
    prompt_lower = prompt.lower()
    if "evaluate" in prompt_lower or "faithfulness" in prompt_lower:
        return '{"evaluations": [{"sentence": "Transformers utilize self-attention mechanisms.", "verdict": "SUPPORTED", "reason": "Verbatim quote confirms attention mechanism."}]}'
    return "Mock LLM synthesis report for research query."


def get_mock_structured_response(prompt: str, schema: Type[T]) -> T:
    """Generates schema-compliant deterministic mock objects."""
    schema_name = schema.__name__
    prompt_lower = prompt.lower()

    # Supervisor plan / sub-questions
    if "SupervisorPlan" in schema_name or "SubQuestion" in schema_name:
        data = {
            "sub_questions": [
                "What is the foundational architectural design?",
                "What are the empirical performance benchmarks and computational trade-offs?",
                "What are the key limitations and open research directions?",
            ],
            "reasoning": "Plan broken down into architecture, performance, and limitations.",
        }
        return schema.model_validate(data)

    # Retrieval queries
    if "SearchQueries" in schema_name or "QueryPlan" in schema_name:
        data = {
            "queries": [
                "transformer self attention mechanism architecture",
                "attention is all you need neural network scaling",
            ]
        }
        return schema.model_validate(data)

    # Claim extraction
    if "ExtractedClaims" in schema_name or "ClaimList" in schema_name:
        # Note: quote must appear verbatim in the paper chunks for testing!
        # The mock returns a quote that will match test chunks.
        data = {
            "claims": [
                {
                    "claim_id": "c1",
                    "claim": "Self-attention replaces recurrent neural networks with constant sequential operations.",
                    "paper_id": "1706.03762",
                    "chunk_id": "1706.03762_chunk_0",
                    "quote": "Self-attention reduces sequential computation to a constant number of operations.",
                    "stance": "supports",
                    "confidence": 0.95,
                },
                {
                    "claim_id": "c2",
                    "claim": "Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
                    "paper_id": "1706.03762",
                    "chunk_id": "1706.03762_chunk_1",
                    "quote": "Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
                    "stance": "supports",
                    "confidence": 0.98,
                },
            ]
        }
        return schema.model_validate(data)

    # Critical comparison
    if "CriticalComparison" in schema_name or "ComparisonResult" in schema_name:
        data = {
            "themes": [
                {
                    "theme_name": "Attention Efficiency vs Recurrence",
                    "agreements": [
                        "Both models discard sequential recurrent loops in favor of parallelized attention layers."
                    ],
                    "contradictions": [],
                    "claim_ids": ["c1", "c2"],
                }
            ],
            "methodology_differences": [
                "Transformer relies strictly on dot-product attention, whereas traditional architectures rely on recurrent gated memory."
            ],
            "limitations_found": [
                "Quadratic memory complexity with respect to sequence length."
            ],
            "open_questions": [
                "How to achieve sub-quadratic complexity for ultra-long context windows?"
            ],
        }
        return schema.model_validate(data)

    # Synthesis report
    if "SynthesisReport" in schema_name or "ResearchSummary" in schema_name:
        data = {
            "title": "Synthesis of Attention and Transformer Architectures",
            "executive_summary": "Transformers represent a paradigm shift from recurrent architectures to fully parallel self-attention networks [1].",
            "key_findings": [
                "Self-attention reduces sequential computation to a constant number of operations [1].",
                "Multi-head attention enables attending to information from different representation subspaces simultaneously [1].",
            ],
            "critical_comparison": "While recurrent networks process tokens sequentially, Transformers execute parallel matrix multiplications across all positions [1].",
            "limitations_and_contradictions": "The core limitation is quadratic computational complexity in sequence length [1].",
            "open_questions": [
                "Can linear attention approximations match full attention performance across all modalities?"
            ],
            "full_markdown": (
                "# Research Synthesis: Transformer Architectures\n\n"
                "## Executive Summary\n"
                "Transformers represent a paradigm shift from recurrent architectures to fully parallel self-attention networks [1].\n\n"
                "## Key Findings\n"
                "- Self-attention reduces sequential computation to a constant number of operations [1].\n"
                "- Multi-head attention enables attending to information from different representation subspaces simultaneously [1].\n\n"
                "## Critical Comparison & Limitations\n"
                "While recurrent networks process tokens sequentially, Transformers execute parallel matrix multiplications across all positions [1]. "
                "The core limitation is quadratic computational complexity in sequence length [1].\n\n"
                "## References\n"
                "- [1] Vaswani et al. (2017). Attention Is All You Need. https://arxiv.org/abs/1706.03762\n"
            ),
        }
        return schema.model_validate(data)

    # Faithfulness evaluation
    if "FaithfulnessEval" in schema_name or "EvaluationJudge" in schema_name:
        data = {
            "verdicts": [
                {
                    "sentence": "Self-attention reduces sequential computation to a constant number of operations.",
                    "citation": "[1]",
                    "verdict": "SUPPORTED",
                    "reason": "Directly matches claim c1 and source quote.",
                },
                {
                    "sentence": "Multi-head attention enables attending to information from different representation subspaces simultaneously.",
                    "citation": "[1]",
                    "verdict": "SUPPORTED",
                    "reason": "Matches claim c2.",
                },
            ]
        }
        return schema.model_validate(data)

    # Default fallback using model fields
    sample_fields: Dict[str, Any] = {}
    for fname, field in schema.model_fields.items():
        if field.annotation == str:
            sample_fields[fname] = f"Mock {fname}"
        elif field.annotation == int:
            sample_fields[fname] = 1
        elif field.annotation == float:
            sample_fields[fname] = 0.95
        elif field.annotation == bool:
            sample_fields[fname] = True
        elif getattr(field.annotation, "__origin__", None) == list:
            sample_fields[fname] = []
        elif getattr(field.annotation, "__origin__", None) == dict:
            sample_fields[fname] = {}
        else:
            sample_fields[fname] = None
    return schema.model_validate(sample_fields)
